/*
 * Copyright (C) 2007 by Pablo Alvarado
 *
 * This file is part of the LTI-Computer Vision Library 2 (LTI-Lib-2)
 *
 * The LTI-Lib-2 is free software; you can redistribute it and/or
 * modify it under the terms of the BSD License.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the authors nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file   surfLocalDescriptor.cpp
 *         Contains an example of use for the class lti::surfLocalDescriptor
 * \author Pablo Alvarado
 * \date   04.11.2007
 * revisions ..: $Id: matrixTransform.cpp,v 1.5 2011-08-29 14:17:33 alvarado Exp $
 */

// LTI-Lib Headers
#include "ltiObject.h"
#include "ltiIOImage.h"
#include "ltiMath.h"
#include "ltiPassiveWait.h"

#include "ltiMatrixTransform.h"
#include "ltiBilinearInterpolation.h"

#include "ltiLispStreamHandler.h"
#include <ltiMaximumFilter.h>

#include "ltiViewer2D.h" // The normal viewer
typedef lti::viewer2D viewer_type;

// Standard Headers
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <stdio.h>

using std::cout;
using std::cerr;
using std::endl;

#include "project.hpp"

using namespace std::chrono;
using namespace std;




typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::microseconds us;
typedef std::chrono::duration<float> fsec;

void maxFilterCPU(lti::channel8 &res, const lti::channel8 &img){
  lti::maximumFilter<lti::channel8::value_type>kernel(5);
  kernel.apply(img,res);
}

void maxFilterTrivialKernelCPU(lti::channel8 &imgDst, const lti::channel8 &imgSrc)
{
    for(int32_t j=0; j<imgSrc.rows(); j++)
    {
        for(int32_t i=0; i<imgSrc.columns(); i++)
        {
            uint8_t max = imgSrc.at(j,i);

            for(int32_t a=i- KERNEL_RADIUS; a<i+ KERNEL_RADIUS; a++)
            {
                for(int32_t b=j- KERNEL_RADIUS; b<j+ KERNEL_RADIUS; b++)
                {
                    uint8_t value=max;
                    if(a >= 0 && a < imgSrc.columns() && b >= 0 && b < imgSrc.rows())
                        value = imgSrc.at(b,a);
                    if(value > max)
                        max = value;
                }
            }
            imgDst.at(j,i) = max;
        }
    }
}

void L2norm(const lti::channel8 &img_base, const lti::channel8 &img_mod, double &L2norm_result){
  double sum = 0;
  double delta = 0;

  for (uint32_t i = 0; i < img_base.rows(); i++)
  {
    for (uint32_t j = 0; j < img_base.columns(); j++) {
      delta += (img_base.at(i,j) - img_mod.at(i,j)) * (img_base.at(i,j) - img_mod.at(i,j));
      sum   += img_base.at(i,j) * img_mod.at(i,j);
    }
  }
  L2norm_result = sqrt(delta / sum);
}

/*
 * Help
 */
void usage() {
  cout << "Usage: matrixTransform [image] [-h]" << endl;
  cout << "Try some image transformations on the given image\n";
  cout << "  -h show this help." << endl;
}

/*
 * Parse the line command arguments
 */
void parseArgs(int argc, char*argv[],
               std::string& filename) {

  filename.clear();
  // check each argument of the command line
  for (int i=1; i<argc; i++) {
    if (*argv[i] == '-') {
      switch (argv[i][1]) {
        case 'h':
          usage();
          exit(EXIT_SUCCESS);
          break;
        default:
          break;
      }
    } else {
      filename = argv[i]; // guess that this is the filename
    }
  }
}

/*
 * Main method
 */
int main(int argc, char* argv[]) {
  high_resolution_clock::time_point t0, t1;
  fsec elapsed_time;
  ofstream time_vs_kernel_file, time_vs_img_size_file;
  std::string filename_time_vs_kernel = "time_vs_kernel.csv";
  std::string filename_time_vs_img_size = "time_vs_img_size.csv";

  static const char* confFile = "project.dat";

  std::string imgFile;
  parseArgs(argc,argv,imgFile);

  lti::ioImage loader; // used to load an image file

  lti::image imgRgba;
  lti::channel8 img;
  if (!loader.load(imgFile,imgRgba)) {
    std::cerr << "Could not read " << imgFile << ": "
              << loader.getStatusString()
              << std::endl;
    usage();
    exit(EXIT_FAILURE);
  }

  // Convert to grayscale
  img.castFrom(imgRgba);

  std::ifstream in(confFile);
  bool write=true;

  // Configuration variables
  float dummy = 80;

  if (in) {
    lti::lispStreamHandler lsh;
    lsh.use(in);
    write=!lti::read(lsh,"dummy",dummy);
  }
  if (write) {
    // something went wrong loading the data, so just write again to fix
    // the errors
    std::ofstream out(confFile);
    lti::lispStreamHandler lsh;
    lsh.use(out);
    lti::write(lsh,"dummy",dummy);
    out<<std::endl;
  }

  lti::channel8 res, res_cpu_lti_lib, res_separable, res_separable_mth,res_separable_mth_shmem,res_cpu_trivial;
  res.resize(img.rows(), img.columns(), 0);
  res_cpu_lti_lib.resize(img.rows(), img.columns(), 0);
  res_cpu_trivial.resize(img.rows(), img.columns(), 0);
  res_separable.resize(img.rows(), img.columns(), 0);
  res_separable_mth.resize(img.rows(), img.columns(), 0);
  res_separable_mth_shmem.resize(img.rows(), img.columns(), 0);
  //tmp.resize(img.rows(), img.columns(), 0);

  lti::viewer2D view_trivial_GPU("Transformed Trivial GPU");
  lti::viewer2D view_trivial_CPU("Transformed Trivial CPU");
  lti::viewer2D view_separable("Transformed Separable");
  lti::viewer2D view_separable_mth("Transformed Separable + MTH");
  lti::viewer2D view_separable_mth_shmem("Transformed Separable + MTH + Shared Memory");
  lti::viewer2D view_origianl("Original");
  lti::viewer2D view_cpu_lti_lib("Transformed CPU LTI::lib");
  //lti::viewer2D view_tmp("Transformed TMP");
  lti::viewer2D::interaction action;
  lti::ipoint pos;

  bool showTransformed= true;
  float dt_ms;
  double l2norm_result;

  uint num_kernel = 5;
  uint num_interactions = 30;

  float data_array [num_kernel][num_interactions];

  do {
    for (uint i = 0; i <= num_kernel; i++){
      if (i == 0) {
        data_array[i][0] = (float)i;
        for(uint j = 0; j <= num_interactions; j++){
          t0 = Time::now();
          maxFilterCPU(res_cpu_lti_lib, img);
          t1 = Time::now();
          elapsed_time = t1 - t0;
          dt_ms = elapsed_time.count();
          data_array[i][j+1] = dt_ms;
        }
      }
      else if (i == 1) {
        data_array[i][0] = (float)i;
        for(uint j = 0; j <= num_interactions; j++){
          t0 = Time::now();
          maxFilterTrivialKernelCPU(res_cpu_trivial, img);
          t1 = Time::now();
          elapsed_time = t1 - t0;
          dt_ms = elapsed_time.count();
          data_array[i][j+1] = dt_ms;
        }
      }
      else if (i == 2) {
        data_array[i][0] = (float)i;
        for(uint j = 0; j <= num_interactions; j++){
          maxFilterTrivial(res, img, dt_ms);
          data_array[i][j+1] = dt_ms;

        }
      }
      else if (i == 3) {
        data_array[i][0] = (float)i;
        for(uint j = 0; j <= num_interactions; j++){
          maxFilterSeparable(res_separable, img, dt_ms);
          data_array[i][j+1] = dt_ms;

        }
      }
      else if (i == 4) {
        data_array[i][0] = (float)i;
        for(uint j = 0; j <= num_interactions; j++){
          maxFilterSeparableMTH(res_separable_mth, img, dt_ms);
          data_array[i][j+1] = dt_ms;

        }
      }
      else{
        data_array[i][0] = (float)i;
        for(uint j = 0; j <= num_interactions; j++){
          maxFilterSeparableMTHShMem(res_separable_mth_shmem, img, dt_ms);
          data_array[i][j+1] = dt_ms;
        }
      }
    }

    // t0 = Time::now();
    // maxFilterCPU(res_cpu_lti_lib, img);
    // t1 = Time::now();
    // elapsed_time = t1 - t0;
    // dt_ms = elapsed_time.count();
    // std::cout << "elapsed time cpu lti::lib: " << dt_ms << '\n';
    // t0 = Time::now();
    // maxFilterTrivialKernelCPU(res_cpu_trivial, img);
    // t1 = Time::now();
    // elapsed_time = t1 - t0;
    // dt_ms = elapsed_time.count();
    // std::cout << "elapsed time cpu trivial: " << dt_ms << '\n';
    // maxFilterTrivial(res, img, dt_ms);
    // std::cout << "elapsed time gpu trivial: " << dt_ms << '\n';
    //
    // maxFilterSeparable(res_separable, img, dt_ms);
    // std::cout << "elapsed time gpu Separable: " << dt_ms << '\n';
    //maxFilterSeparableMTH(res_separable_mth, img, dt_ms);
    // std::cout << "elapsed time gpu Separable + MTH: " << dt_ms << '\n';
    // maxFilterSeparableMTHShMem(res_separable_mth_shmem, img, dt_ms);
    // std::cout << "elapsed time gpu Separable + MTH + Shared Memory: " << dt_ms << '\n';

    L2norm(img, res, l2norm_result);
    printf("L2 norm for img-trivial: %E\n", l2norm_result);

    L2norm(res, res_separable, l2norm_result);
    printf("L2 norm for trivial-separable: %E\n", l2norm_result);

    L2norm(res, res_separable_mth, l2norm_result);
    printf("L2 norm for trivial-separable + MTH: %E\n", l2norm_result);

    L2norm(res, res_separable_mth_shmem, l2norm_result);
    printf("L2 norm for trivial-separable + MTH + Shared Mem: %E\n", l2norm_result);

    L2norm(res, res_cpu_trivial, l2norm_result);
    printf("L2 norm for trivial gpu - trivial cpu: %E\n", l2norm_result);

    L2norm(res_cpu_lti_lib, res_cpu_trivial, l2norm_result);
    printf("L2 norm for lti - trivial cpu: %E\n", l2norm_result);

    L2norm(res_cpu_lti_lib, res_separable_mth_shmem, l2norm_result);
    printf("L2 norm for lti - separable + MTH + Shared Mem: %E\n", l2norm_result);

    view_trivial_CPU.show(res_cpu_trivial);
    view_trivial_GPU.show(res);
    view_cpu_lti_lib.show(res_cpu_lti_lib);
    view_separable.show(res_separable);
    view_separable_mth.show(res_separable_mth);
    view_separable_mth_shmem.show(res_separable_mth_shmem);
    view_origianl.show(img);

    time_vs_kernel_file.open(filename_time_vs_kernel, std::ofstream::out | std::ofstream::app);

    if (time_vs_kernel_file.is_open()) {
      for (uint i = 0; i < num_interactions; i++) {
        for (uint j = 0; j <= num_kernel; j++) {
          time_vs_kernel_file << data_array[j][i] << ",";
        }
        time_vs_kernel_file << "\n";
      }
    }
    time_vs_kernel_file.close();

    if(view_trivial_GPU.waitButtonReleased(action, pos))
    {
        std::cout << "click" << std::endl;
        showTransformed = !showTransformed;
    }
  } while(action.action != lti::viewer2D::Closed);



  return EXIT_SUCCESS;
}
