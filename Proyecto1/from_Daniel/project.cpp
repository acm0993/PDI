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

using std::cout;
using std::cerr;
using std::endl;

#include "project.hpp"

using namespace std::chrono;




  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::microseconds us;
  typedef std::chrono::duration<float> fsec;

void maxFilterCPU(lti::channel8 &res, const lti::channel8 &img){
  lti::maximumFilter<lti::channel8::value_type>kernel(5);
  kernel.apply(img,res);
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

  lti::channel8 res, res_cpu, res_separable, res_separable_mth,res_separable_mth_shmem;
  res.resize(img.rows(), img.columns(), 0);
  res_cpu.resize(img.rows(), img.columns(), 0);
  res_separable.resize(img.rows(), img.columns(), 0);
  res_separable_mth.resize(img.rows(), img.columns(), 0);
  res_separable_mth_shmem.resize(img.rows(), img.columns(), 0);
  //tmp.resize(img.rows(), img.columns(), 0);

  lti::viewer2D view_trivial("Transformed Trivial");
  lti::viewer2D view_separable("Transformed Separable");
  lti::viewer2D view_separable_mth("Transformed Separable + MTH");
  lti::viewer2D view_separable_mth_shmem("Transformed Separable + MTH + Shared Memory");
  lti::viewer2D view_origianl("Original");
  lti::viewer2D view_cpu("Transformed CPU");
  //lti::viewer2D view_tmp("Transformed TMP");
  lti::viewer2D::interaction action;
  lti::ipoint pos;

  bool showTransformed= true;
  float dt_ms;
  do {
    // Apply algorithm;
    t0 = Time::now();
    maxFilterCPU(res_cpu, img);
    t1 = Time::now();
    elapsed_time = t1 - t0;
    std::cout << "elapsed time cpu: " << elapsed_time.count() << '\n';
    maxFilterTrivial(res, img, dt_ms);
    std::cout << "elapsed time gpu trivial: " << dt_ms << '\n';

    maxFilterSeparable(res_separable, img, dt_ms);
    std::cout << "elapsed time gpu Separable: " << dt_ms << '\n';
    maxFilterSeparableMTH(res_separable_mth, img, dt_ms);
    std::cout << "elapsed time gpu Separable + MTH: " << dt_ms << '\n';
    maxFilterSeparableMTHShMem(res_separable_mth_shmem, img, dt_ms);
    std::cout << "elapsed time gpu Separable + MTH + Shared Memory: " << dt_ms << '\n';


    // Show
    //if(showTransformed)
        view_trivial.show(res);
        view_cpu.show(res_cpu);
        view_separable.show(res_separable);
        view_separable_mth.show(res_separable_mth);
        view_separable_mth_shmem.show(res_separable_mth_shmem);


        //view_tmp.show(tmp);
    //else
        view_origianl.show(img);

    if(view_trivial.waitButtonReleased(action, pos))
    {
        std::cout << "click" << std::endl;
        showTransformed = !showTransformed;
    }
  } while(action.action != lti::viewer2D::Closed);

  return EXIT_SUCCESS;
}
