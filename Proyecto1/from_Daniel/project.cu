#include "project.hpp"
#include <cstdint>

struct CudaImage
{
    uint8_t *data;
    uint32_t width;
    uint32_t height;
    size_t pitch;
};

#define N 16

__global__ void maxFilterTrivialKernel(CudaImage imgDst, const CudaImage imgSrc)
{
    for(int32_t j=0; j<imgSrc.height; j++)
    {
        for(int32_t i=0; i<imgSrc.width; i++)
        {
            uint8_t max = imgSrc.data[j*imgSrc.pitch + i];

            for(int32_t a=i-2; a<i+2; a++)
            {
                for(int32_t b=j-2; b<j+2; b++)
                {
                    uint8_t value=max;
                    if(a >= 0 && a < imgSrc.width && b >= 0 && b < imgSrc.height)
                        value = imgSrc.data[b*imgSrc.pitch + a];
                    if(value > max)
                        max = value;
                }
            }
            imgDst.data[j*imgDst.pitch + i] = max;
        }
    }
}

void maxFilterTrivial(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms)
{
    CudaImage imgSrc, imgDst;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    imgSrc.width = imgDst.width = imgCpu.columns();
    imgSrc.height = imgDst.height = imgCpu.rows();

    cudaMallocPitch(&imgSrc.data, &imgSrc.pitch, imgCpu.columns(), imgCpu.rows());
    cudaMallocPitch(&imgDst.data, &imgDst.pitch, imgCpu.columns(), imgCpu.rows());



    cudaMemcpy2D(imgSrc.data, imgSrc.pitch, imgCpu.data(), imgCpu.columns(), imgCpu.columns(), imgCpu.rows(), cudaMemcpyHostToDevice);

    cudaEventRecord(event1, 0);
    maxFilterTrivialKernel<<<1,1>>>(imgDst,imgSrc);
    cudaEventRecord(event2, 0);


    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);

    cudaEventElapsedTime(&dt_ms, event1, event2);

    cudaMemcpy2D(res.data(), res.columns(), imgDst.data, imgDst.pitch, res.columns(), res.rows(), cudaMemcpyDeviceToHost);
    cudaFree(&imgSrc);
    cudaFree(&imgDst);
}

__global__ void maxFilterSeparableKernel(CudaImage imgDst, const CudaImage imgSrc, CudaImage tmp)
{
  for (int32_t j = 0; j < imgSrc.height; j++) {
    for (size_t i = 0; i < imgSrc.width; i++) {
      uint8_t max = imgSrc.data[j*imgSrc.pitch + i];
      for (int32_t b = j-2; b < j+2; b++) {
        uint8_t value = max;
        if(b >= 0 && b < imgSrc.height)
          value = imgSrc.data[b*imgSrc.pitch + i];
        if(value > max)
          max = value;
      }
      tmp.data[j*tmp.pitch + i] = max;

    }

  }

  for (int32_t j = 0; j < tmp.height; j++) {
    for (size_t i = 0; i < tmp.width; i++) {
      uint8_t max = tmp.data[j*tmp.pitch + i];
      for (int32_t a = i-2; a < i+2; a++) {
        uint8_t value = max;
        if(a >= 0 && a < tmp.width)
          value = tmp.data[j*tmp.pitch + a];
        if(value > max)
          max = value;
      }
      imgDst.data[j*imgDst.pitch + i] = max;

    }
  }
}

void maxFilterSeparable(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms)
{
    CudaImage imgSrc, imgDst, tmp;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    imgSrc.width = imgDst.width =  tmp.width = imgCpu.columns();
    imgSrc.height = imgDst.height = tmp.height = imgCpu.rows();

    cudaMallocPitch(&imgSrc.data, &imgSrc.pitch, imgCpu.columns(), imgCpu.rows());
    cudaMallocPitch(&imgDst.data, &imgDst.pitch, imgCpu.columns(), imgCpu.rows());
    cudaMallocPitch(&tmp.data, &tmp.pitch, imgCpu.columns(), imgCpu.rows());


    cudaMemcpy2D(imgSrc.data, imgSrc.pitch, imgCpu.data(), imgCpu.columns(), imgCpu.columns(), imgCpu.rows(), cudaMemcpyHostToDevice);


    // for (size_t i = 0; i < 10; i++) {
    //   maxFilterSeparableKernel<<<1,1>>>(imgDst,imgSrc, tmp);
    // }
    cudaEventRecord(event1, 0);
    maxFilterSeparableKernel<<<1,1>>>(imgDst,imgSrc, tmp);
    cudaEventRecord(event2, 0);


    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);

    cudaEventElapsedTime(&dt_ms, event1, event2);

    cudaMemcpy2D(res.data(), res.columns(), imgDst.data, imgDst.pitch, res.columns(), res.rows(), cudaMemcpyDeviceToHost);

    cudaFree(&imgSrc);
    cudaFree(&imgDst);
    cudaFree(&tmp);
}

__global__ void maxFilterSeparableMTHKernel(CudaImage imgDst, const CudaImage imgSrc, CudaImage tmp)
{
  uint8_t max;
  uint i = threadIdx.x + blockIdx.x * blockDim.x;
  uint j = threadIdx.y + blockIdx.y * blockDim.y;
  uint Idx = j*imgSrc.pitch + i;

  max = imgSrc.data[Idx];
  for (int32_t b = j-2; b < j+2; b++) {
    uint8_t value = max;
    if(b >= 0 && b < imgSrc.height)
      value = imgSrc.data[b*imgSrc.pitch + i];
    if(value > max)
      max = value;
    }
  tmp.data[Idx] = max;

  __syncthreads();
  max = tmp.data[Idx];
  for (int32_t a = i-2; a < i+2; a++) {
    uint8_t value = max;
    if(a >= 0 && a < tmp.width)
      value = tmp.data[j*tmp.pitch + a];
    if(value > max)
      max = value;
    }
  imgDst.data[Idx] = max;



}

void maxFilterSeparableMTH(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms)
{
    CudaImage imgSrc, imgDst, tmp;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    //uint N = 32;

    dim3 blocks(N,N);
    blocks.x = (imgCpu.columns()+N-1)/N;
    blocks.y = (imgCpu.rows()+N-1)/N;
    dim3 threads(N,N);

    imgSrc.width = imgDst.width =  tmp.width = imgCpu.columns();
    imgSrc.height = imgDst.height = tmp.height = imgCpu.rows();

    cudaMallocPitch(&imgSrc.data, &imgSrc.pitch, imgCpu.columns(), imgCpu.rows());
    cudaMallocPitch(&imgDst.data, &imgDst.pitch, imgCpu.columns(), imgCpu.rows());
    cudaMallocPitch(&tmp.data, &tmp.pitch, imgCpu.columns(), imgCpu.rows());


    cudaMemcpy2D(imgSrc.data, imgSrc.pitch, imgCpu.data(), imgCpu.columns(), imgCpu.columns(), imgCpu.rows(), cudaMemcpyHostToDevice);

    cudaEventRecord(event1, 0);
    //maxFilterSeparableMTHKernel<<<(imgCpu.columns()+N-1)/N,N>>>(imgDst,imgSrc, tmp);
    maxFilterSeparableMTHKernel<<<blocks,threads>>>(imgDst,imgSrc, tmp);
    cudaEventRecord(event2, 0);


    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);

    cudaEventElapsedTime(&dt_ms, event1, event2);

    cudaMemcpy2D(res.data(), res.columns(), imgDst.data, imgDst.pitch, res.columns(), res.rows(), cudaMemcpyDeviceToHost);

    cudaFree(&imgSrc);
    cudaFree(&imgDst);
    cudaFree(&tmp);
}


__global__ void maxFilterSeparableMTHShMemKernel(CudaImage imgDst, const CudaImage imgSrc, CudaImage tmp)
{
  uint8_t max_val;
  int x = blockIdx.x*16 + threadIdx.x - 2;
  int y = blockIdx.y*16 + threadIdx.y - 2;

  __shared__ uint8_t MemShared[(N+4)*(N+4)];

  x = max(0,x);
  x = min(x, imgSrc.width-1);
  x = max(y,0);
  x = min(y, imgSrc.height-1);

  uint index = y*imgSrc.width + x;
  uint bindex =  threadIdx.y * blockDim.y + threadIdx.x;

  MemShared[bindex] = imgSrc.data[index];

  __syncthreads();

  // if ((threadIdx.x >= 2) && (threadIdx.x < (N+4) - 2) && (threadIdx.y >= 2) && (threadIdx.y < (N+4) - 2)){
  //   float sum = 0;
  //   for(int dy=-2; dy <= 2; dy++){
  //     for(int dx=-2; dx <= 2; dx++){
  //       float i = MemShared[bindex + (dy*blockDim.x) + dx];
  //       sum += i;
  //     }
  //   }
  //   imgDst.data[index]=sum/25;
  // }
  //__shared__ uint8_t MemShared_tmp[N+4][N+4];
  // if(i<2 || i>=imgSrc.width-2 || j<2 || j>=imgSrc.height-2)
  // {return;}
  //
  // uint shY = threadIdx.y + 2;
  // uint shX = threadIdx.x + 2;
  //
  // /*Only The boundary threads of Thread-Block will do extra effort of padding*/
  // if (threadIdx.x==0 || threadIdx.x==blocks.x-1 || threadIdx.y==0 || threadIdx.y==blocks.y-1){}
  // else{
  //   MemShared[shY][shX] = imgSrc.data[Idx];
  // }
  // __syncthreads();
  //
  max_val = MemShared[index];
  for (int32_t b = y-2; b < y+2; b++) {
    uint8_t value = max_val;
    if(b >= 0)
      value = MemShared[bindex + (b*blockDim.x) + x];
    if(value > max_val)
      max_val = value;
    }
  imgDst.data[index] = max_val;
  //
  // // __syncthreads();
  // // max = tmp.data[Idx];
  // // for (int32_t a = i-2; a < i+2; a++) {
  // //   uint8_t value = max;
  // //   if(a >= 0 && a < tmp.width)
  // //     value = tmp.data[j*tmp.pitch + a];
  // //   if(value > max)
  // //     max = value;
  // //   }
  // // imgDst.data[Idx] = max;



}

void maxFilterSeparableMTHShMem(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms)
{
    CudaImage imgSrc, imgDst, tmp;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    //uint N = 32;

    dim3 blocks;
    blocks.x = (imgCpu.columns()+N-1)/N;
    blocks.y = (imgCpu.rows()+N-1)/N;
    dim3 threads(N,N);

    imgSrc.width = imgDst.width =  tmp.width = imgCpu.columns();
    imgSrc.height = imgDst.height = tmp.height = imgCpu.rows();

    cudaMallocPitch(&imgSrc.data, &imgSrc.pitch, imgCpu.columns(), imgCpu.rows());
    cudaMallocPitch(&imgDst.data, &imgDst.pitch, imgCpu.columns(), imgCpu.rows());
    cudaMallocPitch(&tmp.data, &tmp.pitch, imgCpu.columns(), imgCpu.rows());


    cudaMemcpy2D(imgSrc.data, imgSrc.pitch, imgCpu.data(), imgCpu.columns(), imgCpu.columns(), imgCpu.rows(), cudaMemcpyHostToDevice);

    cudaEventRecord(event1, 0);
    maxFilterSeparableMTHShMemKernel<<<blocks,threads>>>(imgDst,imgSrc,tmp);
    cudaEventRecord(event2, 0);


    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);

    cudaEventElapsedTime(&dt_ms, event1, event2);

    cudaMemcpy2D(res.data(), res.columns(), imgDst.data, imgDst.pitch, res.columns(), res.rows(), cudaMemcpyDeviceToHost);

    cudaFree(&imgSrc);
    cudaFree(&imgDst);
    cudaFree(&tmp);
}
