#include "project.hpp"
#include <cstdint>

struct CudaImage
{
    uint8_t *data;
    uint32_t width;
    uint32_t height;
    size_t pitch;
};


__global__ void maxFilterTrivialKernel(CudaImage imgDst, const CudaImage imgSrc)
{
    for(int32_t j=0; j<imgSrc.height; j++)
    {
        for(int32_t i=0; i<imgSrc.width; i++)
        {
            uint8_t max = imgSrc.data[j*imgSrc.pitch + i];

            for(int32_t a=i- KERNEL_RADIUS; a<i+ KERNEL_RADIUS; a++)
            {
                for(int32_t b=j- KERNEL_RADIUS; b<j+ KERNEL_RADIUS; b++)
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
    for (int32_t i = 0; i < imgSrc.width; i++) {
      uint8_t max = imgSrc.data[j*imgSrc.pitch + i];
      for (int32_t b = j- KERNEL_RADIUS; b < j+ KERNEL_RADIUS; b++) {
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
    for (int32_t i = 0; i < tmp.width; i++) {
      uint8_t max = tmp.data[j*tmp.pitch + i];
      for (int32_t a = i- KERNEL_RADIUS; a < i+ KERNEL_RADIUS; a++) {
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

__global__ void maxFilterSeparableMTHKernel_cols(CudaImage imgDst, const CudaImage imgSrc)//, CudaImage tmp)
{
  uint8_t max = 0,value = 0;
  uint i = threadIdx.x + blockIdx.x * blockDim.x;
  uint j = threadIdx.y + blockIdx.y * blockDim.y;
  uint Idx = j*imgSrc.pitch + i;

  max = imgSrc.data[Idx];
  for (int32_t b = j- KERNEL_RADIUS; b < j+ KERNEL_RADIUS; b++) {
    value = imgSrc.data[b*imgSrc.pitch + i];
    if(value > max)
      max = value;
    }
    imgDst.data[Idx] = max;

}

__global__ void maxFilterSeparableMTHKernel_rows(CudaImage imgDst, const CudaImage imgSrc)//, CudaImage tmp)
{
  uint8_t max = 0,value = 0;
  uint i = threadIdx.x + blockIdx.x * blockDim.x;
  uint j = threadIdx.y + blockIdx.y * blockDim.y;
  uint Idx = j*imgSrc.pitch + i;

  max = imgSrc.data[Idx];
  for (int32_t b = i- KERNEL_RADIUS; b < i+ KERNEL_RADIUS; b++) {
    value = imgSrc.data[j*imgSrc.pitch + b];
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
    maxFilterSeparableMTHKernel_rows<<<blocks,threads>>>(tmp,imgSrc);
    maxFilterSeparableMTHKernel_cols<<<blocks,threads>>>(imgDst,tmp);
    cudaEventRecord(event2, 0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);

    cudaEventElapsedTime(&dt_ms, event1, event2);

    cudaMemcpy2D(res.data(), res.columns(), imgDst.data, imgDst.pitch, res.columns(), res.rows(), cudaMemcpyDeviceToHost);

    cudaFree(&imgSrc);
    cudaFree(&imgDst);
    cudaFree(&tmp);
}

__global__ void maxFilterSeparableMTHShMemKernel_Rows(CudaImage imgDst, const CudaImage imgSrc)
{
    __shared__ uint8_t s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;


    int offsetSrc = baseY * imgSrc.pitch + baseX;
    int offsetDst = baseY * imgSrc.pitch + baseX;

    //Load main data
//#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = imgSrc.data[offsetSrc + i * ROWS_BLOCKDIM_X];
    }

    //Load left halo
//#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? imgSrc.data[offsetSrc + i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
//#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imgSrc.width - baseX > i * ROWS_BLOCKDIM_X) ? imgSrc.data[offsetSrc + i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();
    uint8_t max = 0;//s_Data[threadIdx.y][threadIdx.x];
    uint8_t value = 0;
//#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        value = max;

//#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            value = s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
            if(value > max)
              max = value;
        }

        imgDst.data[offsetDst + i * ROWS_BLOCKDIM_X] = max;
    }
}

__global__ void maxFilterSeparableMTHShMemKernel_Cols(CudaImage imgDst, const CudaImage imgSrc)
{
    __shared__ uint8_t s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    int offsetSrc = baseY * imgSrc.pitch + baseX;
    int offsetDst = baseY * imgSrc.pitch + baseX;

    //Main data
//#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = imgSrc.data[offsetSrc + i * COLUMNS_BLOCKDIM_Y * imgSrc.pitch];
    }

    //Upper halo
//#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? imgSrc.data[offsetSrc + i * COLUMNS_BLOCKDIM_Y * imgSrc.pitch] : 0;
    }

    //Lower halo
//#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imgSrc.height - baseY > i * COLUMNS_BLOCKDIM_Y) ? imgSrc.data[offsetSrc + i * COLUMNS_BLOCKDIM_Y * imgSrc.pitch] : 0;
    }

    //Compute and store results
    __syncthreads();

    uint8_t max = 0;//s_Data[threadIdx.y][threadIdx.x];
    uint8_t value = 0;
//#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        value = max;
//#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            value = s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
            if(value > max)
              max = value;
        }

        imgDst.data[offsetDst + i * COLUMNS_BLOCKDIM_Y * imgSrc.pitch] = max;
    }

}


void maxFilterSeparableMTHShMem(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms)
{
    CudaImage imgSrc, imgDst, tmp;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    //uint N = 32;

    /*dim3 blocks;
    blocks.x = (imgCpu.columns()+N-1)/N;
    blocks.y = (imgCpu.rows()+N-1)/N;
    dim3 threads(N,N);*/

    dim3 blocks_row(imgCpu.columns() / (ROWS_RESULT_STEPS *ROWS_BLOCKDIM_X), imgCpu.rows() / ROWS_BLOCKDIM_Y);
    dim3 threads_row(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
    dim3 blocks_col(imgCpu.columns() / COLUMNS_BLOCKDIM_X, imgCpu.rows() / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads_col(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    imgSrc.width = imgDst.width =  tmp.width = imgCpu.columns();
    imgSrc.height = imgDst.height = tmp.height = imgCpu.rows();

    cudaMallocPitch(&imgSrc.data, &imgSrc.pitch, imgCpu.columns(), imgCpu.rows());
    cudaMallocPitch(&imgDst.data, &imgDst.pitch, imgCpu.columns(), imgCpu.rows());
    cudaMallocPitch(&tmp.data, &tmp.pitch, imgCpu.columns(), imgCpu.rows());


    cudaMemcpy2D(imgSrc.data, imgSrc.pitch, imgCpu.data(), imgCpu.columns(), imgCpu.columns(), imgCpu.rows(), cudaMemcpyHostToDevice);

    cudaEventRecord(event1, 0);
    //maxFilterSeparableMTHShMemKernel<<<blocks,threads>>>(imgDst,imgSrc,tmp);
    maxFilterSeparableMTHShMemKernel_Rows<<<blocks_row,threads_row>>>(tmp,imgSrc);
    maxFilterSeparableMTHShMemKernel_Cols<<<blocks_col,threads_col>>>(imgDst,tmp);
    cudaEventRecord(event2, 0);


    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);

    cudaEventElapsedTime(&dt_ms, event1, event2);

    cudaMemcpy2D(res.data(), res.columns(), imgDst.data, imgDst.pitch, res.columns(), res.rows(), cudaMemcpyDeviceToHost);

    cudaFree(&imgSrc);
    cudaFree(&imgDst);
    cudaFree(&tmp);
}
