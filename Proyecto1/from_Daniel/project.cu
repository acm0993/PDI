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

void maxFilterTrivial(lti::channel8 &res, const lti::channel8 &imgCpu)
{
    CudaImage imgSrc, imgDst;

    imgSrc.width = imgDst.width = imgCpu.columns();
    imgSrc.height = imgDst.height = imgCpu.rows();

    cudaMallocPitch(&imgSrc.data, &imgSrc.pitch, imgCpu.columns(), imgCpu.rows());
    cudaMallocPitch(&imgDst.data, &imgDst.pitch, imgCpu.columns(), imgCpu.rows());

    cudaMemcpy2D(imgSrc.data, imgSrc.pitch, imgCpu.data(), imgCpu.columns(), imgCpu.columns(), imgCpu.rows(), cudaMemcpyHostToDevice);

    maxFilterTrivialKernel<<<1,1>>>(imgDst,imgSrc);

    cudaMemcpy2D(res.data(), res.columns(), imgDst.data, imgDst.pitch, res.columns(), res.rows(), cudaMemcpyDeviceToHost);
}
