#include <ltiChannel8.h>

#define N 32
#define KERNEL_RADIUS 2
#define ROWS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 8
#define ROWS_RESULT_STEPS 1
#define ROWS_HALO_STEPS 1
#define COLUMNS_BLOCKDIM_X 16
#define COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 1
#define COLUMNS_HALO_STEPS 1

void maxFilterTrivial(lti::channel8 &res, const lti::channel8 &img, float &dt_ms);
void maxFilterSeparable(lti::channel8 &res, const lti::channel8 &img, float &dt_ms);
void maxFilterSeparableMTH(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms);
void maxFilterSeparableMTHShMem(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms);
//void maxFilterTrivialKernelCPU(lti::channel8 &imgDst, const lti::channel8 &imgSrc);
