#include <ltiChannel8.h>


void maxFilterTrivial(lti::channel8 &res, const lti::channel8 &img, float &dt_ms);
void maxFilterSeparable(lti::channel8 &res, const lti::channel8 &img, float &dt_ms);
void maxFilterSeparableMTH(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms);
void maxFilterSeparableMTHShMem(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms);
void maxFilterTrivialKernelCPU(lti::channel8 &imgDst, const lti::channel8 &imgSrc);
