#include <ltiChannel8.h>



//void maxFilterCPU(lti::channel8 &res, const lti::channel8 &img, float &dt_ms);
void maxFilterTrivial(lti::channel8 &res, const lti::channel8 &img, float &dt_ms);
void maxFilterSeparable(lti::channel8 &res, const lti::channel8 &img, float &dt_ms);
void maxFilterSeparableMTH(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms);
void maxFilterSeparableMTHShMem(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms);
//void maxFilterSeparable(lti::channel8 &res, const lti::channel8 &imgCpu, float &dt_ms, lti::channel8 &img_tmp);
