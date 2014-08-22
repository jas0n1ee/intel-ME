#include <iostream>
#include <vector>
#include <algorithm>
#include <CL/cl.hpp>
#include <CL/cl_ext.h>
#include "oclobject.hpp"
#include "yuv_utils.h"
#define CL_EXT_DECLARE(name) static name##_fn pfn_##name = 0;
#define lambda 700;
//#define enableMVmov
#define CL_EXT_INIT_WITH_PLATFORM(platform, name) { \
    pfn_##name = (name##_fn) clGetExtensionFunctionAddressForPlatform(platform, #name); \
    if (! pfn_##name ) \
        { \
        std::cout<<"ERROR: can't get handle to function pointer " <<#name<< ", wrong driver version?\n"; \
        } \
};

static const cl_uint kMBBlockType = CL_ME_MB_TYPE_16x16_INTEL;
#define subpixel_mode		CL_ME_SUBPIXEL_MODE_QPEL_INTEL
#define sad_adjust_mode		CL_ME_SAD_ADJUST_MODE_HAAR_INTEL


CL_EXT_DECLARE( clCreateAcceleratorINTEL );
CL_EXT_DECLARE( clReleaseAcceleratorINTEL );

#define SRC_BLOCK_WIDTH 16
#define SRC_BLOCK_HEIGHT 16

typedef cl_short2 MotionVector;
typedef unsigned char      uint8_t;
class ME
{
	public:
		ME(){};
		ME(int width,int height,int searchPath);
		~ME();
		void ExtractMotionEstimation(cl::Image2D refImage,cl::Image2D srcImage,
				std::vector<MotionVector>& ,
				std::vector<MotionVector>& ,
				USHORT*,
				bool);
		void ExtractMotionEstimation_b(void *ref,void *src,
				std::vector<MotionVector>& ,
				std::vector<MotionVector>& ,
				USHORT*,
				bool);
		void ComputeNumMVs( cl_uint nMBType, int nPicWidth, int nPicHeight, int & nMVSurfWidth, int & nMVSurfHeight );
		unsigned int ComputeSubBlockSize( cl_uint nMBType );
		void downsampling(void *src,void *det);
		void downsampling(std::vector<MotionVector>&src,std::vector<MotionVector>&det);
		void resampling(void *src,void *det);
		void resampling(std::vector<MotionVector>&src,std::vector<MotionVector>&det);
		friend std::vector<MotionVector> moveMV(std::vector<MotionVector> src,int direction,int mvImageHeight,int mvImageWidth);
		void costfunction(void *ref,void *src,
				std::vector<MotionVector>& ,
				std::vector<MotionVector>& );
		friend void compare(void *ref,void *src,std::vector<MotionVector>&,std::vector<MotionVector>&,ME &me4,ME &me16);
		friend void compare(void *ref,void *src,std::vector<MotionVector>&,std::vector<MotionVector>&,int width,int height);
		friend void compare_weak(void *ref,void *src,std::vector<MotionVector>&,std::vector<MotionVector>&,int width,int height);
		friend void compare_pro(void *ref,void *src,std::vector<MotionVector>&,std::vector<MotionVector>&,int width,int height);
		friend void compare_MV(void *ref,void *src,std::vector<MotionVector>&,std::vector<MotionVector>&,int *,int width,int height);
		friend void PyramidME(void *ref,void *src,std::vector<MotionVector>& ,ME &me, int );
		friend void PyramidME_weak(void *ref,void *src,std::vector<MotionVector>& ,ME &me,int);
		friend void PyramidME_pro(void *ref,void *src,std::vector<MotionVector>& MVs,std::vector<MotionVector> &ref_MV,ME &me,int);
	private:
		int height,width;
		int mvImageHeight,mvImageWidth;
		cl::CommandQueue queue;
		cl::Context context;
		cl::Device device;
		cl::Program p;
		cl::Kernel kernel;
		cl_accelerator_intel accelerator;
		cl_motion_estimation_desc_intel desc;
		cl::Image2D refImage;
		cl::Image2D srcImage;
		cl::Buffer mvBuffer;
		cl::Buffer pmv;
		cl::Buffer res;
		
};	
