#include <iostream>
#include <vector>
#include <algorithm>
#include <CL/cl.hpp>
#include <CL/cl_ext.h>
#include "oclobject.hpp"
#include "yuv_utils.h"
#define CL_EXT_DECLARE(name) static name##_fn pfn_##name = 0;
#define lambda 400;
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
struct picinfo
{
	int width;
	int height;
	picinfo(int w,int h){width=w;height=h;}
};

class ME
{
	public:
		ME(){};
		ME(int width,int height,int searchPath);
		~ME();
		void ExtractMotionEstimation(cl::Image2D refImage,cl::Image2D srcImage,
				std::vector<MotionVector>& ,
				std::vector<MotionVector> ,
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
		void costfunction(void *ref,void *src,
				std::vector<MotionVector>& ,
				std::vector<MotionVector>& );
		friend void compare_weak(void *ref,void *src,std::vector<MotionVector>&,picinfo info,int *cost,int lam);
		friend void compare(void *ref,void *src,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,picinfo info,int *cost,int lam);
		friend void PyramidME_weak(void *ref,void *src, std::vector<MotionVector> &MVs,picinfo info,int Layers,int *cost,int lam1,int lam2);
		friend void PyramidME_1080p(void *ref,void *src, std::vector<MotionVector> &MVs,std::vector<MotionVector> &ref_MV,picinfo info,int Layers,int *cost,int lam1,int lam2);
		friend void PyramidME_4k(void *ref,void *src, std::vector<MotionVector> &MVs,std::vector<MotionVector> &ref_MV,picinfo info,int Layers,int *cost,int lam1,int lam2);
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
