#include <iostream>
#include <vector>
#include <algorithm>
#include <CL/cl.hpp>
#include <CL/cl_ext.h>
#include "oclobject.hpp"
#define CL_EXT_DECLARE(name) static name##_fn pfn_##name = 0;

#define CL_EXT_INIT_WITH_PLATFORM(platform, name) { \
    pfn_##name = (name##_fn) clGetExtensionFunctionAddressForPlatform(platform, #name); \
    if (! pfn_##name ) \
        { \
        std::cout<<"ERROR: can't get handle to function pointer " <<#name<< ", wrong driver version?\n"; \
        } \
};

static const cl_uint kMBBlockType = CL_ME_MB_TYPE_8x8_INTEL;
#define subpixel_mode		CL_ME_SUBPIXEL_MODE_QPEL_INTEL
#define sad_adjust_mode		CL_ME_SAD_ADJUST_MODE_HAAR_INTEL
#define search_path_type	CL_ME_SEARCH_PATH_RADIUS_4_4_INTEL


CL_EXT_DECLARE( clCreateAcceleratorINTEL );
CL_EXT_DECLARE( clReleaseAcceleratorINTEL );

#define SRC_BLOCK_WIDTH 16
#define SRC_BLOCK_HEIGHT 16

typedef cl_short2 MotionVector;

class ME
{
	public:
		ME(int width,int height);
		~ME();
		void ExtractMotionEstimation(void *,void *,
				std::vector<MotionVector>& ,
				std::vector<MotionVector>& ,
				bool);
		void ComputeNumMVs( cl_uint nMBType, int nPicWidth, int nPicHeight, int & nMVSurfWidth, int & nMVSurfHeight );
		unsigned int ComputeSubBlockSize( cl_uint nMBType );

	private:
		int height,width;
		int mvImageHeight,mvImageWidth;
		cl::Context context;
		cl::Device device;
		cl::CommandQueue queue;
		cl::Kernel kernel;
		cl_accelerator_intel accelerator;
		cl::Image2D refImage;
		cl::Image2D srcImage;
		cl::Buffer mvBuffer;
		cl::Buffer pmv;
		
};	
