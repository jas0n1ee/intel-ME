#include "ME.h"
#include <CL/cl.h>
#include <cmath>
#include "yuv_utils.h"
void ME::ComputeNumMVs( cl_uint nMBType, int nPicWidth, int nPicHeight, int & nMVSurfWidth, int & nMVSurfHeight )
{
    // Size of the input frame in pixel blocks (SRC_BLOCK_WIDTH x SRC_BLOCK_HEIGHT each)
    int nPicWidthInBlk  = (nPicWidth + SRC_BLOCK_WIDTH - 1) / SRC_BLOCK_WIDTH;
    int nPicHeightInBlk = (nPicHeight + SRC_BLOCK_HEIGHT - 1) / SRC_BLOCK_HEIGHT;

    if (CL_ME_MB_TYPE_4x4_INTEL == nMBType) {         // Each Src block has 4x4 MVs
        nMVSurfWidth = nPicWidthInBlk * 4;
        nMVSurfHeight = nPicHeightInBlk * 4;
    }
    else if (CL_ME_MB_TYPE_8x8_INTEL == nMBType) {    // Each Src block has 2x2 MVs
        nMVSurfWidth = nPicWidthInBlk * 2;
        nMVSurfHeight = nPicHeightInBlk * 2;
    }
    else if (CL_ME_MB_TYPE_16x16_INTEL == nMBType) {  // Each Src block has 1 MV
        nMVSurfWidth = nPicWidthInBlk;
        nMVSurfHeight = nPicHeightInBlk;
    }
    else
    {
        throw std::runtime_error("Unknown macroblock type");
    }
}
unsigned int ME::ComputeSubBlockSize( cl_uint nMBType )
{
    switch (nMBType)
    {
    case CL_ME_MB_TYPE_4x4_INTEL: return 4;
    case CL_ME_MB_TYPE_8x8_INTEL: return 8;
    case CL_ME_MB_TYPE_16x16_INTEL: return 16;
    default:
        throw std::runtime_error("Unknown macroblock type");
    }
}
ME::ME(int width,int height,int search_path)
{
	this->height=height;
	this->width=width;
   	OpenCLBasic init("Intel", "GPU");
	context = cl::Context(init.context); clRetainContext(init.context);
    device  = cl::Device(init.device);   clRetainDevice(init.device);
    queue = cl::CommandQueue(init.queue);clRetainCommandQueue(init.queue);
	std::string ext = device.getInfo< CL_DEVICE_EXTENSIONS >();
    if (string::npos == ext.find("cl_intel_accelerator") || string::npos == ext.find("cl_intel_motion_estimation"))
    {
        throw Error("Error, the selected device doesn't support motion estimation or accelerator extensions!");
    }

    CL_EXT_INIT_WITH_PLATFORM( init.platform, clCreateAcceleratorINTEL );
    CL_EXT_INIT_WITH_PLATFORM( init.platform, clReleaseAcceleratorINTEL );
	cl_int err = 0;
    const cl_device_id & d = device();
    p=cl::Program(clCreateProgramWithBuiltInKernels( context(), 1, &d, "block_motion_estimate_intel", &err ));
    /*
	if (err != CL_SUCCESS)
    {
        throw cl::Error(err, "Failed creating builtin kernel(s)");
    }
	*/
	cl_uint search_path_type;
	switch(search_path){
	case 2:search_path_type=0;
		break;
	case 4:search_path_type=1;
		break;
	case 16:search_path_type=5;
		break;
	default:search_path_type=0;
	}
	kernel=cl::Kernel(p,"block_motion_estimate_intel");
	
		cl_motion_estimation_desc_intel desc = {
        kMBBlockType,                                     // Number of motion vectors per source pixel block (the value of CL_ME_MB_TYPE_16x16_INTEL specifies  just a single vector per block )
		subpixel_mode,                // Motion vector precision
		sad_adjust_mode,              // SAD Adjust (none/Haar transform) for the residuals, but we don't compute them in this tutorial anyway
		search_path_type              // Search window radius
    };
	accelerator = pfn_clCreateAcceleratorINTEL(context(), CL_ACCELERATOR_TYPE_MOTION_ESTIMATION_INTEL,
        sizeof(cl_motion_estimation_desc_intel), &desc, &err);
    /*
	if (err != CL_SUCCESS)
    {
        throw cl::Error(err, "Error creating motion estimation accelerator object.");
    }
	*/
    ComputeNumMVs(desc.mb_block_type, width, height, mvImageWidth, mvImageHeight);
	cl::ImageFormat imageFormat(CL_R, CL_UNORM_INT8);
	refImage=cl::Image2D(context, CL_MEM_READ_ONLY, imageFormat, width, height, 0,0);
	srcImage=cl::Image2D(context, CL_MEM_READ_ONLY, imageFormat, width, height, 0,0);
    mvBuffer=cl::Buffer(context, CL_MEM_WRITE_ONLY, mvImageWidth * mvImageHeight * sizeof(MotionVector));
	pmv=cl::Buffer(context, CL_MEM_READ_WRITE, mvImageWidth * mvImageHeight * sizeof(MotionVector));
	res=cl::Buffer(context, CL_MEM_READ_WRITE, mvImageWidth * mvImageHeight * sizeof(USHORT));
}
void ME::ExtractMotionEstimation(cl::Image2D refImage,cl::Image2D srcImage,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,USHORT * residuals,bool preMVEnable)
{
	double time=time_stamp();	
	MVs.resize(mvImageWidth * mvImageHeight);
	// Load next picture
	// Schedule full-frame motion estimation
	kernel.setArg(0, accelerator);
	kernel.setArg(1, srcImage);
	kernel.setArg(2, refImage);
	if(preMVEnable) 
	{
		queue.enqueueWriteBuffer(pmv,CL_TRUE,0,sizeof(MotionVector) * mvImageHeight*mvImageWidth,&preMVs[0],0,0); 
		kernel.setArg(3, pmv);
	}
	else kernel.setArg(3, sizeof(cl_mem), NULL);
	kernel.setArg(4, mvBuffer);

	kernel.setArg(5, res); //in this simple tutorial we don't want to compute residuals

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
	queue.finish();
	void * pMVs = &MVs[0];
	queue.enqueueReadBuffer(mvBuffer,CL_TRUE,0,sizeof(MotionVector) * mvImageWidth * mvImageHeight,pMVs,0,0);
	queue.enqueueReadBuffer(res,CL_TRUE,0,sizeof(USHORT) * mvImageWidth * mvImageHeight,residuals,0,0);
    	
	std::cout<<"ExtractME time \t"<<1000*(time_stamp()-time)<<"(ms)\n";
}
void ME::ExtractMotionEstimation_b(void *ref,void *src,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,USHORT * residuals,bool preMVEnable)
{
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
	region[0] = width;
    region[1] = height;
    region[2] = 1;
	MVs.resize(mvImageWidth * mvImageHeight);
	queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
    // Copy to tiled image memory - this copy (and its overhead) is not necessary in a full GPU pipeline
	queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);
	// Load next picture
	// Schedule full-frame motion estimation
	double time=time_stamp();
	kernel.setArg(0, accelerator);
	kernel.setArg(1, srcImage);
	kernel.setArg(2, refImage);
	if(preMVEnable) 
	{
		queue.enqueueWriteBuffer(pmv,CL_TRUE,0,sizeof(MotionVector) * mvImageHeight*mvImageWidth,&preMVs[0],0,0); 
		kernel.setArg(3, pmv);
	}
	else kernel.setArg(3, sizeof(cl_mem), NULL);
	kernel.setArg(4, mvBuffer);

	kernel.setArg(5, res); //in this simple tutorial we don't want to compute residuals

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
	queue.finish();
	void * pMVs = &MVs[0];
	queue.enqueueReadBuffer(mvBuffer,CL_TRUE,0,sizeof(MotionVector) * mvImageWidth * mvImageHeight,pMVs,0,0);
	queue.enqueueReadBuffer(res,CL_TRUE,0,sizeof(USHORT) * mvImageWidth * mvImageHeight,residuals,0,0);
	std::cout<<"ExtractME time \t"<<1000*(time_stamp()-time)<<"(ms)\n";
}
ME::~ME()
{
	pfn_clReleaseAcceleratorINTEL(accelerator);
}

std::vector<MotionVector> moveMV(std::vector<MotionVector> src,int direction,int mvImageHeight,int mvImageWidth)
{
	/*
	1|2|3
	4|0|6
	7|8|9
	*/
	std::vector<MotionVector> result;
	result.resize(mvImageHeight*mvImageWidth);
	switch(direction){
	case 0:
		return src;
		break;
	case 5:
		return src;
		break;
	case 1:
		for(int i=0;i<mvImageHeight-1;i++)
		{
			for(int j=0;j<mvImageWidth-1;j++)
			{
				result[i*mvImageWidth+j]=src[(i+1)*mvImageWidth+j+1];
			}
			result[(i+1)*mvImageWidth-1].s[0]=0;
			result[(i+1)*mvImageWidth-1].s[1]=0;
		}
		for(int j=(mvImageHeight-1)*mvImageWidth;j<mvImageHeight*mvImageWidth;j++)
		{
			result[j].s[0]=0;
			result[j].s[1]=0;
		}
		return result;
		break;
	case 2:
		for(int i=0;i<mvImageHeight-1;i++)
		{
			for(int j=0;j<mvImageWidth;j++)
			{
				result[i*mvImageWidth+j]=src[(i+1)*mvImageWidth+j];
			}
		}
		for(int i=(mvImageHeight-1)*mvImageWidth;i<mvImageWidth*mvImageHeight;i++)
		{
			result[i].s[0]=0;
			result[i].s[1]=0;
		}
		return result;
		break;
	case 4:
		for(int i=0;i<mvImageHeight;i++)
		{
			for(int j=0;j<mvImageWidth-1;j++)
			{
				result[i*mvImageWidth+j]=src[i*mvImageWidth+j+1];
			}
		}
		for(int i=0;i<mvImageHeight;i++)
		{
			result[(i+1)*mvImageWidth-1].s[0]=0;
			result[(i+1)*mvImageWidth-1].s[1]=0;
		}
		return result;
		break;
	case 6:
		for(int i=0;i<mvImageHeight;i++)
		{
			for(int j=1;j<mvImageWidth;j++)
			{
				result[i*mvImageWidth+j]=src[i*mvImageWidth+j-1];
			}
		}
		for(int i=0;i<mvImageHeight;i++)
		{
			result[i*mvImageWidth].s[0]=0;
			result[i*mvImageWidth].s[1]=0;
		}
		return result;
		break;
	case 8:
		for(int i=1;i<mvImageHeight;i++)
		{
			for(int j=0;j<mvImageWidth;j++)
			{
				result[i*mvImageWidth+j]=src[(i-1)*mvImageWidth+j];
			}
		}
		for(int i=0;i<mvImageWidth;i++)
		{
			result[i].s[0]=0;
			result[i].s[1]=0;
		}
		return result;
		break;
	default :return src;
	}
}
