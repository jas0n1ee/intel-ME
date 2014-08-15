#include "ME.h"
#include <CL/cl.h>
#include <cmath>
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
void ME::ExtractMotionEstimation(void *src,void *ref,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,USHORT * residuals,bool preMVEnable)
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
    	
}
ME::~ME()
{
	pfn_clReleaseAcceleratorINTEL(accelerator);
}
void ME::downsampling(void*s,void*d)
{
	uint8_t * src=(unsigned char *)s;
	uint8_t * det=(unsigned char *)d;
	int w2=width/2;
	for(int i=0;i<height/2;i++)
	{
		for(int j=0;j<width/2;j++)
		{
			det[i*w2+j]=(src[i*2*width+j*2]+src[i*2*width+j*2+1]+src[(i*2+1)*width+j*2]+src[(i*2+1)*width+j*2+1]);
		}
	}
}

void ME::resampling(void*s,void*d)
{
	uint8_t * src=(unsigned char *)s;
	uint8_t * det=(unsigned char *)d;
	int w2=width/2;
	for(int i=0;i<height/2;i++)
	{
		for(int j=0;j<width/2;j++)
		{
			det[2*i*width+j*2]=src[i*w2+j];
			det[2*i*width+j*2+1]=src[i*w2+j];
			det[(2*i+1)*width+j*2]=src[i*w2+j];
			det[(2*i+1)*width+j*2+1]=src[i*w2+j];
		}
	}
}
void ME::resampling(std::vector<MotionVector>&src,std::vector<MotionVector>&det)
{
	int w2=mvImageWidth/2;
	det.resize(mvImageHeight*mvImageWidth);
	for(int i=0;i<mvImageHeight/2;i++)
	{
		for(int j=0;j<mvImageWidth/2;j++)
		{
			det[2*i*mvImageWidth+j*2].s[0]=src[i*w2+j].s[0]*2;
			det[2*i*mvImageWidth+j*2].s[1]=src[i*w2+j].s[1]*2;
			det[2*i*mvImageWidth+j*2+1].s[0]=src[i*w2+j].s[0]*2;
			det[2*i*mvImageWidth+j*2+1].s[1]=src[i*w2+j].s[1]*2;
			det[(2*i+1)*mvImageWidth+j*2].s[0]=src[i*w2+j].s[0]*2;
			det[(2*i+1)*mvImageWidth+j*2].s[1]=src[i*w2+j].s[1]*2;
			det[(2*i+1)*mvImageWidth+j*2+1].s[0]=src[i*w2+j].s[0]*2;
			det[(2*i+1)*mvImageWidth+j*2+1].s[1]=src[i*w2+j].s[1]*2;
		}
	}
}
void ME::costfunction(void *src,void *ref,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs)
{
	USHORT * pre_res = new USHORT[mvImageHeight*mvImageWidth];
	USHORT * non_res = new USHORT[mvImageHeight*mvImageWidth];
	std::vector<MotionVector> MV_pre;
	std::vector<MotionVector> MV_non;
	std::vector<MotionVector> MV_l;
	std::vector<MotionVector> MV_r;
	std::vector<MotionVector> MV_u;
	std::vector<MotionVector> MV_d;
	int pre[2];
	int non[2];
	ExtractMotionEstimation(src,ref,MV_pre,preMVs,pre_res,TRUE);
	ExtractMotionEstimation(src,ref,MV_non,preMVs,non_res,FALSE);
	MVs.resize(mvImageHeight*mvImageWidth);
	for(int i=0;i<mvImageHeight*mvImageWidth;i++)
	{
		pre[0]=abs(MV_pre[i].s[0]-preMVs[i].s[0]);
		pre[1]=abs(MV_pre[i].s[1]-preMVs[i].s[1]);
		non[0]=abs(MV_non[i].s[0]-preMVs[i].s[0]);
		non[1]=abs(MV_non[i].s[1]-preMVs[i].s[1]);
		int MVbit_pre=(int)(log10(pre[0]+1)/log10(2)+log10(pre[1]+1)/log10(2));
		int MVbit_non=(int)(log10(non[0]+1)/log10(2)+log10(non[1]+1)/log10(2));
		MVbit_non=min(MVbit_non,(int)(log10(MV_non[i].s[0]+1)/log10(2)+log10(MV_non[i].s[1]+1)/log10(2)));
		int cost_pre=pre_res[i]+ MVbit_pre * lambda;
		int cost_non=non_res[i]+ MVbit_non * lambda;
		MVs[i]=(cost_pre>cost_non)?MV_non[i]:MV_pre[i];
	}
}
std::vector<MotionVector> ME::moveMV(std::vector<MotionVector> src,int direction)
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
void compare(void *src,void *ref,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,ME &me4,ME &me16)
{
	int mvImageHeight=me4.mvImageHeight;
	int mvImageWidth=me4.mvImageWidth;
	USHORT * pre4_res = new USHORT[mvImageHeight*mvImageWidth];
	USHORT * pre16_res = new USHORT[mvImageHeight*mvImageWidth];
	USHORT * non_res = new USHORT[mvImageHeight*mvImageWidth];
	std::vector<MotionVector> MV_pre4;
	std::vector<MotionVector> MV_pre16;
	std::vector<MotionVector> MV_non;
	std::vector<MotionVector> MV_r;
	std::vector<MotionVector> MV_u;
	std::vector<MotionVector> MV_d;
	int pre4[2];
	int pre16[2];
	int non[2];
	me4.ExtractMotionEstimation(src,ref,MV_pre4,preMVs,pre4_res,TRUE);
	me16.ExtractMotionEstimation(src,ref,MV_pre16,preMVs,pre16_res,TRUE);
	me4.ExtractMotionEstimation(src,ref,MV_non,preMVs,non_res,FALSE);
	MVs.resize(mvImageHeight*mvImageWidth);
	for(int i=0;i<mvImageHeight*mvImageWidth;i++)
	{
		pre4[0]=abs(MV_pre4[i].s[0]-preMVs[i].s[0]);
		pre4[1]=abs(MV_pre4[i].s[1]-preMVs[i].s[1]);
		pre16[0]=abs(MV_pre16[i].s[0]-preMVs[i].s[0]);
		pre16[1]=abs(MV_pre16[i].s[1]-preMVs[i].s[1]);
		non[0]=abs(MV_non[i].s[0]-preMVs[i].s[0]);
		non[1]=abs(MV_non[i].s[1]-preMVs[i].s[1]);
		int MVbit_pre4=(int)(log10(pre4[0]+1)/log10(2)+log10(pre4[1]+1)/log10(2));
		int MVbit_pre16=(int)(log10(pre16[0]+1)/log10(2)+log10(pre16[1]+1)/log10(2));
		int MVbit_non=(int)(log10(non[0]+1)/log10(2)+log10(non[1]+1)/log10(2));
		MVbit_non=min(MVbit_non,(int)(log10(abs(MV_non[i].s[0])+1)/log10(2)+log10(abs(MV_non[i].s[1])+1)/log10(2)));
		int cost_pre4=pre4_res[i]+ MVbit_pre4 * lambda;
		int cost_pre16=pre16_res[i]+ MVbit_pre16 * lambda;
		int cost_non=non_res[i]+ MVbit_non * lambda;
		//std::cout<<cost_non<<"\t"<<cost_pre4<<"\t"<<cost_pre16<<"\t"<<MVbit_non<<"\t"<<MVbit_pre4<<"\t"<<MVbit_pre16<<"\n";
		MVs[i]=(cost_pre4>cost_non)?
			((cost_non>cost_pre16)?MV_pre16[i]:MV_non[i]):MV_pre4[i];
	}
}