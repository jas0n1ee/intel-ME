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
void ME::ExtractMotionEstimation(cl::Image2D refImage,cl::Image2D srcImage,std::vector<MotionVector>& MVs,std::vector<MotionVector>preMVs,USHORT * residuals,bool preMVEnable)
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
void ME::costfunction(void *ref,void *src,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs)
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
	queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);
	USHORT * pre_res = new USHORT[mvImageHeight*mvImageWidth];
	USHORT * non_res = new USHORT[mvImageHeight*mvImageWidth];
	std::vector<MotionVector> MV_pre;
	std::vector<MotionVector> MV_non;
	MotionVector zero;
	zero.s[0]=0;
	zero.s[1]=0;
	ExtractMotionEstimation(refImage,srcImage,MV_pre,preMVs,pre_res,TRUE);
	ExtractMotionEstimation(refImage,srcImage,MV_non,preMVs,non_res,FALSE);
	MVs.resize(mvImageHeight*mvImageWidth);
	for(int i=0;i<mvImageHeight*mvImageWidth;i++)
	{
		int cost_pre=pre_res[i];
		int cost_non=non_res[i];
		MVs[i]=(cost_pre>cost_non)?zero:preMVs[i];
	}
}
void compare(void *ref,void *src,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,picinfo info,int *cost,int lam)
{
	int width=info.width;
	int height=info.height;
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
	region[0] = width;
    region[1] = height;
    region[2] = 1;
	ME me4(width,height,4);
	ME me16(width,height,16);
	cl::Image2D refImage(me4.refImage);
	cl::Image2D srcImage(me4.srcImage);
	int mvImageHeight=me4.mvImageHeight;
	int mvImageWidth=me4.mvImageWidth;
	USHORT * pre4_res = new USHORT[mvImageHeight*mvImageWidth];
	USHORT * pre16_res = new USHORT[mvImageHeight*mvImageWidth];
	USHORT * non_res = new USHORT[mvImageHeight*mvImageWidth];
	std::vector<MotionVector> MV_pre4;
	std::vector<MotionVector> MV_pre16;
	std::vector<MotionVector> MV_non;
	int pre4[2];
	int pre16[2];
	int non[2];
	
	me4.queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
	me4.queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);

	me4.ExtractMotionEstimation(refImage,srcImage,MV_pre4,preMVs,pre4_res,TRUE);
	me4.ExtractMotionEstimation(refImage,srcImage,MV_non,preMVs,non_res,FALSE);

	refImage=me16.refImage;
	srcImage=me16.srcImage;

	me16.queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
	me16.queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);

	me16.ExtractMotionEstimation(refImage,srcImage,MV_pre16,preMVs,pre16_res,TRUE);

	MVs.resize(mvImageHeight*mvImageWidth);
	bool isCostNull=0;
	if(cost==NULL)
	{
		cost=new int [mvImageHeight*mvImageWidth];
		isCostNull=1;
	}
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
		int cost_pre4=pre4_res[i]+ MVbit_pre4 * lam;
		int cost_pre16=pre16_res[i]+ MVbit_pre16 * lam;
		int cost_non=non_res[i]+ MVbit_non * lam;
		//std::cout<<cost_non<<"\t"<<cost_pre4<<"\t"<<cost_pre16<<"\t"<<MVbit_non<<"\t"<<MVbit_pre4<<"\t"<<MVbit_pre16<<"\n";
		MVs[i]=(cost_pre4>cost_non)?
			((cost_non>cost_pre16)?MV_pre16[i]:MV_non[i]):MV_pre4[i];
		cost[i]=(cost_pre4>cost_non)?
			((cost_non>cost_pre16)?cost_pre16:cost_non):cost_pre4;
		//MVs[i]=(cost_pre4>cost_non)?MV_non[i]:MV_pre4[i];
	}
	if(isCostNull) delete [] cost;
	delete [] non_res;
	delete [] pre16_res;
	delete [] pre4_res;
}
void compare_weak(void *ref,void *src,std::vector<MotionVector>& MVs,picinfo info,int *cost,int lam)
{
	int width=info.width;
	int height=info.height;
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
	region[0] = width;
    region[1] = height;
    region[2] = 1;
	ME me4(width,height,4);
	ME me16(width,height,16);
	cl::Image2D refImage(me4.refImage);
	cl::Image2D srcImage(me4.srcImage);
	int mvImageHeight=me4.mvImageHeight;
	int mvImageWidth=me4.mvImageWidth;
	USHORT * pre4_res = new USHORT[mvImageHeight*mvImageWidth];
	USHORT * pre16_res = new USHORT[mvImageHeight*mvImageWidth];
	std::vector<MotionVector> MV_pre4;
	std::vector<MotionVector> MV_pre16;
	std::vector<MotionVector> null_Vector;
	int pre4[2];
	int pre16[2];
	
	me4.queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
	me4.queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);

	me4.ExtractMotionEstimation(refImage,srcImage,MV_pre4,null_Vector,pre4_res,FALSE);

	refImage=me16.refImage;
	srcImage=me16.srcImage;

	me16.queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
	me16.queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);

	me16.ExtractMotionEstimation(refImage,srcImage,MV_pre16,null_Vector,pre16_res,FALSE);

	MVs.resize(mvImageHeight*mvImageWidth);
	bool isCostNull=0;
	if(cost==NULL)
	{
		isCostNull=1;
		cost = new int [mvImageHeight*mvImageWidth];
	}

	for(int i=0;i<mvImageHeight*mvImageWidth;i++)
	{
		pre4[0]=abs(MV_pre4[i].s[0]);
		pre4[1]=abs(MV_pre4[i].s[1]);
		pre16[0]=abs(MV_pre16[i].s[0]);
		pre16[1]=abs(MV_pre16[i].s[1]);
		int MVbit_pre4=(int)(log10(pre4[0]+1)/log10(2)+log10(pre4[1]+1)/log10(2));
		int MVbit_pre16=(int)(log10(pre16[0]+1)/log10(2)+log10(pre16[1]+1)/log10(2));
		int cost_pre4=pre4_res[i]+ MVbit_pre4 * lam;
		int cost_pre16=pre16_res[i]+ MVbit_pre16 * lam;
		//std::cout<<cost_non<<"\t"<<cost_pre4<<"\t"<<cost_pre16<<"\t"<<MVbit_non<<"\t"<<MVbit_pre4<<"\t"<<MVbit_pre16<<"\n";
		MVs[i]=(cost_pre4>cost_pre16)?MV_pre16[i]:MV_pre4[i];
		cost[i]=(cost_pre4>cost_pre16)?cost_pre16:cost_pre4;
	}
	if(isCostNull) delete [] cost;
	delete [] pre4_res;
	delete [] pre16_res;
}
void PyramidME_weak(void *ref,void *src, std::vector<MotionVector> &MVs,picinfo info,int Layers,int *cost,int lam1,int lam2)
{
	std::vector<MotionVector> MV;
	std::vector<MotionVector> MV_ref;
	YUVUtils::PlanarImage*ref_l[2];
	YUVUtils::PlanarImage*src_l[2];
	int width=info.width;
	int height=info.height;
	int w=width/pow(2,1);
	int h=height/pow(2,1);
	ME me(width,height,2);
	ME d2(w,h,4);
	ref_l[0]=YUVUtils::CreatePlanarImage(width/pow(2,1),height/pow(2,1));
	src_l[0]=YUVUtils::CreatePlanarImage(width/pow(2,1),height/pow(2,1));
	me.downsampling(src,src_l[0]->Y);
	me.downsampling(ref,ref_l[0]->Y);
	bool isCostNull=0;
	if(Layers==2)
	{
		w/=2;
		h/=2;
		ref_l[1]=YUVUtils::CreatePlanarImage(width/pow(2,2),height/pow(2,2));
		src_l[1]=YUVUtils::CreatePlanarImage(width/pow(2,2),height/pow(2,2));
		d2.downsampling(src_l[0]->Y,src_l[1]->Y);
		d2.downsampling(ref_l[0]->Y,ref_l[1]->Y);
		compare_weak(ref_l[1]->Y,src_l[1]->Y,MV,picinfo(w,h),NULL,lam1);
		d2.resampling(MV,MV_ref);
		YUVUtils::ReleaseImage(src_l[1]);
		YUVUtils::ReleaseImage(ref_l[1]);
	}
	w=width/pow(2,1);
	h=height/pow(2,1);
	if(Layers==2)
		compare(ref_l[0]->Y,src_l[0]->Y,MV,MV_ref,picinfo(w,h),NULL,lam1);
	else
		compare_weak(ref_l[0]->Y,src_l[0]->Y,MV,picinfo(w,h),NULL,lam1);
	me.resampling(MV,MV_ref);
	compare(ref,src,MVs,MV_ref,picinfo(width,height),cost,lam2);
	YUVUtils::ReleaseImage(src_l[0]);
	YUVUtils::ReleaseImage(ref_l[0]);
	
}
void PyramidME_1080p(void *ref,void *src, std::vector<MotionVector> &MVs,std::vector<MotionVector> &ref_MV,picinfo info,int Layers,int *cost,int lam1,int lam2)
{
	std::vector<MotionVector> MV;
	std::vector<MotionVector> MV_sub;
	std::vector<MotionVector> MV_d2;
	std::vector<MotionVector> MV_d4;
	std::vector<MotionVector> MV_d8;
	YUVUtils::PlanarImage*ref_l[3];
	YUVUtils::PlanarImage*src_l[3];
	int width=info.width;
	int height=info.height;
	int w=width/pow(2,1);
	int h=height/pow(2,1);
	ME me(width,height,2);
	ME d2(w,h,4);
	ref_l[0]=YUVUtils::CreatePlanarImage(width/pow(2,1),height/pow(2,1));
	src_l[0]=YUVUtils::CreatePlanarImage(width/pow(2,1),height/pow(2,1));
	me.downsampling(src,src_l[0]->Y);
	me.downsampling(ref,ref_l[0]->Y);
	me.downsampling(ref_MV,MV_d2);
	if(Layers>=2)
	{
		w/=2;
		h/=2;
		ME d4(w,h,2);

		ref_l[1]=YUVUtils::CreatePlanarImage(width/pow(2,2),height/pow(2,2));
		src_l[1]=YUVUtils::CreatePlanarImage(width/pow(2,2),height/pow(2,2));
		d2.downsampling(src_l[0]->Y,src_l[1]->Y);
		d2.downsampling(ref_l[0]->Y,ref_l[1]->Y);
		d2.downsampling(MV_d2,MV_d4);
		if(Layers==3)
		{
			ME d8(w/2,h/2,2);
			ref_l[2]=YUVUtils::CreatePlanarImage(width/pow(2,3),height/pow(2,3));
			src_l[2]=YUVUtils::CreatePlanarImage(width/pow(2,3),height/pow(2,3));
			d4.downsampling(src_l[1]->Y,src_l[2]->Y);
			d4.downsampling(ref_l[1]->Y,ref_l[2]->Y);
			d4.downsampling(MV_d4,MV_d8);
			d8.costfunction(ref_l[2]->Y,src_l[2]->Y,MV_sub,MV_d8);
			compare(ref_l[2]->Y,src_l[2]->Y,MV,MV_sub,picinfo(w/2,h/2),NULL,lam1);
			d4.resampling(MV,MV_sub);
		}
		else d4.costfunction(ref_l[1]->Y,src_l[1]->Y,MV_sub,MV_d4);
		compare(ref_l[1]->Y,src_l[1]->Y,MV,MV_sub,picinfo(w,h),NULL,lam1);
		d2.resampling(MV,MV_sub);
		YUVUtils::ReleaseImage(src_l[1]);
		YUVUtils::ReleaseImage(ref_l[1]);
	}
	else
	{
		d2.costfunction(ref_l[0]->Y,src_l[0]->Y,MV_sub,MV_d2);
	}
	w=width/pow(2,1);
	h=height/pow(2,1);
	compare(ref_l[0]->Y,src_l[0]->Y,MV,MV_sub,picinfo(w,h),NULL,lam1);
	me.resampling(MV,MV_sub);
	compare(ref,src,MVs,MV_sub,picinfo(width,height),cost,lam2);
	YUVUtils::ReleaseImage(src_l[0]);
	YUVUtils::ReleaseImage(ref_l[0]);
}

	