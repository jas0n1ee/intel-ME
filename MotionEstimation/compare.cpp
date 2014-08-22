#include "ME.h"
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
	int pre[2];
	int non[2];
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
void compare(void *ref,void *src,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,ME &me4,ME &me16)
{
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
	region[0] = me4.width;
    region[1] = me4.height;
    region[2] = 1;
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
	double cp_time=time_stamp();
	me4.queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
	me4.queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);
	std::cout<<"cp time\t"<<1000*(time_stamp()-cp_time)<<"\n";
	me4.ExtractMotionEstimation(refImage,srcImage,MV_pre4,preMVs,pre4_res,TRUE);
	
	cp_time=time_stamp();
	//me4.queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
	//me4.queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);
	std::cout<<"cp time\t"<<1000*(time_stamp()-cp_time)<<"\n";
	me4.ExtractMotionEstimation(refImage,srcImage,MV_non,preMVs,non_res,FALSE);

	refImage=me16.refImage;
	srcImage=me16.srcImage;
	cp_time=time_stamp();
	me16.queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
	me16.queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);
	std::cout<<"cp time\t"<<1000*(time_stamp()-cp_time)<<"\n";
	me16.ExtractMotionEstimation(refImage,srcImage,MV_pre16,preMVs,pre16_res,TRUE);

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
//		std::cout<<cost_non<<"\t"<<cost_pre4<<"\t"<<cost_pre16<<"\t"
//				<<non_res[i]<<"\t"<<pre4_res[i]<<"\t"<<pre16_res[i]<<"\t"<<MVbit_non<<"\t"<<MVbit_pre4<<"\t"<<MVbit_pre16<<"\n";
		MVs[i]=(cost_pre4>cost_non)?
			((cost_non>cost_pre16)?MV_pre16[i]:MV_non[i]):MV_pre4[i];
	}
	delete [] non_res;
	delete [] pre16_res;
	delete [] pre4_res;
}
void compare(void *ref,void *src,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,int width,int height)
{
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
		//MVs[i]=(cost_pre4>cost_non)?
		//	((cost_non>cost_pre16)?MV_pre16[i]:MV_non[i]):MV_pre4[i];
		MVs[i]=(cost_pre4>cost_non)?MV_non[i]:MV_pre4[i];
	}
	delete [] non_res;
	delete [] pre16_res;
	delete [] pre4_res;
}
void compare(void *ref,void *src,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,int width,int height,int lam)
{
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
		//MVs[i]=(cost_pre4>cost_non)?MV_non[i]:MV_pre4[i];
	}
	delete [] non_res;
	delete [] pre16_res;
	delete [] pre4_res;
}
void compare_weak(void *ref,void *src,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,int width,int height)
{
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
	int pre4[2];
	int pre16[2];
	
	me4.queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
	me4.queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);

	me4.ExtractMotionEstimation(refImage,srcImage,MV_pre4,preMVs,pre4_res,FALSE);

	refImage=me16.refImage;
	srcImage=me16.srcImage;

	me16.queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
	me16.queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);

	me16.ExtractMotionEstimation(refImage,srcImage,MV_pre16,preMVs,pre16_res,FALSE);

	MVs.resize(mvImageHeight*mvImageWidth);
	for(int i=0;i<mvImageHeight*mvImageWidth;i++)
	{
		pre4[0]=abs(MV_pre4[i].s[0]);
		pre4[1]=abs(MV_pre4[i].s[1]);
		pre16[0]=abs(MV_pre16[i].s[0]);
		pre16[1]=abs(MV_pre16[i].s[1]);
		int MVbit_pre4=(int)(log10(pre4[0]+1)/log10(2)+log10(pre4[1]+1)/log10(2));
		int MVbit_pre16=(int)(log10(pre16[0]+1)/log10(2)+log10(pre16[1]+1)/log10(2));
		int cost_pre4=pre4_res[i]+ MVbit_pre4 * lambda;
		int cost_pre16=pre16_res[i]+ MVbit_pre16 * lambda;
		//std::cout<<cost_non<<"\t"<<cost_pre4<<"\t"<<cost_pre16<<"\t"<<MVbit_non<<"\t"<<MVbit_pre4<<"\t"<<MVbit_pre16<<"\n";
		MVs[i]=(cost_pre4>cost_pre16)?MV_pre16[i]:MV_pre4[i];
	}
	delete [] pre4_res;
	delete [] pre16_res;
}
void compare_pro(void *ref,void *src,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,int width,int height)
{
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
	USHORT * pyr4_res = new USHORT[mvImageHeight*mvImageWidth];
	USHORT * pyr16_res = new USHORT[mvImageHeight*mvImageWidth];
	std::vector<MotionVector> MV_pre4;
	std::vector<MotionVector> MV_pre16;
	std::vector<MotionVector> MV_pyr4;
	std::vector<MotionVector> MV_pyr16;
	std::vector<MotionVector> MV_non;
	std::vector<MotionVector> MV_mov;
	int pre4[2];
	int pre16[2];
	int non[2];
	int pyr4[2];	
	int pyr16[2];	
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

	int Layers=1;
	std::vector<MotionVector> MV;
	std::vector<MotionVector> MV_ref;
	YUVUtils::PlanarImage*ref_l[2];
	YUVUtils::PlanarImage*src_l[2];
	int w=width/pow(2,1);
	int h=height/pow(2,1);
	ME d2(w,h,4);
	ref_l[0]=YUVUtils::CreatePlanarImage(width/pow(2,1),height/pow(2,1));
	src_l[0]=YUVUtils::CreatePlanarImage(width/pow(2,1),height/pow(2,1));
	me4.downsampling(src,src_l[0]->Y);
	me4.downsampling(ref,ref_l[0]->Y);
	if(Layers==2)
	{
		w/=2;
		h/=2;
		ref_l[1]=YUVUtils::CreatePlanarImage(width/pow(2,2),height/pow(2,2));
		src_l[1]=YUVUtils::CreatePlanarImage(width/pow(2,2),height/pow(2,2));
		d2.downsampling(src_l[0]->Y,src_l[1]->Y);
		d2.downsampling(ref_l[0]->Y,ref_l[1]->Y);
		compare_weak(ref_l[1]->Y,src_l[1]->Y,MV,MV_ref,w,h);
		d2.resampling(MV,MV_ref);
		YUVUtils::ReleaseImage(src_l[1]);
		YUVUtils::ReleaseImage(ref_l[1]);
	}
	w=width/pow(2,1);
	h=height/pow(2,1);
	
	if(Layers==2)
		compare(ref_l[0]->Y,src_l[0]->Y,MV,MV_ref,w,h);
	else
		compare_weak(ref_l[0]->Y,src_l[0]->Y,MV,MV_ref,w,h);
	me4.resampling(MV,MV_ref);
	me4.ExtractMotionEstimation_b(ref,src,MV_pyr4,MV_ref,pyr4_res,TRUE);
	me16.ExtractMotionEstimation_b(ref,src,MV_pyr16,MV_ref,pyr16_res,TRUE);
	YUVUtils::ReleaseImage(src_l[0]);
	YUVUtils::ReleaseImage(ref_l[0]);
#ifdef enableMVmov
	int *tcost=new int[mvImageHeight*mvImageWidth];	
	compare_MV(ref,src,MV_mov,preMVs,tcost,width,height);
#endif
	for(int i=0;i<mvImageHeight*mvImageWidth;i++)
	{
		pre4[0]=abs(MV_pre4[i].s[0]-preMVs[i].s[0]);
		pre4[1]=abs(MV_pre4[i].s[1]-preMVs[i].s[1]);
		pre16[0]=abs(MV_pre16[i].s[0]-preMVs[i].s[0]);
		pre16[1]=abs(MV_pre16[i].s[1]-preMVs[i].s[1]);
		non[0]=abs(MV_non[i].s[0]-preMVs[i].s[0]);
		non[1]=abs(MV_non[i].s[1]-preMVs[i].s[1]);
		pyr4[0]=abs(MV_pyr4[i].s[0]-preMVs[i].s[0]);
		pyr4[1]=abs(MV_pyr4[i].s[1]-preMVs[i].s[1]);
		pyr16[0]=abs(MV_pyr16[i].s[0]-preMVs[i].s[0]);
		pyr16[1]=abs(MV_pyr16[i].s[1]-preMVs[i].s[1]);
		int MVbit_pre4=(int)(log10(pre4[0]+1)/log10(2)+log10(pre4[1]+1)/log10(2));
		int MVbit_pre16=(int)(log10(pre16[0]+1)/log10(2)+log10(pre16[1]+1)/log10(2));
		int MVbit_non=(int)(log10(non[0]+1)/log10(2)+log10(non[1]+1)/log10(2));
		int MVbit_pyr4=(int)(log10(pyr4[0]+1)/log10(2)+log10(pyr4[1]+1)/log10(2));
		int MVbit_pyr16=(int)(log10(pyr16[0]+1)/log10(2)+log10(pyr16[1]+1)/log10(2));
		MVbit_non=min(MVbit_non,(int)(log10(abs(MV_non[i].s[0])+1)/log10(2)+log10(abs(MV_non[i].s[1])+1)/log10(2)));
		int cost[6];
		cost[0]=non_res[i]+ MVbit_non * lambda;
		cost[1]=pre4_res[i]+ MVbit_pre4 * lambda;
		cost[2]=pre16_res[i]+ MVbit_pre16 * lambda;
		cost[3]=pyr4_res[i]+ MVbit_pyr4 * lambda;
		cost[4]=pyr16_res[i]+ MVbit_pyr16 * lambda;
#ifdef enableMVmov
		cost[5]=tcost[i];
#endif
		int min=0;
		for(int i=0;i<5;i++) if(cost[i]<cost[min]) min=i;
		switch(min)
		{
		case 0:
			MVs[i]=MV_non[i];
			break;
		case 1:
			MVs[i]=MV_pre4[i];
			break;
		case 2:
			MVs[i]=MV_pre16[i];
			break;
		case 3:
			MVs[i]=MV_pyr4[i];
			break;
		case 4:
			MVs[i]=MV_pyr16[i];
			break;
#ifdef enableMVmov
		case 5:
			MVs[i]=MV_mov[i];
			break;
#endif
		}

		/*
		int cost_pre4=pre4_res[i]+ MVbit_pre4 * lambda;
		int cost_pre16=pre16_res[i]+ MVbit_pre16 * lambda;
		int cost_non=non_res[i]+ MVbit_non * lambda;
		int cost_pyr4=pyr4_res[i]+ MVbit_pyr4 * lambda;
		int cost_pyr16=pyr16_res[i]+ MVbit_pyr16 * lambda;
		*/
		//std::cout<<non_res[i]<<"\t"<<pre4_res[i]<<"\t"<<pre16_res[i]<<"\t"<<pyr4_res[i]<<"\t"<<pyr16_res[i]<<"\t"<<MVbit_non<<"\t"<<MVbit_pre4<<"\t"<<MVbit_pre16<<"\t"<<MVbit_pyr4<<"\t"<<MVbit_pyr16<<"\n";
			
	}

#ifdef enableMVmov
	delete []tcost;
#endif
	delete [] non_res;
	delete [] pre16_res;
	delete [] pre4_res;
	delete [] pyr4_res;
	delete [] pyr16_res;
}
void compare_MV(void *ref,void *src,std::vector<MotionVector>& MVs,std::vector<MotionVector>&preMVs,int *tcost,int width,int height)
{
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
	USHORT * res_l = new USHORT[mvImageHeight*mvImageWidth];
	USHORT * res_r = new USHORT[mvImageHeight*mvImageWidth];
	USHORT * res_u = new USHORT[mvImageHeight*mvImageWidth];
	USHORT * res_p = new USHORT[mvImageHeight*mvImageWidth];
	std::vector<MotionVector> MV_l;
	std::vector<MotionVector> MV_r;
	std::vector<MotionVector> MV_u;
	std::vector<MotionVector> MV_p;
	std::vector<MotionVector> p_MV_l=moveMV(preMVs,4,mvImageHeight,mvImageWidth);
	std::vector<MotionVector> p_MV_r=moveMV(preMVs,6,mvImageHeight,mvImageWidth);
	std::vector<MotionVector> p_MV_u=moveMV(preMVs,2,mvImageHeight,mvImageWidth);
	std::vector<MotionVector> p_MV_p=moveMV(preMVs,8,mvImageHeight,mvImageWidth);
	int l4[2];
	int r4[2];
	int u4[2];
	int p4[2];
	int l16[2];
	int r16[2];
	int u16[2];
	int p16[2];
	me4.queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
	me4.queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);

	me4.ExtractMotionEstimation(refImage,srcImage,MV_l,p_MV_l,res_l,TRUE);
	me4.ExtractMotionEstimation(refImage,srcImage,MV_r,p_MV_r,res_r,TRUE);
	me4.ExtractMotionEstimation(refImage,srcImage,MV_u,p_MV_u,res_u,TRUE);
	me4.ExtractMotionEstimation(refImage,srcImage,MV_p,p_MV_p,res_p,TRUE);

	MVs.resize(mvImageHeight*mvImageWidth);
	for(int i=0;i<mvImageHeight*mvImageWidth;i++)
	{
		l4[0]=abs(MV_l[i].s[0]-p_MV_l[i].s[0]);
		l4[1]=abs(MV_l[i].s[1]-p_MV_l[i].s[1]);
		r4[0]=abs(MV_r[i].s[0]-p_MV_r[i].s[0]);
		r4[1]=abs(MV_r[i].s[1]-p_MV_r[i].s[1]);
		u4[0]=abs(MV_u[i].s[0]-p_MV_u[i].s[0]);
		u4[1]=abs(MV_u[i].s[1]-p_MV_u[i].s[1]);
		p4[0]=abs(MV_p[i].s[0]-p_MV_p[i].s[0]);
		p4[1]=abs(MV_p[i].s[1]-p_MV_p[i].s[1]);
		int MVbit_l4=(int)(log10(l4[0]+1)/log10(2)+log10(l4[1]+1)/log10(2));
		int MVbit_r4=(int)(log10(r4[0]+1)/log10(2)+log10(r4[1]+1)/log10(2));
		int MVbit_u4=(int)(log10(u4[0]+1)/log10(2)+log10(u4[1]+1)/log10(2));
		int MVbit_p4=(int)(log10(p4[0]+1)/log10(2)+log10(p4[1]+1)/log10(2));
		int cost[4];
		cost[0]=res_l[i]+MVbit_l4 * lambda;
		cost[1]=res_r[i]+MVbit_r4 * lambda;
		cost[2]=res_u[i]+MVbit_u4 * lambda;
		cost[3]=res_p[i]+MVbit_p4 * lambda;
		int min=0;
		for(int i=0;i<4;i++) if(cost[i]<cost[min]) min=i;
		switch(min)
		{
		case 0:
			MVs[i]=MV_l[i];
			tcost[i]=cost[0];
			break;
		case 1:
			MVs[i]=MV_r[i];
			tcost[i]=cost[1];
			break;
		case 2:
			MVs[i]=MV_u[i];
			tcost[i]=cost[2];
			break;
		case 3:
			MVs[i]=MV_p[i];
			tcost[i]=cost[3];
			break;
		}
	}


		
	refImage=me16.refImage;
	srcImage=me16.srcImage;

	me16.queue.enqueueWriteImage(refImage, CL_TRUE, origin, region, 0, 0, ref);
	me16.queue.enqueueWriteImage(srcImage, CL_TRUE, origin, region, 0, 0, src);


	me16.ExtractMotionEstimation(refImage,srcImage,MV_l,p_MV_l,res_l,TRUE);
	me16.ExtractMotionEstimation(refImage,srcImage,MV_r,p_MV_r,res_r,TRUE);
	me16.ExtractMotionEstimation(refImage,srcImage,MV_u,p_MV_u,res_u,TRUE);
	me16.ExtractMotionEstimation(refImage,srcImage,MV_p,p_MV_p,res_p,TRUE);
	
	for(int i=0;i<mvImageHeight*mvImageWidth;i++)
	{
		l16[0]=abs(MV_l[i].s[0]-p_MV_l[i].s[0]);
		l16[1]=abs(MV_l[i].s[1]-p_MV_l[i].s[1]);
		r16[0]=abs(MV_r[i].s[0]-p_MV_r[i].s[0]);
		r16[1]=abs(MV_r[i].s[1]-p_MV_r[i].s[1]);
		u16[0]=abs(MV_u[i].s[0]-p_MV_u[i].s[0]);
		u16[1]=abs(MV_u[i].s[1]-p_MV_u[i].s[1]);
		p16[0]=abs(MV_p[i].s[0]-p_MV_p[i].s[0]);
		p16[1]=abs(MV_p[i].s[1]-p_MV_p[i].s[1]);
		int MVbit_l16=(int)(log10(l16[0]+1)/log10(2)+log10(l16[1]+1)/log10(2));
		int MVbit_r16=(int)(log10(r16[0]+1)/log10(2)+log10(r16[1]+1)/log10(2));
		int MVbit_u16=(int)(log10(u16[0]+1)/log10(2)+log10(u16[1]+1)/log10(2));
		int MVbit_p16=(int)(log10(p16[0]+1)/log10(2)+log10(p16[1]+1)/log10(2));
		int cost[5];
		cost[0]=res_l[i]+MVbit_l16 * lambda;
		cost[1]=res_r[i]+MVbit_r16 * lambda;
		cost[2]=res_u[i]+MVbit_u16 * lambda;
		cost[3]=res_p[i]+MVbit_p16 * lambda;
		cost[4]=tcost[i];
		int min=0;
		for(int i=0;i<5;i++) if(cost[i]<cost[min]) min=i;
		switch(min)
		{
		case 0:
			MVs[i]=MV_l[i];
			tcost[i]=cost[0];
			break;
		case 1:
			MVs[i]=MV_r[i];
			tcost[i]=cost[1];
			break;
		case 2:
			MVs[i]=MV_u[i];
			tcost[i]=cost[2];
			break;
		case 3:
			MVs[i]=MV_p[i];
			tcost[i]=cost[3];
			break;
		}
	}
	delete []res_l;
	delete []res_r;
	delete []res_u;
	delete []res_p;
}