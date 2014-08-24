#include "ME.h"

void PyramidME(void *ref,void *src, std::vector<MotionVector> &MVs,ME &me, int Layers)
{
	//std::vector<ME> Pyramid;
	//std::vector<YUVUtils::PlanarImage *>ref_l;
	//std::vector<YUVUtils::PlanarImage *>src_l;
	std::vector<MotionVector> MV;
	std::vector<MotionVector> MV_ref;
	ME *Pyramid=new ME[Layers+1];
	YUVUtils::PlanarImage**ref_l=new YUVUtils::PlanarImage* [Layers+1];
	YUVUtils::PlanarImage**src_l=new YUVUtils::PlanarImage* [Layers+1];
	int width=me.width;
	int height=me.height;
	int w=width/pow(2,1);
	int h=height/pow(2,1);
	Pyramid[0]=ME(width,height,4);
	Pyramid[1]=ME(w,h,4);
	ref_l[1]=YUVUtils::CreatePlanarImage(width/pow(2,1),height/pow(2,1));
	src_l[1]=YUVUtils::CreatePlanarImage(width/pow(2,1),height/pow(2,1));
	Pyramid[0].downsampling(src,src_l[1]->Y);
	Pyramid[0].downsampling(ref,ref_l[1]->Y);

	for(int i=2;i<=Layers;i++)
	{
		Pyramid[i]=ME(width/pow(2,i),height/pow(2,i),4);
		ref_l[i]=YUVUtils::CreatePlanarImage(width/pow(2,i),height/pow(2,i));
		src_l[i]=YUVUtils::CreatePlanarImage(width/pow(2,i),height/pow(2,i));
		Pyramid[i-1].downsampling(src_l[i-1]->Y,src_l[i]->Y);
		Pyramid[i-1].downsampling(ref_l[i-1]->Y,ref_l[i]->Y);
	}
	Pyramid[Layers].ExtractMotionEstimation_b(ref_l[Layers]->Y,src_l[Layers]->Y,MV,MV_ref,NULL,FALSE);
	YUVUtils::ReleaseImage(src_l[Layers-1]);
	YUVUtils::ReleaseImage(ref_l[Layers-1]);
	for(int i=Layers-1;i>0;i--)
	{
		Pyramid[i].resampling(MV,MV_ref);
		Pyramid[i].ExtractMotionEstimation_b(ref_l[i]->Y,src_l[i]->Y,MV,MV_ref,NULL,TRUE);
		YUVUtils::ReleaseImage(src_l[i]);
		YUVUtils::ReleaseImage(ref_l[i]);
	}
	Pyramid[0].ExtractMotionEstimation_b(ref,src,MVs,MV_ref,NULL,TRUE);
	me.resampling(MV,MV_ref);
 	me.ExtractMotionEstimation_b(ref,src,MVs,MV_ref,NULL,TRUE);
	YUVUtils::ReleaseImage(src_l[0]);
	YUVUtils::ReleaseImage(ref_l[0]);
	
}
void PyramidME_weak(void *ref,void *src, std::vector<MotionVector> &MVs,ME &me,int Layers)
{
	std::vector<MotionVector> MV;
	std::vector<MotionVector> MV_ref;
	YUVUtils::PlanarImage*ref_l[2];
	YUVUtils::PlanarImage*src_l[2];
	int width=me.width;
	int height=me.height;
	int w=width/pow(2,1);
	int h=height/pow(2,1);
	ME d2(w,h,4);
	ref_l[0]=YUVUtils::CreatePlanarImage(width/pow(2,1),height/pow(2,1));
	src_l[0]=YUVUtils::CreatePlanarImage(width/pow(2,1),height/pow(2,1));
	me.downsampling(src,src_l[0]->Y);
	me.downsampling(ref,ref_l[0]->Y);
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
	me.resampling(MV,MV_ref);
	compare(ref,src,MVs,MV_ref,width,height);
	YUVUtils::ReleaseImage(src_l[0]);
	YUVUtils::ReleaseImage(ref_l[0]);
	
}
void PyramidME_pro(void *ref,void *src, std::vector<MotionVector> &MVs,std::vector<MotionVector> &ref_MV,ME &me,int Layers,int lam1,int lam2)
{
	std::vector<MotionVector> MV;
	std::vector<MotionVector> MV_sub;
	std::vector<MotionVector> MV_d2;
	std::vector<MotionVector> MV_d4;
	std::vector<MotionVector> MV_d8;
	YUVUtils::PlanarImage*ref_l[3];
	YUVUtils::PlanarImage*src_l[3];
	int width=me.width;
	int height=me.height;
	int w=width/pow(2,1);
	int h=height/pow(2,1);
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
			compare(ref_l[2]->Y,src_l[2]->Y,MV,MV_sub,w/2,h/2,lam1);
			d4.resampling(MV,MV_sub);
		}
		else d4.costfunction(ref_l[1]->Y,src_l[1]->Y,MV_sub,MV_d4);
		compare(ref_l[1]->Y,src_l[1]->Y,MV,MV_sub,w,h,lam1);
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
	compare(ref_l[0]->Y,src_l[0]->Y,MV,MV_sub,w,h,lam1);
	//me.resampling(MV,MVs);
	me.resampling(MV,MV_sub);
	compare(ref,src,MVs,MV_sub,width,height,lam2);
	YUVUtils::ReleaseImage(src_l[0]);
	YUVUtils::ReleaseImage(ref_l[0]);
	
}
