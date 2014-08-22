#include "ME.h"
void ME::downsampling(void*s,void*d)
{
	double time=time_stamp();
	uint8_t * src=(unsigned char *)s;
	uint8_t * det=(unsigned char *)d;
	int w2=width/2;
	for(int i=0;i<height/2;i++)
	{
		for(int j=0;j<width/2;j++)
		{
			det[i*w2+j]=(src[i*2*width+j*2]+src[i*2*width+j*2+1]+src[(i*2+1)*width+j*2]+src[(i*2+1)*width+j*2+1])/4;
		}
	}
	std::cout<<"downsample time \t"<<1000*(time_stamp()-time)<<"(ms)\n";
}
void ME::downsampling(std::vector<MotionVector>&src,std::vector<MotionVector>&det)
{
	double time=time_stamp();
	int w2,h2;
	ComputeNumMVs(kMBBlockType, width/2, height/2, w2, h2);
	det.resize(w2*h2);
	for(int i=0;i<mvImageHeight/2;i++)
	{
		for(int j=0;j<mvImageWidth/2;j++)
		{
			det[i*w2+j].s[0]=(src[i*2*mvImageWidth+j*2].s[0]+src[i*2*mvImageWidth+j*2+1].s[0]+src[(i*2+1)*mvImageWidth+j*2].s[0]+src[(i*2+1)*mvImageWidth+j*2+1].s[0])/8;
			det[i*w2+j].s[1]=(src[i*2*mvImageWidth+j*2].s[1]+src[i*2*mvImageWidth+j*2+1].s[1]+src[(i*2+1)*mvImageWidth+j*2].s[1]+src[(i*2+1)*mvImageWidth+j*2+1].s[1])/8;
		}
	}
	std::cout<<"downsample time \t"<<1000*(time_stamp()-time)<<"(ms)\n";
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
	double time=time_stamp();
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
	std::cout<<"resample time \t\t"<<1000*(time_stamp()-time)<<"(ms)\n";
}
