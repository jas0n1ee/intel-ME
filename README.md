intel-ME
========
### MotionEstmation function defination
	void MotionEstimation( void *src,void *ref, 
							std::vector<MotionVector> & MVs,
 							std::vector<MotionVector>& preMVs, 
							bool preMVEnable,
							const int width,const int height)
### PicYuv defination used in this program

	typedef struct
	    {
		uint8_t * Y;
		uint8_t * U;
		uint8_t * V;
		unsigned int Width;
		unsigned int Height;
		int PitchY=0;
		int PitchU=0;
		int PitchV=0;
	    } PlanarImage;
we only use `uint8_t * Y` part in this program.
### MotionVector defination
	typedef cl_short2 MotionVector;
	typedef signed   __int16        cl_short;
	typedef union
	{
	    cl_short  CL_ALIGNED(4) s[2];
	#if defined( __GNUC__) && ! defined( __STRICT_ANSI__ )
	   __extension__ struct{ cl_short  x, y; };
	   __extension__ struct{ cl_short  s0, s1; };
	   __extension__ struct{ cl_short  lo, hi; };
	#endif
	#if defined( __CL_SHORT2__) 
	    __cl_short2     v2;
	#endif
	}cl_short2;

#function defined in ME.h
		void ComputeNumMVs( cl_uint nMBType, int nPicWidth, int nPicHeight, int & nMVSurfWidth, int & nMVSurfHeight );
		//origin function in intel Motion Estimation
		unsigned int ComputeSubBlockSize( cl_uint nMBType );
		//same as above
		void downsampling(void *src,void *det);
		//downsampling the image, will be replace by cl code in the future
		void resampling(void *src,void *det);
		void resampling(std::vector<MotionVector>&src,std::vector<MotionVector>&det);
		//resampling the image and MV
		friend std::vector<MotionVector> moveMV(std::vector<MotionVector> src,int direction,int mvImageHeight,int mvImageWidth);
		//move MV map to different direction, less usage
		void costfunction(void *ref,void *src,
				std::vector<MotionVector>& ,
				std::vector<MotionVector>& );
		//first version to compute cost and select the least one
		friend void compare(void *ref,void *src,std::vector<MotionVector>&,std::vector<MotionVector>&,ME &me4,ME &me16);
		//version that needs input of ME object
		friend void compare(void *ref,void *src,std::vector<MotionVector>&,std::vector<MotionVector>&,int width,int height);
		//version that dont need ME object
		friend void compare_weak(void *ref,void *src,std::vector<MotionVector>&,std::vector<MotionVector>&,int width,int height);
		//version that dont need ref_MV
		friend void compare_pro(void *ref,void *src,std::vector<MotionVector>&,std::vector<MotionVector>&,int width,int height);
		//version that contain pyramid and original compare function
		friend void compare_MV(void *ref,void *src,std::vector<MotionVector>&,std::vector<MotionVector>&,int *,int width,int height);
		//work with moveMV, little usage
		friend void PyramidME(void *ref,void *src,std::vector<MotionVector>& ,ME &me, int );
		//unfinished Pyramid function, lots of bugs
		friend void PyramidME_weak(void *ref,void *src,std::vector<MotionVector>& ,ME &me,int);
		//small size Pyramid, work well