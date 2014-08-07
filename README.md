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
