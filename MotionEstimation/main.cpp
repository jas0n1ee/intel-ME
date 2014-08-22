// Copyright (c) 2009-2013 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly

#define __CL_ENABLE_EXCEPTIONS
#define videofile "Cobra_3840x2160_30frames.yuv"

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <CL/cl.hpp>
#include <CL/cl_ext.h>

#include "yuv_utils.h"
#include "cmdparser.hpp"
#include "oclobject.hpp"
#include "ME.h"
using namespace YUVUtils;
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4355)    // 'this': used in base member initializer list
#endif
// All command-line options for the sample
class CmdParserMV : public CmdParser
{
public:
    CmdOption<std::string>         fileName;
    CmdOption<std::string>         overlayFileName;
    CmdOption<int>		width;
    CmdOption<int>      height;
    CmdOption<bool>		help;
    CmdOption<bool>		out_to_bmp;

    CmdParserMV  (int argc, const char** argv) :
    CmdParser(argc, argv),
        out_to_bmp(*this,		'b',"nobmp","","Do not output frames to the sequence of bmp files (in addition to the yuv file), by default the output is on", ""),
        help(*this,				'h',"help","","Show this help text and exit."),
		fileName(*this,			0,"input", "string", "Input video sequence filename (.yuv file format)",videofile),
        overlayFileName(*this,	0,"output","string", "Output video sequence with overlaid motion vectors filename ","output.yuv"),
        width(*this,			0, "width",	"<integer>", "Frame width for the input file", 3840),
        height(*this,			0, "height","<integer>", "Frame height for the input file",2160)
    {
    }
    virtual void parse ()
    {
        CmdParser::parse();
        if(help.isSet())
        {
            printUsage(std::cout);
        }
    }
};
#ifdef _MSC_VER
#pragma warning (pop)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Overlay routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Draw a pixel on Y picture
typedef uint8_t U8;
void DrawPixel(int x, int y, U8 *pPic, int nPicWidth, int nPicHeight, U8 u8Pixel)
{
    int nPixPos;

    if (x<0 || x>=nPicWidth || y<0 || y>=nPicHeight)
        return;         // Don't draw out of bound pixels
    nPixPos = y * nPicWidth + x;
    *(pPic+nPixPos) = u8Pixel;
}
// Bresenham's line algorithm
void DrawLine(int x0, int y0, int dx, int dy, U8 *pPic, int nPicWidth, int nPicHeight, U8 u8Pixel)
{
    using std::swap;

    int x1 = x0 + dx;
    int y1 = y0 + dy;
    bool bSteep = abs(dy) > abs(dx);
    if (bSteep)
    {
        swap(x0, y0);
        swap(x1, y1);
    }
    if (x0 > x1)
    {
        swap(x0, x1);
        swap(y0, y1);
    }
    int nDeltaX = x1 - x0;
    int nDeltaY = abs(y1 - y0);
    int nError = nDeltaX / 2;
    int nYStep;
    if (y0 < y1)
        nYStep = 1;
    else
        nYStep = -1;

    for (x0; x0 <= x1; x0++)
    {
        if (bSteep)
            DrawPixel(y0, x0, pPic, nPicWidth, nPicHeight, u8Pixel);
        else
            DrawPixel(x0, y0, pPic, nPicWidth, nPicHeight, u8Pixel);

        nError -= nDeltaY;
        if (nError < 0)
        {
            y0 += nYStep;
            nError += nDeltaX;
        }
    }
}

void OverlayVectors(unsigned int subBlockSize, const MotionVector* pMV, PlanarImage* srcImage, const int& mvImageWidth, const int& mvImageHeight, const int& width, const int& height)
{
    const int nHalfBlkSize = subBlockSize/2;
    for (int i = 0; i < mvImageHeight; i++)
    {
        for (int j = 0; j < mvImageWidth; j++)
        {
            DrawLine (j*subBlockSize + nHalfBlkSize, i*subBlockSize + nHalfBlkSize,
                (pMV[j+i*mvImageWidth].s[0] + 2) >> 2, (pMV[j+i*mvImageWidth].s[1]+ 2) >> 2,
                srcImage->Y, width, height, 200);
        }
    }
}

int main( int argc, const char** argv )
{
	CmdParserMV cmd(argc, argv);
	cmd.parse();

	// Immediatly exit if user wanted to see the usage information only.

	const int width = cmd.width.getValue();
	const int height = cmd.height.getValue();
	// Open input sequence
	Capture * pCapture = Capture::CreateFileCapture(cmd.fileName.getValue(), width, height);
	int numPics=pCapture->GetNumFrames();
	// Process sequence
	std::cout << "Processing " << numPics << " frames ..." << std::endl;
			
	PlanarImage * refImage = CreatePlanarImage(width, height);
	PlanarImage * srcImage = CreatePlanarImage(width, height);
	//remember to release!!!!!!!!!!!!!!
	pCapture->GetSample(0,refImage);
	pCapture->GetSample(1,srcImage);



	ME me(width,height,16);
	ME me4(width,height,4);
	ME me16(width,height,16);
	
	int mvImageWidth, mvImageHeight;
	me.ComputeNumMVs(kMBBlockType, width, height, mvImageWidth, mvImageHeight);

	std::vector<MotionVector> MVs;
	std::vector<MotionVector> MV_tmp;
	std::vector<MotionVector> MV_tmp2;
	std::vector<MotionVector> MV_ref;
	MVs.resize(mvImageHeight*mvImageWidth);
	MV_ref.resize(mvImageHeight*mvImageWidth);

	double meTime=0;
	double meStart=time_stamp();
	PyramidME_weak(refImage->Y,srcImage->Y,MVs,me16,2);
	std::cout<<"ME Time\t\t"<<1000*(time_stamp()-meStart)<<"ms"<<std::endl;
	FrameWriter * pWriter = FrameWriter::CreateFrameWriter(width, height, pCapture->GetNumFrames(), cmd.out_to_bmp.getValue());
	unsigned int subBlockSize = me.ComputeSubBlockSize(kMBBlockType);

	OverlayVectors(subBlockSize, &MVs[0], srcImage, mvImageWidth, mvImageHeight, width, height);
	pWriter->AppendFrame(refImage);
	pWriter->AppendFrame(srcImage);

	for(int i=2;i<numPics;i++)
	{
		std::swap(refImage,srcImage);
		std::swap(MV_ref,MVs);	
		pCapture->GetSample(i,srcImage);
		
/*
		std::vector<MotionVector> temp;
		me4.downsampling(MV_ref,temp);
		me4.resampling(temp,MVs);
*/
		meStart=time_stamp();
		//compare(refImage->Y,srcImage->Y,MVs,MV_ref,me4,me16);
		//compare_pro(refImage->Y,srcImage->Y,MVs,MV_ref,width,height);
		PyramidME_pro(refImage->Y,srcImage->Y,MVs,MV_ref,me16,2);
		std::cout<<"ME Time\t\t"<<1000*(time_stamp()-meStart)<<"ms"<<std::endl;
		OverlayVectors(subBlockSize, &MVs[0], srcImage, mvImageWidth, mvImageHeight, width, height);
		pWriter->AppendFrame(srcImage);
	}
	// Generate sequence with overlaid motion vectors

	// Overlay MVs on Src picture, except the very first one
	std::cout << "Writing " << pCapture->GetNumFrames() << " frames to " << cmd.overlayFileName.getValue() << "..." << std::endl;
	pWriter->WriteToFile(cmd.overlayFileName.getValue().c_str());

	FrameWriter::Release(pWriter);
	Capture::Release(pCapture);
	ReleaseImage(srcImage);
	ReleaseImage(refImage);

    std::cout << "Done!" << std::endl;

    return 0;
}