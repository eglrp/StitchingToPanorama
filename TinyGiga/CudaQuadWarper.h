/**
@brief quad warper using cuda npp
@author Shane Yuna
@date Mar 3, 2018
*/

#ifndef  __TINY_GIGA_CUDA_QUAD_WARPER_H__
#define __TINY_GIGA_CUDA_QUAD_WARPER_H__

// std
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

// cuda
#ifdef WIN32
#include <windows.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <npp.h>
#include <nppi.h>

// opencv
#include <opencv2/opencv.hpp>

namespace CudaQuadWarperKernel {

};

class QuadMesh {
private:
	int rows;
	int cols;
	cv::Mat meshCtrlPts;
public:

private:

public:
	QuadMesh();
	~QuadMesh();

	/**
	@brief init mesh
	@param int rows: rows of mesh
	@param int cols: cols of mesh
	@return int
	*/
	int init(int rows, int cols);

	/**
	@brief get mesh rows
	@return int
	*/
	int getRows() { return rows; }

	/**
	@brief get mesh cols
	@return int
	*/
	int getCols() { return cols; }

	/**
	@brief set control points
	@param int rowInd: row index of control point need to set
	@param int colInd: col index of control point need to set
	@param cv::Point2f pt: input mesh control point
	@return int
	*/
	int setCtrlPt(int rowInd, int colInd, cv::Point2f pt);

	/**
	@brief set control points
	@param int rowInd: row index of control point need to set
	@param int colInd: col index of control point need to set
	@return cv::Point2f pt: input mesh control point
	*/
	cv::Point2f getCtrlPt(int rowInd, int colInd);
};

class CudaQuadWarper {
private:

public:

private:

public:
	CudaQuadWarper();
	~CudaQuadWarper();

	/**
	@brief warp image using mesh
	@param cv::Mat input: input image
	@param cv::Mat & output: output image
	@param QuadMesh mesh: input mesh for image warping
	@return int
	*/
	static int warp(cv::Mat input, cv::Mat & output, QuadMesh mesh);

};


#endif