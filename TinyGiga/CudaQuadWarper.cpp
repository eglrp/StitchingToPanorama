/**
@brief quad warper using cuda npp
@author Shane Yuna
@date Mar 3, 2018
*/

#include "SysUtil.hpp"
#include "CudaQuadWarper.h"

#include "Exceptions.h"
#include "helper_string.h"
#include "helper_cuda.h"

QuadMesh::QuadMesh() {}
QuadMesh::~QuadMesh() {}

/**
@brief init mesh
@param int rows: rows of mesh
@param int cols: cols of mesh
@return int
*/
int QuadMesh::init(int rows, int cols) {
	this->rows = rows;
	this->cols = cols;
	meshCtrlPts.create(rows + 1, cols + 1, CV_32FC2);
	return 0;
}

/**
@brief set control points
@param int rowInd: row index of control point need to set
@param int colInd: col index of control point need to set
@param cv::Point2f pt: input mesh control point
@return int
*/
int QuadMesh::setCtrlPt(int rowInd, int colInd, cv::Point2f pt) {
	if (rowInd < 0 || colInd < 0 || rowInd > rows || colInd > cols) {
		SysUtil::errorOutput(SysUtil::sprintf("Mesh control point set error, "\
			"mesh size (%ld, %ld), access (%d, %d)", meshCtrlPts.cols, meshCtrlPts.rows,
			rowInd, colInd));
		exit(-1);
	}
	meshCtrlPts.at<cv::Point2f>(rowInd, colInd) = pt;
	return 0;
}

/**
@brief set control points
@param int rowInd: row index of control point need to set
@param int colInd: col index of control point need to set
@return cv::Point2f pt: input mesh control point
*/
cv::Point2f QuadMesh::getCtrlPt(int rowInd, int colInd) {
	if (rowInd < 0 || colInd < 0 || rowInd > rows || colInd > cols) {
		SysUtil::errorOutput(SysUtil::sprintf("Mesh control point get error, "\
			"mesh size (%ld, %ld), access (%d, %d)", meshCtrlPts.cols, meshCtrlPts.rows,
			rowInd, colInd));
		exit(-1);
	}
	return meshCtrlPts.at<cv::Point2f>(rowInd, colInd);
}


CudaQuadWarper::CudaQuadWarper() {}
CudaQuadWarper::~CudaQuadWarper() {}

/**
@brief warp image using mesh
@param cv::Mat input: input image
@param cv::Mat & output: output image
@param Mesh mesh: input mesh for image warping
@return int
*/
int CudaQuadWarper::warp(cv::Mat input, cv::Mat & output, QuadMesh mesh) {
	// calculate quad size
	float quadWidth = static_cast<float>(input.cols) / mesh.getCols();
	float quadHeight = static_cast<float>(input.rows) / mesh.getRows();
	// prepare GPU data
	cv::cuda::GpuMat input_d;
	cv::cuda::GpuMat output_d(input.size(), input.type());
	input_d.upload(input);
	output_d.create(input_d.size(), input.type());
	for (size_t i = 0; i < mesh.getRows(); i++) {
		for (size_t j = 0; j < mesh.getCols(); j++) {
			// calculate dst quad
			const Npp8u* pSrc = input_d.data;
			NppiSize oSrcSize; oSrcSize.width = input_d.cols; oSrcSize.height = input_d.rows;
			NppiRect oSrcROI; oSrcROI.x = 0; oSrcROI.y = 0; oSrcROI.width = input_d.cols; oSrcROI.height = input_d.rows;
			int nSrcStep = input_d.step;
			Npp8u* pDst = output_d.data;
			int nDstStep = output_d.step;
			NppiRect oDstROI; oDstROI.x = 0; oDstROI.y = 0; oDstROI.width = output_d.cols; oDstROI.height = output_d.rows;
			
			double aDstQuad[4][2];
			double aSrcQuad[4][2];
			aSrcQuad[0][1] = mesh.getCtrlPt(i, j).x * input.cols; 
			aSrcQuad[0][0] = mesh.getCtrlPt(i, j).y * input.rows;
			aSrcQuad[3][1] = mesh.getCtrlPt(i, j + 1).x * input.cols; 
			aSrcQuad[3][0] = mesh.getCtrlPt(i, j + 1).y * input.rows;
			aSrcQuad[1][1] = mesh.getCtrlPt(i + 1, j).x * input.cols; 
			aSrcQuad[1][0] = mesh.getCtrlPt(i + 1, j).y * input.rows;
			aSrcQuad[2][1] = mesh.getCtrlPt(i + 1, j + 1).x * input.cols; 
			aSrcQuad[2][0] = mesh.getCtrlPt(i + 1, j + 1).y * input.rows;
			aDstQuad[0][1] = j * quadWidth; aDstQuad[0][0] = i * quadHeight;
			aDstQuad[3][1] = (j + 1) * quadWidth - 1; aDstQuad[3][0] = i * quadHeight;
			aDstQuad[1][1] = j * quadWidth; aDstQuad[1][0] = (i + 1) * quadHeight - 1;
			aDstQuad[2][1] = (j + 1) * quadWidth - 1; aDstQuad[2][0] = (i + 1) * quadHeight - 1;

			// warp quad
			if (input.type() == CV_8UC3) {
				NPP_CHECK_NPP(nppiWarpPerspectiveQuad_8u_C3R(pSrc, oSrcSize, nSrcStep,
					oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, NPPI_INTER_LINEAR));
			}
			else if (input.type() == CV_8UC1) {
				NPP_CHECK_NPP(nppiWarpPerspectiveQuad_8u_C1R(pSrc, oSrcSize, nSrcStep,
					oSrcROI, aSrcQuad, pDst, nDstStep, oDstROI, aDstQuad, NPPI_INTER_LINEAR));
			}
			output_d.download(output);
			int a = 0;
			a++;
		}
	}
	// download to CPU
	output_d.download(output);
	input_d.release();
	output_d.release();
	return 0;
}
