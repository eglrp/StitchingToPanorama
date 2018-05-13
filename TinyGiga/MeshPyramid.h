/**
@brief mesh pyramid class
@author Shane Yuna
@date Mar 3, 2018
*/

#ifndef  __TINY_GIGA_MESH_PYRAMID_H__
#define __TINY_GIGA_MESH_PYRAMID_H__

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include "SysUtil.hpp"
#include "OpenGLImageWarper.h"

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

	/**
	@brief get mesh grid control points which stored in cv::Mat
	@return cv::Mat: output control points
	*/
	cv::Mat getMesh(int width, int height);
};

struct float9 {
	float H[9];
	float invH[9];
	// get perspective transform matrix
	cv::Mat getPerspectiveMatrix() {
		return cv::Mat(3, 3, CV_32F, H);
	}
	// get perspective transform matrix
	cv::Mat getInvPerspectiveMatrix() {
		return cv::Mat(3, 3, CV_32F, invH);
	}
	// calculate inverse H
	int calcInverseH() {
		// calculate inverse H
		cv::Mat invHmat = this->getPerspectiveMatrix().inv();
		std::memcpy(invH, invHmat.data, 9 * sizeof(float));
		return 0;
	}
	// set perspective transform matrix
	int setPerspectiveMatrix(cv::Mat hmat) {
		// set H
		if (hmat.type() != CV_32F)
			hmat.convertTo(hmat, CV_32F);
		std::memcpy(H, hmat.data, 9 * sizeof(float));
		// calculate inverse H
		this->calcInverseH();
		return 0;
	}
	// apply perspective warp
	cv::Point2f applyPerspectiveWarp(cv::Point2f input, float scale = 1.0f) {
		cv::Point2f pt;
		input.x *= scale; input.y *= scale;
		float x = H[0] * input.x + H[1] * input.y + H[2];
		float y = H[3] * input.x + H[4] * input.y + H[5];
		float z = H[6] * input.x + H[7] * input.y + H[8];
		pt.x = x / z / scale;
		pt.y = y / z / scale;
		return pt;
	}
	// apply inverse perspective warp
	cv::Point2f applyInvPerspectiveWarp(cv::Point2f input, float scale = 1.0f) {
		cv::Point2f pt;
		input.x *= scale; input.y *= scale;
		float x = invH[0] * input.x + invH[1] * input.y + invH[2];
		float y = invH[3] * input.x + invH[4] * input.y + invH[5];
		float z = invH[6] * input.x + invH[7] * input.y + invH[8];
		pt.x = x / z / scale;
		pt.y = y / z / scale;
		return pt;
	}
};

struct KeyPoint {
	cv::Point2f pt;
	float confidence;
	
	// construction function
	KeyPoint(cv::Point2f pt, float confidence) {
		this->pt = pt;
		this->confidence = confidence;
	}

	// function used to sort
	static bool less(KeyPoint pt1, KeyPoint pt2) {
		return pt1.confidence < pt2.confidence;
	}
	static bool larger(KeyPoint pt1, KeyPoint pt2) {
		return pt1.confidence > pt2.confidence;
	}
};

struct ImagePyramidPair {
	size_t layerNum;
	int patchSize;
	std::vector<cv::Size> gridSizes;
	std::vector<cv::Size> imgSizes;
	std::vector<cv::Mat> localImgs;
	std::vector<cv::Mat> refImgs;
	std::vector<cv::Mat> localEdges;
	std::vector<cv::Mat> refEdges;
	std::vector<std::vector<std::vector<float9>>> flowMaps;
	std::vector<QuadMesh> ctrlMeshes;

	/**
	@brief construction function
	*/
	ImagePyramidPair();

	/**
	@brief clear function
	@return int
	*/
	int clear();

	/**
	@brief function to calculate control meshes from flowMaps 
		(multiple homography)
	@param int layerInd: index of pyramid layer that need to calculate mesh
	@return int
	*/
	int calcCtrlMesh(int layerInd);
};

class KeyPointsPyramid {
private:
	// status
	bool isInit;
public:
	// image size
	size_t imgWidth;
	size_t imgHeight;
	// mesh size	
	size_t meshrows;
	size_t meshcols;
	// quad size
	float quadWidth;
	float quadHeight;
	std::vector<std::vector<std::vector<KeyPoint>>> pyrPt;
private:

public:
	// construction function
	KeyPointsPyramid();
	~KeyPointsPyramid();

	/**
	@brief clear function
	@return int
	*/
	int clear();

	/**
	@brief init function
	@return int
	*/
	int init(size_t imgWidth, size_t imgHeight,
		size_t meshrows, size_t meshcols);

	/**
	@brief build pyramid
	@param std::vector<KeyPoint> & keyPts: input key points
	@return int
	*/
	int buildKeyPointsPyramid(std::vector<KeyPoint> & keyPts);

	/**
	@brief pop keypoints with highest confidence for input meshsize
	@param cv::Size meshsize: input meshsize
	@param size_t num: keypoint numbers in every quad
	@param std::vector<KeyPoint> & keypts: output keypoints
	@return int
	*/
	int popKeyPoints(cv::Size meshsize, size_t num, std::vector<KeyPoint> & keypts);

	/**
	@brief debug function
		visualize selected key points
	@cv::Mat img: input image used as background
	@return cv::Mat: visualization result
	*/
	cv::Mat visualizeKeyPoints(cv::Mat img);
};

class MeshPyramid {
private:
	// input variables
	cv::Mat localImgOrig; // original local image
	cv::Mat localImg; // color corrected local image
	cv::Mat refImg;
	cv::Mat localEdge;
	cv::Mat refEdge;
	cv::Mat warpedImg;
	float scale;

	// reference block
	cv::Size sizeBlock;
	cv::Rect refRect;
	cv::Ptr<cv::ximgproc::StructuredEdgeDetection> SEDPtr;
	cv::Mat refBlk;
	cv::Mat edgeRefBlk;

	// image pyramid
	int patchSize;
	ImagePyramidPair pairs;

	// key point pyramid
	KeyPointsPyramid ptPyr;

	// global homography matrix
	cv::Mat globalH;

	// OpenGL warper
	std::shared_ptr<gl::OpenGLImageWarper> imgWarper;

public:

private:
	/*******************************************************************************/
	/*                              basic common functions                         */
	/*******************************************************************************/
	/**
	@brief apply SED detector on input image
	@param cv::Mat img: input image
	@param float scale: resize scale (apply SED detector on original size image is
	too time consuming)
	@return cv::Mat: returned edge image
	*/
	cv::Mat SEDDetector(cv::Mat img, float scale);

	/**
	@brief first iteration, find reference block
	@return int
	*/
	int findRefBlk();

	/**
	@brief visualize optical flow
	@param cv::Mat flow: input optical flow
	@return cv::Mat: output visualization result
	*/
	static cv::Mat visualizeOpticalFlow(cv::Mat flow);

	/*******************************************************************************/
	/*                 functions for pyramid based quadtree matching               */
	/*******************************************************************************/
	/**
	@brief build image pyramid
	@return int
	*/
	int buildPyramid();

	/**
	@brief quad tree patch match
	@param cv::Mat refPatch: image patch cropped from reference image
	@param cv::Mat refEdgePatch: edge map patch cropped from reference edege image
	@param cv::Mat localPatch: image patch cropped from local-view image
	@param cv::Mat localEdgePatch: edge map patch cropped from local-view edge image
	@param cv::Point2f tlPos: top left corner position of input patch
	@param cv::Size size: size of image which patch cropped from
	@param cv::Mat & shift: output shift 
	@param cv::Mat & H: output homography
	@param std::vector<cv::Point2f> & refPts: key points on reference image 
	@param std::vector<cv::Point2f> & localPts: key points on localview image
	@return int
	*/
	int quadTreeMatch(cv::Mat refPatch, cv::Mat localPatch, 
		cv::Mat refEdgePatch, cv::Mat localEdgePatch, cv::Point2f tlPos,
		cv::Size size, cv::Mat & shift, cv::Mat & H, 
		std::vector<cv::Point2f> & refPts, std::vector<cv::Point2f> & localPts);
	
	/**
	@brief feature matching on pyramid
	@return int
	*/
	int pyramidFeatureMatch();

	/*******************************************************************************/
	/*                 functions for keypoints detection and keypoints             */
	/*                              pyramid construction                           */
	/*******************************************************************************/
	/**
	@brief detect keypoints
	@param cv::Mat img: input image
	@param std::vector<KeyPoint> & keyPts: output key points
	@param int maxNum: input max number of corner points
	@param int minDist: input min distance among corner points
	@param float scale = 1.0f: input scale, resize the image to speed up corner detection
	@return int
	*/
	int detectKeyPts(cv::Mat img, std::vector<KeyPoint> & keyPts,
		int maxNum, int minDist, float scale = 1.0f);

	/**
	@brief calculate confidence of keypoints
	@param std::vector<KeyPoint> & keyPts: output key points
	@return int
	*/
	int calcKeyPtsConfidence(std::vector<KeyPoint> & keyPts);

	/**
	@brief estimate global homography matrix
	@return int
	*/
	int estimateGlobalH();

	/**
	@brief estimate image region with large parallax
	@return int
	*/
	int estimateParallax();

	/*******************************************************************************/
	/*                   function to do warping and get final result               */
	/*******************************************************************************/
	/**
	@brief warp final image
	@return int
	*/
	int warpFinelImage();

public:
	MeshPyramid();
	~MeshPyramid();

	/**
	@brief init function, input local-view image, reference image and scale
	@param cv::Mat localImg: input local-view image
	@param cv::Mat refImg: input reference image
	@param float scale: input scale gap between local-view iamge and reference image
	@return int
	*/
	int init(cv::Mat localImg, cv::Mat refImg, float scale);

	/**
	@brief construct keypoints pyramid
	@param cv::Size meshsize: input size of the mesh
	@return int
	*/
	int constructKeyPtsPyramid(cv::Size meshsize);

	/**
	@brief cross resolution warping function
	@return int
	*/
	int process();

	/**
	@brief get final warped function
	@return cv::Mat: final warped image
	*/
	cv::Mat getFinalWarpedImage();

	/**
	@brief get reference block image
	@return cv::Mat: final warped image
	*/
	cv::Mat getRefBlkImage();
};


#endif // ! 