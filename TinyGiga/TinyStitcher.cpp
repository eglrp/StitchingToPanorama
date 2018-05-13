/**
@brief class for Stitcher
@author: Shane Yuan
@date: Dec 11, 2017
*/

#include "TinyStitcher.h"
#include <opencv2/ximgproc.hpp>

#define _DEBUG_TINY_STITCHER

TinyStitcher::TinyStitcher() {}
TinyStitcher::~TinyStitcher() {}

/**
@brief init tiny giga
@param cv::Mat refImg: input reference image
@param cv::Mat localImg: input local-view image
@param std::string modelname: modelname used for structure edge detector
@param float scale: input scale
@return int
*/
int TinyStitcher::init(cv::Mat refImg, cv::Mat localImg,
	std::string modelname, float scale) {
	this->refImg = refImg;
	this->localImg = localImg;
	this->modelname = modelname;
	this->scale = scale;
	ptr = cv::ximgproc::createStructuredEdgeDetection(modelname);
	return 0;
}

/**
@brief apply SED detector on input image
@param cv::Mat img: input image
@param float scale: resize scale (apply SED detector on original size image is
too time consuming)
@return cv::Mat: returned edge image
*/
cv::Mat TinyStitcher::SEDDetector(cv::Mat img, float scale) {
	cv::Mat edgeImg;
#ifdef _DEBUG
	cv::Size size_large = img.size();
	cv::Size size_small = cv::Size(size_large.width * scale, size_large.height * scale);
	cv::resize(img, img, size_small);
#endif
	img.convertTo(img, cv::DataType<float>::type, 1 / 255.0);
	ptr->detectEdges(img, edgeImg);
	edgeImg = edgeImg * 255;
#ifdef _DEBUG
	cv::resize(edgeImg, edgeImg, size_large);
#endif
	edgeImg.convertTo(edgeImg, CV_8U);
	return edgeImg;
}

/**
@brief color correction (gray image only)
@param cv::Mat & srcImg: input/output src gray image
@param cv::Mat dstImg: input dst gray image
@return int
*/
int TinyStitcher::colorCorrect(cv::Mat & srcImg, cv::Mat dstImg) {
	cv::Scalar meanSrc, stdSrc, meanDst, stdDst;
	cv::meanStdDev(srcImg, meanSrc, stdSrc);
	cv::meanStdDev(dstImg, meanDst, stdDst);

	srcImg.convertTo(srcImg, -1, stdDst.val[0] / stdSrc.val[0], 
		meanDst.val[0] - stdDst.val[0] / stdSrc.val[0] * meanSrc.val[0]);
	return 0;
}

/**
@brief re-sample feature points from optical flow fields
@param cv::Mat refinedflowfield: input refined flow fields
@param cv::Mat preflowfield: input prewarped flow fields
@param std::vector<cv::Point2f> & refPts_fourthiter: output reference feature points
@param std::vector<cv::Point2f> & localPts_fourthiter: output local feature points
@param int meshrows: rows of mesh
@param int meshcols: cols of mesh
@return int
*/
int TinyStitcher::resampleFeaturePoints(cv::Mat refinedflowfield, cv::Mat preflowfield,
	std::vector<cv::Point2f> & refPts_fourthiter,
	std::vector<cv::Point2f> & localPts_fourthiter,
	int meshrows, int meshcols) {
	// calculate quad size
	float quadWidth = static_cast<float>(refinedflowfield.cols) / static_cast<float>(meshcols);
	float quadHeight = static_cast<float>(refinedflowfield.rows) / static_cast<float>(meshrows);
	// init matching points vectors
	std::vector<cv::Point2f> refPts_final;
	std::vector<cv::Point2f> localPts_final;
	// add matching points on key point positions
	for (size_t i = 0; i < refPts_fourthiter.size(); i++) {
		cv::Point2f p = refPts_fourthiter[i];
		cv::Point2f initflowVal = preflowfield.at<cv::Point2f>(p.y, p.x);
		cv::Point2f refinedflowVal = refinedflowfield.at<cv::Point2f>(p.y, p.x);
		if (cv::norm(initflowVal - refinedflowVal) > 400)
			continue;
		cv::Point2f p1 = refinedflowVal;
		localPts_final.push_back(p1);
		refPts_final.push_back(p);
	}
	// add matching points on quad center
	std::vector<cv::Point2f> borderPoints;
	borderPoints.push_back(cv::Point2f(0.5, 0.5));
	borderPoints.push_back(cv::Point2f(0.25, 0.25));
	borderPoints.push_back(cv::Point2f(0.75, 0.25));
	borderPoints.push_back(cv::Point2f(0.75, 0.75));
	borderPoints.push_back(cv::Point2f(0.75, 0.75));
	for (size_t i = 0; i < meshrows; i++) {
		for (size_t j = 0; j < meshcols; j++) {
			int pointNum = 1;
			if (i == 0 || j == 0 || i == meshrows - 1 || j == meshcols - 1)
				pointNum = 5;
			for (size_t k = 0; k < pointNum; k++) {
				cv::Point2f p = cv::Point2f(quadWidth * (j + borderPoints[k].x), 
					quadHeight * (i + borderPoints[k].y));
				cv::Point2f initflowVal = preflowfield.at<cv::Point2f>(p.y, p.x);
				cv::Point2f refinedflowVal = refinedflowfield.at<cv::Point2f>(p.y, p.x);
				if (cv::norm(initflowVal - refinedflowVal) > 400)
					continue;
				cv::Point2f p1 = refinedflowVal;
				localPts_final.push_back(p1);
				refPts_final.push_back(p);
			}
		}
	}
	refPts_fourthiter = refPts_final;
	localPts_fourthiter = localPts_final;
	return 0;
}

/**
@brief first iteration, find reference block
@return int
*/
int TinyStitcher::firstIteration() {
	SysUtil::infoOutput("First iteration ...\n");
	// calculate edge map
	cv::Mat refEdge = this->SEDDetector(refImg, 0.25);
	cv::Mat localEdge = this->SEDDetector(localImg, 0.25);
	// resize localview image
	cv::Mat templ, templEdge;
	sizeBlock = cv::Size(localImg.cols * scale, localImg.rows * scale);
	cv::resize(localImg, templ, sizeBlock);
	cv::resize(localEdge, templEdge, sizeBlock);

	cv::Mat result, resultEdge;
	cv::matchTemplate(refImg, templ, result, cv::TM_CCOEFF_NORMED);
	cv::matchTemplate(refEdge, templEdge, resultEdge, cv::TM_CCOEFF_NORMED);
	result = result.mul(resultEdge);

	cv::Point maxLoc;
	cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
	refRect = cv::Rect(maxLoc.x, maxLoc.y, sizeBlock.width, sizeBlock.height);
	refImg(refRect).copyTo(refBlk);
	cv::resize(refBlk, refBlk, localImg.size());

	// compute edge map and save
	cv::Mat refblkEdge;
	refEdge(refRect).copyTo(refblkEdge);
	cv::Mat smallLocalImg = localImg.clone();
	cv::resize(smallLocalImg, smallLocalImg, cv::Size(localImg.cols * 0.15,
		localImg.rows * 0.15));
	cv::Mat smallLocalEdge = this->SEDDetector(smallLocalImg, 1);
	cv::imwrite("refblk_edge.png", refblkEdge);
	cv::imwrite("local_edge.png", smallLocalEdge);

	return 0;
}


/**
@brief second iteration, find global homography
@return int
*/
int TinyStitcher::secondIteration() {
	SysUtil::infoOutput("Second iteration ...\n");
	featureMatchPtr->setSize(cv::Size(256, 256), cv::Size(512, 512));
	featureMatchPtr->buildCorrespondence(refBlk, localImg, 0.5f, refPts_seconditer, localPts_seconditer);
#ifdef _DEBUG_TINY_STITCHER
	cv::Mat visual = TinyStitcher::visualMatchingPts(refBlk, localImg, refPts_seconditer,
		localPts_seconditer, 0);
#endif
	globalH = cv::findHomography(localPts_seconditer, refPts_seconditer, CV_RANSAC, 3.0f);
	globalH.convertTo(globalH, CV_32F);
	return 0;
}

/**
@brief third iteration, find local homography matrices
@return int
*/
int TinyStitcher::thirdIteration() {
	SysUtil::infoOutput("Third iteration ...\n");
	// build correspondence
	featureMatchPtr->buildCorrespondence(refBlk, localImg, 1.0f, globalH,
		refPts_thirditer, localPts_thirditer);
#ifdef _DEBUG_TINY_STITCHER
	cv::Mat visual = TinyStitcher::visualMatchingPts(refBlk, localImg, refPts_seconditer,
		localPts_seconditer, 0);
#endif
	// apply asap to get deformation mesh
	int width = refBlk.cols;
	int height = refBlk.rows;
	int meshrows = 8;
	int meshcols = 8;
	float quadWidth = static_cast<float>(width) / static_cast<float>(meshcols);
	float quadHeight = static_cast<float>(height) / static_cast<float>(meshrows);
	float smoothWeight = 0.5;
	asapPtr_thirditer->setMesh(height, width, quadHeight, quadWidth, smoothWeight);
	asapPtr_thirditer->setControlPoints(localPts_thirditer, refPts_thirditer);
	asapPtr_thirditer->solve();
	asapPtr_thirditer->calcFlowField();
	cv::Mat flow = asapPtr_thirditer->getFlowfield();
	cv::remap(localImg, warpImg_thirditer, flow, cv::Mat(), cv::INTER_LINEAR);
	return 0;
}

/**
@brief fourth iteration, use optical flow to refine local homography matrices
@return int
*/
int TinyStitcher::fourthIteration() {
	SysUtil::infoOutput("Fourth iteration ...\n");
	// optical flow refine
	float flowCalcScale = 0.25;
	cv::Mat refineflow;
	cv::Mat localImgGray, refImgGray;
	cv::Size newsize(static_cast<float>(refBlk.cols) * flowCalcScale,
		static_cast<float>(refBlk.rows) * flowCalcScale);
	cv::resize(warpImg_thirditer, localImgGray, newsize);
	cv::resize(refBlk, refImgGray, newsize);
	cv::cvtColor(refImgGray, refImgGray, CV_BGR2GRAY);
	cv::cvtColor(localImgGray, localImgGray, CV_BGR2GRAY);
	// color correction
	colorCorrect(localImgGray, refImgGray);

	// calculate optical flow
	cv::Mat updateflow;
	cv::Ptr<cv::optflow::DeepFlow> deepFlow = cv::optflow::createDeepFlow();
	deepFlow->calc(refImgGray, localImgGray, updateflow);
	cv::resize(updateflow, updateflow, refBlk.size());
	updateflow = updateflow / flowCalcScale;
	for (int i = 0; i < updateflow.rows; i++) {
		for (int j = 0; j < updateflow.cols; j++) {
			cv::Point2f _flow = updateflow.at<cv::Point2f>(i, j);
			cv::Point2f _newFlow;
			_newFlow.x = _flow.x + j;
			_newFlow.y = _flow.y + i;
			updateflow.at<cv::Point2f>(i, j) = _newFlow;
		}
	}
	cv::remap(asapPtr_thirditer->getFlowfield(), refineflow, updateflow, cv::Mat(), cv::INTER_LINEAR);
#ifdef _DEBUG_TINY_STITCHER
	cv::Mat visual;
	cv::remap(localImg, warpImg_fourthiter, refineflow, cv::Mat(), cv::INTER_LINEAR);
#endif
	// resample final feature points
	int meshrows = 16;
	int meshcols = 16;
	refPts_fourthiter = refPts_thirditer;
	localPts_fourthiter = localPts_thirditer;
	resampleFeaturePoints(refineflow, asapPtr_thirditer->getFlowfield(),
		refPts_fourthiter, localPts_fourthiter, meshrows, meshcols);
#ifdef _DEBUG_TINY_STITCHER
	visual = TinyStitcher::visualMatchingPts(refBlk, localImg, refPts_fourthiter,
		localPts_fourthiter, 0);
#endif

	// apply asap to generate final warped image
	int width = refBlk.cols;
	int height = refBlk.rows;
	float quadWidth = static_cast<float>(width) / static_cast<float>(meshcols);
	float quadHeight = static_cast<float>(height) / static_cast<float>(meshrows);
	float smoothWeight = 0.2;
	asapPtr_fourthiter->setMesh(height, width, quadHeight, quadWidth, smoothWeight);
	asapPtr_fourthiter->setControlPoints(localPts_thirditer, refPts_thirditer);
	asapPtr_fourthiter->solve();
	asapPtr_fourthiter->calcFlowField();
	cv::Mat flow = asapPtr_fourthiter->getFlowfield();
	cv::remap(localImg, warpImg_fourthiter, flow, cv::Mat(), cv::INTER_LINEAR);
	return 0;
}

/**
@brief get final warped image
@return cv::Mat warpImg: finally warped image
*/
cv::Mat TinyStitcher::getFinalWarpImg() {
	return warpImg_fourthiter;
}

/**
@brief get reference block image
@return cv::Mat warpImg: reference block image
*/
cv::Mat TinyStitcher::getRefBlkImg() {
	return refBlk;
}

/********************************************************************/
/*                     visualization functions                      */
/********************************************************************/
/**
@brief visualize matching points
@param cv::Mat img1: first image
@param cv::Mat img2: second image
@param std::vector<cv::Point2f> pt1: matching points of the first image
@param std::vector<cv::Point2f> pt2: matching points of the second image
@param int direction, 0: horizontal, 1: vertical
*/
cv::Mat TinyStitcher::visualMatchingPts(cv::Mat img1, cv::Mat img2,
	std::vector<cv::Point2f> pt1, std::vector<cv::Point2f> pt2, int direction) {
	cv::Mat showImg;
	cv::Size showSize;
	if (direction == 0) {// horizontal
		showSize = cv::Size(img1.cols + img2.cols, img1.rows);
		showImg.create(showSize, CV_8UC3);
		cv::Rect rect(0, 0, img1.cols, img1.rows);
		img1.copyTo(showImg(rect));
		rect.x += img1.cols;
		img2.copyTo(showImg(rect));
	}
	else {// vertical
		showSize = cv::Size(img1.cols, img1.rows + img2.rows);
		showImg.create(showSize, CV_8UC3);
		cv::Rect rect(0, 0, img1.cols, img1.rows);
		img1.copyTo(showImg(rect));
		rect.y += img1.rows;
		img2.copyTo(showImg(rect));
	}
	cv::RNG rng(12345);
	int r = 14;
	for (int i = 0; i < pt1.size(); i++) {
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		cv::Point2f p1 = pt1[i];
		cv::Point2f p2 = pt2[i];
		if (p1.x < 0 || p1.x >= img1.cols || p1.y < 0 || p1.y >= img1.rows)
			continue;
		if (p2.x < 0 || p2.x >= img2.cols || p2.y < 0 || p2.y >= img2.rows)
			continue;
		if (direction == 0)
			p2.x += img1.cols;
		else p2.y += img1.rows;
		circle(showImg, p1, r, color, -1, 8, 0);
		circle(showImg, p2, r, color, -1, 8, 0);
	}
	return showImg;
}