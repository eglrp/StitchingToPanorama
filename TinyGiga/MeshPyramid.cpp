/**
@brief mesh pyramid class
@author Shane Yuna
@date Mar 3, 2018
*/

#include <chrono>

#include "ColorCorrect.h"
#include "MeshPyramid.h"

#include <opencv2/optflow.hpp>

#define _DEBUG_MESH_PYRAMID

QuadMesh::QuadMesh() {}
QuadMesh::~QuadMesh() {}

/***************************************************************************************/
/*                                    QuadMesh struct                                  */
/***************************************************************************************/
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

/**
@brief get mesh grid control points which stored in cv::Mat
@return cv::Mat: output control points
*/
cv::Mat QuadMesh::getMesh(int width, int height) {
	cv::Mat mesh(rows + 1, cols + 1, CV_32FC2);
	for (size_t row = 0; row <= rows; row++) {
		for (size_t col = 0; col <= cols; col++) {
			mesh.at<cv::Point2f>(row, col) =
				cv::Point2f(meshCtrlPts.at<cv::Point2f>(row, col).x * width,
					meshCtrlPts.at<cv::Point2f>(row, col).y * height);
		}
	}
	return mesh;
}

/***************************************************************************************/
/*                                 ImagePyramidPair struct                             */
/***************************************************************************************/
/**
@brief construction function
*/
ImagePyramidPair::ImagePyramidPair() : layerNum(0), patchSize(64), gridSizes(std::vector<cv::Size>()),
	imgSizes(std::vector<cv::Size>()), localImgs(std::vector<cv::Mat>()), 
	refImgs(std::vector<cv::Mat>()), localEdges(std::vector<cv::Mat>()),
	refEdges(std::vector<cv::Mat>()), 
	ctrlMeshes(std::vector<QuadMesh>()) {}

/**
@brief clear function
@return int
*/
int ImagePyramidPair::clear() {
	layerNum = 0;
	patchSize = 64;
	gridSizes.clear(); imgSizes.clear(); localImgs.clear();
	refImgs.clear(); localEdges.clear(); refEdges.clear();
	ctrlMeshes.clear();
	return 0;
}

/**
@brief function to calculate control meshes from flowMaps
	(multiple homography)
@param int layerInd: index of pyramid layer that need to calculate mesh
@return int
*/
int ImagePyramidPair::calcCtrlMesh(int layerInd) {
	size_t rows = ctrlMeshes[0].getRows();
	size_t cols = ctrlMeshes[0].getCols();
	float quadWidth = 1.0f / static_cast<float>(cols);
	float quadHeight = 1.0f / static_cast<float>(rows);
	if (layerInd == 0) {
		// calculate rectangle and prewarp local image
		// start from regular mesh
		cv::Mat regularPts(rows + 1, cols + 1, CV_32FC2);
		float9 Hlast = flowMaps[0][0][0];
		// generate regular mesh grid
		for (size_t row = 0; row <= rows; row++) {
			for (size_t col = 0; col <= cols; col++) {
				regularPts.at<cv::Point2f>(row, col) = cv::Point2f(quadWidth * col, quadHeight * row);
			}
		}
		// use homography matrix to inverse warp mesh control points
		for (size_t row = 0; row <= rows; row++) {
			for (size_t col = 0; col <= cols; col++) {
				cv::Point2f pt = regularPts.at<cv::Point2f>(row, col);
				pt = Hlast.applyPerspectiveWarp(pt);
				ctrlMeshes[0].setCtrlPt(row, col, pt);
			}
		}
	}
	else {
		// prepare variables
		cv::Mat ctrlMesh(rows + 1, cols + 1, CV_32FC2);
		cv::Mat ctrlMeshWeight(rows + 1, cols + 1, CV_32S);
		ctrlMesh.setTo(cv::Scalar(0, 0));
		ctrlMeshWeight.setTo(cv::Scalar(0));
		// get gridsize
		cv::Size gridSize = gridSizes[layerInd];
		for (size_t gridrow = 0; gridrow < gridSize.height; gridrow++) {
			for (size_t gridcol = 0; gridcol < gridSize.width; gridcol++) {
				float rowIndBegin = static_cast<float>(gridrow) / gridSize.height * rows;
				float rowIndEnd = static_cast<float>(gridrow + 1) / gridSize.height * rows;
				float colIndBegin = static_cast<float>(gridcol) / gridSize.width * cols;
				float colIndEnd = static_cast<float>(gridcol + 1) / gridSize.width * cols;
				// inverse warp mesh control points
				float9 Hlast = flowMaps[layerInd][gridrow][gridcol];
				for (int row = rowIndBegin; row <= rowIndEnd; row++) {
					for (int col = colIndBegin; col <= colIndEnd; col++) {
						cv::Point2f pt = ctrlMeshes[layerInd - 1].getCtrlPt(row, col);
						pt = Hlast.applyPerspectiveWarp(pt);
						ctrlMesh.at<cv::Point2f>(row, col) += pt;
						ctrlMeshWeight.at<int>(row, col) += 1;
					}
				}
			}
		}
		// calculate final control mesh
		for (size_t row = 0; row < ctrlMesh.rows; row++) {
			for (size_t col = 0; col < ctrlMesh.cols; col++) {
				 ctrlMeshes[layerInd].setCtrlPt(row, col, ctrlMesh.at<cv::Point2f>(row, col) /
					 ctrlMeshWeight.at<int>(row, col));
			}
		}
	}
	return 0;
}

/***************************************************************************************/
/*                                    MeshPyramid class                                */
/***************************************************************************************/
KeyPointsPyramid::KeyPointsPyramid() {}
KeyPointsPyramid::~KeyPointsPyramid() {}

/**
@brief init function
@param size_t imgWidth: width of the input image
@param size_t imgHeight: height of the input image
@param size_t meshrows: rows of mesh
@param size_t meshcols: cols of mesh
@return int
*/
int KeyPointsPyramid::init(size_t imgWidth, size_t imgHeight,
	size_t meshrows, size_t meshcols) {
	// init key points vector
	this->meshrows = meshrows;
	this->meshcols = meshcols;
	pyrPt.resize(meshrows);
	for (size_t i = 0; i < meshrows; i ++) {
		pyrPt[i].resize(meshcols);
	}
	isInit = true;
	// calculate quad width
	quadWidth = static_cast<float>(imgWidth) / meshcols; 
	quadHeight = static_cast<float>(imgHeight) / meshrows; 
	return 0;
}

/**
@brief clear function
@return int
*/
int KeyPointsPyramid::clear() {
	for (size_t i = 0; i < meshrows; i ++) {
		for (size_t j = 0; j < meshcols; j ++) {
			pyrPt[i][j].clear();
		}
		pyrPt[i].clear();
	}
	pyrPt.clear();
	isInit = false;
	return 0;	
}

/**
@brief build pyramid
@param std::vector<KeyPoint> & keyPts: input key points
@return int
*/
int KeyPointsPyramid::buildKeyPointsPyramid(std::vector<KeyPoint> & keyPts) {
	// assign key point to different blocks
	for (size_t i = 0; i < keyPts.size(); i ++) {
		cv::Point2f pt = keyPts[i].pt;
		int row = static_cast<int>(pt.y / quadHeight);
		int col = static_cast<int>(pt.x / quadWidth);
		pyrPt[row][col].push_back(keyPts[i]);
	}
	// resort key points by resorting
	for (size_t i = 0; i < meshrows; i++) {
		for (size_t j = 0; j < meshcols; j++) {
			if (pyrPt[i][j].size() > 0) {
				std::sort(pyrPt[i][j].begin(), pyrPt[i][j].end(), KeyPoint::larger);
			}
		}
	}
	return 0;
}

/**
@brief pop keypoints with highest confidence for input meshsize
@param cv::Size meshsize: input meshsize
@param size_t num: keypoint numbers in every quad
@param std::vector<KeyPoint> & keypts: output keypoints
@return int
*/
int KeyPointsPyramid::popKeyPoints(cv::Size meshsize, size_t num, std::vector<KeyPoint> & keypts) {

	return 0;
}

/**
@brief debug function
visualize selected key points
@cv::Mat img: input image used as background
@return cv::Mat: visualization result
*/
cv::Mat KeyPointsPyramid::visualizeKeyPoints(cv::Mat img) {
	cv::Mat visual = img.clone();
	cv::RNG rng(12345);
	int r = 14;
	for (size_t i = 0; i < meshrows; i++) {
		for (size_t j = 0; j < meshcols; j++) {
			if (pyrPt[i][j].size() > 0) {
				cv::Point2f pt = pyrPt[i][j][0].pt;
				cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				cv::circle(visual, pt, r, color, -1, 8, 0);
			}
		}
	}
	return visual;
}

/***************************************************************************************/
/*                                    MeshPyramid class                                */
/***************************************************************************************/
MeshPyramid::MeshPyramid() {
	patchSize = 256;
}
MeshPyramid::~MeshPyramid() {
	imgWarper->release();
}

/**
@brief apply SED detector on input image
@param cv::Mat img: input image
@param float scale: resize scale (apply SED detector on original size image is
too time consuming)
@return cv::Mat: returned edge image
*/
cv::Mat MeshPyramid::SEDDetector(cv::Mat img, float scale) {
	cv::Mat edgeImg;
#ifdef _DEBUG
	cv::Size size_large = img.size();
	cv::Size size_small = cv::Size(size_large.width * scale, size_large.height * scale);
	cv::resize(img, img, size_small);
#endif
	img.convertTo(img, cv::DataType<float>::type, 1 / 255.0);
	SEDPtr->detectEdges(img, edgeImg);
	edgeImg = edgeImg * 255;
#ifdef _DEBUG
	cv::resize(edgeImg, edgeImg, size_large);
#endif
	edgeImg.convertTo(edgeImg, CV_8U);
	return edgeImg;
}

/**
@brief first iteration, find reference block
@return int
*/
int MeshPyramid::findRefBlk() {
	SysUtil::infoOutput("Find reference block ...\n");
	// calculate edge map
	refEdge = this->SEDDetector(refImg, 0.25);
	localEdge = this->SEDDetector(localImg, 0.25);
	// resize localview image
	cv::Mat templ, templEdge;
	sizeBlock = cv::Size(localImg.cols * scale, localImg.rows * scale);
	cv::resize(localImg, templ, sizeBlock);
	cv::resize(localEdge, templEdge, sizeBlock);
	// apply zncc template matching
	cv::Mat result, resultEdge;
	cv::matchTemplate(refImg, templ, result, cv::TM_CCOEFF_NORMED);
	cv::matchTemplate(refEdge, templEdge, resultEdge, cv::TM_CCOEFF_NORMED);
	result = result.mul(resultEdge);
	// crop reference block out
	cv::Point maxLoc;
	cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
	refRect = cv::Rect(maxLoc.x, maxLoc.y, sizeBlock.width, sizeBlock.height);
	refImg(refRect).copyTo(refBlk);
	refEdge(refRect).copyTo(edgeRefBlk);
	cv::resize(refBlk, refBlk, localImg.size());
	cv::resize(edgeRefBlk, edgeRefBlk, localImg.size());
	// color correction
	ColorCorrect::correct(localImgOrig, refBlk, localImg);
	return 0;
}

/**
@brief visualize optical flow
@param cv::Mat
*/
cv::Mat MeshPyramid::visualizeOpticalFlow(cv::Mat flow) {
	cv::Mat vis;
	//extraxt x and y channels
	cv::Mat xy[2]; //X,Y
	cv::split(flow, xy);
	//calculate angle and magnitude
	cv::Mat magnitude, angle;
	cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);
	//translate magnitude to range [0;1]
	double mag_max;
	cv::minMaxLoc(magnitude, 0, &mag_max);
	magnitude.convertTo(magnitude, -1, 1.0 / mag_max);
	//build hsv image
	cv::Mat _hsv[3], hsv;
	_hsv[0] = angle;
	_hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
	_hsv[2] = magnitude;
	cv::merge(_hsv, 3, hsv);
	//convert to BGR and show
	cv::cvtColor(hsv, vis, cv::COLOR_HSV2BGR);
	return vis;
}


/**
@brief init function
@param cv::Mat localImg: local view image
@param cv::Mat refImg: reference image
@param float scale: scale between local view image and reference image
@return int
*/
int MeshPyramid::init(cv::Mat localImg, cv::Mat refImg, float scale) {
	this->localImg = localImg.clone();
	this->localImgOrig = localImg.clone();
	this->refImg = refImg.clone();
	this->scale = scale;
	SEDPtr = cv::ximgproc::createStructuredEdgeDetection("E:/Project/PanoramaStitcher/model/model.yml");
	imgWarper = std::make_shared<gl::OpenGLImageWarper>();
	imgWarper->init();
	return 0;
}

/*******************************************************************************/
/*                 functions for pyramid based quadtree matching               */
/*******************************************************************************/
/**
@brief build image pyramid
@return int
*/
int MeshPyramid::buildPyramid() {
	// calculate pyramid layers
	int maxSize = 2048;
	pairs.clear();
	pairs.patchSize = 256;
	int layerInd = 0;
	for (;;) {
		// calculate size 
		cv::Size gridSize = cv::Size(std::pow(2, layerInd), std::pow(2, layerInd));
		cv::Size imgsize = cv::Size(patchSize * 2 * gridSize.width, patchSize * 2 * gridSize.height);
		// check size
		if (imgsize.width <= maxSize && imgsize.height <= maxSize) {
			layerInd++;
			// resize image and edge image
			pairs.gridSizes.push_back(gridSize);
			pairs.imgSizes.push_back(imgsize);
			cv::Mat localImgPyr, refImgPyr, localEdgePyr, refEdgePyr;
			cv::resize(localImg, localImgPyr, imgsize);
			cv::resize(refBlk, refImgPyr, imgsize);
			cv::resize(localEdge, localEdgePyr, imgsize);
			cv::resize(edgeRefBlk, refEdgePyr, imgsize);
			pairs.localImgs.push_back(localImgPyr);
			pairs.refImgs.push_back(refImgPyr);
			pairs.localEdges.push_back(localEdgePyr);
			pairs.refEdges.push_back(refEdgePyr);
			pairs.ctrlMeshes.push_back(QuadMesh());
		}
		else {
			break;
		}
	}
	pairs.layerNum = layerInd;
	// set mesh size and flowmaps
	pairs.flowMaps.resize(pairs.layerNum);
	cv::Size meshSize = cv::Size(std::pow(2, pairs.layerNum - 1), std::pow(2, pairs.layerNum - 1));
	for (size_t ind = 0; ind < pairs.layerNum; ind++) {
		pairs.ctrlMeshes[ind].init(meshSize.height, meshSize.width);
		pairs.flowMaps[ind].resize(pairs.gridSizes[ind].height);
		for (size_t ind2 = 0; ind2 < pairs.gridSizes[ind].height; ind2++) {
			pairs.flowMaps[ind][ind2].resize(pairs.gridSizes[ind].width);
		}
	}
	return 0;
}


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
int MeshPyramid::quadTreeMatch(cv::Mat refPatch, cv::Mat localPatch,
	cv::Mat refEdgePatch, cv::Mat localEdgePatch, cv::Point2f tlPos, 
	cv::Size size, cv::Mat & shift, cv::Mat & H,
	std::vector<cv::Point2f> & refPts, std::vector<cv::Point2f> & localPts) {
	// init variable 
	std::vector<cv::Rect> searchRects(4);
	std::vector<cv::Rect> patchRects(4);
	cv::Size imgSize = refPatch.size();
	cv::Size searchSize = imgSize / 2;
	cv::Size patchSize = imgSize / 4;
	// compute search rects and patch rects
	searchRects[0] = cv::Rect(0, 0, searchSize.width, searchSize.height);
	searchRects[1] = cv::Rect(searchSize.width, 0, searchSize.width, searchSize.height);
	searchRects[2] = cv::Rect(0, searchSize.height, searchSize.width, searchSize.height);
	searchRects[3] = cv::Rect(searchSize.width, searchSize.height, searchSize.width, searchSize.height);
	patchRects[0] = cv::Rect(patchSize.width / 2, patchSize.height / 2, patchSize.width, patchSize.height);
	patchRects[1] = cv::Rect(patchSize.width / 2 + searchSize.width, patchSize.height / 2, 
		patchSize.width, patchSize.height);
	patchRects[2] = cv::Rect(patchSize.width / 2, patchSize.height / 2 + searchSize.height,
		patchSize.width, patchSize.height);
	patchRects[3] = cv::Rect(patchSize.width / 2 + searchSize.width,
		patchSize.height / 2 + searchSize.height, 
		patchSize.width, patchSize.height);
	// apply template matching
	shift = cv::Mat::zeros(4, 2, CV_32F);
	for (size_t i = 0; i < 4; i++) {
		cv::Mat searchBlk = refPatch(searchRects[i]);
		cv::Mat searchEdgeBlk = refEdgePatch(searchRects[i]);
		cv::Mat patchBlk = localPatch(patchRects[i]);
		cv::Mat patchEdgeBlk = localEdgePatch(patchRects[i]);
		cv::Mat resultEdge, result;
		cv::matchTemplate(searchEdgeBlk, patchEdgeBlk, resultEdge, cv::TM_CCOEFF_NORMED);
		cv::matchTemplate(searchBlk, patchBlk, result, cv::TM_CCOEFF_NORMED);
		// result = result.mul(resultEdge);
		// find max position
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
		cv::Point2f center = cv::Point2f(static_cast<float>(result.cols) / 2,
			static_cast<float>(result.rows) / 2);
		shift.at<float>(i, 0) = static_cast<float>(maxLoc.x) - center.x;
		shift.at<float>(i, 1) = static_cast<float>(maxLoc.y) - center.y;
		// push to vector
		refPts.push_back(cv::Point2f(searchRects[i].x + searchRects[i].width / 2 + shift.at<float>(i, 0),
			searchRects[i].y + searchRects[i].height / 2 + shift.at<float>(i, 1)) + tlPos);
		localPts.push_back(cv::Point2f(patchRects[i].x + patchRects[i].width / 2, 
			patchRects[i].y + patchRects[i].height / 2) + tlPos);
		refPts[i].x /= size.width;
		refPts[i].y /= size.height;
		localPts[i].x /= size.width;
		localPts[i].y /= size.height;
	}
	cv::Mat hmat = cv::getPerspectiveTransform(localPts, refPts);
	hmat.convertTo(H, CV_32F);
	// check quadtree match confidence
	// get original corners
	cv::Mat src(3, 4, CV_32F);
	cv::Mat dst(3, 4, CV_32F);
	src.at<float>(0, 0) = (tlPos.x) / size.width; 
	src.at<float>(1, 0) = (tlPos.y) / size.height; 
	src.at<float>(2, 0) = 1.0f;
	src.at<float>(0, 1) = (tlPos.x + imgSize.width) / size.width; 
	src.at<float>(1, 1) = (tlPos.y) / size.height; 
	src.at<float>(2, 1) = 1.0f;
	src.at<float>(0, 2) = (tlPos.x) / size.width; 
	src.at<float>(1, 2) = (tlPos.y + imgSize.height) / size.height; 
	src.at<float>(2, 2) = 1.0f;
	src.at<float>(0, 3) = (tlPos.x + imgSize.width) / size.width;
	src.at<float>(1, 3) = (tlPos.y + imgSize.height) / size.height; 
	src.at<float>(2, 3) = 1.0f;
	dst = H * src;
	float shiftPercents[4];
	for (size_t i = 0; i < 4; i++) {
		dst.at<float>(0, i) = dst.at<float>(0, i) / dst.at<float>(2, i);
		dst.at<float>(1, i) = dst.at<float>(1, i) / dst.at<float>(2, i);
		shiftPercents[i] = std::max<float>(abs(dst.at<float>(0, i) - src.at<float>(0, i)) 
			/ (static_cast<float>(imgSize.width) / static_cast<float>(size.width)),
			abs(dst.at<float>(1, i) - src.at<float>(1, i))
			/ (static_cast<float>(imgSize.height) / static_cast<float>(size.height)));
	}

	int returnVal = 0;
	for (size_t i = 0; i < 4; i++) {
		if (shiftPercents[i] > 0.05) {
			returnVal = 1;
			break;
		}
	}

#ifdef _DEBUG_MESH_PYRAMID
	std::vector<cv::Point2f> refPtsDebug(4);
	std::vector<cv::Point2f> localPtsDebug(4);
	for (size_t i = 0; i < 4; i++) {
		refPtsDebug[i].x = refPts[i].x * size.width - tlPos.x;
		refPtsDebug[i].y = refPts[i].y  * size.height - tlPos.y;
		localPtsDebug[i].x = localPts[i].x * size.width - tlPos.x;
		localPtsDebug[i].y = localPts[i].y * size.height - tlPos.y;
	}
	cv::Mat hmat_debug = cv::getPerspectiveTransform(localPtsDebug, refPtsDebug);
	cv::Mat out_debug;
	cv::warpPerspective(localPatch, out_debug, hmat_debug, localPatch.size());
#endif
	return returnVal;
}

/**
@brief feature matching on pyramid
@return int
*/
int MeshPyramid::pyramidFeatureMatch() {
	// for loop to do match
	cv::Mat mesh;
	for (size_t i = 0; i < pairs.layerNum; i++) {
		// get grid size
		cv::Size gridsize = pairs.gridSizes[i];
		// prewarp local image
		cv::Mat localImg;
		cv::Mat localEdge;
		if (i == 0) {
			localImg = pairs.localImgs[i].clone();
			localEdge = pairs.localEdges[i].clone();
		}
		else {
			// calculate mesh control points
			pairs.calcCtrlMesh(i - 1);
			// apply mesh based image warping
			cv::Size outputSize = pairs.localImgs[i].size();
			mesh = pairs.ctrlMeshes[i - 1].getMesh(outputSize.width, outputSize.height);
			imgWarper->warp(pairs.localImgs[i], localImg, outputSize, mesh);
			cv::Mat temp = pairs.localEdges[i];
			cv::cvtColor(temp, temp, cv::COLOR_GRAY2BGR);
			imgWarper->warp(temp, localEdge, outputSize, mesh);
			cv::cvtColor(localEdge, localEdge, cv::COLOR_BGR2GRAY);
		}
		// apply feature match
		float quadWidth = static_cast<float>(pairs.imgSizes[i].width) / gridsize.width;
		float quadHeight = static_cast<float>(pairs.imgSizes[i].height) / gridsize.height;
		for (size_t row = 0; row < gridsize.height; row++) {
			for (size_t col = 0; col < gridsize.width; col++) {
				// calculate rect
				cv::Rect rect;
				rect.x = col * quadWidth;
				rect.y = row * quadHeight;
				rect.width = static_cast<int>(quadWidth);
				rect.height = static_cast<int>(quadHeight);
				// first layer
				cv::Mat shift;
				float9 H;
				std::vector<cv::Point2f> refPts;
				std::vector<cv::Point2f> localPts;
				int val = quadTreeMatch(pairs.refImgs[i](rect), localImg(rect), pairs.refEdges[i](rect),
					localEdge(rect), rect.tl(), pairs.imgSizes[i], shift, H.getPerspectiveMatrix(),
					refPts, localPts);
				if (val == 0 || i == 0) {
					H.calcInverseH();
					std::memcpy(&pairs.flowMaps[i][row][col], &H, sizeof(H));
				}
				else {
					SysUtil::infoOutput(SysUtil::sprintf("Reject H estimation result in "\
						"layer:%d, row:%d, col:%d\n", i, row, col));
					std::memcpy(&pairs.flowMaps[i][row][col], &pairs.flowMaps[i - 1][row / 2][col / 2],
						sizeof(H));
				}
			}
		}
	}
	// calculate mesh control points
	pairs.calcCtrlMesh(pairs.layerNum - 1);
	return 0;
}

/*******************************************************************************/
/*                 functions for keypoints detection and keypoints             */
/*                              pyramid construction                           */
/*******************************************************************************/
/**
@brief detect keypoints
@param cv::Mat img: input image
@param std::vector<KeyPoint> & keyPts: output key points
@param int maxNum: max number of corner points
@param int minDist: min distance among corner points
@param float scale = 1.0f: input scale, resize the image to speed up corner detection
@return int
*/
int MeshPyramid::detectKeyPts(cv::Mat img, std::vector<KeyPoint> & keyPts,
	int maxNum, int minDist, float scale) {
	// convert to 
	cv::Mat imgGray;
	cvtColor(img, imgGray, CV_BGR2GRAY);
	// calculate new image size
	cv::Size newSize(img.cols * scale, img.rows * scale);
	cv::resize(imgGray, imgGray, newSize);
	// Apply corner detection
	cv::Mat feature;
	cv::goodFeaturesToTrack(imgGray, feature, maxNum, 0.01, minDist, cv::Mat(),
		5, false, 0.01);
	for (int i = 0; i < feature.rows; i++) {
		keyPts.push_back(KeyPoint(feature.at<cv::Point2f>(i, 0) / scale, 0.0f));
	}
	return 0;
}

/**
@brief calculate confidence of keypoints
@param std::vector<KeyPoint> & keyPts: output key points
@return int
*/
int MeshPyramid::calcKeyPtsConfidence(std::vector<KeyPoint> & keyPts) {
	// compute confidence map
	cv::Mat gradX, gradY;
	cv::Sobel(localEdge, gradX, CV_16SC1, 1, 0, 5);
	cv::Sobel(localEdge, gradY, CV_16SC1, 0, 1, 5);
	gradX = cv::abs(gradX);
	gradY = cv::abs(gradY);
	gradX.convertTo(gradX, CV_32F);
	gradY.convertTo(gradY, CV_32F);
	// compute integral map
	cv::Mat gradIntX, gradIntY;
	cv::integral(gradX, gradIntX, CV_64F);
	cv::integral(gradY, gradIntY, CV_64F);
	// compute confidence
	int patchsize = 256;
	cv::Rect imgRect(0, 0, localImg.cols, localImg.rows);
	for (size_t i = 0; i < keyPts.size(); i ++) {
		cv::Point2f pt = keyPts[i].pt;
		cv::Rect2i rect;
		rect.x = pt.x - patchSize / 2;
		rect.y = pt.y - patchSize / 2;
		rect.width = patchsize;
		rect.height = patchsize;
		// check if rect is inside the image
		// if not, set the confidence to 0
		if ((rect | imgRect) != imgRect) {
			keyPts[i].confidence = 0;
		}
		// if yes, use integral map the compute the confidence
		else {
			cv::Point2i tl = rect.tl();
			cv::Point2i br = rect.br();
			cv::Point2i tr = cv::Point2i(br.x, tl.y);
			cv::Point2i bl = cv::Point2i(tl.x, br.y);
			double confX = gradIntX.at<double>(tl.y, tl.x) - gradIntX.at<double>(tr.y, tr.x)
				- gradIntX.at<double>(bl.y, bl.x) + gradIntX.at<double>(br.y, br.x);
			double confY = gradIntY.at<double>(tl.y, tl.x) - gradIntY.at<double>(tr.y, tr.x)
				- gradIntY.at<double>(bl.y, bl.x) + gradIntY.at<double>(br.y, br.x);
			keyPts[i].confidence = static_cast<float>(std::min<double>(confX, confY));
		}
	}
	return 0;
}

/**
@brief estimate global homography matrix
@return int
*/
int MeshPyramid::estimateGlobalH() {
	// resize image
	cv::Mat refBlkSmall, localImgSmall;
	cv::Mat edgeRefBlkSmall, localEdgeSmall;
	cv::Size sizeSmall = cv::Size(localImg.cols / 5, localImg.rows / 5);
	cv::resize(refBlk, refBlkSmall, sizeSmall);
	cv::resize(edgeRefBlk, edgeRefBlkSmall, sizeSmall);
	cv::resize(localImg, localImgSmall, sizeSmall);
	cv::resize(localEdge, localEdgeSmall, sizeSmall);
	// quad tree match
	cv::Mat shift;
	cv::Mat normalH;
	std::vector<cv::Point2f> refPts;
	std::vector<cv::Point2f> localPts;
	int val = quadTreeMatch(refBlkSmall, localImgSmall, edgeRefBlkSmall,
		localEdgeSmall, cv::Point2f(0, 0), localImgSmall.size(), shift, normalH,
		refPts, localPts);
	// calculate H on original image size;
	for (size_t i = 0; i < 4; i++) {
		refPts[i].x = refPts[i].x * localImg.cols;
		refPts[i].y = refPts[i].y  * localImg.rows;
		localPts[i].x = localPts[i].x * localImg.cols;
		localPts[i].y = localPts[i].y * localImg.rows;
	}
	globalH = cv::getPerspectiveTransform(localPts, refPts);
#ifdef _DEBUG_MESH_PYRAMID
	cv::Mat out_debug;
	cv::warpPerspective(localImg, out_debug, globalH, localImg.size());
#endif
	return 0;
}

/**
@brief estimate image region with large parallax
@return int
*/
int MeshPyramid::estimateParallax() {
	// warp local image
	cv::Mat warpedImg;
	cv::warpPerspective(localImg, warpedImg, globalH, localImg.size());
	// change to gray image and adjust the color
	cv::Mat warpedImgGray, refBlkGray;
	cv::cvtColor(warpedImg, warpedImgGray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(refBlk, refBlkGray, cv::COLOR_BGR2GRAY);
	// calculate mean and std of two images
	cv::Scalar meanLocal, meanRef, stdLocal, stdRef;
	cv::meanStdDev(warpedImgGray, meanLocal, stdLocal);
	cv::meanStdDev(refBlkGray, meanRef, stdRef);
	// adjust the color
	double alpha = stdRef.val[0] / stdLocal.val[0];
	double beta = -stdRef.val[0] / stdLocal.val[0] * meanLocal.val[0] + meanRef.val[0];
	warpedImgGray.convertTo(warpedImgGray, CV_8U, alpha, beta);

	// resize image to do faster optical flow
	cv::Size sizeSmall = cv::Size(warpedImgGray.cols / 4, warpedImgGray.rows / 4);
	cv::resize(warpedImgGray, warpedImgGray, sizeSmall);
	cv::resize(refBlkGray, refBlkGray, sizeSmall);
	// apply optical flow to estimate parallax
	cv::Ptr<cv::DenseOpticalFlow> flowPtr = cv::createOptFlow_DualTVL1();
	cv::Mat flow;
	flowPtr->calc(warpedImgGray, refBlkGray, flow);

#ifdef _DEBUG_MESH_PYRAMID
	cv::Mat vis = MeshPyramid::visualizeOpticalFlow(flow);
#endif
	
	return 0;
}

/**
@brief construct keypoints pyramid
@param cv::Size meshsize: input size of the mesh
@return int
*/
int MeshPyramid::constructKeyPtsPyramid(cv::Size meshsize) {
	// init point pyramid
	ptPyr.init(localImg.cols, localImg.rows, meshsize.height, meshsize.width);
	// construct pyramid
	std::vector<KeyPoint> keyPts;
	this->detectKeyPts(this->localImg, keyPts, 4000, 10, 0.5);
	this->calcKeyPtsConfidence(keyPts);
	// build pyramid
	ptPyr.buildKeyPointsPyramid(keyPts);
#ifdef _DEBUG_MESH_PYRAMID
	// debug
	cv::Mat result = ptPyr.visualizeKeyPoints(this->localImg);
	cv::imwrite("keypt_pyramid.png", result);
#endif
	return 0;
}

/**
@brief warp final image
@return int
*/
int MeshPyramid::warpFinelImage() {
	cv::Size outputSize = localImg.size();
	cv::Mat mesh = pairs.ctrlMeshes[pairs.layerNum - 1].getMesh(outputSize.width, outputSize.height);
	imgWarper->warp(localImg, warpedImg, outputSize, mesh);
	return 0;
}

/**
@brief get reference block image
@return cv::Mat: final warped image
*/
cv::Mat MeshPyramid::getRefBlkImage() {
	return this->refBlk;
}

/**
@brief debug function
@return int
*/
int MeshPyramid::process() {
	// record start time
	auto start = std::chrono::system_clock::now();
	// find reference block
	this->findRefBlk();

	//// build pyramid
	//this->buildPyramid();
	//// feature match on pyramid
	//this->pyramidFeatureMatch();
	//// warp finel local image
	//this->warpFinelImage();

	// estimate global homography matrix
	this->estimateGlobalH();
	// estimate parallax
	this->estimateParallax();

	// build keypoints pyramid
	//this->constructKeyPtsPyramid(cv::Size(16, 16));


	// record end time
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << "Time elapsed " << diff.count() * 1000 << " ms\n";
	return 0;
}

/**
@brief get final warped function
@return cv::Mat: final warped image
*/
cv::Mat MeshPyramid::getFinalWarpedImage() {
	return warpedImg;
}