/**
@brief mesh pyramid class
@author Shane Yuna
@date Mar 3, 2018
*/

#include "ColorCorrect.h"

ColorCorrect::ColorCorrect() {}
ColorCorrect::~ColorCorrect() {}

/**
@brief static function to calculate image color mean, covariance matrix
@param cv::Mat input: input image
@param cv::Mat & outmean: output color mean vector
@param cv::Mat & outcov: output color covariance matrix
@param cv::Mat mask = cv::Mat(): input mask for image
@return int
*/
int ColorCorrect::calcMeanCov(cv::Mat input, cv::Mat & outmean,
	cv::Mat & outcov, cv::Mat mask) {
	// check mask
	size_t pixelnum = 0;
	if (mask.empty()) {
		mask = cv::Mat::ones(input.size(), CV_8U);
		pixelnum = input.size().area();
	}
	else {
		pixelnum = cv::countNonZero(mask);
	}
	// reshape image to vector
	cv::Mat imgvec(pixelnum, 3, CV_32F);
	size_t ind = 0;
	outmean = cv::Mat::zeros(3, 1, CV_32F);
	outcov = cv::Mat::zeros(3, 3, CV_32F);
	for (size_t i = 0; i < input.rows; i++) {
		for (size_t j = 0; j < input.cols; j++) {
			if (mask.at<uchar>(i, j) != 0) {
				cv::Vec3b val = input.at<cv::Vec3b>(i, j);
				float b = static_cast<float>(val.val[0]);
				float g = static_cast<float>(val.val[1]);
				float r = static_cast<float>(val.val[2]);
				imgvec.at<float>(ind, 0) = b;
				imgvec.at<float>(ind, 1) = g;
				imgvec.at<float>(ind, 2) = r;
				outmean.at<float>(0, 0) += b;
				outmean.at<float>(1, 0) += g;
				outmean.at<float>(2, 0) += r;
				ind++;
			}
		}
	}
	outmean.at<float>(0, 0) /= pixelnum;
	outmean.at<float>(1, 0) /= pixelnum;
	outmean.at<float>(2, 0) /= pixelnum;
	// calculate outcov
	for (size_t i = 0; i < pixelnum; i++) {
		imgvec.at<float>(i, 0) -= outmean.at<float>(0, 0);
		imgvec.at<float>(i, 1) -= outmean.at<float>(1, 0);
		imgvec.at<float>(i, 2) -= outmean.at<float>(2, 0);
	}
	outcov = imgvec.t() * imgvec / pixelnum;
	return 0;
}

/**
@brief function to make eigen values positive
@param cv::Mat & eigvalue: input/output eigen value matrix (3x3)
@return int
*/
int ColorCorrect::makeEigvaluesPositive(cv::Mat & eigvalues) {
	for (size_t i = 0; i < 3; i++) {
		if (eigvalues.at<float>(i, 0) < POSITIVE) {
			eigvalues.at<float>(i, 0) = POSITIVE;
		}
	}
	return 0;
}

/**
@brief static function for color correction
@param cv::Mat src: input source image
@param cv::Mat dst: input destination image
@param cv::Mat & out: output color corrected image
@return int
*/
int ColorCorrect::correct(cv::Mat src, cv::Mat dst, cv::Mat & out) {
	// calculate mean and covariance matrix
	cv::Mat srcmean, srccov, dstmean, dstcov;
	ColorCorrect::calcMeanCov(src, srcmean, srccov);
	ColorCorrect::calcMeanCov(dst, dstmean, dstcov);
	// apply eigen decomposition first time
	cv::Mat srcEigVal, srcEigVec;
	cv::eigen(srccov, srcEigVal, srcEigVec);
	ColorCorrect::makeEigvaluesPositive(srcEigVal);
	// compute C matrix
	cv::Mat srcEigValSqrt = cv::Mat::zeros(3, 3, CV_32F);
	srcEigValSqrt.at<float>(0, 0) = std::sqrt(srcEigVal.at<float>(0, 0));
	srcEigValSqrt.at<float>(1, 1) = std::sqrt(srcEigVal.at<float>(1, 0));
	srcEigValSqrt.at<float>(2, 2) = std::sqrt(srcEigVal.at<float>(2, 0));
	cv::Mat C = srcEigValSqrt * srcEigVec * dstcov * srcEigVec.t() * srcEigValSqrt;
	// apply eigen decomposition second time
	cv::Mat eigValC, eigVecC;
	cv::eigen(C, eigValC, eigVecC);
	ColorCorrect::makeEigvaluesPositive(eigValC);
	// compute tranform matrix
	cv::Mat eigValSqrtC = cv::Mat::zeros(3, 3, CV_32F);
	eigValSqrtC.at<float>(0, 0) = std::sqrt(eigValC.at<float>(0, 0));
	eigValSqrtC.at<float>(1, 1) = std::sqrt(eigValC.at<float>(1, 0));
	eigValSqrtC.at<float>(2, 2) = std::sqrt(eigValC.at<float>(2, 0));
	cv::Mat srcEigValSqrtInv = cv::Mat::zeros(3, 3, CV_32F);
	srcEigValSqrtInv.at<float>(0, 0) = 1 / srcEigValSqrt.at<float>(0, 0);
	srcEigValSqrtInv.at<float>(1, 1) = 1 / srcEigValSqrt.at<float>(1, 1);
	srcEigValSqrtInv.at<float>(2, 2) = 1 / srcEigValSqrt.at<float>(2, 2);
	cv::Mat A = srcEigVec.t() * srcEigValSqrtInv * eigVecC.t() * eigValSqrtC * eigVecC *
		srcEigValSqrtInv * srcEigVec;
	cv::Mat bias = - A * srcmean + dstmean;
	// apply color correction
	cv::Mat pixelVec(3, src.size().area(), CV_32F);
	int ind = 0;
	for (size_t i = 0; i < src.rows; i++) {
		for (size_t j = 0; j < src.cols; j++) {
			cv::Vec3b val = src.at<cv::Vec3b>(i, j);
			pixelVec.at<float>(0, ind) = static_cast<float>(val.val[0]);
			pixelVec.at<float>(1, ind) = static_cast<float>(val.val[1]);
			pixelVec.at<float>(2, ind) = static_cast<float>(val.val[2]);
			ind++;
		}
	}
	ind = 0;
	pixelVec = A * pixelVec;
	for (size_t i = 0; i < src.rows; i++) {
		for (size_t j = 0; j < src.cols; j++) {
			cv::Rect rect(ind, 0, 1, 3);
			pixelVec(rect) += bias;
			cv::Vec3b val;
			val.val[0] = static_cast<uchar>(std::min<float>(std::max<float>
				(pixelVec.at<float>(0, ind), 0), 255));
			val.val[1] = static_cast<uchar>(std::min<float>(std::max<float>
				(pixelVec.at<float>(1, ind), 0), 255));
			val.val[2] = static_cast<uchar>(std::min<float>(std::max<float>
				(pixelVec.at<float>(2, ind), 0), 255));
			out.at<cv::Vec3b>(i, j) = val;
			ind++;
		}
	}
	return 0;
}