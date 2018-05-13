/**
@brief mesh pyramid class
@author Shane Yuna
@date Mar 3, 2018
*/

#ifndef  __TINY_GIGA_COLOR_CORRECT_H__
#define __TINY_GIGA_COLOR_CORRECT_H__

// include stl
#include <iostream>
#include <cstdlib>
#include <memory>

// opencv
#include <opencv2/opencv.hpp>

class ColorCorrect {
#define POSITIVE 2.2204e-16
private:

public:

private:
	/**
	@brief function to make eigen values positive
	@param cv::Mat & eigvalue: input/output eigen value matrix (3x3)
	@return int
	*/
	static int makeEigvaluesPositive(cv::Mat & eigvalues);

public:
	ColorCorrect();
	~ColorCorrect();

	/**
	@brief static function to calculate image color mean, covariance matrix
	@param cv::Mat input: input image
	@param cv::Mat & outmean: output color mean vector
	@param cv::Mat & outcov: output color covariance matrix
	@param cv::Mat mask = cv::Mat(): input mask for image
	@return int
	*/
	static int calcMeanCov(cv::Mat input, cv::Mat & outmean,
		cv::Mat & outcov, cv::Mat mask = cv::Mat());

	/**
	@brief static function for color correction
	@param cv::Mat src: input source image
	@param cv::Mat dst: input destination image
	@param cv::Mat & out: output color corrected image
	@return int
	*/
	static int correct(cv::Mat src, cv::Mat dst, cv::Mat & out);
};

#endif