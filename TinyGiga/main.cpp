/**
@brief main function for tiny giga project 
@author: Shane Yuan
@date: Dec 11, 2017
*/

#include <iostream>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "SysUtil.hpp"
#include "TinyStitcher.h"
#include "MeshPyramid.h"

int main(int argc, char* argv[]) {

	argv[1] = "E:/data/bb/global_80_210.png";
	argv[2] = "E:/data/bb/local_80_210.png";
	//argv[1] = "E:/Project/PanoramaStitcher/TinyGiga/example_data/ref_00.jpg";
	//argv[2] = "E:/Project/PanoramaStitcher/TinyGiga/example_data/local_02.jpg";
	argv[3] = "0.118";
	argv[4] = "warped.jpg";
	argv[5] = "refblk.jpg";


	cudaDeviceReset();
	cudaDeviceSynchronize();

	std::string refname = std::string(argv[1]);
	std::string localname = std::string(argv[2]);
	cv::Mat refImg = cv::imread(refname);
	cv::Mat localImg = cv::imread(localname);
	std::string modelname = "E:/Project/PanoramaStitcher/model/model.yml";
	float scale = atof(argv[3]);


	MeshPyramid pyramid;
	pyramid.init(localImg, refImg, scale);
	pyramid.process();
	//cv::Mat warpedImg = pyramid.getFinalWarpedImage();
	//cv::Mat refBlkImg = pyramid.getRefBlkImage();

	//cv::imwrite(argv[4], warpedImg);
	//cv::imwrite(argv[5], refBlkImg);

	return 0;
}