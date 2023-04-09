#pragma once
#if defined (_WINDOWS)
  // windows api import/export
  #ifndef HYDRANET_IMPORTS
    #define HYDRANET_API  __declspec(dllexport)
  #else
    #define HYDRANET_API  __declspec(dllimport)
  #endif
#else
  // linux api
  #define HYDRANET_API __attribute__((visibility("default")))
#endif

#define IN
#define OUT
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

struct Lane_Info
{
	// points coords
	std::vector<cv::Point> lane_pts;

	// one line confidence score
	float conf_score;

	Lane_Info()
	{
		lane_pts.clear();
		conf_score = 0.0f;
	}

};

struct Seg_Info
{

	cv::Mat seg_mask;

};

struct Detection_Info
{
	// bounnding box
	int x1 = 0;
	int x2 = 0;
	int y1 = 0;
	int y2 = 0;

	// class score
	float conf_score;
	// class name 
	std::string class_name;
	// class id
	int class_id;

};

struct Output_Info
{
	// 车道线输出信息
	std::vector<Lane_Info> lanes;

	// 语义分割输出信息
	cv::Mat seg_mask;

	// 目标检测输出信息
	std::vector<Detection_Info> detector;

};


/******************************************************************************************************************
* 功  能：LaneATT_Init函数, 车道线检测初始化函数
* 参  数：( IN/输入参数)：
						handle			单线程空句柄指针
		 (OUT/输出参数)：
						handle			指向lane detector对象的指针
* 返回值：0(正确);非0(不正确)
* 备  注：
******************************************************************************************************************/
HYDRANET_API int Hydranet_Init(IN void **handle, std::string model_path);

/******************************************************************************************************************
* 功  能：LaneATT_Detect函数, 车道线检测
* 参  数：( IN/输入参数)：
						handle					单线程句柄指针
						input_image				单帧图像
						session_detection		车道线检测模型
		 (OUT/输出参数)：
						visual_image		visual information
						output_pts			output points
* 返回值：0(正确);非0(不正确)
* 备  注：
******************************************************************************************************************/
HYDRANET_API int Hydranet_Detect(IN void *handle,
								 IN cv::Mat& input_image,
								 OUT cv::Mat& visual_image,
								 OUT Output_Info & output);

/******************************************************************************************************************
* 功  能：LaneATT_Uinit函数, 车道线检测反始化函数
* 参  数：( IN/输入参数)：
						handle			单线程句柄指针
		 (OUT/输出参数)：

* 返回值：0(正确);非0(不正确)
* 备  注：
******************************************************************************************************************/
HYDRANET_API int Hydranet_Uinit(IN void *handle);
