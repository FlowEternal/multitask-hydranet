// ——————————————————————————————————————————————————————————————————————————————
// File Name	:yolo_model.h
// Abstract 	:yolo model
// Version  	:1.0
// Author		:zhan dong xu
// Date			:2021/01/12
// ——————————————————————————————————————————————————————————————————————————————
#ifndef ONNX_YOLO_MODEL_H
#define ONNX_YOLO_MODEL_H

#include <memory>
#include <string>
// ——————————————————————————————————
// ———————— opencv和ONNX头文件 ————————
// ——————————————————————————————————
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>

// ———————————————————————
// ———————— 宏定义 ————————
// ———————————————————————

// alogrithm相关宏
#define INPUT_NAME				"input:0"
#define OUTPUT_ONE_NAME			"Identity:0"
#define OUTPUT_TWO_NAME			"Identity_1:0"
#define OUTPUT_THREE_NAME		"Identity_2:0"
#define NET_INPUT_HEIGHT		416
#define NET_INPUT_WIDTH			416
#define TOTAL_ANCHOR_NUM		2535
#define CLASS_NUM				80
#define BBOX_DIM				4
#define MERGE_OUTPUT_DIM		6

// 维度宏参
static constexpr const int INPUT_WIDTH = NET_INPUT_WIDTH;
static constexpr const int INPUT_HEIGHT = NET_INPUT_HEIGHT;
static constexpr const int INPUT_CHANNEL = 3;
static constexpr const int Anchor_num = TOTAL_ANCHOR_NUM;
static constexpr const int Class_num = CLASS_NUM;
static constexpr const int Bbox_dim = BBOX_DIM;

// 算法参数
#define SCORE_THRESHOLD			0.5
#define	IOU_THRESHOLD			0.5
#define MAX_SINGLE_CLASS_NUM	100

namespace yolo 
{
	namespace yolo_detection 
	{

		class YOLO
		{
		public:
			YOLO();

			void detect(const cv::Mat& input_image,
				Ort::Session& session_,
				cv::Mat& visual_image,
				std::vector<std::array<float, MERGE_OUTPUT_DIM>> & process_result);

		private:
			// ONNX UFLD input graph node tensor size
			cv::Size _m_input_node_size_host = cv::Size(INPUT_WIDTH,INPUT_HEIGHT);
			// successfully init model flag
			bool _m_successfully_initialized = false;

			// 输入张量定义
			std::array<float, INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNEL> input_image_{};
			std::array<int64_t, 4> input_image_shape_{ 1,INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL };
			Ort::Value input_image_tensor_{ nullptr };

			// output

			std::array<float, 1 * Anchor_num *Bbox_dim> pred_bbox_{};
			std::array<int64_t, 3> pred_bbox_shape_{ 1,Anchor_num, Bbox_dim };
			Ort::Value pred_bbox_tensor_{ nullptr };


			std::array<float, 1 * Anchor_num *1> pred_confidence_score_{};
			std::array<int64_t, 3> pred_confidence_score_shape_{ 1,Anchor_num, 1 };
			Ort::Value pred_confidence_score_tensor_{ nullptr };


			std::array<float, 1 * Anchor_num *Class_num> pred_class_score_{};
			std::array<int64_t, 3> pred_class_score_shape_{ 1,Anchor_num, Class_num };
			Ort::Value pred_class_score_tensor_{ nullptr };

			/***
			 * Preprocess image, resize image and scale image according to python script
			 * @param input_image
			 * @param output_image
			 */
			void preprocess(const cv::Mat& input_image, cv::Mat& output_image);

			/***
			 * Postprocess output tensor and get result matrix for point abstract and visualization
			 * @param output_tensor
			 * @param result_mat
			 */
			void postprocess(float* output_ptr_bbox,					// [1,2535,4]
				float* output_ptr_class_score,			// [1,2535,80]
				float* output_ptr_confidence_score,		// [1,2535,1]
				std::vector<std::array<float, MERGE_OUTPUT_DIM>> & process_result,
				cv::Mat & visual_image);

				
		};

	}

}


#endif //ONNX_YOLO_MODEL_H
