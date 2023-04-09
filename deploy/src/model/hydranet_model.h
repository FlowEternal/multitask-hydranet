// ——————————————————————————————————————————————————————————————————————————————
// File Name	:hydranet_model.h
// Abstract 	:hydranet
// Version  	:1.0
// Author		:zhan dong xu
// Date			:2021/09/12
// ——————————————————————————————————————————————————————————————————————————————
#ifndef ONNX_Hydranet_MODEL_H
#define ONNX_Hydranet_MODEL_H

#include <string>
#include <algorithm>  

#if defined (_WINDOWS)
#include <Windows.h>
#else
#include <sys/time.h> 
#endif

// —————————————————————————
// ———————— 接口头文件 ———————
// —————————————————————————
#include <Hydranet.h>

// ——————————————————————————
// ———————— ONNX头文件 ———————
// ——————————————————————————
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>

#define USING_TENSORRT_ARM true
#if not defined (_WINDOWS)
#if USING_TENSORRT_ARM
#include <tensorrt_provider_factory.h>
#endif
#endif 

// ————————————————————————————————
// ———————— Network  宏定义 ————————
// ————————————————————————————————
#define NET_INPUT_NAME					"input"
#define NET_INPUT_HEIGHT				640
#define NET_INPUT_WIDTH					640
#define NET_INPUT_CHANNEL				3
static constexpr const int INPUT_WIDTH = NET_INPUT_WIDTH;
static constexpr const int INPUT_HEIGHT = NET_INPUT_HEIGHT;
static constexpr const int INPUT_CHANNEL = NET_INPUT_CHANNEL;


// ——————————————————————————————
// ———————— 车道线分支参数 ————————
// ——————————————————————————————
// 宏参
#define OUTPUT_LANE_REG							"lane_reg"
#define OUTPUT_LANE_CLS							"lane_cls"
#define LANE_EXIST_DIM							2		// 这里恒定为2
#define ANCHOR_INTERVAL							8
#define POINTS_PER_ANCHOR						4
#define STRIDE									32
#define STRIDE_WIDTH							32
#define STRIDE_HEIGHT							32
#define POINTS_PER_LINE							int(NET_INPUT_HEIGHT/float(ANCHOR_INTERVAL))
#define LANE_FEATURE_WIDTH						int(NET_INPUT_WIDTH/float(STRIDE))
#define LANE_FEATURE_HEIGHT						int(NET_INPUT_HEIGHT/float(STRIDE))
#define LANE_ANCHOR								LANE_FEATURE_WIDTH * LANE_FEATURE_HEIGHT
#define LANE_REGRESSION_DIM						(POINTS_PER_LINE + 1) * 2
// 算法参数
#define CONF_THRESHOLD_DEFAULT					0.90		// 需要调节
#define NMS_THRESHOLD							100
#define USE_MEAN_DISTANCE						false
#define MIN_LANE_LENGTH							2
#define MARGINE_DOWN_BRANCH						100
#define NMS_THRESHOLD_TOP_K						14		// 设定的车道线最大检测数量
#define LANE_WIDTH								20
#define RATIO_THRESHOLD							0.01
static constexpr const int lane_anchor = LANE_ANCHOR;
static constexpr const int lane_feature_width = LANE_FEATURE_WIDTH;
static constexpr const int lane_feature_height = LANE_FEATURE_HEIGHT;
static constexpr const int lane_exist_dim = LANE_EXIST_DIM;
static constexpr const int lane_regression_dim = LANE_REGRESSION_DIM;

// NMS结构体定义
typedef struct {
	std::vector<cv::Point> lane_pts;
	float score;
	int start_pos;
	int end_pos;
	float anchor_center_x;
	float anchor_center_y;
	int index;
}Lane;

// ————————————————————————————————
// ———————— 语义分割分支参数 ————————
// ————————————————————————————————
#define OUTPUT_SEG									"seg"
#define OUTPUT_SEG_HEIGHT							NET_INPUT_HEIGHT
#define OUTPUT_SEG_WIDTH							NET_INPUT_WIDTH
#define MERGE_MASK_IMG								true
static constexpr const int output_seg_width = OUTPUT_SEG_WIDTH;
static constexpr const int output_seg_height = OUTPUT_SEG_HEIGHT;

// ————————————————————————————————
// ———————— 目标检测分支参数 ————————
// ————————————————————————————————
#define OUTPUT_DETECTION_ANCHOR						"anchors"
#define OUTPUT_DETECTION_REGRESSION					"regression"
#define OUTPUT_DETECTION_CLASSIFICATION				"classification"
// alogrithm相关宏
#define TOTAL_ANCHOR_NUM		76725
#define CLASS_NUM				9
#define BBOX_DIM				4
#define MERGE_OUTPUT_DIM		6
#define SCORE_THRESHOLD			0.4
#define	IOU_THRESHOLD			0.3
#define MAX_SINGLE_CLASS_NUM	100
static constexpr const int Anchor_num = TOTAL_ANCHOR_NUM;
static constexpr const int Class_num = CLASS_NUM;
static constexpr const int Bbox_dim = BBOX_DIM;


namespace hydranet 
{

	namespace hydranet_detection 
	{

		class hydranet_model
		{

		public:
			// default constructor
			hydranet_model(std::string model_path);

			void detect(const cv::Mat& input_image,cv::Mat& visual_image, Output_Info output_info);

		private:

			// inference engine
			Ort::Session session_{ nullptr };
			Ort::Env env_{nullptr};

			// timer
			std::chrono::steady_clock::time_point tic;
			std::chrono::steady_clock::time_point tac;
			std::chrono::steady_clock::time_point tic_inner;
			std::chrono::steady_clock::time_point tac_inner;
			double time_used = 0;

			// image original size
			cv::Size _m_input_node_size_host = cv::Size(INPUT_WIDTH,INPUT_HEIGHT);

			// original image dimension
			float org_img_width = 0.0f;
			float org_img_height = 0.0f;

			// ———————————————————————————————————————
			// ———————————  input related ———————————
			// ———————————————————————————————————————
			// member variable
			std::array<float, INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNEL> input_image_{};
			std::array<int64_t, 4> input_image_shape_{ 1,INPUT_CHANNEL,INPUT_HEIGHT, INPUT_WIDTH };
			Ort::Value input_image_tensor_{ nullptr };
			// member function
			void preprocess(const cv::Mat& input_image, cv::Mat& output_image, cv::Mat &visual_img);

			// ———————————————————————————————————————
			// ———————————  lane detection ———————————
			// ———————————————————————————————————————
			// output lane exist
			std::array<float, 1 * lane_anchor * lane_exist_dim> pred_cls_{};
			std::array<int64_t, 3> pred_cls_shape_{ 1,LANE_ANCHOR, LANE_EXIST_DIM };
			Ort::Value pred_cls_tensor_{ nullptr };

			// output reg
			std::array<float, 1 * lane_anchor * lane_regression_dim> pred_reg_{};
			std::array<int64_t, 3> pred_reg_shape_{ 1,LANE_ANCHOR, LANE_REGRESSION_DIM };
			Ort::Value pred_reg_tensor_{ nullptr };

			// threshold related
			float conf_threshold = CONF_THRESHOLD_DEFAULT;
			float nms_top_k = NMS_THRESHOLD_TOP_K;

			void postprocess_lane(float* output_cls_ptr, float* output_reg_ptr, 
				std::vector< Lane_Info > & process_result, cv::Mat & visual_image,
				cv::Mat & seg_mask_postprocess);

			void draw_lane_line(Lane_Info & one_lane, cv::Mat& visual_img);

			// ———————————————————————————————————————
			// ———————————  segmentation  ———————————
			// ———————————————————————————————————————
			std::array<int64_t, output_seg_width * output_seg_height> seg_results_{};
			std::array<int64_t, 3> seg_output_shape_{ 1,OUTPUT_SEG_HEIGHT, OUTPUT_SEG_WIDTH };
			Ort::Value seg_output_tensor_{ nullptr };

			void postprocess_seg(int64_t* output_data_ptr, cv::Mat& visual_img, cv::Mat& seg_mask, cv::Mat& seg_mask_postprocess);

			// ——————————————————————————————————————————
			// ———————————  object detection  ———————————
			// ——————————————————————————————————————————
			// detection anchor
			std::array<float, 1 * Anchor_num *Bbox_dim> det_anchor_{};
			std::array<int64_t, 3> det_anchor_shape_{ 1,Anchor_num, Bbox_dim };
			Ort::Value det_anchor_tensor_{ nullptr };

			// detection regression
			std::array<float, 1 * Anchor_num *Bbox_dim> det_pred_bbox_{};
			std::array<int64_t, 3> det_pred_bbox_shape_{ 1,Anchor_num, Bbox_dim };
			Ort::Value det_pred_bbox_tensor_{ nullptr };

			// detection class score
			std::array<float, 1 * Anchor_num *Class_num> det_pred_class_score_{};
			std::array<int64_t, 3> det_pred_class_score_shape_{ 1,Anchor_num, Class_num };
			Ort::Value det_pred_class_score_tensor_{ nullptr };

			void postprocess_detection(float* output_anchor_ptr, 
										float * output_reg_ptr, 
										float * output_cls_ptr,
										std::vector<Detection_Info>& detect_result,
										cv::Mat& visual_img);

		};

	}

}

#endif //ONNX_Hydranet_MODEL_H
