// ——————————————————————————————————————————————————————————————————————————————
// File Name	:yolo_model.cpp
// Abstract 	:yolo
// Version  	:1.0
// Author		:zhan dong xu
// Date			:2021/01/15
// ——————————————————————————————————————————————————————————————————————————————

#include "yolo_model.h"
#include <iostream>     // std::cout
#include <numeric>      // std::iota

#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>


// 算法和显示
#define INIT_MAX_VALUE		-100000
#define MININAL_LANE_POINT	2
#define VISUAL_POINT_RADIUS	1
#define VISUAL_POINT_WIDTH	-1
#define VISUAL_POINT_COLOR	cv::Scalar(0,255,0)
#define DRAW_POINT_NUM		20
#define RATIO_MAX_Y			1.4

#define MAX(a,b)	(a>b)?a:b
#define MIN(a,b)	(a<b)?a:b

std::vector<std::string> my_class_name = {

	"person",
	"bicycle",
	"car",
	"motorbike",
	"aeroplane",
	"bus",
	"train",
	"truck",
	"boat",
	"traffic light",
	"fire hydrant",
	"stop sign",
	"parking meter",
	"bench",
	"bird",
	"cat",
	"dog",
	"horse",
	"sheep",
	"cow",
	"elephant",
	"bear",
	"zebra",
	"giraffe",
	"backpack",
	"umbrella",
	"handbag",
	"tie",
	"suitcase",
	"frisbee",
	"skis",
	"snowboard",
	"sports ball",
	"kite",
	"baseball bat",
	"baseball glove",
	"skateboard",
	"surfboard",
	"tennis racket",
	"bottle",
	"wine glass",
	"cup",
	"fork",
	"knife",
	"spoon",
	"bowl",
	"banana",
	"apple",
	"sandwich",
	"orange",
	"broccoli",
	"carrot",
	"hot dog",
	"pizza",
	"donut",
	"cake",
	"chair",
	"sofa",
	"pottedplant",
	"bed",
	"diningtable",
	"toilet",
	"tvmonitor",
	"laptop",
	"mouse",
	"remote",
	"keyboard",
	"cell phone",
	"microwave",
	"oven",
	"toaster",
	"sink",
	"refrigerator",
	"book",
	"clock",
	"vase",
	"scissors",
	"teddy bear",
	"hair drier",
	"toothbrush",
};

// nms部分
typedef struct {
	cv::Rect2f box;
	float score;
	int index;
}BBOX;

static float get_iou_value(cv::Rect2f rect1, cv::Rect2f rect2)
{
	float xx1, yy1, xx2, yy2;

	xx1 = MAX(rect1.x, rect2.x);
	yy1 = MAX(rect1.y, rect2.y);
	xx2 = MIN(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
	yy2 = MIN(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

	float insection_width, insection_height;
	insection_width = MAX(0, xx2 - xx1 + 1);
	insection_height = MAX(0, yy2 - yy1 + 1);

	float insection_area, union_area, iou;
	insection_area = float(insection_width) * insection_height;
	union_area = float(rect1.width*rect1.height + rect2.width*rect2.height - insection_area);
	iou = insection_area / union_area;
	return iou;
}

bool cmpScore(BBOX lsh, BBOX rsh) {
	if (lsh.score > rsh.score)
		return true;
	else
		return false;
}

//input:  boxes: 原始检测框集合;
//input:  score：confidence * class_prob
//input:  confThreshold 和 nmsThreshold 分别是 检测框置信度阈值以及做nms时的阈值
//output: indices  经过上面两个阈值过滤后剩下的检测框的index
void nms_boxes(std::vector<cv::Rect2f> &boxes,
	std::vector<float> &scores,
	float scoreThreshold,
	float nmsThreshold,
	std::vector<int> &indices)
{
	BBOX bbox;
	std::vector<BBOX> bboxes;
	int i, j;
	for (i = 0; i < boxes.size(); i++)
	{
		bbox.box = boxes[i];
		bbox.score = scores[i];
		bbox.index = i;
		if (bbox.score > scoreThreshold)
		{
			bboxes.push_back(bbox);
		}

	}
	sort(bboxes.begin(), bboxes.end(), cmpScore);



	int updated_size = bboxes.size();
	for (i = 0; i < updated_size; i++)
	{

		indices.push_back(bboxes[i].index);
		for (j = i + 1; j < updated_size; j++)
		{
			float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
			if (iou > nmsThreshold)
			{
				bboxes.erase(bboxes.begin() + j);
				updated_size = bboxes.size();
			}
		}
	}


}

namespace yolo
{
	namespace yolo_detection
	{
		YOLO::YOLO()
		{



		}
		void YOLO::detect(const cv::Mat& input_image,
			Ort::Session& session_,
			cv::Mat& visual_image,
			std::vector<std::array<float, MERGE_OUTPUT_DIM>> & process_result)
		{

			// preprocess
			cv::Mat input_image_copy;
			input_image.copyTo(input_image_copy);

			preprocess(input_image, input_image_copy);

			float* output = input_image_.data();

			// error in linux
#if defined(_WINDOWS)
			fill(input_image_.begin(), input_image_.end(), 0.f);
#else
#endif	
			const int row = INPUT_HEIGHT;
			const int col = INPUT_WIDTH;

			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < col; j++)
				{

					for (int c = 0; c < 3; c++)
					{
						output[i*col * 3 + j * 3 + c] = (input_image_copy.ptr<float>(i)[j * 3 + c]);
					}

				}
			}

			const char* input_names[] = { INPUT_NAME };
			const char* output_names[] = { OUTPUT_ONE_NAME,OUTPUT_TWO_NAME,OUTPUT_THREE_NAME };
			auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
			input_image_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				input_image_.data(),
				input_image_.size(),
				input_image_shape_.data(),
				input_image_shape_.size());

			std::vector<Ort::Value> inputs_tensor;
			std::vector<Ort::Value> outputs_tensor;

			inputs_tensor.push_back(std::move(input_image_tensor_));

			// output
			pred_bbox_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				pred_bbox_.data(),
				pred_bbox_.size(),
				pred_bbox_shape_.data(),
				pred_bbox_shape_.size());

			pred_confidence_score_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				pred_confidence_score_.data(),
				pred_confidence_score_.size(),
				pred_confidence_score_shape_.data(),
				pred_confidence_score_shape_.size());

			pred_class_score_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				pred_class_score_.data(),
				pred_class_score_.size(),
				pred_class_score_shape_.data(),
				pred_class_score_shape_.size());

			outputs_tensor.push_back(std::move(pred_bbox_tensor_));
			outputs_tensor.push_back(std::move(pred_confidence_score_tensor_));
			outputs_tensor.push_back(std::move(pred_class_score_tensor_));

			clock_t start, end;
			start = clock();			//程序开始计时

			session_.Run(Ort::RunOptions{ nullptr },
				input_names,
				inputs_tensor.data(), inputs_tensor.size(),
				output_names,
				outputs_tensor.data(), outputs_tensor.size());

			end = clock();				//程序结束用时
			double endtime = (double)(end - start) / CLOCKS_PER_SEC;
			std::cout << "3.run session time: " << endtime * 1000 << " ms" << std::endl;

			// ————————————————————————————
			// ———————— 处理输出张量 ————————
			// ————————————————————————————
			start = clock();			//程序开始计时
			// 创建结果矩阵并进行后处理
			float* output_ptr_bbox = outputs_tensor[0].GetTensorMutableData<float>();
			float* output_ptr_confidence_score = outputs_tensor[1].GetTensorMutableData<float>();
			float* output_ptr_class_score = outputs_tensor[2].GetTensorMutableData<float>();
			visual_image = input_image.clone();
			postprocess(output_ptr_bbox, output_ptr_class_score, 
				output_ptr_confidence_score, process_result, visual_image);
			end = clock();				//程序结束用时
			endtime = (double)(end - start) / CLOCKS_PER_SEC;
			std::cout << "4.post processing time: " << endtime * 1000 << " ms" << std::endl;


		}



		/***************Private Function Sets*******************/

		/***
		 * Resize image and scale image into [-1.0, 1.0]
		 * @param input_image
		 * @param output_image
		 */
		void YOLO::preprocess(const cv::Mat &input_image, cv::Mat& output_image)
		{

			clock_t start, end;
			start = clock();			//程序开始计时
			if (input_image.type() != CV_32FC3)
			{
				input_image.convertTo(output_image, CV_32FC3);
			}

			if (output_image.size() != _m_input_node_size_host)
			{
				cv::resize(output_image, output_image, _m_input_node_size_host);
			}
			end = clock();				//程序结束用时
			double endtime = (double)(end - start) / CLOCKS_PER_SEC;
			std::cout << "1.preprocess image resize time: " << endtime * 1000 << " ms" << std::endl;

			start = clock();			//程序开始计时
			cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
			cv::divide(output_image, cv::Scalar(255.0f, 255.0f, 255.0f), output_image);
			end = clock();				//程序结束用时
			endtime = (double)(end - start) / CLOCKS_PER_SEC;
			std::cout << "2.preprocess image normalization time: " << endtime * 1000 << " ms" << std::endl;


		}

		/***
		 * Postprocess output tensor and get result matrix for point abstract and visualization
		 * @param output_tensor
		 * @param result_mat
		 */
		void YOLO::postprocess(float* output_ptr_bbox,					// [1,2535,4]
			float* output_ptr_class_score,			// [1,2535,80]
			float* output_ptr_confidence_score,		// [1,2535,1]
			std::vector<std::array<float, MERGE_OUTPUT_DIM>> & process_result,
			cv::Mat & visual_image)
		{

			int org_width = visual_image.cols;
			int org_height = visual_image.rows;

			std::vector<cv::Rect2f> boxes;
			for (int anchor_idx = 0; anchor_idx < Anchor_num; anchor_idx++)
			{
				float * tmp_bbx_ptr = output_ptr_bbox + anchor_idx * 4;
				float _x = tmp_bbx_ptr[0];
				float _y = tmp_bbx_ptr[1];
				float _width = tmp_bbx_ptr[2] - tmp_bbx_ptr[0];
				float _height = tmp_bbx_ptr[3] - tmp_bbx_ptr[1];

				cv::Rect2f tmp_rect(_x, _y, _width, _height);
				boxes.push_back(tmp_rect);

			}

			// 循环
			for (int class_idx = 0; class_idx < CLASS_NUM; class_idx++)
			{


				std::vector<float> scores;
				for (int anchor_idx = 0; anchor_idx < Anchor_num; anchor_idx++)
				{
					float confidence = *(output_ptr_confidence_score + anchor_idx);
					float * one_class_prob_ptr = (output_ptr_class_score + anchor_idx * CLASS_NUM);

					float tmp_score = confidence * one_class_prob_ptr[class_idx];
					scores.push_back(tmp_score);

				}

				float score_threshold = SCORE_THRESHOLD;
				float nms_threshold = IOU_THRESHOLD;
				std::vector<int> indices;
				nms_boxes(boxes, scores, score_threshold, nms_threshold, indices);

				// max 100 
				int max_iter = MIN((int)indices.size(), MAX_SINGLE_CLASS_NUM);
				for (int iter_num = 0; iter_num < max_iter; iter_num++)
				{

					std::array<float, MERGE_OUTPUT_DIM> tmp_array;
					cv::Rect2f tmp_rect = boxes[indices[iter_num]];
					tmp_array[0] = tmp_rect.x;
					tmp_array[1] = tmp_rect.y;
					tmp_array[2] = tmp_rect.x + tmp_rect.width;
					tmp_array[3] = tmp_rect.y + tmp_rect.height;
					tmp_array[4] = scores[indices[iter_num]];
					tmp_array[5] = static_cast<float>(class_idx);
					process_result.push_back(tmp_array);

					float x_min = (tmp_array[0] * org_width); // x_min 
					float y_min = (tmp_array[1] * org_height); // y_min
					float x_max = (tmp_array[2] * org_width); // x_min 
					float y_max = (tmp_array[3] * org_height); // y_min
					float score = tmp_array[4]; // score
					std::string cls_name = my_class_name[(int)tmp_array[5]];

					cv::rectangle(visual_image, cv::Point(x_min, y_min), cv::Point(x_max, y_max),
						cv::Scalar(0, 255, 0), 2);

					std::string info_str = cls_name + " : " + std::to_string(score);
					cv::putText(visual_image, info_str, cv::Point(x_min, MAX(y_min - 10, 0)),
						cv::FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1);

				}


			}


		}


	}
}