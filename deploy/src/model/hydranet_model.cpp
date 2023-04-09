// ——————————————————————————————————————————————————————————————————————————————
// File Name	:hydranet_model.cpp
// Abstract 	:hydranet
// Version  	:1.0
// Author		:zhan dong xu
// Date			:2021/09/12
// ——————————————————————————————————————————————————————————————————————————————

#include "hydranet_model.h"

// ——————————————————————————————————————
// ———————————lane相关的显示参数———————————
// ——————————————————————————————————————
#define VISUALIZATION_BOX		false
#define OUTPUT_PT				false
#define VISUAL_POINT_RADIUS		6
#define VISUAL_POINT_WIDTH		-1
#define VISUAL_POINT_COLOR		cv::Scalar(0,255,0)
// line显示参数
#define VISUAL_LINE_WIDTH	6					
// box显示参数
#define WIDTH_BOX_BASE		float(40)		
#define HEIGHT_BOX_BASE		float(40)	
#define BOX_COLOR			cv::Scalar(255,255,0)
#define LINE_COLOR			cv::Scalar(255,255,0)
#define TEXT_SCALE			2					// "Lane"有四个字符
#define BOX_IOU_THRESHOLD	0.000001			// 不让box重合
// 文本显示参数
#define FONT_SCALE_TXT		0.7			
#define THICKNESS_TXT		2					
#define FONT_TYPE			cv::FONT_HERSHEY_COMPLEX
#define TEXT_COLOR			cv::Scalar(0,0,0)
// 虚线相关
#define RATIO_INTERPOLATE	0.5

// —————————————————————————————————————
// ———————————seg相关的显示参数———————————
// —————————————————————————————————————
std::vector<cv::Scalar> draw_color_vec = {
	cv::Scalar(0, 0, 0),
	cv::Scalar(128, 0, 128),
	cv::Scalar(255, 255, 255),
	cv::Scalar(0, 255, 255),
	cv::Scalar(0, 255, 0) 
};

// —————————————————————————————————————
// ———————————det相关的显示参数———————————
// —————————————————————————————————————
std::vector<std::string> detect_vec = {
					  "roadtext",
				  "pedestrian",
				  "guidearrow",
				  "traffic",
				  "obstacle",
				  "vehicle_wheel",
				  "roadsign",
				  "vehicle",
				  "vehicle_light"
	
};

std::vector<cv::Scalar> detect_color_list = {
	cv::Scalar(0,252,124),
	cv::Scalar(0,255,127),
	cv::Scalar(255,255,0),
	cv::Scalar(220,245,245),
	cv::Scalar(255, 255, 240),
	cv::Scalar(205, 235, 255),
	cv::Scalar(196, 228, 255),
	cv::Scalar(212, 255, 127),
	cv::Scalar(226, 43, 138),
	cv::Scalar(135, 184, 222),

};


template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length);

// line iou
inline bool devIoU(Lane a, Lane b, const float threshold);

void nms_boxes(std::vector<Lane> & lanes_input);

// box iou
float cal_iou(cv::Rect rect1, cv::Rect rect2);

bool is_overlap_with_any(std::vector<cv::Rect> box_list, cv::Rect target_rect);

// detection 
static float get_iou_value(cv::Rect2f rect1, cv::Rect2f rect2);
//input:  boxes: 原始检测框集合;
//input:  score：confidence * class_prob
//input:  confThreshold 和 nmsThreshold 分别是 检测框置信度阈值以及做nms时的阈值
//output: indices  经过上面两个阈值过滤后剩下的检测框的index
void nms_boxes_detect(std::vector<Detection_Info> & nms_detect_infos, std::vector<int> & index_choose);



#if defined (_WINDOWS)
wchar_t * char2wchar(const char* cchar)
{
	wchar_t *m_wchar;
	int len = MultiByteToWideChar(CP_ACP, 0, cchar, strlen(cchar), NULL, 0);
	m_wchar = new wchar_t[len + 1];
	MultiByteToWideChar(CP_ACP, 0, cchar, strlen(cchar), m_wchar, len);
	m_wchar[len] = '\0';
	return m_wchar;
}
#endif

namespace hydranet
{

	namespace hydranet_detection
	{

		hydranet_model::hydranet_model(std::string model_path)
		{

			env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Default");
			Ort::SessionOptions session_option;
			session_option.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

#if not defined (_WINDOWS)
			if (USING_TENSORRT_ARM)
			{
				Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_option, 0));
			}
#endif
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));


			// 测量模型加载时间 start
			tic = std::chrono::steady_clock::now();
			std::cout << std::endl;
			std::cout << "(1) Start Model Loading" << std::endl;

#if defined (_WINDOWS)
			const ORTCHAR_T* model_path_convert = char2wchar(model_path.c_str());
			session_ = Ort::Session(env_, model_path_convert, session_option);

#else
			session_ = Ort::Session(env_, model_path.c_str(), session_option);
#endif 


			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Model Loading Time Cost: " << time_used << "ms!" << std::endl;
			std::cout << std::endl;
			// 测量模型加载时间 end


		}


		void hydranet_model::preprocess(const cv::Mat &input_image, cv::Mat& output_image, cv::Mat & visual_img)
		{

			// start
			tic_inner = std::chrono::steady_clock::now();

			if (input_image.size() != _m_input_node_size_host)
			{
				cv::resize(input_image, output_image, _m_input_node_size_host, 0, 0, cv::INTER_LINEAR);
			}


			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Image Resize Time Cost: " << time_used << "ms!" << std::endl;
			std::cout << "-- Original Image Size: Height = " << input_image.size().height << ", Width = " << input_image.size().width << std::endl;
			std::cout << "-- Resized Image Size: Height = " << output_image.size().height << ", Width = " << output_image.size().width << std::endl;
			// end


			// start
			tic_inner = std::chrono::steady_clock::now();
			visual_img = output_image.clone();
			if (output_image.type() != CV_32FC3)
			{
				// 首先转化为RGB
				cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
				// 然后转化为float32
				output_image.convertTo(output_image, CV_32FC3);
				// 进行normalization
				cv::divide(output_image, cv::Scalar(255.0f, 255.0f, 255.0f), output_image);
				cv::subtract(output_image, cv::Scalar(0.485, 0.456, 0.406), output_image);
				cv::divide(output_image, cv::Scalar(0.229, 0.224, 0.225), output_image);
			}

			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Converting Resized Image To Float32 Time Cost: " << time_used << "ms!" << std::endl;
			// end


		}

		void hydranet_model::detect(const cv::Mat& input_image,
			cv::Mat& visual_image,Output_Info output_info)
		{

			// 这里首先做一个可视化备份并获取原始图像尺寸
			org_img_height = input_image.rows;
			org_img_width = input_image.cols;
			std::cout << std::endl;
			std::cout << "(1) Start Input Tensor Preprocess" << std::endl;
			// —————————————————————————————————————
			// ——————————— prepare input ———————————
			// —————————————————————————————————————
			cv::Mat input_image_copy;
			input_image.copyTo(input_image_copy);

			// 测量preprocess时间 start
			tic = std::chrono::steady_clock::now();
			preprocess(input_image, input_image_copy, visual_image);
			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Input Tensor Preprocess Time Cost: " << time_used << "ms!" << std::endl;
			// 测量preprocess时间 end



			// 测量填充tensor时间 start
			tic = std::chrono::steady_clock::now();
			std::cout << std::endl;
			std::cout << "(2) Start Input Tensor Filling" << std::endl;
			float* input_image_ptr = input_image_.data();
#if defined(_WINDOWS)
			fill(input_image_.begin(), input_image_.end(), 0.f);
#else
#endif	
			const int row = INPUT_HEIGHT;
			const int col = INPUT_WIDTH;
			const int channel = INPUT_CHANNEL;

			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < col; j++)
				{
					input_image_ptr[0 * row*col + i * col + j] = (input_image_copy.ptr<float>(i)[j * 3 + 0]);
					input_image_ptr[1 * row*col + i * col + j] = (input_image_copy.ptr<float>(i)[j * 3 + 1]);
					input_image_ptr[2 * row*col + i * col + j] = (input_image_copy.ptr<float>(i)[j * 3 + 2]);
				}
			}

			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Input Tensor Filling Time Cost: " << time_used << "ms!" << std::endl;
			// 测量填充tensor时间 end




			// 测量创建ORT Tensor时间 start
			tic = std::chrono::steady_clock::now();
			std::cout << std::endl;
			std::cout << "(3) Start Tensor Convert" << std::endl;

			// ——————————————————————————————
			// ——————————— tensor ———————————
			// ——————————————————————————————
			const char* input_names[] = { NET_INPUT_NAME };
			const char* output_names[] = { 
				OUTPUT_SEG, 
				OUTPUT_DETECTION_ANCHOR,
				OUTPUT_DETECTION_REGRESSION,
				OUTPUT_DETECTION_CLASSIFICATION,
				OUTPUT_LANE_CLS,
				OUTPUT_LANE_REG
			};

			auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);
			input_image_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				input_image_.data(),
				input_image_.size(),
				input_image_shape_.data(),
				input_image_shape_.size());

			std::vector<Ort::Value> inputs_tensor;
			std::vector<Ort::Value> outputs_tensor;
			inputs_tensor.push_back(std::move(input_image_tensor_));

			////////////////////////////////
			// output seg
			seg_output_tensor_ = Ort::Value::CreateTensor<int64>(memory_info,
				seg_results_.data(),
				seg_results_.size(),
				seg_output_shape_.data(),
				seg_output_shape_.size());

			outputs_tensor.push_back(std::move(seg_output_tensor_));

			////////////////////////////////
			// output detection anchor
			det_anchor_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				det_anchor_.data(),
				det_anchor_.size(),
				det_anchor_shape_.data(),
				det_anchor_shape_.size());


			// output detection regression
			det_pred_bbox_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				det_pred_bbox_.data(),
				det_pred_bbox_.size(),
				det_pred_bbox_shape_.data(),
				det_pred_bbox_shape_.size());

			// output detection classification
			det_pred_class_score_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				det_pred_class_score_.data(),
				det_pred_class_score_.size(),
				det_pred_class_score_shape_.data(),
				det_pred_class_score_shape_.size());

			outputs_tensor.push_back(std::move(det_anchor_tensor_));
			outputs_tensor.push_back(std::move(det_pred_bbox_tensor_));
			outputs_tensor.push_back(std::move(det_pred_class_score_tensor_));

			////////////////////////////////
			// output lane exist
			pred_cls_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				pred_cls_.data(),
				pred_cls_.size(),
				pred_cls_shape_.data(),
				pred_cls_shape_.size());

			// output lane regression
			pred_reg_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				pred_reg_.data(),
				pred_reg_.size(),
				pred_reg_shape_.data(),
				pred_reg_shape_.size());

			outputs_tensor.push_back(std::move(pred_cls_tensor_));
			outputs_tensor.push_back(std::move(pred_reg_tensor_));


			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Tensor Convert Time Cost: " << time_used << "ms!" << std::endl;
			// 测量创建ORT Tensor时间 end


			// Single Forward Inference start
			tic = std::chrono::steady_clock::now();

			std::cout << std::endl;

			std::cout << "(4) Start Single Forward Inference" << std::endl;

			session_.Run(Ort::RunOptions{ nullptr },
				input_names,
				inputs_tensor.data(), inputs_tensor.size(),
				output_names,
				outputs_tensor.data(), outputs_tensor.size());

			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Single Forward Inference Time Cost: " << time_used << "ms!" << std::endl;
			// Single Forward Inference end


			// PostProcess start
			tic = std::chrono::steady_clock::now();

			std::cout << std::endl;

			std::cout << "(5) Start PostProcessing" << std::endl;

			// ————————————————————————————
			// ———————— 处理输出张量 ————————
			// ————————————————————————————

			// 处理语义分割
			int64* output_seg_ptr = outputs_tensor[0].GetTensorMutableData<int64>();
			cv::Mat seg_mask_postprocess;
			postprocess_seg(output_seg_ptr, visual_image, output_info.seg_mask, seg_mask_postprocess);

			// 处理目标
			float* output_det_anchor_ptr = outputs_tensor[1].GetTensorMutableData<float>();
			float* output_det_reg_ptr = outputs_tensor[2].GetTensorMutableData<float>();
			float* output_det_cls_ptr = outputs_tensor[3].GetTensorMutableData<float>();
			postprocess_detection(output_det_anchor_ptr, 
									output_det_reg_ptr, 
									output_det_cls_ptr,
									output_info.detector, 
									visual_image);


			// 处理车道线
			float* output_cls_ptr = outputs_tensor[4].GetTensorMutableData<float>();
			float* output_reg_ptr = outputs_tensor[5].GetTensorMutableData<float>();
			postprocess_lane(output_cls_ptr, output_reg_ptr, output_info.lanes, visual_image, seg_mask_postprocess);

			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--PostProcessing Time Cost: " << time_used << "ms!" << std::endl;
			// Postprocess end

		}

		// ————————————————————————————————
		// ———————— 车道线decode函数 ————————
		// ————————————————————————————————
		void hydranet_model::postprocess_lane(float* output_cls_ptr,
										float* output_reg_ptr, 
										std::vector< Lane_Info > & process_result,
										cv::Mat & visual_image, cv::Mat & seg_mask_postprocess)
		{

			// start
			tic_inner = std::chrono::steady_clock::now();
			std::vector<Lane>		choose_lane_info;

			for (int index_hegiht = 0; index_hegiht < LANE_FEATURE_HEIGHT; index_hegiht++)
				for (int index_width = 0; index_width < LANE_FEATURE_WIDTH; index_width++)
				{

					int anchor_index = index_hegiht * LANE_FEATURE_WIDTH + index_width;

					// anchor cls
					float * tmp_cls_ptr = output_cls_ptr + anchor_index * LANE_EXIST_DIM;
					// anchor reg
					float * tmp_reg_ptr = output_reg_ptr + anchor_index * LANE_REGRESSION_DIM;
					// anchor y pos
					int anchor_y_pos = int((LANE_FEATURE_HEIGHT - 1 - index_hegiht) * POINTS_PER_ANCHOR);
					// anchor center x
					float anchor_center_x = (1.0 * index_width + 0.5) * STRIDE_WIDTH;
					// anchor center y
					float anchor_center_y = (1.0 * index_hegiht + 0.5) * STRIDE_HEIGHT;

					// score filter
					float * softmax_array_conf = new float[LANE_EXIST_DIM];
					activation_function_softmax<float>(tmp_cls_ptr, softmax_array_conf, LANE_EXIST_DIM);

					// 如果confidence小于阈值 直接pass
					if (softmax_array_conf[LANE_EXIST_DIM - 1] < conf_threshold)
					{
						delete[] softmax_array_conf;
						continue;
					}

					Lane lane_obj;

					std::vector<cv::Point> lane_pts;

					// up process
					int end_pos = anchor_y_pos;
					int relative_up_end_pos = tmp_reg_ptr[POINTS_PER_LINE + 1];
					for (int up_index = 0; up_index< POINTS_PER_LINE; up_index++)
					{
						// out of range then break
						if ((up_index > relative_up_end_pos) || ((anchor_y_pos + up_index) > POINTS_PER_LINE))
						{
							break;
						}

						float relative_x = tmp_reg_ptr[POINTS_PER_LINE + 2 + up_index] * ANCHOR_INTERVAL; // scale invariable
						float abs_x = relative_x + anchor_center_x; // out

						// out of range break
						if ((abs_x < 0) || (abs_x > NET_INPUT_WIDTH))
						{
							break;
						}

						float abs_y = NET_INPUT_HEIGHT - 1 - (anchor_y_pos + up_index) * ANCHOR_INTERVAL; // out

						// insert pt
						lane_pts.push_back(cv::Point2f(abs_x, abs_y));

						// refresh
						end_pos += 1;

					}

					// down process
					int start_pos = anchor_y_pos;
					int relative_down_end_pos = tmp_reg_ptr[POINTS_PER_LINE];
					for (int down_index = 0; down_index < anchor_y_pos; down_index++)
					{
						// out of range then break
						if ((down_index > relative_down_end_pos) || ((anchor_y_pos -1 - down_index)< 0))
						{
							break;
						}


						float relative_x = tmp_reg_ptr[down_index] * ANCHOR_INTERVAL; // scale invariable
						float abs_x = relative_x + anchor_center_x; // out

						// out of range break
						if ((abs_x < 0) || (abs_x >= NET_INPUT_WIDTH + MARGINE_DOWN_BRANCH))
						{
							break;
						}

						float abs_y = NET_INPUT_HEIGHT - 1 - (anchor_y_pos - 1 - down_index) * ANCHOR_INTERVAL; // out

						// insert pt
						lane_pts.insert(lane_pts.begin(), cv::Point2f(abs_x, abs_y));

						// refresh
						start_pos -=1;

					}

					// wheater total len > 2
					if (lane_pts.size() > MIN_LANE_LENGTH )
					{
						// save 
						lane_obj.index = anchor_index;
						lane_obj.anchor_center_x = anchor_center_x;
						lane_obj.anchor_center_y = anchor_center_y;
						lane_obj.start_pos = start_pos;
						lane_obj.end_pos = end_pos;
						lane_obj.score = softmax_array_conf[LANE_EXIST_DIM - 1];
						lane_obj.lane_pts = lane_pts;
						int max_index = 0;
						float max_value = 0.0;
						choose_lane_info.push_back(lane_obj);

					}


					delete[] softmax_array_conf;


				}


			std::cout << "-- proposal anchor line number before nms: " << choose_lane_info.size() << std::endl;

			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Anchor Thresholding And Preparing before NMS Time Cost: " << time_used << "ms!" << std::endl;
			// end



			// start
			tic_inner = std::chrono::steady_clock::now();
			nms_boxes(choose_lane_info);
			std::cout << "-- proposal anchor line number after nms: " << choose_lane_info.size() << std::endl;

			if (choose_lane_info.size() == 0)
			{
				return;
			}

			int further_rm_num = choose_lane_info.size() - nms_top_k;
			while (further_rm_num-- > 0)
			{
				choose_lane_info.pop_back();
			}

			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Line NMS Time Cost: " << time_used << "ms!" << std::endl;
			// end


			// 通过分割网络过滤过检的车道线
			std::vector<Lane>		choose_lane_info_refine;
			for (int index = 0; index < choose_lane_info.size(); index++)
			{
				Lane one_lane_info = choose_lane_info[index];

				cv::Mat painter = cv::Mat::zeros(NET_INPUT_HEIGHT, NET_INPUT_WIDTH, CV_8UC1);
				for (int lane_idx = 0; lane_idx < one_lane_info.lane_pts.size()-1; lane_idx++)
				{
					// scale back
					int x_coord1 = float(one_lane_info.lane_pts[lane_idx].x);
					int y_coord1 = float(one_lane_info.lane_pts[lane_idx].y);
					cv::Point tmp_pts1 = cv::Point(x_coord1, y_coord1);

					int x_coord2 = float(one_lane_info.lane_pts[lane_idx+1].x);
					int y_coord2 = float(one_lane_info.lane_pts[lane_idx+1].y);
					cv::Point tmp_pts2 = cv::Point(x_coord2, y_coord2);

					cv::line(painter, tmp_pts1, tmp_pts2, 1, LANE_WIDTH);;


				}
				
				cv::Mat merge_lane_seg;
				cv::bitwise_and(seg_mask_postprocess, painter, merge_lane_seg);
				cv::Scalar sum_pixel = cv::sum(merge_lane_seg);
				long int sum_pixel_gray = sum_pixel[0];

				cv::Scalar sum_pixel_lane = cv::sum(painter);
				long int sum_pixel_lane_gray = sum_pixel_lane[0];

				float ratio = float(sum_pixel_gray) / float(sum_pixel_lane_gray);
				std::cout << ratio << std::endl;

				if (ratio > RATIO_THRESHOLD)
				{
					choose_lane_info_refine.push_back(one_lane_info);
				}

			}


			// start
			tic_inner = std::chrono::steady_clock::now();
			for (int index = 0; index < choose_lane_info_refine.size(); index++)
			{
				Lane one_lane_info = choose_lane_info_refine[index];

				Lane_Info output_one_lane_info;

				std::vector<cv::Point> output_one_lane_pts;
				for (int lane_idx = 0; lane_idx < one_lane_info.lane_pts.size() ; lane_idx++)
				{
					// scale back
					int scaled_x_coord = float(one_lane_info.lane_pts[lane_idx].x) / NET_INPUT_WIDTH * org_img_width;
					int scaled_y_coord = float(one_lane_info.lane_pts[lane_idx].y) / NET_INPUT_HEIGHT * org_img_height;
					output_one_lane_pts.push_back(cv::Point(scaled_x_coord, scaled_y_coord));
					continue;

				}

				output_one_lane_info.lane_pts = output_one_lane_pts;
				output_one_lane_info.conf_score = one_lane_info.score;
				process_result.push_back(output_one_lane_info);

			}


			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Decoding Line Information Time Cost: " << time_used << "ms!" << std::endl;
			// end


			// start
			tic_inner = std::chrono::steady_clock::now();

			// —————————————————————————————————————
			// ——————————— 可视化显示车道线 ———————————
			// —————————————————————————————————————
			std::vector<cv::Rect>	box_list;
			for (int lane_idx = 0; lane_idx < process_result.size(); lane_idx++)
			{

				std::vector<cv::Point> & tmp_pts = process_result[lane_idx].lane_pts;

				if (tmp_pts.size() < 2)
				{
					continue;
				}


				// —————————————————————————————————————
				// ——————————— 车道线点连成线 —————————————
				// —————————————————————————————————————
				draw_lane_line(process_result[lane_idx], visual_image);

				std::string info_type = "Lane"; // 这里用choose_lane_info的type

				int width_box = int(WIDTH_BOX_BASE / TEXT_SCALE * (info_type.length()));
				int height_box = int(HEIGHT_BOX_BASE);

				// —————————————————————————————————————————
				// ——————————— 显示车道线的置信度 —————————————
				// —————————————————————————————————————————
				float cof_score_round = process_result[lane_idx].conf_score;
				std::string info_conf = std::to_string(cof_score_round).substr(0, 4);


				// 这里开始进行box的画图
				// 保证每个box不相交
				float text_x = 0;
				float text_y = 0;
				int counter = -1;
				cv::Rect text_box;
				cv::Point pt1;
				cv::Point pt2;

				do
				{

					counter++; // 从0 开始
					if (counter >= process_result[lane_idx].lane_pts.size())
					{
						break;
					}

					text_x = process_result[lane_idx].lane_pts[counter].x / org_img_width * NET_INPUT_WIDTH;
					text_y = process_result[lane_idx].lane_pts[counter].y / org_img_height * NET_INPUT_HEIGHT;
					pt1 = cv::Point(text_x, text_y);
					pt2 = cv::Point(text_x + width_box, text_y - 2 * height_box);
					text_box = cv::Rect(pt1, pt2);

				} while ((text_x + width_box >= NET_INPUT_WIDTH) || (is_overlap_with_any(box_list, text_box)));

				box_list.push_back(text_box);
				cv::rectangle(visual_image, text_box, BOX_COLOR, -1); // TODO
				cv::Point text_center_conf = cv::Point(text_x, text_y);
				cv::Point text_center_type = cv::Point(text_x, text_y - height_box);

				// line type
				cv::putText(visual_image, info_conf, text_center_conf,
					FONT_TYPE, FONT_SCALE_TXT,
					TEXT_COLOR, THICKNESS_TXT, 8, 0);

				// confidence
				cv::putText(visual_image, info_type, text_center_type,
					FONT_TYPE, FONT_SCALE_TXT,
					TEXT_COLOR, THICKNESS_TXT, 8, 0);


			}

			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Visualization Time Cost: " << time_used << "ms!" << std::endl;
			// end


		}


		void hydranet_model::draw_lane_line(Lane_Info & one_lane,cv::Mat & visual_img)
		{
			std::vector<cv::Point> tmp_pts = one_lane.lane_pts;

			for (int idx = 0; idx < tmp_pts.size() - 1; idx++)
			{

				float coord_x_ = tmp_pts[idx].x / org_img_width * NET_INPUT_WIDTH;
				float coord_y_ = tmp_pts[idx].y / org_img_height * NET_INPUT_HEIGHT;
				cv::Point2f pt1 = cv::Point2f(coord_x_, coord_y_);

				float coord_x__ = tmp_pts[idx + 1].x / org_img_width * NET_INPUT_WIDTH;
				float coord_y__ = tmp_pts[idx + 1].y / org_img_height * NET_INPUT_HEIGHT;
				cv::Point2f pt2 = cv::Point2f(coord_x__, coord_y__);

				int x_new = int(pt2.x * RATIO_INTERPOLATE + (1 - RATIO_INTERPOLATE)*pt1.x);
				int y_new = int(pt2.y * RATIO_INTERPOLATE + (1 - RATIO_INTERPOLATE)*pt1.y);
				cv::Point2f pt2_new = cv::Point2f(x_new, y_new);

				cv::line(visual_img, pt1, pt2, LINE_COLOR, VISUAL_LINE_WIDTH, cv::LineTypes::FILLED);

				}

			}

		// ——————————————————————————————————
		// ———————— 语义分割decode函数 ————————
		// ——————————————————————————————————
		void hydranet_model::postprocess_seg(int64_t* output_data_ptr, cv::Mat& visual_img, cv::Mat& seg_mask, cv::Mat& seg_mask_postprocess)
		{
			seg_mask = cv::Mat::zeros(NET_INPUT_HEIGHT, NET_INPUT_WIDTH, CV_8UC1);
			seg_mask_postprocess = cv::Mat::zeros(NET_INPUT_HEIGHT, NET_INPUT_WIDTH, CV_8UC1);

			uchar* seg_mask_ptr = seg_mask.data;
			uchar* seg_mask_ptr_postprocess = seg_mask_postprocess.data;

			cv::Mat org_img = visual_img.clone();

			// 进行softmax和期望值计算
			for (int height = 0; height < seg_mask.rows; height++)
			{
				for (int width = 0; width < seg_mask.cols; width++)
				{

					int64 tmp_cls_id = *(output_data_ptr + height * seg_mask.cols + width);
					*(seg_mask_ptr + height * seg_mask.cols + width) = (uchar)tmp_cls_id;

					// lane segmentation
					*(seg_mask_ptr_postprocess + height * seg_mask_postprocess.cols + width) = (tmp_cls_id == 2) ? 1 : 0;

				}


			}


			// 可视化
			for (int height = 0; height < visual_img.rows; height++)
			{
				for (int width = 0; width < visual_img.cols; width++)
				{

					int index = *((uchar*)seg_mask.data + height * visual_img.cols + width);
					visual_img.ptr<uchar>(height)[width * 3 + 0] = draw_color_vec[index].val[0];
					visual_img.ptr<uchar>(height)[width * 3 + 1] = draw_color_vec[index].val[1];
					visual_img.ptr<uchar>(height)[width * 3 + 2] = draw_color_vec[index].val[2];

				}


			}


			// resize回原尺寸并可视化
			cv::resize(seg_mask, seg_mask, cv::Size(org_img_width, org_img_height), cv::INTER_NEAREST);

#if MERGE_MASK_IMG
			// mask和原图进行叠加
			cv::addWeighted(org_img, 0.8, visual_img, 0.5, 0.0, visual_img);
#endif


		}

		// ——————————————————————————————————
		// ———————— 目标检测decode函数 ————————
		// ——————————————————————————————————
		void hydranet_model::postprocess_detection(float* output_anchor_ptr,
													float * output_reg_ptr,
													float * output_cls_ptr,
													std::vector<Detection_Info>& detect_result,
													cv::Mat& visual_img)
		{

			std::vector<Detection_Info> detect_infos;

			for (long int anchor_index =0;anchor_index< TOTAL_ANCHOR_NUM;anchor_index++)
			{
				float * tmp_conf_score_ptr = output_cls_ptr + CLASS_NUM * anchor_index;

				// get max score and corresponding index
				int max_index = 0;
				float max_value = 0.0;
				for (int i = 0; i < CLASS_NUM; i++)
				{
					if (tmp_conf_score_ptr[i] > max_value)
					{
						max_value = tmp_conf_score_ptr[i];
						max_index = i;
					}
				}

				// filter
				if (max_value <= SCORE_THRESHOLD)
				{
					continue;
				}

				Detection_Info one_proposal, nms_proposal;
				one_proposal.conf_score = max_value;
				one_proposal.class_name = detect_vec[max_index];
				one_proposal.class_id = max_index;

				// regression bounding box
				float * tmp_anchor_ptr = output_anchor_ptr + BBOX_DIM * anchor_index;
				float * tmp_bbx_relative_ptr = output_reg_ptr + BBOX_DIM * anchor_index;

				// decode bounding box
				float y_centers_a = (tmp_anchor_ptr[0] + tmp_anchor_ptr[2]) / 2;
				float x_centers_a = (tmp_anchor_ptr[1] + tmp_anchor_ptr[3]) / 2;
				float ha = tmp_anchor_ptr[2] - tmp_anchor_ptr[0];
				float wa = tmp_anchor_ptr[3] - tmp_anchor_ptr[1];
				float w = exp(tmp_bbx_relative_ptr[3]) * wa;
				float h = exp(tmp_bbx_relative_ptr[2]) * ha;
				float y_centers = tmp_bbx_relative_ptr[0] * ha + y_centers_a;
				float x_centers = tmp_bbx_relative_ptr[1] * wa + x_centers_a;
				float ymin = y_centers - h / 2.;
				float xmin = x_centers - w / 2.;
				float ymax = y_centers + h / 2.;
				float xmax = x_centers + w / 2.;

				// clip out of box
				if (xmin < 0)
				{
					xmin = 0;
				}

				if (ymin < 0)
				{
					ymin = 0;
				}

				if (xmax > NET_INPUT_WIDTH-1)
				{
					xmax = NET_INPUT_WIDTH - 1;
				}


				if (ymax > NET_INPUT_HEIGHT - 1)
				{
					ymax = NET_INPUT_HEIGHT - 1;
				}

				one_proposal.x1 = xmin;
				one_proposal.y1 = ymin;
				one_proposal.x2 = xmax;
				one_proposal.y2 = ymax;

				detect_infos.push_back(one_proposal);

			}

			std::vector<int>  index_choose;
			nms_boxes_detect(detect_infos, index_choose);

			std::vector<Detection_Info> detect_infos_refine;
			for (int i = 0;i<index_choose.size();i++)
			{
				detect_infos_refine.push_back(detect_infos[index_choose[i]]);
			}

			// 输出结果
			for (int j = 0; j < detect_infos_refine.size(); j++)
			{
				Detection_Info tmp_info;
				tmp_info.class_id = detect_infos_refine[j].class_id;
				tmp_info.class_name = detect_infos_refine[j].class_name;
				tmp_info.conf_score = detect_infos_refine[j].conf_score;
				tmp_info.x1 = detect_infos_refine[j].x1 / float(NET_INPUT_WIDTH) * org_img_width;
				tmp_info.x2 = detect_infos_refine[j].x2 / float(NET_INPUT_WIDTH) * org_img_width;

				tmp_info.y1 = detect_infos_refine[j].y1 / float(NET_INPUT_HEIGHT) * org_img_height;
				tmp_info.y2 = detect_infos_refine[j].y2 / float(NET_INPUT_HEIGHT) * org_img_height;

				detect_result.push_back(tmp_info);

			}

			// 可视化
			for (int j = 0; j < detect_infos_refine.size(); j++)
			{

				std::string class_name = detect_infos_refine[j].class_name;
				int class_id = detect_infos_refine[j].class_id;

				float conf_score = detect_infos_refine[j].conf_score;
				int x1 = detect_infos_refine[j].x1;
				int y1 = detect_infos_refine[j].y1;
				int x2 = detect_infos_refine[j].x2;
				int y2 = detect_infos_refine[j].y2;


				// 开始显示
				int t1 = int(round(0.003 * MAX(NET_INPUT_HEIGHT, NET_INPUT_WIDTH)));
				cv::Scalar color = detect_color_list[class_id];
				cv::Point pt1 = cv::Point(x1, y1);
				cv::Point pt2 = cv::Point(x2, y2);
				rectangle(visual_img, pt1, pt2, color, t1);

				int tf = MAX( t1 - 2, 1);

				std::string conf_score_str = std::to_string(int(conf_score * 100));
				int baseline;
				cv::Size s_size = cv::getTextSize(conf_score_str, cv::FONT_HERSHEY_SIMPLEX, (float(t1) / 3), tf, &baseline);
				cv::Size t_size = cv::getTextSize(class_name, cv::FONT_HERSHEY_SIMPLEX, float(t1) / 3, tf, &baseline);
				cv::Point pt3 = cv::Point(x1 + t_size.width + s_size.width + 15, y1 - t_size.height - 3);
				cv::rectangle(visual_img, pt1, pt3, color, -1);


				std::string txt_str = class_name + ""+ conf_score_str + "%";
				cv::putText(visual_img, txt_str, cv::Point(x1, y1 - 2), 0, float(t1) / 3, cv::Scalar(0, 0, 0),tf, cv::FONT_HERSHEY_SIMPLEX);
				

			}

		}


	}
}


#pragma region =============================== utility function lane ===============================

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
	const _Tp alpha = *std::max_element(src, src + length);
	_Tp denominator{ 0 };

	for (int i = 0; i < length; ++i) {
		dst[i] = std::exp(src[i] - alpha);
		denominator += dst[i];
	}

	for (int i = 0; i < length; ++i) {
		dst[i] /= denominator;
	}

	return 0;
}

// IoU部分
inline bool devIoU(Lane a, Lane b, const float threshold) 
{

	int max_start_pos = MAX(a.start_pos, b.start_pos);
	int min_end_pos = MIN(a.end_pos, b.end_pos);

	// quit if no intersection
	if ((min_end_pos <= max_start_pos) || (max_start_pos < 0) || (min_end_pos < 1))
	{
		return false;
	}

	// cal mean dist
	auto pts_a = a.lane_pts;
	auto pts_b = b.lane_pts;
	float dis_mean = 0.0;
	for (int i = max_start_pos; i < min_end_pos ; i++)
	{
		dis_mean += abs(pts_a[i - a.start_pos].x - pts_b[i - b.start_pos].x);

	}

	dis_mean /= (min_end_pos - max_start_pos);

	// cal max distance
	float dis_start = abs(pts_a[max_start_pos - a.start_pos].x - pts_b[max_start_pos - b.start_pos].x);
	float dis_end = abs(pts_a[min_end_pos - 1 - a.start_pos].x - pts_b[min_end_pos - 1 - b.start_pos].x);
	float dis_max = MAX(dis_start, dis_end);
	float dis_another = MAX(dis_mean, dis_max);

	if (USE_MEAN_DISTANCE)
	{
		// based on mean distance
		return (dis_mean > threshold) ? false : true;
	}
	else
	{
		// based on max distance
		return (dis_another > threshold) ? false : true;

	}

}

bool cmpScore_lane(Lane lsh, Lane rsh) 
{
	if (lsh.score > rsh.score)
		return true;
	else
		return false;
}

void nms_boxes(std::vector<Lane> & lanes_input)

{
	Lane lane;
	std::vector<Lane> lanes_process;
	int i, j;
	for (i = 0; i < lanes_input.size(); i++)
	{
		lanes_process.push_back(lanes_input[i]);
	}

	sort(lanes_process.begin(), lanes_process.end(), cmpScore_lane);
	lanes_input.clear();

	int updated_size = lanes_process.size();
	for (i = 0; i < updated_size; i++)
	{

		lanes_input.push_back(lanes_process[i]);
		for (j = i + 1; j < updated_size; j++)
		{
			bool is_suppressed = false;
			is_suppressed = devIoU(lanes_process[i], lanes_process[j], NMS_THRESHOLD);

			if (is_suppressed)
			{
				// 删除掉重复的
				lanes_process.erase(lanes_process.begin() + j);
				updated_size = lanes_process.size();
				j--;
			}

		}
	}


}

// box iou
float cal_iou(cv::Rect rect1, cv::Rect rect2)
{
	//计算两个矩形的交集
	cv::Rect rect_intersect = rect1 & rect2;
	float area_intersect = rect_intersect.area();

	//计算两个举行的并集
	cv::Rect rect_union = rect1 | rect2;
	float area_union = rect_union.area();

	//计算IOU
	double IOU = area_intersect * 1.0 / area_union;

	return IOU;
}

bool is_overlap_with_any(std::vector<cv::Rect> box_list, cv::Rect target_rect)
{

	for (int index = 0; index < box_list.size(); index++)
	{
		float iou = cal_iou(box_list[index], target_rect);
		if (iou > BOX_IOU_THRESHOLD)
		{
			return true;
		}
	}

	return false;

}


#pragma endregion



#pragma region =============================== utility function object ===============================
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

bool cmpScore_object(BBOX lsh, BBOX rsh) {
	if (lsh.score > rsh.score)
		return true;
	else
		return false;
}


//input:  boxes: 原始检测框集合;
//input:  score：confidence * class_prob
//input:  confThreshold 和 nmsThreshold 分别是 检测框置信度阈值以及做nms时的阈值
//output: indices  经过上面两个阈值过滤后剩下的检测框的index
void nms_boxes_detect(std::vector<Detection_Info> & nms_detect_infos, std::vector<int> & index_choose)
{
	BBOX bbox;
	std::vector<BBOX> bboxes;
	int i, j;
	for (i = 0; i < nms_detect_infos.size(); i++)
	{
		Detection_Info tmp_info = nms_detect_infos[i];
		float _x = tmp_info.x1;
		float _y = tmp_info.y1;
		float _width = tmp_info.x2 - tmp_info.x1;
		float _height = tmp_info.y2 - tmp_info.y1;
		int class_index = tmp_info.class_id;

		cv::Rect2f tmp_rect(_x + class_index * NET_INPUT_WIDTH, _y + class_index * NET_INPUT_WIDTH, _width, _height);
		bbox.box = tmp_rect;
		bbox.score = tmp_info.conf_score;
		bbox.index = i;
		bboxes.push_back(bbox);
	}

	sort(bboxes.begin(), bboxes.end(), cmpScore_object);

	int updated_size = bboxes.size();
	for (i = 0; i < updated_size; i++)
	{

		index_choose.push_back(bboxes[i].index);
		for (j = i + 1; j < updated_size; j++)
		{

			float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
			if (iou > IOU_THRESHOLD)
			{

				bboxes.erase(bboxes.begin() + j);
				updated_size = bboxes.size();
				j--;

			}
		}
	}


}

#pragma endregion