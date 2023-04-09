// ————————————————————————————————————————
// ———————— LaneDetection 头文件 ———————————
// ————————————————————————————————————————
#include <Hydranet.h>

#define INPUT_WIDTH 640
#define INPUT_HEIGHT 640

int main(int argc, char* argv[])
{
	int mode = std::atoi(argv[1]);

	// 输入参数
	std::string model_path = "../model/hydranet_big.onnx";
	std::string test_img_path = "../data/test_img.jpg";
	std::string test_video_path = "../data/test_video.avi";

	// 计时参数
	std::chrono::steady_clock::time_point tic;
	std::chrono::steady_clock::time_point tac;
	double time_used = 0.0;
	double time_used_total = 0.0;

	// 调节参数
	int output_fps = 30;
	int weight_ms = 5;
	int iteration = 20;
	int gpu_warm_up = 10;
	int camera_id = 0;
	cv::namedWindow("visual", cv::WINDOW_FREERATIO);

	// video writer 
	cv::VideoWriter video_writer;
	video_writer.open(
		"../data/visualization.avi", 
		cv::VideoWriter::fourcc('M', 'P', '4', '2'), 
		20.0, 
		cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
		true);



	// ———————————————————————————
	// ———————— 算法初始化 ————————
	// ———————————————————————————
	IN void* handle;
	int ret = Hydranet_Init(&handle, model_path);

	// 变量定义
	cv::Mat						src_image;
	cv::Mat						visual_img;
	cv::VideoCapture			video_reader;

	if (mode == 2)
	{
		video_reader.open(test_video_path);
	}

	if (mode ==3)
	{
		video_reader.open(camera_id);

	}


	unsigned int counter = 0;
	while (true)
	{


		// —————————————————————————
		// ———————— 准备输入 ————————
		// —————————————————————————
		counter++;
		if (mode ==1)
		{
			if (counter < (iteration + gpu_warm_up))
			{

				src_image = cv::imread(test_img_path, cv::IMREAD_COLOR);

			}

			else
			{
				return 0;
			}
		}

		else
		{
			video_reader >> src_image;

			if (src_image.empty())
			{
				break;
			}
		}


		tic = std::chrono::steady_clock::now();

		// ——————————————————————————————
		// ———————— 进行多任务检测 ————————
		// ——————————————————————————————
		std::cout << "++++++++++++++++++++ ITER " << counter << " ++++++++++++++++++++" << std::endl;
		OUT cv::Mat visual_img;
		OUT Output_Info process_result;
		ret = Hydranet_Detect(handle, src_image, visual_img, process_result);

		tac = std::chrono::steady_clock::now();
		time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
		std::cout << "Hydranet_Detect Interface Total Time Cost: " << time_used << "ms!" << std::endl;

		if (counter > gpu_warm_up)
		{

			time_used_total += time_used;
			double average_time = (time_used_total / (counter - gpu_warm_up) );
			std::cout << "Hydranet_Detect Interface Average Time Cost: " << average_time << " ms" << std::endl;

		}

		std::cout << std::endl;
		std::cout << std::endl;

		cv::imshow("visual", visual_img);
		char c = cv::waitKey(weight_ms);
		if (c == 27) break;
		std::cout << std::endl;

		video_writer << visual_img;

	}

	// ————————————————————————————
	// ———————— 算法反初始化 ————————
	// ————————————————————————————
	ret = Hydranet_Uinit(handle);
	return 0;

}





