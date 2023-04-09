#include "Hydranet.h"
#include <hydranet_model.h>

using hydranet::hydranet_detection::hydranet_model;

HYDRANET_API int Hydranet_Init(IN void **handle, std::string model_path)
{

	hydranet_model * detector = new hydranet_model(model_path);

	if (detector == NULL)
	{
		std::cout << "输入的Handle指针为空" << std::endl;
		return -1;
	}

	*handle = (void *)detector;

	return 0;
}

HYDRANET_API int Hydranet_Detect(IN void *handle,
							     IN cv::Mat& input_image,
							     OUT cv::Mat& visual_image,
								 OUT Output_Info & output)
{

	hydranet_model * detector = (hydranet_model *)handle;
	detector->detect(input_image, visual_image, output);
	return 0;

}


HYDRANET_API int Hydranet_Uinit(IN void * handle)
{

	hydranet_model * detector = (hydranet_model *)handle;
	detector->~hydranet_model();
	return 0;

}
