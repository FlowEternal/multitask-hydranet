import yaml, os
import numpy as np
import cv2
np.random.seed(1991)
import torch.utils.data.dataloader
import warnings

warnings.filterwarnings("ignore")

from model import HydraNet

# ---------------------------------------------------#
#  validation packages
# ---------------------------------------------------#

# segmentation


# detection


# lane
from head_lane.lane_codec import LaneCodec


def imagenet_normalize( img):
    """Normalize image.

    :param img: img that need to normalize
    :type img: RGB mode ndarray
    :return: normalized image
    :rtype: numpy.ndarray
    """
    pixel_value_range = np.array([255, 255, 255])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img / pixel_value_range
    img = img - mean
    img = img / std
    return img

def deparallel_model(dict_param):
    ck_dict_new = dict()
    for key, value in dict_param.items():
        temp_list = key.split(".")[1:]
        new_key = ""
        for tmp in temp_list:
            new_key += tmp + "."
        ck_dict_new[new_key[0:-1]] = value
    return ck_dict_new

if __name__ == '__main__':
    import time
    deploy = False
    #---------------------------------------------------#
    #  参数设定
    #---------------------------------------------------#
    # log_path = "logs/joint_pretrain"
    # model_name = "epoch_5.pth"

    log_path = "logs/joint_pretrain_large"
    model_name = "epoch_0.pth"

    img_test = False
    display = True
    img_folder = "demo/images"
    video_path = "demo/video/test_video.avi"
    use_fix_color = True

    # 导出参数
    cfg_path = os.path.join(log_path, "config.yml")
    cfgs = yaml.safe_load(open(cfg_path))
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    net_input_size = (net_input_width, net_input_height)
    train_detect = cfgs["train"]["train_detect"]
    train_seg = cfgs["train"]["train_seg"]
    train_lane = cfgs["train"]["train_lane"]

    segment_class_list = cfgs["segment"]["class_list"]
    # seg_class_color_id = {0: (0, 0, 0),
    #                       1: (0, 0, 0),
    #                       2: (255, 255, 255),
    #                       3: (0, 0, 0),
    #                       4:(0,0,0)
    #                       }

    seg_class_color_id = {0: (0, 0, 0),
                          1: (128, 0, 128),
                          2: (255, 255, 255),
                          3: (0, 255, 255),
                          4:(0,255,0)
                          }

    obj_list = cfgs["detection"]["class_list"][1:]

    # 加载车道线相关
    anchor_stride = cfgs["lane"]["anchor_stride"]
    interval = cfgs["lane"]["interval"]
    anchor_lane_num = cfgs["lane"]["anchor_lane_num"]
    interpolate = cfgs["lane"]["interpolate"]  # 是否插值 如果插值 即为沿长到版边 和原代码就完全一致
    scale_invariance = cfgs["lane"]["scale_invariance"]
    points_per_line = int(net_input_height / interval)

    lane_coder = LaneCodec( input_width=net_input_width,
                            input_height=net_input_height,
                            anchor_stride=anchor_stride,
                            points_per_line=points_per_line,
                            do_interpolate=interpolate,
                            anchor_lane_num=anchor_lane_num,
                            scale_invariance=scale_invariance )

    if not use_fix_color:
        seg_class_color_id = dict()
        for idx in range(len(segment_class_list)):
            color = (
                int(np.random.randint(128, 255)), int(np.random.randint(128, 255)), int(np.random.randint(128, 255)))
            seg_class_color_id.update({idx: color})

    #---------------------------------------------------#
    #  网络模型
    #---------------------------------------------------#
    if deploy:
        hydranet = HydraNet(cfgs=cfgs,onnx_export=True).cuda()
    else:
        hydranet = HydraNet(cfgs=cfgs,onnx_export=False).cuda()

    dict_old = torch.load(os.path.join(log_path,"model",model_name))
    dict_new = deparallel_model(dict_old)
    hydranet.load_state_dict(dict_new)
    hydranet.eval()


    if deploy:
        import torch.onnx
        output_list = ["seg", "anchors", "regression", "classification", "lane_cls", "lane_reg"]
        dummy_input = torch.randn([1, 3, 640, 640]).to("cuda:0")
        torch.onnx.export(
            hydranet,(dummy_input, "deploy"),
            "hydraNET.onnx",
            export_params=True,
            input_names=["input", "mode"],
            output_names=output_list,
            opset_version=12,
            verbose=False,
        )

        exit()

    if img_test:
        img_list = os.listdir(img_folder)
        if not os.path.exists("demo/images_vis"): os.makedirs("demo/images_vis")
        vid = None
        video_writer = None
    else:
        vid = cv2.VideoCapture(video_path)
        video_output = video_path.replace("video","video_vis")
        if not os.path.exists("demo/video_vis"): os.makedirs("demo/video_vis")
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter(video_output, codec, 10, (1920,1080))
        img_list = None

        if display:
            cv2.namedWindow("visual",cv2.WINDOW_FREERATIO)

    counter = 0
    while True:

        if img_test:
            if counter > len(img_list) - 1:
                break

            img_name = img_list[counter]
            img_path = os.path.join("demo/images",img_name)
            input_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            print("process %s" % img_name)
        else:
            _, input_img = vid.read()
            img_path = None

        if input_img is None:
            break
        counter+=1

        org_height, org_width = input_img.shape[0:2]
        org_size = (org_width , org_height)

        #---------------------------------------------------#
        #  preprocess
        #---------------------------------------------------#
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, net_input_size)
        img = img.astype(np.float32)
        img = imagenet_normalize(img)
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        img = torch.tensor(img).cuda().float()

        #---------------------------------------------------#
        #  inference
        #---------------------------------------------------#
        tic = time.time()
        outputs = hydranet(img)
        print("inference time is %i" %(1000*(time.time() - tic)))
        #---------------------------------------------------#
        #  decoder
        #---------------------------------------------------#
        imgs = [input_img]


        if train_lane:
            # conf_thres = cfgs["lane"]["conf_thres"]
            # nms_thres = cfgs["lane"]["nms_thres"]
            conf_thres = 0.90
            nms_thres = 80

            cls_preds = outputs["lane"]['predict_cls']
            loc_preds = outputs["lane"]['predict_loc']
            batch_size = len(imgs)

            predict_jsons = list()
            for batch_idx in range(batch_size):
                cls_pred = cls_preds[batch_idx]
                loc_pred = loc_preds[batch_idx]
                lane_nms_set = hydranet.laneheader.decode(cls_pred, loc_pred, lane_coder , conf_thres, nms_thres, False)
                predict_json = \
                hydranet.laneheader.scale_to_org(lane_nms_set, net_input_width, net_input_height, org_size[0], org_size[1])[
                    "Lines"]
                predict_jsons.append(predict_json)

            imgs = hydranet.laneheader.visual(imgs, predict_jsons, org_size[0],filter_vertical=True)

        if train_seg:
            # display
            output_seg = outputs["seg"]
            imgs = hydranet.segheader.decode(imgs, output_seg, org_size, seg_class_color_id)


        if train_detect:
            anchors = outputs["detection"]["anchors"]
            regression = outputs["detection"]["regression"]
            classification = outputs["detection"]["classification"]
            target_size = (net_input_width, net_input_height)
            preds = hydranet.detectheader.decode(img,regression,classification, anchors,conf_thres=0.4, iou_thres=0.3)
            imgs = hydranet.detectheader.display(preds, imgs, obj_list, org_size, target_size)

        print("total process time is %i" %(1000*(time.time() - tic)))

        # ---------------------------------------------------#
        #  保存显示
        # ---------------------------------------------------#
        if not img_test:
            if display:
                cv2.imshow('visual', imgs[0])
                cv2.waitKey(1)
            video_writer.write(imgs[0])

            # if counter > 500:
            #     break

        else:
            cv2.imwrite(img_path.replace("images", "images_vis"), imgs[0])

