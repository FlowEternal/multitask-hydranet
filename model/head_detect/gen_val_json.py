import json
import os
import cv2

dataset = {'info': {
    'description': "This is stable 1.0 version of the 2014 MS COCO dataset.",
    'url': "http://mscoco.org",
    'version': "1.0",
    'year': 2021,
    'contributor': "Group",
    'date_created': "2021-09-01 11:35:00.000000"},
    'images': [],
    'annotations': [],
    'categories': [
        {'supercategory:': 'road', 'id': 1, 'name': 'roadtext'},
        {'supercategory:': 'person', 'id': 2, 'name': 'pedestrian'},
        {'supercategory:': 'road', 'id': 3, 'name': 'guidearrow'},
        {'supercategory:': 'traffic', 'id': 4, 'name': 'traffic'},
        {'supercategory:': 'obstacle', 'id': 5, 'name': 'obstacle'},
        {'supercategory:': 'vehicle', 'id': 6, 'name': 'vehicle_wheel'},
        {'supercategory:': 'road', 'id': 7, 'name': 'roadsign'},
        {'supercategory:': 'vehicle', 'id': 8, 'name': 'vehicle'},
        {'supercategory:': 'traffic', 'id': 9, 'name': 'vehicle_light'}
    ]

}


def gen_coco_label(root_dir, list_name = "valid.txt"):
    list_txt = os.path.join(root_dir,"list",list_name)
    img_list = open(list_txt).readlines()

    target_anno_file = os.path.join(root_dir, "eval_detect")
    if not os.path.exists(target_anno_file): os.makedirs(target_anno_file)

    json_name = os.path.join(target_anno_file, "gt_bbox_results.json")
    if os.path.exists(json_name):return json_name

    count = 1
    cnt = 0
    annoid = 0
    for index in range(len(img_list)):
        img_full_path = img_list[index].strip("\n")
        txtpath = img_full_path.replace("images","labels_object").replace(".jpg",".txt")

        with open(txtpath) as annof:
            annos = annof.readlines()
            if len(annos) == 0:
                print("skip image %s" % img_full_path)
                continue

        print("process image %s" % img_full_path)
        cnt += 1
        im = cv2.imread(img_full_path)
        height, width, _ = im.shape

        dataset['images'].append(
            {'license': 5, 'file_name': img_full_path, 'coco_url': "local", 'height': height, 'width': width,
             'date_captured': "2018_08_29 10:10:10", 'flickr_url': "local", 'id': cnt})

        # dataset['images'].append({'file_name': imagepath})
        # line + .txt
        # txtpath = os.path.join(annopath, line.strip() + '.txt')

        for ii, anno in enumerate(annos):
            # vehicle,1100,363,1177,363,1177,395,1100,395
            parts = anno.strip("\n").split(",")
            # resize bbox *2/3
            # x1
            x1 = float(parts[0])  # * 2 / 3
            # y1
            y1 = float(parts[1])  # * 2 / 3
            # x2
            x2 = float(parts[2])  # * 2 / 3
            # y2
            y2 = float(parts[3])  # * 2 / 3

            # category
            category = int(parts[4])
            wid = max(0, int(x2 - x1))
            hei = max(0, int(y2 - y1))
            iscrowd = 0

            annoid = annoid + 1
            # print category
            # print category
            # print classes[category]
            '''
            x1 = x1 / 3
            y1 = y1 / 3
            wid = wid / 3
            hei = hei / 3
            '''
            dataset['annotations'].append({
                'segmentation': [],
                'iscrowd': iscrowd,
                'area': wid * hei,
                'image_id': cnt,
                'bbox': [x1, y1, wid, hei],
                'category_id': category,
                'id': annoid
            })
        count += 1



    with open(json_name, 'w') as f:
        json.dump(dataset, f)
    return json_name

if __name__ == '__main__':
    gen_coco_label(root_dir="/data/zdx/Data/MULTITASK_7",list_name="valid.txt")