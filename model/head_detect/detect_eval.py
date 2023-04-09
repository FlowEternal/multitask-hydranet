from pycocotools.cocoeval import COCOeval

def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    print(pred_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
