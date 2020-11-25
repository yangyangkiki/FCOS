#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pycocotools.coco import COCO
import json
import cv2, os

# gt_bbox
coco = COCO('./test.json')  # prw_test.json
ids = list(coco.imgs.keys())
imgToAnns = coco.imgToAnns

gt_id_to_predict = []

# # check bbox number = 18048
# ct = 0
# for id in ids:
#     anns = imgToAnns[id]
#     ct = ct + len(anns)
# print(ct)

for id in ids:
    anns = imgToAnns[id]
    image_name = coco.imgs[id]['file_name']
    bbox_gt = []
    for ann in anns:
        bbox_gt.append([ann['bbox'],ann['person_re_id']])  # gt_bbox, re-id

    gt_id_to_predict.append([id, image_name, bbox_gt])

    print(id)

# show_gt_predict_imgs:

for j in gt_id_to_predict:
    im_name = j[1]
    img_dir = '/home/yang/PycharmProjects/FCOS/datasets/PRW/frames/' + j[1]
    bbox_gt = j[2]

    image = cv2.imread(img_dir)
    color_gt = [1, 127, 31]
    for b_gt in bbox_gt:
        gt = b_gt[0]
        top_left = [int(gt[0]), int(gt[1])]
        bottom_right = [int(gt[0] + gt[2]), int(gt[1] + gt[3])]
        # gt_bbox
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color_gt), 3  # 线的粗细程度
        )
        # gt_id

        template = "{}: {:.1f}"
        s = template.format('id', b_gt[1])

        # s = b_gt[1]
        cv2.putText(
            image, str(s), (int(gt[0]), int(gt[1])+30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )


    # cv2.imshow(im_name, image)
    cv2.imwrite('/home/yang/PycharmProjects/FCOS/re-id/check_test_imgs/' + im_name, image)
cv2.waitKey()
cv2.destroyAllWindows()

print('Done')




