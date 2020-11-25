#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pycocotools.coco import COCO
import json
import cv2, os

# gt_bbox
coco = COCO('./prw_train_old.json')  # prw_test.json
ids = list(coco.imgs.keys())
imgToAnns = coco.imgToAnns
# predicted_bbox
file = open('./bbox_old.json','r')  # bbox_test.json
bboxes_predict = json.load(file)  # 保存的是每一张图片的bbox

# 用score的threshold去掉一些框框
# bboxes_predict_new = []
# for i in bboxes_predict:
#     if i['score'] >= 0.45:  #0.23860901594161987:
#         bboxes_predict_new.append(i)
# bboxes_predict = bboxes_predict_new

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

    bbox_predict = []
    for i in bboxes_predict:
        if i['image_id'] == id:
            bbox_predict.append([i['bbox'], i['score']])  # predict_bbox, score

    gt_id_to_predict.append([id, image_name, bbox_gt, bbox_predict])

    print(id)

# show_gt_predict_imgs:

for j in gt_id_to_predict:
    im_name = j[1]
    img_dir = '/home/yang/PycharmProjects/FCOS/datasets/PRW/frames/' + j[1]
    bbox_gt = j[2]
    bbox_predict = j[3]

    image = cv2.imread(img_dir)
    color_gt = [1, 127, 31]
    for b_gt in bbox_gt:
        gt = b_gt[0]
        top_left = [int(gt[0]), int(gt[1])]
        bottom_right = [int(gt[0] + gt[2]), int(gt[1] + gt[3])]
        # gt_bbox
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color_gt), 6  # 线的粗细程度
        )
        # gt_id

        template = "{}: {:.1f}"
        s = template.format('id', b_gt[1])

        # s = b_gt[1]
        cv2.putText(
            image, str(s), (int(gt[0]), int(gt[1])+30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    color_predict = [16, 247, 241]  # [66,222,6]
    for b_predict in bbox_predict:
        predict = b_predict[0]
        top_left = [int(predict[0]), int(predict[1])]
        bottom_right = [int(predict[0] + predict[2]), int(predict[1] + predict[3])]
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color_predict), 2
        )

        score = b_predict[1]
        template = "{:.2f}"
        s = template.format(score)
        cv2.putText(
            image, str(s), (int(predict[0]), int(predict[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    # cv2.imshow(im_name, image)
    cv2.imwrite('/home/yang/PycharmProjects/FCOS/re-id/show_train_predict_gt_bbox/' + im_name, image)
cv2.waitKey()
cv2.destroyAllWindows()

print('Done')




