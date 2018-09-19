from sklearn import metrics
import base64
import os
import os.path
import cv2
import argparse
import numpy as np
import base64
import io
from io import BytesIO
from PIL import Image
import re
import json
import uuid

def rect_area(rc):
    return (rc[2]-rc[0] + 1)*(rc[3]-rc[1] + 1)

def IoU(rc1, rc2):
    rc_inter =  [max(rc1[0],rc2[0]), max(rc1[1],rc2[1]),min(rc1[2],rc2[2]), min(rc1[3],rc2[3]) ]
    iw = rc_inter [2] - rc_inter [0] + 1;
    ih = rc_inter [3] - rc_inter [1] + 1;
    return (float(iw))*ih/(rect_area(rc1)+rect_area(rc2)-iw*ih) if (iw>0 and ih>0) else 0

def construct_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # necessary inputs
    parser.add_argument('--workdir', required=True, type=str, metavar='PATH', help='image list file')

    parser.add_argument('--preds', default='results/', type=str, metavar='PATH', help='prediction folder')
    parser.add_argument('--labelmap', default='labelmap.txt', type=str, metavar='PATH', help='labelmap of the dataset')
    parser.add_argument('--testfile', type=str, default="test.paths", help = "test file name")
    parser.add_argument('--iou', default=0.5, type=float, help='iou')

    return parser

def load_detections(det_file_folder, labelmap):
    detections_by_class = dict()
    for label in labelmap:
        detections_by_class[label] = []
        det_file_path = os.path.join(det_file_folder, "comp4_det_test_" + label + ".txt")
        if not os.path.exists(det_file_path):
            print("No detections for " + label)
            continue

        with open(det_file_path, "r") as det_in:
            for line in det_in:
                parts = line.strip().split(" ")
                if len(parts) != 6:
                    raise ValueError("Wrong format in detection file for " + label + ": " + line)

                img_id = parts[0]
                conf = float(parts[1])
                # coords from darknet output is [L, T, R, B]
                ltrb_coords = [float(c) for c in parts[2:6]]
                detections_by_class[label].append((img_id, conf, ltrb_coords))

    return detections_by_class

def load_groundtruths(workdir, test_img_list_file, labelmap):
    gt_by_class = dict()
    for label in labelmap:
        gt_by_class[label] = dict()

    with open(os.path.join(workdir, test_img_list_file), 'r') as test_img_list:
        for test_img_path in test_img_list:
            test_img_path = test_img_path.strip()
            file_name = os.path.basename(test_img_path)
            img_id, _ = os.path.splitext(file_name)
            label_path = os.path.join(workdir, "labels", img_id + ".txt")
            width, height = Image.open(test_img_path).size
            with open(label_path, "r") as labels:
                for label in labels:
                    parts = label.strip().split(' ')
                    label = labelmap[int(parts[0])]
                    re_cxcywh_coords = [float(c) for c in parts[1:5]]
                    ltrb_coords = [(re_cxcywh_coords[0]-re_cxcywh_coords[2]/2.0) * width, \
                        (re_cxcywh_coords[1]-re_cxcywh_coords[3]/2.0) * height, \
                        (re_cxcywh_coords[0]+re_cxcywh_coords[2]/2.0) * width, \
                        (re_cxcywh_coords[1]+re_cxcywh_coords[3]/2.0) * height]
                    if img_id not in gt_by_class[label]:
                        gt_by_class[label][img_id] = []
                    gt_by_class[label][img_id].append(ltrb_coords)
        
    return gt_by_class

def decide_on_right_predictions(detections_sorted, gts, iou_thresh):
    captured_gts = set()
    right_det_or_not = np.repeat(0, len(detections_sorted))
    for i, det in enumerate(detections_sorted):
        img_id = det[0]
        coords = det[2]
        if img_id not in gts:
            continue
        
        ious = np.array([IoU(coords, gt) for gt in gts[img_id]])
        bbox_id_with_max_iou = np.argmax(ious)
        if ious[bbox_id_with_max_iou] < iou_thresh or (img_id, bbox_id_with_max_iou) in captured_gts:
            continue

        right_det_or_not[i] = 1
        captured_gts.add((img_id, bbox_id_with_max_iou))

    return right_det_or_not

def calculate_map(labelmap, gt_by_class, detections_by_class, iou):
    ap_by_class = dict()
    for label in labelmap:
        if label not in gt_by_class or len(gt_by_class[label]) == 0:
            print("Label: " + label + " warning: not in ground truth boxes")
            ap_by_class[label] = 0
            continue
        if label not in detections_by_class or len(detections_by_class[label]) == 0:
            print("Label: " + label + " warning: no detections made")
            ap_by_class[label] = 0
            continue
        
        detections_sorted = sorted(detections_by_class[label], key=lambda x:-x[1])
        gts = gt_by_class[label]
        right_det_or_not = decide_on_right_predictions(detections_sorted, gts, iou)
        n_right_dets = np.sum(right_det_or_not)
        n_gt = np.sum(np.array([len(gts[img_id]) for img_id in gts]))

        ap_by_class[label] = metrics.average_precision_score(right_det_or_not, np.array([det[1] for det in detections_sorted])) * n_right_dets/n_gt if n_right_dets > 0 else 0

        print("Label: " + label + ", AP@" + str(iou) + ": " + str(ap_by_class[label]) + ", # dets: " + str(len(detections_sorted)) + ", # right dets: " + str(n_right_dets) + ", # ground truths: " + str(n_gt))
    return ap_by_class

def main():
    parser = construct_arg_parser()
    args = parser.parse_args()
    labelmap = []
    with open(os.path.join(args.workdir, "labelmap.txt"), 'r') as labels:
        labelmap = [label.strip("\r\n") for label in labels]
    print(labelmap)
    gt_by_class = load_groundtruths(args.workdir, args.testfile, labelmap)
    dets_by_class = load_detections(args.preds, labelmap)
    ap_by_class = calculate_map(labelmap, gt_by_class, dets_by_class, args.iou)
    mAP = np.average(np.array([ap_by_class[label] for label in ap_by_class]))
    print("mAP@" + str(args.iou) + ": " + str(mAP))

if __name__ == '__main__':
    main()
