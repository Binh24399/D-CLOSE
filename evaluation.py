import math
import numpy as np
import cv2
import torch
import torchvision
from scipy import spatial
from tqdm import trange
from yolox.utils import postprocess
from tool import bbox_iou
kernel_width = 0.25

def auc(arr):
    '''Returns normalized Area Under Curve of the array.'''
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)  # auc formula 

def causal_metric(model, img, bbox, saliency_map, mode, step, kernel_width=0.25):
    '''
    model: type(nn.Module)
    img: type(np.ndarray) - shape:[H, W, 3]
    bbox: type(tensor) - shape:[num_boxes, (4 + 1 + num_classes + 1)] - Predicted bboxes
    saliency_map: type(np.ndarray) - shape:[num_boxes, H, W]
    mode: type(str) - Select deletion or insertion metric ('del' or 'ins')
    step: number of pixels modified per one iteration
    kernel_width: (0-1) - Control parameter (default=0.25)
    Return: deletion/insertion metric and number of objects.
    '''
    del_ins = np.zeros(80)
    count = np.zeros(80)
    HW = saliency_map.shape[1] * saliency_map.shape[2]
    n_steps = (HW + step - 1) // step
    for idx in range(saliency_map.shape[0]):
        target_cls = bbox[idx][-1]
        if mode == 'del':
            start = img.copy()
            finish = np.zeros_like(start)
        else:
            start = cv2.GaussianBlur(img, (51, 51), 0)
            finish = img.copy()
        salient_order = np.flip(np.argsort(saliency_map[idx].reshape(HW, -1), axis=0), axis=0)
        y = salient_order // img.shape[1]
        x = salient_order - y*img.shape[1]
        scores = np.zeros(n_steps + 1)
        with torch.no_grad():
            for i in range(n_steps + 1):
                temp_ious = []
                temp_score = []
                torch_start = torch.from_numpy(start.transpose(2, 0, 1)).unsqueeze(0).float()
                out = model(torch_start.cuda())
                p_box, index = postprocess(out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True)
                p_box = p_box[0]
                if p_box is None:
                    scores[i] = 0
                else:  
                    for b in p_box:
                        sample_cls = b[-1]
                        sample_box = b[:4]
                        sample_score = b[5:-1] 
                        iou = torchvision.ops.box_iou(sample_box[:4].unsqueeze(0), bbox[idx][:4].unsqueeze(0)).cpu().item()
                        distances = spatial.distance.cosine(sample_score.cpu(), bbox[idx][5:-1].cpu())
                        weights = math.sqrt(math.exp(-(distances**2)/kernel_width**2)) 
                        if target_cls != sample_cls:
                            iou = 0
                            sample_score = torch.tensor(0.)
                        temp_ious.append(iou)
                        s_score = iou * weights
                        temp_score.append(s_score)      
                    max_score = temp_score[np.argmax(temp_ious)]
                    scores[i] = max_score
                x_coords = x[step * i:step * (i+1), :]
                y_coords = y[step * i:step * (i+1), :]
                start[y_coords, x_coords, :] = finish[y_coords, x_coords, :]
        del_ins[int(target_cls)] += auc(scores)
        count[int(target_cls)] += 1
    return del_ins, count

def metric(bbox, saliency_map):
    '''
    bbox:  type(np.ndarray) - shape:[num_boxes, (4 + 1 + num_classes + 1)] - The ground-truth box matches the prediction box
    saliency_map: type(np.ndarray) - shape:[num_boxes, H, W]
    Return: EBPG/PG metric and number of objects.
    '''
    empty = np.zeros_like(saliency_map)
    proportion = np.zeros(80)
    count_idx = np.zeros(80)
    pg = np.zeros(80)
    for idx in range(bbox.shape[0]):
        x1, y1, x2, y2 = bbox[idx][:4]
        max_point = np.where(saliency_map[idx] == np.max(saliency_map[idx]))
        cls = int(bbox[idx][-1])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x1 <= max_point[1][0] <= x2 and y1 <= max_point[0][0] <= y2:
            pg[cls] += 1     
        empty[idx][y1:y2, x1:x2] = 1
        mask_bbox = saliency_map[idx] * empty[idx]   
        energy_bbox =  mask_bbox.sum()
        energy_whole = saliency_map[idx].sum()
        if energy_whole == 0:
            proportion[cls] += 0
            count_idx[cls] += 1
        else:
            proportion[cls] += energy_bbox / energy_whole
            count_idx[cls] += 1
    return proportion, pg, count_idx

def correspond_box(predictbox, groundtruthboxes):
    '''
    predictbox: type(np.ndarray) - shape:[num_boxes, (4 + 1 + num_classes + 1)] - Predicted bounding boxes
    groundtruthboxes: type(np.ndarray) - shape:[num_boxes, (4 + 1 + num_classes + 1)] - Ground-truth bounding boxes
    Return: The ground-truth box matches the prediction box and the corresponding index of the prediction box.
    '''
    gt_boxs = []
    det = np.zeros(len(groundtruthboxes))
    idx_predictbox = []
    for d in range(len(predictbox)):
        iouMax = 0
        for i in range(len(groundtruthboxes)):
            if predictbox[d][-1] != groundtruthboxes[i][-1]:
                continue
            iou = bbox_iou(predictbox[d][:4], groundtruthboxes[i][:4])
            if iou > iouMax:
                iouMax = iou
                index = i
        if iouMax > 0.5:
            if det[index] == 0:
                det[index] == 1
                gt_boxs.append(groundtruthboxes[index])
                idx_predictbox.append(d)
    return np.array(gt_boxs), idx_predictbox