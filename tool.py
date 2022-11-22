import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import cv2
import numpy as np
from yolox.data.datasets import COCO_CLASSES
coco = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
def visual(img, saliency_map, target_box, arch, save_file=None):
    '''
    img: [h, w, 3]
    saliency_map: [num_boxes, h, w]
    target_box[num_boxes, 86]
    '''
    cp_img = img.copy()
    num_boxes = target_box.shape[0]
    fig = plt.figure(figsize=(16, 10))
    row = round(math.sqrt(num_boxes)+0.5)
    col = row
    i = 1

    for idx, b in enumerate(target_box):
        fig.add_subplot(col, row, i)

        x_min, y_min, x_max, y_max = b[:4]
        id_obj = b[-1]
        temp_sal = saliency_map[idx]
        m = np.min(temp_sal)
        M = np.max(temp_sal)
        if M == m:
            temp_sal = ((temp_sal - m) / (M - m + 1e-8)) * 255
        else:
            temp_sal = ((temp_sal - m) / (M - m)) * 255
        temp_sal = temp_sal.astype(np.uint8)
        heatmap = cv2.applyColorMap(temp_sal, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        fig.axes[i-1].imshow(cp_img)
        fig.axes[i-1].imshow(heatmap, alpha=0.5)

        w = (x_max - x_min)
        h = (y_max - y_min)
        
        rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor='r', facecolor='none')
        fig.axes[i-1].add_patch(rect)
        if arch=="yolox":
            scores = b[4] * b[5+int(id_obj)]
            fig.axes[i-1].text(x_min, y_min, '{} - {:.2f}'.format(COCO_CLASSES[int(id_obj)], scores), size = 8, style='italic', bbox=dict(boxstyle="round,pad=0.2", facecolor='green', alpha=0.5))
        else:
            scores = b[4]
            fig.axes[i-1].text(x_min, y_min, '{} - {:.2f}'.format(coco[int(id_obj)], scores), size = 8, style='italic', bbox=dict(boxstyle="round,pad=0.2", facecolor='green', alpha=0.5))
        i += 1
    
    if save_file is not None:
        plt.tight_layout()
        plt.savefig(save_file)
    # return cp_img

def get_prediction(pred, threshold):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.
      
    """
    pred_class = pred[0]['labels'].cpu().numpy()
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    pred_boxes = [[i[0], i[1], i[2], i[3], pred_score[j], pred_class[j]] for (j, i) in enumerate(list(pred[0]['boxes'].cpu().detach().numpy()))]
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold]
    if len(pred_t) == 0:
        return np.array([])
    else:
        pred_t = pred_t[-1]
    pred_boxes = pred_boxes[:pred_t+1]
    # pred_class = pred_class[:pred_t+1]
    # scores = pred_score[:pred_t+1]
    return np.array(pred_boxes)


def bbox_iou(box1, box2, x1y1x2y2=True):
    
    # print('iou box1:', box1)
    # print('iou box2:', box2)

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea
