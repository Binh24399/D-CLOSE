import math
import glob
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from scipy import spatial
from skimage.segmentation import slic,mark_boundaries
from PIL import Image
from yolox.utils import postprocess
from tool import get_prediction, bbox_iou

class DCLOSE(object):
    def __init__(self, arch, model, img_size=(640, 640), n_segments=150, n_levels=5, n_samples=4000, resize_offset = 2.2, batch_size=32, prob=0.5, kernel_width = 0.25, seed=0, device='cuda', **kwargs):
        '''
        arch: type(str) - Model's name
        model: type(nn.Module)
        img_size: type(tuple) - Input image size
        n_segments: type(int) - Segment coefficient
        n_levels: type(int) - Number of segment levels
        n_samples: type(int) - Total number of random masks generated
        resize_offset: type(float) - Mask resize ratio (default=2.2)
        batch_size: type(int) -  Number of masks in a batch (default=32)
        prob: (0-1) - Probability of 0 and 1 in mask generating (default=0.5)
        kernel_width: (0-1) - Control parameter (default=0.25)
        device: type(str) - Whether use cuda or cpu.
        '''
        self.arch = arch
        self.model = model.eval()
        self.img_size = img_size
    
        self.n_segments = n_segments
        self.n_levels = n_levels
        self.n_samples = n_samples
        # self.total_samples = self.n_samples*self.n_levels
        self.seed = seed
        self.r = resize_offset
        self.p = prob
        self.batch_size = batch_size
        self.kernel_width = kernel_width
        self.device = device

    def __call__(self, image, box):
        return self.generate_saliency_map(image, box)

    def get_features(self, img):
        '''
        img: type(tensor) - shape:[1, 3, H, W]
        Returns the subdivided superpixels in the input image and the number of superpixels generated.
        '''
        img_np = img.squeeze().numpy().transpose(1, 2, 0).astype(np.uint8)
        segments = []
        n_spixels = []
        for i in range(self.n_levels):
            num_segments=self.n_segments*(2**(i+0))
            slic_segment = slic(img_np,n_segments=num_segments,compactness=10,sigma=1)
            segments.append(slic_segment)
            n_spixels.append(np.unique(np.asarray(slic_segment)).shape[0])
        return (segments, n_spixels)
    
    def generate_saliency_map(self, img, box):
        '''
        box: type(tensor) - shape:[num_boxes, (4 + 1 + num_classes + 1)] - Predicted bboxes
            Ex: [x_min, y_min, x_max, y_max, p_obj, class_0, class_1, ....., class_n, label]
        img: type(tensor) - shape:[1, 3, H, W]
        Return saliency maps corresponding to each input bounding box - shape:[num_boxes, H, W].
        '''
        assert self.arch in ["yolox", "faster-rcnn"], "This function only supports yolox and faster-rcnn."
        torch.manual_seed(self.seed)
        h, w = self.img_size  # height, width of input image
        # info target vector
        num_objs = box.shape[0]
        target_box = box[:,:4] # ---> shape[num_boxes, 4]
        target_scores = box[:,5:-1] # ---> shape[num_boxes, 80]
        target_id = box[:,-1].reshape(num_objs, 1) # ---> shape[num_boxes, 1]
        # Create array to save results
        res = np.zeros((self.n_levels, num_objs, h, w), dtype=np.float32) # ---> shape[num_levels, num_objs, h, w]
        # Max score for each bounding box
        max_score = np.zeros((num_objs,), dtype=np.float32) # ---> shape[num_objs,]
        # Define density map
        density_map = torch.zeros((self.n_levels, 1, 1 ,h, w)).cuda()

        # resize mask with a resize offset
        h_mask, w_mask = h + math.floor(self.r*h), w + math.floor(self.r*w)
        # generate and get the number of superpixels for each segment level
        slic_seg, n_spixels = self.get_features(img)
        num_chunks = (self.n_samples + self.batch_size - 1) // self.batch_size 
        level_group = max(num_chunks // self.n_levels, 1)
        # loop for samples
        for chunk in range(num_chunks):
            mask_bs = min(self.n_samples - self.batch_size * chunk, self.batch_size)
            level_idx = chunk // level_group
            level_idx = min(level_idx, self.n_levels - 1)

            data = np.random.choice([0, 1], size=n_spixels[level_idx], p=[1 - self.p, self.p])
            zeros = np.where(data == 0)[0]
            mask = np.zeros(slic_seg[level_idx].shape).astype(float)
            for z in zeros:
              mask[slic_seg[level_idx] == z] = 1.0
            mask = Image.fromarray(mask * 255.)
            mask = mask.resize((w_mask,h_mask),Image.BILINEAR)
            mask = np.array(mask)
            for b in range(self.batch_size):
                # crop mask
                w_crop = np.random.randint(0, self.r*w + 1)
                h_crop = np.random.randint(0, self.r*h + 1)
                masks_np = mask[h_crop:h_crop+h, w_crop:w_crop+w]
                masks_np /= 255.0
                masks_ts = torch.from_numpy(masks_np).to(self.device)
                masks_ts = masks_ts.resize(1,1,h, w)
                density_map[level_idx] += masks_ts

                per_img = masks_ts * img.cuda()
                if self.arch == 'yolox':
                    p = self.model(per_img.to(self.device))
                    p_box, p_index = postprocess(p, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True)         
                    p_box = p_box[0]
                    if p_box is None:
                        continue
                    # proposal vector
                    n = p_box.shape[0]
                    coord = p_box[:, :4] # ---> shape[num_boxes, 4]
                    p_obj = p_box[:, 4]
                    all_scores = p_box[:, 5:-1] # ---> shape[num_boxes, 80]
                    cls_id = p_box[:, -1].reshape(n, 1) # ---> shape[num_boxes, 1]
                    # loop for proposal boxes
                    for idx, value in enumerate(target_id):    
                        value = value.cpu().item()
                        indices, _ = torch.where(cls_id == value)
                        temp = coord[indices] # ---> shape[num_boxes, 4]
                        if len(all_scores[indices]) == 0:
                            continue
                        score_obj = 0.
                        for k in range(temp.shape[0]):
                            # similarity score for each box
                            distances = spatial.distance.cosine(all_scores[indices][k].cpu(), target_scores[idx].cpu())
                            weights = math.sqrt(math.exp(-(distances**2)/self.kernel_width**2)) 
                            iou = torchvision.ops.box_iou(temp[k].unsqueeze(0), target_box[idx].unsqueeze(0)).cpu().item()
                            score_obj = max(score_obj, iou * weights * p_obj[indices][k].cpu().item())             
                        max_score[idx] = score_obj
                        res[level_idx][idx] += masks_ts.cpu().squeeze().numpy() * max_score[idx]
                else:
                    p = self.model(per_img.to(self.device))
                    p_box = get_prediction(p, 0.8)
                    if len(p_box) == 0:
                        continue
                    n = p_box.shape[0]
                    coord = p_box[:, :4] # ---> shape[num_boxes, 4]
                    p_obj = p_box[:, 4]
                    cls_id = p_box[:, -1].reshape(n, 1) # ---> shape[num_boxes, 1]

                    for idx, value in enumerate(target_id):
                        indices, _ = np.where(cls_id == value)
                        temp = coord[indices] # ---> shape[num_boxes, 4]
                        if len(p_obj[indices]) == 0:
                            continue
                        score_obj = 0.
                        for k in range(temp.shape[0]):
                            iou = bbox_iou(temp[k], target_box[idx])
                            score_obj = max(score_obj, iou * p_obj[indices][k])     
                        max_score[idx] = score_obj
                        res[level_idx][idx] += masks_ts.cpu().squeeze().numpy() * max_score[idx]

        heatmap = np.zeros((num_objs, h, w), dtype=np.float32)
        # cascading feature block
        for i in range(self.n_levels):
          for idx_obj in range(res.shape[1]):
            if res[4-i][idx_obj].max() == res[4-i][idx_obj].min():
              res[4-i][idx_obj] = np.zeros_like(res[4-i][idx_obj])
            else:
              res[4-i][idx_obj] /= density_map[4-i].cpu().squeeze().numpy()
              res[4-i][idx_obj] = (res[4-i][idx_obj] - res[4-i][idx_obj].min()) / (res[4-i][idx_obj].max() - res[4-i][idx_obj].min())      
            heatmap[idx_obj] += res[4-i][idx_obj]
            if i != 0:
              heatmap[idx_obj] *= res[4-i][idx_obj]
        return heatmap
