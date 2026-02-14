# -*- encoding: utf-8 -*-
"""
@File    : model_predict.py
@Time    : 2021/12/20 13:30
@Author  : 高中需
@Usage   :
"""
import cv2
import torch
import numpy as np
import time
from PIL import Image
from models.utils import select_device
from yolo5_development_pt2 import yolo_predict
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
import warnings
from torch.nn import functional as F

from PIL import Image

warnings.filterwarnings('ignore')


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression_1(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None,
                        agnostic=False, multi_label=False, labels=(), max_det=1000):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


class Predict():
    def __init__(self,yolo_model_1280_ds1):
        self.yolo_model_1280_ds1 = yolo_model_1280_ds1
        # self.yolo_model_1280_ds32 = yolo_model_1280_ds32
        self.device = select_device(0)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    @torch.no_grad()
    def yolo_predict(self, im,
                     model,
                     device,
                     conf_thres=0.2,  # confidence threshold
                     iou_thres=0.3,  # NMS IOU threshold
                     ):
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im)
        # print('pred',pred)
        # exit()

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        return pred


    # @torch.no_grad()
    # def yolo_predict_32(self, im,
    #                  model,
    #                  device,
    #                  conf_thres=0.25,  # confidence threshold
    #                  iou_thres=0.45,  # NMS IOU threshold
    #                  ):
    #     im = torch.from_numpy(im).to(model.device)
    #     # im = im.half() if model.fp16 else im.float()
    #     im = im.float()
    #     im /= 255  # 0 - 255 to 0.0 - 1.0
    #     if len(im.shape) == 3:
    #         im = im[None]  # expand for batch dimmodel
    #
    #     # Inference
    #     pred,proto= model(im,augment=False,visualize=False)[:2]
    #     # print('pred',pred)
    #     # print('proto',proto)
    #     # exit()
    #
    #     # NMS
    #     # pred = non_max_suppression(pred, conf_thres, iou_thres)
    #     pred = non_max_suppression_1(pred, conf_thres, iou_thres, max_det=1000, nm=32)
    #     return pred,proto





    @torch.no_grad()
    def yolo_predict_4(self, im,
                     model,
                     device,
                     conf_thres=0.25,  # confidence threshold
                     iou_thres=0.45,  # NMS IOU threshold
                     ):
        im = torch.from_numpy(im.copy()).to(model.device)
        # im = torch.from_numpy(np.ascontiguousarray(im)).to(model.device)
        # im = im.half() if model.fp16 else im.float()
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dimmodel

        # print(im.shape)

        # Inference
        pred,out= model(im,augment=False,visualize=False)
        proto = out[1]

        # NMS
        # pred = non_max_suppression(pred, conf_thres, iou_thres)
        pred = non_max_suppression_1(pred, conf_thres, iou_thres, max_det=1000, nm=32)
        return pred,proto,im



    #     @torch.no_grad()
    #     def yolo_predict_1(self, im,
    #                      model,
    #                      device,
    #                      conf_thres=0.2,  # confidence threshold
    #                      iou_thres=0.3,  # NMS IOU threshold
    #                      ):

    #         img = Image.fromarray(np.uint8(im.transpose(1, 2, 0)))
    #         # img.save(r'C:\Users\Administrator\Desktop\22/4.png')
    #         im = torch.from_numpy(np.array(img).transpose(2, 0, 1))
    #         im = im.to(device)
    #         im = im.float()
    #         im /= 255  # 0 - 255 to 0.0 - 1.0
    #         if len(im.shape) == 3:
    #             im = im[None]  # expand for batch dim

    #         # Inference
    #         pred = model(im)
    #         # print('pred',pred)
    #         # exit()

    #         # NMS
    #         pred = non_max_suppression(pred, conf_thres, iou_thres)
    #         return pred
    # def cls_oophoron(self, img):
    #     label_dict = {
    #         0: 'AF', 1: 'PF', 2: 'PFs', 3: 'SF', 4: 'TF', 5: 'VF'
    #     }
    #     img = cv2.resize(img[:, :, ::-1], (224, 224))
    #     img_tensor = self.transform(img)
    #     img_tensor = img_tensor.unsqueeze(dim=0)
    #     img_tensor = img_tensor.cuda()
    #     pred = self.cls_oop_model(img_tensor)
    #
    #     y_pred = np.argmax(pred.detach().cpu().numpy(), axis=1)  # 将概率分布转化为类（分类）
    #     # print(label_dict[int(y_pred)])
    #     return label_dict[int(y_pred)]

    def cls_oophoron(self, img):
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        label_dict = {
            0: 'AF', 1: 'other'
        }
        #
        img = Image.fromarray(img)

        with torch.no_grad():
            img = trans(img).unsqueeze(0)
            img = img.cuda(non_blocking=True)
            # compute output
            output = self.cls_model(img)
            predict = F.softmax(output, dim=1)  # (1, class_num)
            result = torch.argmax(predict, dim=1)  # (1, class_num)
        return label_dict[int(result)]

    def seg_luteum(self, img_list):
        def normPRED(d):
            ma = torch.max(d)
            mi = torch.min(d)
            dn = (d - mi) / (ma - mi)
            return dn

        trans = transforms.Compose([
            ToTensorLab(flag=0)])
        mask_list = []
        for img in img_list:
            data = trans(img)
            data = data.unsqueeze(0).type(torch.FloatTensor)
            data = Variable(data.cuda())
            d1, d2, d3, d4, d5, d6, d7 = self.seg_luteum_model(data)
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)
            del d1, d2, d3, d4, d5, d6, d7
            pred = pred.detach().cpu().numpy()
            mask_list.append(pred)
        return mask_list


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, image):
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        image = image / np.max(image)
        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
            tmpImg = tmpImg.transpose((2, 0, 1))
        return torch.from_numpy(tmpImg)
