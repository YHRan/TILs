import os
import torch.nn as nn
from models.utils import attempt_load_v7, select_device
# from models.common import DetectMultiBackend
import torch


class LoadYOLOModelv7(nn.Module):
    def __init__(self, weights='yolov5s.pt', device=None, dnn=False):
        super().__init__()
        pt = True
        device = 0
        stride = 64
        device = select_device(device)
        model = attempt_load_v7(weights, map_location=device)
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        y = self.model(im, augment=augment)
        return y if val else y[0]


class LoadModel():

    @staticmethod
    def load_yolo7_model(model_path):
        model = LoadYOLOModelv7(model_path)
        return model

    @staticmethod
    def kick_model(model_path_list):
        model_set = []
        model_set.append(LoadModel.load_yolo7_model(model_path_list[0]))
        # model_set.append(LoadModel.load_yolo7_model(model_path_list[1]))
        if len(model_set) != len(model_path_list):
            model_set = []

        return model_set

