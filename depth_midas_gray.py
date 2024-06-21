# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 15:09:03 2024

@author: zahra
"""

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

class method(object):
    def __init__(self, name):
        self.name=name 
    def depth_midas(img):
        model_type = "DPT_Large" 
        
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        
        transform = midas_transforms.dpt_transform
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        input_batch = transform(img).to(device)
        
        with torch.no_grad():
            prediction = midas(input_batch)
        
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        output = prediction.cpu().numpy()
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
        output = output.astype(np.uint8)
        colored_output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
        return colored_output,output



