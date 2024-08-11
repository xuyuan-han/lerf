import os
import typing
from pathlib import Path
import numpy as np
import torch
#from einops import rearrange
import cv2
import math
from lerf.segment_anything import sam_model_registry, SamPredictor
import glob
from lerf.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm

class SAMDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        #image_list: torch.Tensor,
        cache_path: str=None,
        image_paths: list=None,
    ):
        """
        img_list = []
        for img_path in glob.glob(os.path.join(self.image_paths, "*")):
            if img_path.endswith(".png") or img_path.endswith(".jpg") or img_path.endswith(".JPG"):
                img_list.append(img_path)

        self.image_paths = img_list
        self.predictor = self.init_sam()

        self.features = self.load_or_extract_features()
        """
        assert "image_shape" in cfg
        self.predictor = self.init_sam(device=device,cfg=cfg)
        super().__init__(cfg, device, image_paths, cache_path)


    def init_sam(self, device, cfg):
        sam = sam_model_registry[cfg["model_type"]](checkpoint=cfg["sam_checkpoint"])
        sam.to(device)
        return SamPredictor(sam)
    
    def create(self, image_list):
        features = []
        for image_path in tqdm(image_list, desc="sam", total=len(image_list), leave=False):
            
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            self.predictor.set_image(image)
            feature = self.predictor.features
            #align to aspect ratio of image
            if h < w:
                H = int(math.ceil((h / w) * feature.shape[-1]))
                feature = feature[:, :, :H, :]
            elif h > w:
                W = int(math.ceil((w / h) * feature.shape[-1]))
                feature = feature[:, :, :, :W]

            feature = feature.squeeze().movedim(0,-1)
            features.append(feature.cpu().detach())
        self.data = torch.stack(features, axis=0)  # n h w c

    def __call__(self, img_points):
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind = (img_points[:, 1] * img_scale[0]).long()
        y_ind = (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)


