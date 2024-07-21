import os
import typing
from abc import ABC
from pathlib import Path
import numpy as np
import torch
from einops import rearrange
import cv2
import math
from lerf.segment_anything import sam_model_registry, SamPredictor
import glob
class SAMDataloader(ABC):
    def __init__(
        self,
        device: torch.device,
        npy_paths,
        image_paths,
        npy_directory,
        image_shape: typing.Tuple[int, int],
        sam_checkpoint: str =  "lerf\segment_anything\sam_vit_h_4b8939.pth",
        model_type: str = "vit_h",
    ):
        self.device = device
        self.npy_paths = npy_paths
        self.image_paths = image_paths
        self.image_shape = image_shape
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.npy_directory = npy_directory
        img_list = []
        for img_path in glob.glob(os.path.join(self.image_paths, "*")):
            if img_path.endswith(".png") or img_path.endswith(".jpg") or img_path.endswith(".JPG"):
                img_list.append(img_path)

        self.image_paths = img_list
        self.predictor = self.init_sam()

        self.features = self.load_or_extract_features()

    def init_sam(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        return predictor

    def get_embeddings(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        self.predictor.set_image(image)
        feature = self.predictor.features
        if h < w:
            H = int(math.ceil((h / w) * feature.shape[-1]))
            feature = feature[:, :, :H, :]
        elif h > w:
            W = int(math.ceil((w / h) * feature.shape[-1]))
            feature = feature[:, :, :, :W]
        return feature

    def extract_and_save_features(self, image_path, npy_directory):
        feature = self.get_embeddings(image_path)
        base_name = os.path.basename(image_path).split(".")[0] + ".npy"
        save_path = os.path.join(npy_directory, base_name)
        np.save(save_path, feature.squeeze().cpu().numpy())
        print(f"Extracted and saved features for {image_path} at {save_path}")
        return save_path

    def load_or_extract_features(self):
        features = []
        if not os.path.exists(self.npy_directory):
            os.mkdir(self.npy_directory)
        for npy_path, image_path in zip(self.npy_paths, self.image_paths):
            if not os.path.exists(npy_path):
                npy_path = self.extract_and_save_features(image_path, self.npy_directory)
            feature = np.load(npy_path)
            feature = rearrange(feature, "c h w -> h w c")
            features.append(feature)
        features = np.stack(features, axis=0)  # n h w c
        return torch.from_numpy(features).to(self.device)

    def __call__(self, img_points):
        img_scale = (
            self.features.shape[1] / self.image_shape[0],
            self.features.shape[2] / self.image_shape[1],
        )
        x_ind = (img_points[:, 1] * img_scale[0]).long()
        y_ind = (img_points[:, 2] * img_scale[1]).long()
        return (self.features[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)


