# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Datamanager.
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from copy import copy

import torch
import yaml
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper
from rich.progress import Console

CONSOLE = Console(width=120)

from lerf.data.utils.dino_dataloader import DinoDataloader
from lerf.data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from lerf.encoders.image_encoder import BaseImageEncoder
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from lerf.data.utils.colmap_dataloader import ColmapDataloader
from lerf.data.utils.sam_dataloader import SAMDataloader

# For SAM
from lerf.data.utils.feature_dataloader import FeatureDataloader

@dataclass
class LERFDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: LERFDataManager)
    patch_tile_size_range: Tuple[int, int] = (0.05, 0.5)
    patch_tile_size_res: int = 7
    patch_stride_scaler: float = 0.5
    percent_depth_rays: float = 0.25
    compute_other_losses_for_depth_rays: bool = False
    generate_depth_rays: bool = True
    generate_sam_masks: bool = True
    use_dinov2: bool = True


class LERFDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: LERFDataManagerConfig

    def __init__(
        self,
        config: LERFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):

        #adapt folders if using scannetpp
        if not (config.data / "colmap" / "sparse" / "0").exists():
            config.dataparser.colmap_path = Path("dslr/colmap")
            config.dataparser.images_path = Path("dslr/resized_images")

        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

        if config.generate_depth_rays:
            #calculate number of depth rays to sample based on input training rays and adapt config for datamanager accordingly
            self.num_depth_rays_per_batch = int(config.percent_depth_rays * config.train_num_rays_per_batch)

            if config.compute_other_losses_for_depth_rays and self.train_pixel_sampler:
                    num_train_rays = config.train_num_rays_per_batch - self.num_depth_rays_per_batch
                    self.train_pixel_sampler.set_num_rays_per_batch(num_train_rays)

        self.image_encoder: BaseImageEncoder = kwargs["image_encoder"]
        images = [self.train_dataset[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)

        if config.use_dinov2:
            dino_model_type = "dinov2_vitb14"
            dino_stride = 14
        else:
            dino_model_type = "dino_vits8"
            dino_stride = 8

        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        clip_cache_path = Path(osp.join(cache_dir, f"clip_{self.image_encoder.name}"))
        dino_cache_path = Path(osp.join(cache_dir, f"{dino_model_type}.npy"))
        # NOTE: cache config is sensitive to list vs. tuple, because it checks for dict equality

        if config.generate_depth_rays:
            colmap_cache_path = Path(osp.join(cache_dir, "depth.npy"))

            self.colmap_dataloader = ColmapDataloader(
                image_list=images.permute(0,2,3,1),
                device=self.device,
                cfg={
                    "dir_path": str(config.dataparser.data / config.dataparser.colmap_path),
                    "image_shape": list(images.permute(0,2,3,1).shape),
                },
                train_outputs=self.train_dataparser_outputs,
                cache_path=colmap_cache_path,
            )

        if config.generate_sam_masks:
            sam_cache_path = Path(osp.join(cache_dir, "sam.npy"))

            # Load SAM data
            self.sam_loader = SAMDataloader(
                device=self.device,
                cfg={"image_shape": list(images.shape[2:4]),
                     "sam_checkpoint": "lerf\segment_anything\sam_vit_h_4b8939.pth",
                     "model_type": "vit_h"},
                cache_path=sam_cache_path,
                image_paths=self.train_dataparser_outputs.image_filenames
            )

        self.dino_dataloader = DinoDataloader(
            image_list=images,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4]),
                 "model_type": dino_model_type,
                 "dino_stride": dino_stride},
            cache_path=dino_cache_path,
        )


        torch.cuda.empty_cache()
        self.clip_interpolator = PyramidEmbeddingDataloader(
            image_list=images,
            device=self.device,
            cfg={
                "tile_size_range": [0.05, 0.5],
                "tile_size_res": 7,
                "stride_scaler": 0.5,
                "image_shape": list(images.shape[2:4]),
                "model_name": self.image_encoder.name,
            },
            cache_path=clip_cache_path,
            model=self.image_encoder,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]

        if self.config.generate_depth_rays:
            #sample colmap rays
            colmap_ray_indices,colmap_rgb,colmap_depths,colmap_weights = self.colmap_dataloader(self.num_depth_rays_per_batch)

            #Append ray indices of colmap rays to ray_indices.
            ray_indices_sum = torch.cat((ray_indices,colmap_ray_indices),dim=0)
            ray_bundle = self.train_ray_generator(ray_indices_sum)           

            #Add weights and depths, as well as indicator how many colmap rays we have to batch.
            batch["depths"] = colmap_depths
            batch["sigma"] = colmap_weights
            batch["num_depth_rays"] = self.num_depth_rays_per_batch

            #append config option related to computing losses for depth rays to metadata
            ray_bundle.metadata["num_depth_rays"] = self.num_depth_rays_per_batch
        else:
            ray_bundle = self.train_ray_generator(ray_indices)

        if self.config.generate_depth_rays and self.config.compute_other_losses_for_depth_rays:
            #append color of colmap rays to gt colors
            batch["image"] = torch.cat((batch["image"],colmap_rgb.to(batch["image"])),dim=0)
            batch["clip"], clip_scale = self.clip_interpolator(ray_indices_sum)
            batch["dino"] = self.dino_dataloader(ray_indices_sum)
            batch["sam"] = self.sam_loader(ray_indices)
        else:
            batch["clip"], clip_scale = self.clip_interpolator(ray_indices)
            
            if self.config.generate_depth_rays:
                #needed since otherwise we get problems when calculating outputs
                clip_scale = torch.cat((clip_scale,torch.ones((self.num_depth_rays_per_batch)).to(clip_scale).unsqueeze(1)),dim=0)

            batch["dino"] = self.dino_dataloader(ray_indices)

        if self.config.generate_sam_masks:
            batch["sam"] = self.sam_loader(ray_indices)


        ray_bundle.metadata["clip_scales"] = clip_scale
        # assume all cameras have the same focal length and image width
        ray_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
        ray_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
        ray_bundle.metadata["fy"] = self.train_dataset.cameras[0].fy.item()
        ray_bundle.metadata["height"] = self.train_dataset.cameras[0].height.item()
        return ray_bundle, batch
