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


@dataclass
class LERFDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: LERFDataManager)
    patch_tile_size_range: Tuple[int, int] = (0.05, 0.5)
    patch_tile_size_res: int = 7
    patch_stride_scaler: float = 0.5
    percent_depth_rays: float = 0.5
    compute_other_losses_for_depth_rays: bool = False


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
        
        #calculate number of depth rays to sample based on input training rays and adapt config for datamanager accordingly
        self.num_depth_rays_per_batch = int(config.percent_depth_rays * config.train_num_rays_per_batch)
        config.train_num_rays_per_batch = config.train_num_rays_per_batch - self.num_depth_rays_per_batch

        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

        #TODO: check if load_llff_data , Load the data using the utilities
        #images, poses, bds, render_poses, i_test = load_llff_data(config.dataparser.data_dir)

        self.image_encoder: BaseImageEncoder = kwargs["image_encoder"]
        images = [self.train_dataset[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)

        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        clip_cache_path = Path(osp.join(cache_dir, f"clip_{self.image_encoder.name}"))
        dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))
        # NOTE: cache config is sensitive to list vs. tuple, because it checks for dict equality


        # TODO: do we have to explicitly set the data loader for depth load_llff? Put it into utils?
        # Load depth data
        #depth_list/depth_gts = DepthDataLoader(depth_list, self.device, cache_path=depth_cache_path) where (load_colmap_depth(config.dataparser.data_dir))
        # depth_gts = load_colmap_depth(args.datadir, factor=args.factor, bd_factor=.75)
        self.colmap_dataloader = ColmapDataloader(image_list=images.permute(0,2,3,1),num_rays_per_batch=self.num_depth_rays_per_batch,train_outputs=self.train_dataparser_outputs, device=self.device, directory_path=config.dataparser.data)
        self.dino_dataloader = DinoDataloader(
            image_list=images,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4])},
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

        #sample colmap rays
        colmap_ray_indices,colmap_rgb,colmap_depths,colmap_weights = self.colmap_dataloader()

        #Append ray indices of colmap rays to ray_indices.
        ray_indices_sum = torch.cat((ray_indices,colmap_ray_indices),dim=0)
        ray_bundle = self.train_ray_generator(ray_indices_sum)

        #TODO current problem: if we filter here based on compute_other_losses, clip scales will only be defined for ray_indices sampled by pixel sampler and this will lead to issues in getOutputs() function.
        #                       Could avoid this by filtering out outputs for depth rays in getOutputs function. Or alternatively we could just compute everything for depth rays and filter in getLossdict() prior to loss computation

        if self.config.compute_other_losses_for_depth_rays:
            #append color of colmap rays to gt colors
            batch["image"] = torch.cat((batch["image"],colmap_rgb),dim=0)
            batch["clip"], clip_scale = self.clip_interpolator(ray_indices_sum)
            batch["dino"] = self.dino_dataloader(ray_indices_sum)
        else:
            batch["clip"], clip_scale = self.clip_interpolator(ray_indices)
            batch["dino"] = self.dino_dataloader(ray_indices)

        #Add weights and depths, as well as indicator how many colmap rays we have to batch.
        batch["depths"] = colmap_depths
        batch["weights"] = colmap_weights
        #append config option related to computing losses for depth rays to metadata
        ray_bundle.metadata["compute_other_losses_for_depth_rays"] = self.config.compute_other_losses_for_depth_rays
        ray_bundle.metadata["num_depth_rays"] = self.num_depth_rays_per_batch
        
        ray_bundle.metadata["clip_scales"] = clip_scale
        # assume all cameras have the same focal length and image width
        ray_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
        ray_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
        ray_bundle.metadata["fy"] = self.train_dataset.cameras[0].fy.item()
        ray_bundle.metadata["height"] = self.train_dataset.cameras[0].height.item()

        return ray_bundle, batch
