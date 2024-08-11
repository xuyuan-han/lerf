"""
LERF configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.dataparsers.scannetpp_dataparser import ScanNetppDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from lerf.data.lerf_datamanager import LERFDataManagerConfig
from lerf.lerf import LERFModelConfig
from lerf.lerf_pipeline import LERFPipelineConfig

"""
Swap out the network config to use OpenCLIP or CLIP here.
"""
from lerf.encoders.clip_encoder import CLIPNetworkConfig
from lerf.encoders.openclip_encoder import OpenCLIPNetworkConfig

lerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                generate_depth_rays = False,
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(12, 12),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_lerf_samples=24,
            ),
            network=OpenCLIPNetworkConfig(
                clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512
            ),
            #  You can swap the type of input encoder by specifying different NetworkConfigs, the one below uses OpenAI CLIP, the one above uses OpenCLIP
            # network=CLIPNetworkConfig(
            #     clip_model_type="ViT-B/16", clip_n_dims=512
            # )
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=4000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for LERF",
)
lerf_method_big = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf-big",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4,
                eval_num_rays_per_batch=4,
                generate_depth_rays = False,
                generate_sam_masks=False
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(16, 16),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_lerf_samples=32,
                sam_masks=False
            ),
            network=OpenCLIPNetworkConfig(
                clip_model_type="ViT-L-14", clip_model_pretrained="laion2b_s32b_b82k", clip_n_dims=768
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=3000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="A larger version of LERF with a higher memory footprint, bigger CLIP model, and more hashgrid capacity",
)

lerf_method_lite = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf-lite",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=ColmapDataParserConfig(train_split_fraction=0.99, max_2D_matches_per_3D_point=0,
                                                  eval_mode="fraction"), #ScanNetppDataParserConfig(), TODO: Write own scannetpp dataparser that supports downsampling and setting train_split_fraction
                train_num_rays_per_batch= 4096,  #4096,
                eval_num_rays_per_batch= 4096,         #4096,
                generate_depth_rays = False,
                generate_sam_masks=False
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                hashgrid_sizes=(19,),
                hashgrid_layers=(16,),
                hashgrid_resolutions=((16, 512),),
                num_lerf_samples=12,
                sam_masks=False
            ),
            network=OpenCLIPNetworkConfig(
                clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=7000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="A lightweight version of LERF designed to work on smaller GPUs",
)    

# TODO the config for depth supervised learning is missing
lerf_method_depth = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf-depth",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=ColmapDataParserConfig(train_split_fraction=0.99,max_2D_matches_per_3D_point=-1,eval_mode="fraction"),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                #generate_depth_rays = False,
                #compute_other_losses_for_depth_rays=True, #TODO: currently generated radiance field gets negatively impacted when computing other losses for colmap rays. Semantic predictions seem unaffected.
                generate_sam_masks=False,
                use_dinov2=False
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                hashgrid_sizes=(19,),
                hashgrid_layers=(16,),
                hashgrid_resolutions=((16, 512),),
                num_lerf_samples=12,
                sam_masks=False
            ),
            network=OpenCLIPNetworkConfig(
                clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=7000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="A depth version of LERF designed to work on smaller GPUs",
)


lerf_method_sam = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf-sam",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=ColmapDataParserConfig(train_split_fraction=0.99, max_2D_matches_per_3D_point=0,
                                                  eval_mode="fraction"), #ScanNetppDataParserConfig(), TODO: Write own scannetpp dataparser that supports downsampling and setting train_split_fraction
                train_num_rays_per_batch= 4096,  #4096,
                eval_num_rays_per_batch= 4096,         #4096,
                generate_depth_rays = False,
                generate_sam_masks=True,
                use_dinov2=False

            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                hashgrid_sizes=(19,),
                hashgrid_layers=(16,),
                hashgrid_resolutions=((16, 512),),
                num_lerf_samples=12,
                sam_masks=True
            ),
            network=OpenCLIPNetworkConfig(
                clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=7000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="A lightweight version of LERF designed to work on smaller GPUs by using SAM masks",
)

lerf_method_dino = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf-dino",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=ColmapDataParserConfig(train_split_fraction=0.99, max_2D_matches_per_3D_point=0,
                                                  eval_mode="fraction"), #ScanNetppDataParserConfig(), TODO: Write own scannetpp dataparser that supports downsampling and setting train_split_fraction
                train_num_rays_per_batch= 4096,  #4096,
                eval_num_rays_per_batch= 4096,         #4096,
                generate_depth_rays = False,
                generate_sam_masks=False,
                use_dinov2= True
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                hashgrid_sizes=(19,),
                hashgrid_layers=(16,),
                hashgrid_resolutions=((16, 512),),
                num_lerf_samples=12,
                sam_masks=False
            ),
            network=OpenCLIPNetworkConfig(
                clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=7000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="A lightweight version of LERF designed to work on smaller GPUs by leveraging DINOv2",
)
