import typing
import imageio
import torch
from lerf.data.utils.dino_extractor import ViTExtractor
from lerf.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm
from nerfstudio.data.utils.colmap_parsing_utils import *
from nerfstudio.data.dataparsers.colmap_dataparser import *
from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset
from nerfstudio.process_data.colmap_utils import *
from nerfstudio.data.utils.colmap_parsing_utils import *
from nerfstudio.cameras import camera_utils

from nerfstudio.data.dataparsers.scannetpp_dataparser import *

class ColmapDataloader:

    def __init__(
            self,
            train_outputs,
            num_rays_per_batch,
            image_list,
            device: torch.device,
            directory_path: str = None, 
    ):
        
        self.device = device

        self.num_rays_per_batch = num_rays_per_batch

        self.cameras = train_outputs.cameras
        self.image_list = image_list
        
        #map colmap ids to dataparser output ids
        if (directory_path / "images.txt").exists():
            images = read_images_text(directory_path / "images.txt")
        else:
            images = read_images_binary( directory_path / "images.bin")

        image_filenames = train_outputs.image_filenames

        self.colmapId2TrainId = {}
        for i in images:
            #initialize with -1, so that we can indicate unmatched frames that are not in the training dataset
            self.colmapId2TrainId[images[i].id] = -1
            for index, j in enumerate(image_filenames):
                if images[i].name == j.name:
                    self.colmapId2TrainId[images[i].id] = index
                    break
        
        #colmap_to_json(recon_dir=(Path(directory_path) / "colmap" / "sparse" / "0"),output_dir=Path(directory_path) / "test") #still discrepancy between own generated json transforms and supplied ones from nerfstudio

        #prepare colmap data
        self.ray_indices,self.depths,self.weights = self.load_colmap_depth(train_outputs)
        print("Conversion complete.")

        #TODO: Look at how to cache these results



    def load_colmap_depth(self, dataparser_outputs, manual_near_far=None):
        # Retrieve metadata and camera information
        cameras = dataparser_outputs.cameras
        metadata = dataparser_outputs.metadata

        points3D_xyz = metadata['points3D_xyz']
        points3D_errors = metadata['points3D_error']
        points3D_image_ids = metadata.get('points3D_image_ids', None)
        points3D_image_xy = metadata.get('points3D_points2D_xy', None)
        #p = metadata["points3D_num_points2D"]

        # Calculate mean projection error
        Err_mean = torch.mean(points3D_errors)
        print("Mean Projection Error:", Err_mean.item())

        #TODO: check how to compute bounds. In DS-Nerf they precompute percentiles for min and max depths for each posed image (close_depth, inf_depth = np.percentile(zs, .5), np.percentile(zs, 99.5)). This is then used in the following to filter points. Also they employ some scaling factor bd_factor (usually set to 0.75).


        #preprocess data to be in shape (samples, 3) where 3 is cameraindex and coords in y, x order. Additionally keep one array for depth and weight values of shape (samples, 1)
        ray_indices = []
        weights = []
        depths =[]
        sorted_out = 0
        for i in range(points3D_image_ids.shape[0]):
            for j in range(points3D_image_ids.shape[1]):

                if points3D_image_ids[i,j] != -1 and self.colmapId2TrainId[int(points3D_image_ids[i,j])] != -1:
                    mapped_id = self.colmapId2TrainId[int(points3D_image_ids[i,j])]
                    point2D = points3D_image_xy[i,j]

                    #sanity check that point coordinates are within width and height bounds
                    if point2D[0] < cameras.width[mapped_id].item() and point2D[0] >= 0. and point2D[1] < cameras.height[mapped_id].item() and point2D[1] >= 0.:
                        #compute depth in camera frame
                        c2w = cameras.camera_to_worlds[mapped_id]
                        depth = c2w[:3, 2].unsqueeze(0) @ (points3D_xyz[i, :] - c2w[:3, 3])

                        err = points3D_errors[i]
                        weight = 2 * np.exp(-(err/Err_mean)**2)

                        #swap x and y since render_rays() expects tensor([y,x])
                        ray_indices.append(torch.tensor([mapped_id,point2D[1],point2D[0]]))
                        depths.append(depth)
                        weights.append(weight)
                    else:
                        sorted_out +=1

        depths = torch.tensor(depths)
        weights = torch.tensor(weights)
        ray_indices = torch.stack(ray_indices,dim=0)

        return ray_indices,depths,weights


    def __call__(self):

        #generate n samples between 0 and ray_indices.shape[0]. DS-Nerf assumes half of all rays per iteration to be depth rays by default.
        indices = torch.randperm(self.ray_indices.shape[0])[:self.num_rays_per_batch]
        
        #index in datastructures
        selected_ray_indices = self.ray_indices[indices]
        selected_weights = self.weights[indices]
        selected_depths = self.depths[indices]
        
        #TODO: How to cope with colmap rays that do not directly align with pixel grid?
        #      For generating clip and dino gt we currently require rays aligned to the pixel grid.
        #      Nerfstudio assumes integer coordinates between 0 and width-1/height-1 in ray indices array. These get then mapped to image coordinates by applying +0.5
        #      Two options: 1.Adapt dino and clip loaders to interpolate ray values between grids.
        #                   2.Generate ray bundle from floored rays and assume generated depth values to be for one whole pixel
        #      For now use option two...

        #convert image coordinates in ray_indices tensor to integer coordinates
        selected_ray_indices[:,1:] = torch.floor(selected_ray_indices[:,1:])
        selected_ray_indices = selected_ray_indices.type(torch.IntTensor)

        #compute gt rgb values
        c, y, x = (i.flatten() for i in torch.split(selected_ray_indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        gt_rgb = self.image_list[c, y, x]

        #return ray bundle and depth and weights. Also return ray indices
        return selected_ray_indices,gt_rgb,selected_depths,selected_weights
