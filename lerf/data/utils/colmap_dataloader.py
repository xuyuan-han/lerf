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
from nerfstudio.data.utils.pixel_sampling_utils import divide_rays_per_image

from nerfstudio.data.dataparsers.scannetpp_dataparser import *

class ColmapDataloader:

    def __init__(
            self,
            cfg: dict,
            device: torch.device,
            image_list: torch.tensor,
            train_outputs: DataparserOutputs,
            cache_path: Path
    ):
        
        self.cfg = cfg
        self.device = device
        self.cache_path = cache_path
        self.data = None
        self.try_load(image_list,train_outputs) # don't save image_list, avoid duplicates

    def create(self, image_list, dataparser_outputs, manual_near_far=None):
        
        #colmap_to_json(recon_dir=(Path(directory_path) / "colmap" / "sparse" / "0"),output_dir=Path(directory_path) / "test") #still discrepancy between own generated json transforms and supplied ones from nerfstudio

        #map colmap ids to dataparser output ids
        if (Path(self.cfg["dir_path"]) / "images.txt").exists():
            images = read_images_text(Path(self.cfg["dir_path"]) / "images.txt")
        else:
            images = read_images_binary(Path(self.cfg["dir_path"]) / "images.bin")

        image_filenames = dataparser_outputs.image_filenames

        colmapId2TrainId = {}
        for i in images:
            #initialize with -1, so that we can indicate unmatched frames that are not in the training dataset
            colmapId2TrainId[images[i].id] = -1
            for index, j in enumerate(image_filenames):
                if images[i].name == j.name:
                    colmapId2TrainId[images[i].id] = index
                    break


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

        #preprocess data to be in shape (samples, 3) where 3 is cameraindex and coords in y, x order. Additionally keep one array for depth and weight values of shape (samples, 1)
        ray_indices = []
        weights = []
        depths =[]
        for i in range(points3D_image_ids.shape[0]):
            for j in range(points3D_image_ids.shape[1]):

                if points3D_image_ids[i,j] != -1 and colmapId2TrainId[int(points3D_image_ids[i,j])] != -1:
                    mapped_id = colmapId2TrainId[int(points3D_image_ids[i,j])]
                    point2D = points3D_image_xy[i,j]
                    #convert height to nerfstudio indexing
                    #point2D[1]= cameras.height[mapped_id].item() - point2D[1]

                    #sanity check that point coordinates are within width and height bounds
                    if point2D[0] < cameras.width[mapped_id].item() and point2D[0] >= 0. and point2D[1] < cameras.height[mapped_id].item() and point2D[1] >= 0.:
                        #compute depth in camera frame (we need positive depth for optimizing network)
                        c2w = cameras.camera_to_worlds[mapped_id]
                        depth = c2w[:3, 2].unsqueeze(0) @ (points3D_xyz[i, :] - c2w[:3, 3])
                        depth = -depth #depth loss expects positive depths

                        err = points3D_errors[i]
                        weight = 2 * np.exp(-(err/Err_mean)**2)

                        #swap x and y since render_rays() expects tensor([y,x])
                        ray_indices.append(torch.tensor([mapped_id,point2D[1],point2D[0]]))
                        depths.append(depth)
                        weights.append(weight)

        depths = torch.tensor(depths)
        weights = torch.tensor(weights)
        ray_indices = torch.stack(ray_indices,dim=0)

        #TODO: check how to compute bounds. In DS-Nerf they precompute percentiles for min and max depths for each posed image (close_depth, inf_depth = np.percentile(zs, .5), np.percentile(zs, 99.5)). This is then used in the following to filter points. Also they employ some scaling factor bd_factor (usually set to 0.75).
        """
        for i in range(len(image_filenames)):
            rays_for_cam = (ray_indices[:,0] == i)
            depths_for_cam = depths[rays_for_cam]
            close_depth, inf_depth = np.percentile(depths_for_cam, .5), np.percentile(depths_for_cam, 99.5)
            valid_depth_indices = depth[rays_for_cam] >= close_depth & depth[rays_for_cam] <= inf_depth
            depths[rays_for_cam] = valid_depth_indices * depths[rays_for_cam]
        
        valid_depths = depths > 0
        ray_indices = ray_indices[valid_depths]
        depths = depths[valid_depths]
        weights = weights[valid_depths]
        """

        #TODO: How to cope with colmap rays that do not directly align with pixel grid?
        #      For generating clip and dino gt we currently require rays aligned to the pixel grid.
        #      Nerfstudio assumes integer coordinates between 0 and width-1/height-1 in ray indices array. These get then mapped to image coordinates by applying +0.5
        #      Two options: 1.Adapt dino and clip loaders to interpolate ray values between grids.
        #                   2.Generate ray bundle from floored rays and assume generated depth values to be for one whole pixel
        #      For now use option two...

        #convert image coordinates in ray_indices tensor to integer coordinates
        ray_indices[:,1:] = torch.floor(ray_indices[:,1:])

        #compute gt rgb values
        c, y, x = (i.flatten() for i in torch.split(ray_indices, 1, dim=-1))
        c, y, x = c.cpu().type(torch.IntTensor), y.cpu().type(torch.IntTensor), x.cpu().type(torch.IntTensor)
        rgbs = image_list[c, y, x]

        self.data = torch.cat((ray_indices,rgbs,depths.unsqueeze(1),weights.unsqueeze(1)),dim=1)

    def __call__(self, num_depth_rays):
        #generate n samples between 0 and ray_indices.shape[0]. DS-Nerf assumes half of all rays per iteration to be depth rays by default.
        indices = (
            torch.rand((num_depth_rays), device=self.device)
            * torch.tensor([self.data.shape[0]], device=self.device)
        ).long()
        
        #index in datastructures
        selected_data = self.data[indices].to(self.device)

        selected_ray_indices = selected_data[:,:3].type(torch.LongTensor)
        selected_rgbs = selected_data[:,3:6]
        selected_depths = selected_data[:,6]
        selected_weights = selected_data[:,7]

        #return ray indices, colors, depths, and weights 
        return selected_ray_indices,selected_rgbs,selected_depths,selected_weights

    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            raise ValueError("Config mismatch")
        self.data = torch.from_numpy(np.load(self.cache_path)).to(self.device)

    def save(self):
        os.makedirs(self.cache_path.parent, exist_ok=True)
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
        np.save(self.cache_path, self.data)

    def try_load(self,image_list, dataparser_outputs):
        try:
            self.load()
        except (FileNotFoundError, ValueError):
            self.create(image_list,dataparser_outputs)
            self.save()
