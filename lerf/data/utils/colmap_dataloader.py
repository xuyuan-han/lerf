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


        """
        def get_rays_by_coord_np(cx, cy, fx, fy, c2w, coords):
            i, j = (coords[:,0]-cx)/fx, -(coords[:,1]-cy)/fy
            dirs = np.stack([i,j,-np.ones_like(i)],-1)
            rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
            rays_d, norm = camera_utils.normalize_with_norm(torch.tensor(rays_d), -1)
            rays_d = np.array(rays_d)
            rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
            return rays_o, rays_d

        rays_depth_list = []
        for i, pose in enumerate(train_outputs.cameras.camera_to_worlds):
            rays_depth = np.stack(get_rays_by_coord_np(train_outputs.cameras.cx[i].item(), train_outputs.cameras.cy[i].item(),train_outputs.cameras.fx[i].item(),train_outputs.cameras.fy[i].item(), np.array(pose), np.stack(data_list[i]['coord'],axis=0)), axis=0) # 2 x N x 3
            rays_depth = np.transpose(rays_depth, [1,0,2]) # N x 2 x 3
            depth_value = np.repeat(np.stack(data_list[i]['depth'],axis=0)[:,None], 3, axis=2) # N x 1 x 3
            weights = np.repeat(np.stack(data_list[i]['error'],axis=0)[:,None,None], 3, axis=2) # N x 1 x 3
            rays_depth = np.concatenate([rays_depth, depth_value, weights], axis=1) # N x 4 x 3
            rays_depth_list.append(rays_depth)

            if i == 1:
                #swap x and y, since generate_rays() assumes coord[:,0] is y and coord[:,1] is x
                coords = np.stack(data_list[i]['coord'],axis=0)
                coords[:,[0,1]] = coords[:,[1,0]]
                
        self.rays_depth = np.concatenate(rays_depth_list, axis=0)
        print('rays_weights mean:', np.mean(rays_depth[:,3,0]))
        print('rays_weights std:', np.std(rays_depth[:,3,0]))
        print('rays_weights max:', np.max(rays_depth[:,3,0]))
        print('rays_weights min:', np.min(rays_depth[:,3,0]))
        print('rays_depth.shape:', rays_depth.shape)
        self.rays_depth = rays_depth.astype(np.float32)
        print('shuffle depth rays')
        np.random.shuffle(self.rays_depth)

        self.max_depth = np.max(self.rays_depth[:,3,0])
        """



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

        """
        data_list = {}
        # Process each camera
        for i in range(points3D_image_ids.shape[0]):
            for j in range(points3D_image_ids.shape[1]):
                if points3D_image_ids[i,j] != -1 and self.colmapId2TrainId[int(points3D_image_ids[i,j])] != -1:
                    
                    mapped_id = self.colmapId2TrainId[int(points3D_image_ids[i,j])]
                    point2D = points3D_image_xy[i,j]

                    #compute depth in camera frame
                    c2w = cameras.camera_to_worlds[mapped_id]
                    depth = c2w[:3, 2].unsqueeze(0) @ (points3D_xyz[i, :] - c2w[:3, 3])

                    err = points3D_errors[i]
                    weight = 2 * np.exp(-(err/Err_mean)**2)

                    if mapped_id in data_list.keys():
                        data_list[mapped_id]["depth"].append(np.array(depth))
                        data_list[mapped_id]["coord"].append(np.array(point2D))
                        data_list[mapped_id]["error"].append(weight)
                    else:
                        data_list[mapped_id] = {"depth":[np.array(depth)], "coord":[np.array(point2D)], "error":[weight]}

        
        # Define or calculate near and far planes
        if manual_near_far:
            near, far = manual_near_far
        else:
            near = torch.quantile(depth_list, 0.01)
            far = torch.quantile(depth_list, 0.99)

        # Validate depth range
        valid_mask = (depths >= near) & (depths <= far)
        valid_depths = depths[valid_mask]
        valid_errors = points3D_errors[valid_mask]
        weights = 2 * torch.exp(-torch.pow((valid_errors / Err_mean), 2))

        # Collect valid data
        if points3D_image_xy is not None:
            # Adjusted to handle mismatch in dimensions
            image_xy = points3D_image_xy[i]
            valid_mask = valid_mask[:image_xy.shape[0]]  # Ensure mask is of correct size
            valid_coords = image_xy[valid_mask]

        if len(valid_depths) > 0:
            data_list.append({
                "depth": valid_depths.numpy(),
                "coord": valid_coords.numpy() if points3D_image_xy is not None else None,
                "error": weights.numpy()
            })
        else:
            print(f'Camera {i}: No valid depths found within range.')

        """


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


        """
        #prepare rays
        c = selected_ray_indices[:, 0].type(torch.IntTensor)  # camera indices
        coords = selected_ray_indices[:, 1:] + 0.5  # coords in row indices, col indices (y,x)
        ray_bundle = self.cameras.generate_rays(camera_indices=c.unsqueeze(-1),coords=coords)
        """

        #return ray bundle and depth and weights. Also return ray indices
        return selected_ray_indices,gt_rgb,selected_depths,selected_weights

"""
def load_colmap_depth(dataparser, factor=8, bd_factor=.75):
    #data_file = Path(basedir) / 'colmap_depth.npy'
    #images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    #points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')
    Errs = dataparser.metadata["points3D_error"]
    Err_mean = np.mean(Errs)
    #Errs = np.array([point3D.error for point3D in points.values()])
    #Err_mean = np.mean(Errs)
    #print("Mean Projection Error:", Err_mean)
    poses = dataparser.cameras.camera_to_worlds
    #poses = get_poses(images)
    _, bds_raw, _ = _load_data(basedir, factor=factor)  # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds_raw.min() * bd_factor)

    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    data_list = []
    for id_im in range(1, len(images) + 1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im - 1, :3, 2].T @ (point3D - poses[id_im - 1, :3, 3])) * sc
            if depth < bds_raw[id_im - 1, 0] * sc or depth > bds_raw[id_im - 1, 1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err / Err_mean) ** 2)
            depth_list.append(depth)
            coord_list.append(point2D / factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append(
                {"depth": np.array(depth_list), "coord": np.array(coord_list), "error": np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list
"""
"""
def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_cameras_binary(camerasfile)

    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print('Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h, w, f]).reshape([3, 1])

    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)

    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_points3d_binary(points3dfile)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]],
                           1)

    return poses, pts3d, perm


def load_colmap_depth(basedir, factor=8, bd_factor=.75):
    poses, points, perm = load_colmap_data(directory_path)
    #data_file = Path(basedir) / 'colmap_depth.npy'

    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
   # points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)

    poses = get_poses(images)

    #_, bds_raw, _ = _load_data(basedir, factor=factor)  # factor=8 downsamples original imgs by 8x
    #load_colmap_data(directory_path)
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds_raw.min() * bd_factor)

    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    data_list = []
    for id_im in range(1, len(images) + 1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im - 1, :3, 2].T @ (point3D - poses[id_im - 1, :3, 3])) * sc
            if depth < bds_raw[id_im - 1, 0] * sc or depth > bds_raw[id_im - 1, 1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err / Err_mean) ** 2)
            depth_list.append(depth)
            coord_list.append(point2D / factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append(
                {"depth": np.array(depth_list), "coord": np.array(coord_list), "error": np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list


def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3, 1])
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)


def _load_data(bounds, factor=None, width=None, height=None, load_imgs=True):
    #poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = bounds[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # 3 x 5 x N
    bds = bounds[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def get_poses_bounds( poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind - 1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr == 1]
    print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())

    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]
        close_depth, inf_depth = np.percentile(zs, .5), np.percentile(zs, 99.5)
        # print( i, close_depth, inf_depth )

        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    #np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)
    return save_arr
"""
