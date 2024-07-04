import typing
import imageio
import torch
from lerf.data.utils.dino_extractor import ViTExtractor
from lerf.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm
from nerfstudio.data.utils.colmap_parsing_utils import *
from nerfstudio.data.dataparsers.colmap_dataparser import *
from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset
from nerfstudio.data.utils.colmap_parsing_utils import *

class ColmapDataloader:

    def __init__(
            self,
            device: torch.device,
            directory_path: str = None,
    ):
        self.device = device
        # use SfM to get depth maps
        # sparse_path = Path(directory_path) /"colmap"/ "sparse" / "0"
        #cmd = ImagesToNerfstudioDataset(
        #    data=directory_path / "images", output_dir=directory_path / "converted",

        #     skip_colmap=True,colmap_model_path=sparse_path , sfm_tool="colmap", use_sfm_depth=True, skip_image_processing=True,
        #  )

        #cmd.main()
        #exctract the data from images










        self.cfg = ColmapDataParserConfig(data=Path(directory_path), train_split_fraction=1.0,
                                          max_2D_matches_per_3D_point=-1,eval_mode="all")
        self.dataparser = self.cfg.setup()
        self.data_parser_outputs = self.dataparser.get_dataparser_outputs(split="train")

        images = read_images_binary(Path(directory_path) / "colmap" / "sparse" / "0" / "images.bin")
        image_filenames = self.data_parser_outputs.image_filenames

        self.names = {}
        for i in images:
            for index, j in enumerate(image_filenames):
                if images[i].name == j.name:
                    self.names[images[i].id] = index

        print(self.names)


        self.load_colmap_depth(self.data_parser_outputs)


        print("Conversion complete.")



    def load_colmap_depth(self, dataparser_outputs, manual_near_far=None):
        # Retrieve metadata and camera information
        cameras = dataparser_outputs.cameras
        metadata = dataparser_outputs.metadata

        points3D_xyz = metadata['points3D_xyz']
        points3D_errors = metadata['points3D_error']
        points3D_image_ids = metadata.get('points3D_image_ids', None)
        points3D_image_xy = metadata.get('points3D_points2D_xy', None)
        points3D_points2D_xy = metadata.get('points3D_points2D_xy', None)
        p = metadata["points3D_num_points2D"]
        print(p, "points")
        print(points3D_points2D_xy.shape, " points3D_points2D_xy")
        print(points3D_errors.shape, " points3D_errors")
        print(points3D_image_ids)
        # Calculate mean projection error
        Err_mean = torch.mean(points3D_errors)
        print("Mean Projection Error:", Err_mean.item())

        data_list = []
       # print(cameras.camera_to_worlds.shape)
        print(cameras.shape, " cameras")
        print(torch.max(points3D_image_ids))
        print(points3D_xyz[0:19], " points3D_xyz")




        # Process each camera
        for i in range(points3D_image_ids.shape[0]):
            depth_list = []
            for j in range(points3D_image_ids.shape[1]):
                if points3D_image_ids[i,j] != -1:
                    c2w = cameras.camera_to_worlds[self.names[int(points3D_image_ids[i,j])]]
                    depth = c2w[:3, 2].T @ (points3D_xyz[i, :] - c2w[:3, 3])
                    if depth > 0:
                        print(depth)


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

        return data_list


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
def __call__(self, img_points):
    # img_points: (B, 3) # (img_ind, x, y)
    img_scale = (
        self.data.shape[1] / self.cfg["image_shape"][0],
        self.data.shape[2] / self.cfg["image_shape"][1],
    )
    x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
    return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)
