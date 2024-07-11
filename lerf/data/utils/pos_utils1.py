import numpy as np
import os
import sys
import imageio
import skimage.transform

from nerfstudio.data.utils.colmap_parsing_utils import *


def load_colmap_data(realdir):
    #load camera data binaries
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_cameras_binary(camerasfile)
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print('Cameras', len(cam))

    #extract the camera parameters
    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h, w, f]).reshape([3, 1])


    #load image data binaries
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print('Images #', len(names))

    perm = np.argsort(names)
    #extract extrinsic and intrinsic camera parameters
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

    #extract 3d points data
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_points3D_binary(points3dfile)

    # exctract IDs of images that are relevant to the 3D points and map IDs to indices
    relevant_image_ids = set()
    for k in pts3d:
        relevant_image_ids.update(pts3d[k].image_ids)
    print('Relevant images:', relevant_image_ids)
    id_to_index = {}
    index = 0
    for k in imdata:
        if imdata[k].id in relevant_image_ids:
            id_to_index[imdata[k].id] = index
            index += 1
    print("Mapping between images and indices: " ,id_to_index)
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]],
                           1)

    # Verify correspondence and create a new pts3d dictionary with updated image_ids
    new_pts3d = {}
    for pt_id in pts3d:
        pt = pts3d[pt_id]
        new_image_ids = []
        for img_id in pt.image_ids:
            if img_id in id_to_index:
                new_image_ids.append(id_to_index[img_id])
            else:
                new_image_ids.append(-1)
        # Create a new Point3D object with the updated image_ids
        new_pts3d[pt_id] = Point3D(
            id=pt.id,
            xyz=pt.xyz,
            rgb=pt.rgb,
            error=pt.error,
            image_ids=np.array(new_image_ids),
            point2D_idxs=pt.point2D_idxs
        )
    #for image in imdata:
     #   print('Image:', imdata[image].name, 'ID:', imdata[image].id)
    for id in pts3d:
        pt_old = pts3d[id]
        pt_new = new_pts3d[id]
        print('Old:', pt_old.image_ids, 'New:', pt_new.image_ids)
        break


    return poses, new_pts3d, perm


def save_poses(basedir, poses, pts3d, perm):
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

    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)



def gen_poses(basedir,  factors=None):
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    #assume that the colmap exists and we want to create poses
    if not all([f in files_had for f in files_needed]):
        print('Need to run COLMAP')

        #run_colmap(basedir, match_type)
    else:
        print('Don\'t need to run COLMAP')

    print('Post-colmap')

    poses, pts3d, perm = load_colmap_data(basedir)

    save_poses(basedir, poses, pts3d, perm)

    print('Done with imgs2poses')

    return  poses, pts3d, perm

    def load_colmap_depth(basedir, factor=8, bd_factor=.75):
        data_file = Path(basedir) / 'colmap_depth.npy'

        images = read_images_binary(Path(basedir) / 'colmap' / 'sparse' / '0' / 'images.bin')
        # points = read_points3D_binary(Path(basedir) / 'colmap' / 'sparse' / '0' / 'points3D.bin')
        # points = self.pts3d
        Errs = np.array([point3D.error for point3D in points.values()])
        Err_mean = np.mean(Errs)
        print("Mean Projection Error:", Err_mean)

        poses = get_poses(images)
        _, bds_raw, _ = _load_data(basedir, images, factor=factor)  # factor=8 downsamples original imgs by 8x
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
            if id_im in images:
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
                    # data_list.append(
                    #   {"depth": np.array(depth_list), "coord": np.array(coord_list), "error": np.array(weight_list)})
                    # print(depth_list)
                    data_list.append(
                        {"depth": np.array(depth_list)})
                else:
                    print(id_im, len(depth_list))
        # json.dump(data_list, open(data_file, "w"))
        np.save(data_file, data_list)
        return data_list


def _load_data(basedir, validated_images, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'colmap', 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # 3 x 5 x N
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
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

    # Helper function to extract numeric part of the filename
    def extract_numeric_part(filename):
        basename = os.path.splitext(filename)[0]
        # Assuming the filename is in the format 'frame_00001', we extract the numeric part
        numeric_part = ''.join(filter(str.isdigit, basename))
        return int(numeric_part)

    # Filter the images based on validated_images
    validated_images = set(validated_images)
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))
                if (f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')) and
                extract_numeric_part(f) in validated_images]

    if poses.shape[-1] != len(validated_images):
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

    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')