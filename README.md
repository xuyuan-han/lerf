# Extending LERF: Language Embedded Radiance Fields
This is our modified version of [LERF](https://lerf.io) for the semester project in Machine Learning for 3D Geometry (IN2392) at TUM.\
Our version of LeRF contains the following modifications:
- A depth supervision based on SFM point cloud points similar to DS-NeRF aiming at improving novel view synthesis quality
- An upgrade of DINO to DINOv2 for improved language field quality and stronger regularization
- An additional regularization based on SAM features to further improve language field quality

<div align='center'>
<img src="https://www.lerf.io/data/nerf_render.svg" height="230px">
</div>

#### TODOs
- [ ] Filter relevancy maps based on SAM mask predictions to acquire more precise results

# Installation
LERF follows the integration guidelines described [here](https://docs.nerf.studio/en/latest/developer_guides/new_methods.html) for custom methods within Nerfstudio. 
### 0. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html) up to and including "tinycudann" to install dependencies and create an environment
### 1. Clone this repo
`git clone https://github.com/xuyuan-han/lerf`
### 2. Install this repo as a python package
Navigate to this folder and run `python -m pip install -e .`

### 3. Run `ns-install-cli`

### 4. Download SAM weights from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and move them to `lerf\segment_anything`

### Checking the install
Run `ns-train -h`: you should see a list of "subcommands" with lerf, lerf-big, and lerf-lite included among them.

# Using LERF
Since we were constrained by the available GPU memory, we provide multiple configurations, one per modification: `lerf-depth`, `lerf-sam`, `lerf-dino`. All of our configurations are based on `lerf-lite`. Additionally, for `lerf-depth`, we assume that preprocessed COLMAP data is already available within the dataset, which is the case for all ScanNet++ scenes and basic Nerfstudio scenes, but not for the scenes provided by LeRF.

- Launch training with `ns-train <configuration> --data <data_folder>`. This specifies a data folder to use. For more details, see [Nerfstudio documentation](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html). 
- Connect to the viewer by forwarding the viewer port (we use VSCode to do this), and click the link to `viewer.nerf.studio` provided in the output of the train script
- Within the viewer, you can type text into the textbox, then select the `relevancy_0` output type to visualize relevancy maps.

## Relevancy Map Normalization
By default, the viewer shows **raw** relevancy scaled with the turbo colormap. As values lower than 0.5 correspond to irrelevant regions, **we recommend setting the `range` parameter to (-1.0, 1.0)**. To match the visualization from the paper, check the `Normalize` tick-box, which stretches the values to use the full colormap.

The images below show the rgb, raw, centered, and normalized output views for the query "Lily".


<div align='center'>
<img src="readme_images/lily_rgb.jpg" width="150px">
<img src="readme_images/lily_raw.jpg" width="150px">
<img src="readme_images/lily_centered.jpg" width="150px">
<img src="readme_images/lily_normalized.jpg" width="150px">
</div>

## Computing Localization Accuracy
We provide our own script for computing the localization accuracy like it is described in the paper. This is, however, only possible on the original LeRF scenes `bouquet`, `figurines`, `ramen`, `teatime`, and `waldo_kitchen` since we adapted their already available evaluation data.

The localization accuracy can be computed by running `python eval_location.py --load-config <config> --camera-path-filename eval\<scene_name>\keyframes.json --query_path_filename eval\<scene_name>\queries.json`, where `config` is a path to the config file of the run on the respective scene that is to be evaluated.


## References
Our work is based on the LeRF paper:
<pre id="codecell0">@inproceedings{lerf2023,
&nbsp;author = {Kerr, Justin and Kim, Chung Min and Goldberg, Ken and Kanazawa, Angjoo and Tancik, Matthew},
&nbsp;title = {LERF: Language Embedded Radiance Fields},
&nbsp;booktitle = {International Conference on Computer Vision (ICCV)},
&nbsp;year = {2023},
} </pre>
Our modification build upon previous work from DS-NeRF, DINO, DINOv2, and SAM:
<pre id="codecell0">@inproceedings{kangle2021dsnerf,
    author    = {Deng, Kangle and Liu, Andrew and Zhu, Jun-Yan and Ramanan, Deva},
    title     = {Depth-supervised {NeRF}: Fewer Views and Faster Training for Free},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
} </pre>
<pre id="codecell0">@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021},
} </pre>
<pre id="codecell0">@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
} </pre>
<pre id="codecell0">@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
} </pre>
