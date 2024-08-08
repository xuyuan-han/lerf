# Extending LERF: Language Embedded Radiance Fields
This is our modified version of [LERF](https://lerf.io) for the semester project in Machine Learning for 3D Geometry (IN2392) at TUM.\
Our version of LeRF contains the following modifications:
- A depth supervision based on SFM point cloud points similar to DS-NeRF aiming at improving Novel View Synthesis quality
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


## Bibtex
Our work is based on the LeRF paper:
<pre id="codecell0">@inproceedings{lerf2023,
&nbsp;author = {Kerr, Justin and Kim, Chung Min and Goldberg, Ken and Kanazawa, Angjoo and Tancik, Matthew},
&nbsp;title = {LERF: Language Embedded Radiance Fields},
&nbsp;booktitle = {International Conference on Computer Vision (ICCV)},
&nbsp;year = {2023},
} </pre>
