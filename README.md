# The Project of Enhanced Learning Interaction-aware Motion Prediction Model with LSTM for Autonomous Driving - Ziyang Zheng

This repository contains the baseline code for the following paper:

**Learning Interaction-aware Motion Prediction Model for Decision-making in Autonomous Driving**
<br> [Zhiyu Huang](https://mczhi.github.io/), [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Jingda Wu](https://wujingda.github.io/), [Wenhui Huang](https://scholar.google.co.kr/citations?user=Hpatee0AAAAJ&hl=en), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[arXiv]](https://arxiv.org/abs/2302.03939)**

If you are looking for or interested in their winning solutions (Team AID) at [NeurIPS 2022 Driving SMARTS Competition](https://smarts-project.github.io/archive/2022_nips_driving_smarts/), please go to [track 1 solution](https://github.com/MCZhi/Predictive-Decision/tree/smarts-comp-track1) and [track 2 solution](https://github.com/MCZhi/Predictive-Decision/tree/smarts-comp-track2).

## Baseline Framework
The interaction-aware predictor is proposed in 2022 to forecast the neighboring agents' future trajectories around the ego vehicle conditioned on the ego vehicle's potential plans. A sampling-based planner will do collision checking and select the optimal trajectory considering the distance to the goal, ride comfort, and safety.

## How to use
### Create a new Conda environment
```bash
conda create -n smarts python=3.8
```

### Re-set the Version of tools
```bash
pip install --upgrade pip wheel==0.38.0
pip install --upgrade pip setuptools==65.5.0
```
#### (Reminder) The version of these tools is set for SMARTS of 2.01, if it is updated in the future, it may need other adjustments

### Install the SMARTS simulator (2.01)
```bash
conda activate smarts
```

Install the [SMARTS simulator](https://smarts.readthedocs.io/en/latest/setup.html). 
```bash
# Download SMARTS
git clone https://github.com/huawei-noah/SMARTS.git
cd <path/to/SMARTS>
git checkout comp-1

# Install the system requirements.
bash utils/setup/install_deps.sh

# Install smarts with comp-1 branch.
pip install "smarts[camera-obs] @ git+https://github.com/huawei-noah/SMARTS.git@comp-1"
pip install 'smarts[camera-obs,sumo,examples,envision]'
```
A lot of bugs still exist in SMARTS of 2.01, some parts of commands list above maybe are unneccessary but make sure the project can work 

### Install Pytorch
```bash
conda install pytorch==1.12.0 -c pytorch
```
make sure your other tools like CUDA work normally
### Training
Run `train.py`. Leave other arguments vacant to use the default setting.
```bash
python train.py --use_exploration --use_interaction
```

### Testing
Run `test.py`. You need specify the path to the trained predictor `--model_path`. You can aslo set `--envision_gui` to visualize the performance of the framework in envision or set `--sumo_gui` to visualize in sumo. 
Check arguments in train.py and test.py to look for more settings.
```bash
python test.py --model_path /training_log/Exp/model.pth
```
To visualize in Envision (some bugs exist in showing the road map), you need to manually start the envision server and then go to `http://localhost:8081/`.
```bash
scl envision start &
```
This coomand enables the use of "--envison_gui"
## Citation
the baseline model used in this project
```
@article{huang2023learning,
  title={Learning Interaction-aware Motion Prediction Model for Decision-making in Autonomous Driving},
  author={Huang, Zhiyu and Liu, Haochen and Wu, Jingda and Huang, Wenhui and Lv, Chen},
  journal={arXiv preprint arXiv:2302.03939},
  year={2023}
}
```
