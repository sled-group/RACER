## Getting Started

### Install RACER
- Tested (Recommended) Versions: Ubuntu 20.04, Python 3.8 and CUDA 11.6. 

- **Step 1:**
We recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) and creating a virtual environment.
```
conda create --name racer python=3.8.18
conda activate racer
```

- **Step 2:** Install PyTorch. For more information visit [here](https://pytorch.org/).
```
conda install install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```

- **Step 3:** Install PyTorch3D. For more information visit [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
```
conda install iopath==0.1.9 -c iopath
conda install nvidiacub==1.10.0 -c bottler 
conda install pytorch3d==0.7.5 -c pytorch3d
```

- **Step 4:** Install CoppeliaSim. Download and unzip [CoppeliaSim_Edu_V4_1_0](https://coppeliarobotics.com/previousVersions) according to your Ubuntu version. If your system is 22.04, please see the [Trouble shooting](#trouble-shooting).
Once you have downloaded CoppeliaSim, add the following to your *~/.bashrc* file and source it.

```
export COPPELIASIM_ROOT=<PATH/TO/COPPELIASIM/DIR>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:<DISPLAY_ID>
```
And remember to start the vncserver use `vncserver :<DISPLAY_ID>`. DISPLAY_ID is a positive int number.

- **Step 5:** Clone the submodules and install all packages.

```
cd <PATH_TO_RACER>
git submodule update --init
pip install -e . 
pip install -e libs/PyRep 
pip install -e libs/RLbench 
pip install -e libs/YARR 
pip install -e libs/peract_colab
``` 





### Prepare Data 
- **Step 1:**
Download [RLbench](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ) dataset provided by [PerAct](https://github.com/peract/peract#download). Please download and place them under `RACER/racer/data/rlbench/xxx` where `xxx` is either `train`, `test`, or `val`.   
You can only download `test` for evaluation if you want to save some space.

- **Step 2:**
Download our augmentation training data. put it under `RACER/racer/data/augment_data`.   
The data will be processed into replay buffers by [YARR](https://github.com/stepjam/YARR).To save data processing time, you can also download our generated replay buffer data [racer_replay_augment]() here, and put place it under  `RACER/racer/replay_buffers`.   
This is useful only if you want to train RACER by yourself and not needed if you just want to evaluate the pre-trained model.



## Train & Eval

### Training RACER
##### Default command
To train RVT on all RLBench tasks, use the following command (from folder `RACER/racer`):
```
python train.py --exp_cfg_path configs/all.yaml --device 0,1,2,3,4,5,6,7
```




### Use LLAVA to talk

### Eval
First, set up a language model serive following [here]()
lang_model_address

Evaluate RVT
```
./scripts/eval_rvt.sh # for RVT eval
```

Evaluate RACER, you first need to set up llava service and language encoder service following [here](), and get the service host address.
```
./scripts/eval_racer.sh  # for RACER eval
```


### Model ckpt
Download the official [RVT](https://drive.google.com/drive/folders/1lf1znYM5I-_WSooR4VeJjzvydINWPj6B) model and place it into `racer/runs/rvt_ckpt`.  
Download our racer [model]() and place it into `racer/runs/racer_ckpt`.


## Trouble shooting
### CoppeliaSim issue


1. "Cannot load library /home/daiyp/manipulation/RACER/coppeliasim/CoppeliaSim_Edu_V4_5_1_rev4_Ubuntu22_04/libsimExtIM.so: libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by libopencv_core.so.406))"
 CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/libsimExtBlueZero.so: (libicui18n.so.66: cannot open shared object file: No such file or directory)"


libGL error: failed to load driver: swrast



# Acknowledgement

This code is adapted and modified upon the released  [RVT](https://github.com/NVlabs/RVT/tree/0b170d7f1e27a13299a5a06134eeb9f53d494e54) code.

We really appreciate their open-sourcing such high-quality code, which is very helpful to our research!
