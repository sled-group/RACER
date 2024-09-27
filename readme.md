## Getting Started

### Install RACER
- Tested (Recommended) Versions: Python 3.9. We used CUDA 11.7. 

- **Step 1:**
We recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) and creating a virtual environment.
```
conda create --name racer python=3.9
conda activate racer
```

- **Step 2:** Install PyTorch. Make sure the PyTorch version is compatible with the CUDA version. More instructions to install PyTorch can be found [here](https://pytorch.org/).
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- **Step 3:** Install PyTorch3D. For more instructions visit [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
```
conda install iopath==0.1.9 -c iopath
conda install pytorch3d==0.7.5 -c pytorch3d
```

- **Step 4:** Install CoppeliaSim. PyRep requires version **4.1** of CoppeliaSim. Download and unzip [CoppeliaSim_Edu_V4_1_0](https://coppeliarobotics.com/previousVersions) according to your Ubuntu version. If your system is 22.04, please see the [Trouble shooting](#trouble-shooting).
Once you have downloaded CoppeliaSim, add the following to your *~/.bashrc* file. (__NOTE__: the 'EDIT ME' in the first line)

```
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISLAY=:1.0
```
Remember to source your .bashrc (`source ~/.bashrc`) or  .zshrc (`source ~/.zshrc`) after this.

- **Step 5:** Clone the repository with the submodules using the following command.

```
git clone --recurse-submodules git@github.com:NVlabs/RVT.git && cd RVT && git submodule update --init
```

Now, locally install RVT and other libraries using the following command. Make sure you are in folder RVT.
```
source setup_env.bash
pip install -e . 
pip install -e libs/PyRep 
pip install -e libs/RLbench 
pip install -e libs/YARR 
pip install -e libs/peract_colab
``` 





### Prepare Data 
- **Step 1:**
Download [RLbench](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ) dataset provided by [PerAct](https://github.com/peract/peract#download). Please download and place them under `RACER/racer/data/rlbench/xxx` where `xxx` is either `train`, `test`, or `val`. You can only download `test` for evaluation if you want to save some space.

- **Step 2:**
Download our augmentation training data. put it under `RACER/racer/data/augment_data`.   
Note that we we use the same dataloader as PerAct, which is based on [YARR](https://github.com/stepjam/YARR). YARR creates a replay buffer on the fly which can increase the startup time.  We provide an option to directly load the replay buffer from the disk. You can download our replay buffer data [racer_replay_augment]() here, and put place it under  `RACER/racer/replay_buffers`. This is useful only if you want to train RACER by yourself and not needed if you want to evaluate the pre-trained model.



## Train & Eval

### Training RACER
##### Default command
To train RVT on all RLBench tasks, use the following command (from folder `RACER/racer`):
```
python train.py --exp_cfg_path configs/all.yaml --device 0,1,2,3,4,5,6,7
```




### Use LLAVA to talk
```
pip install openai==1.27.0
```


### Eval

```
cd RACER
source scripts/setup_env.bash
vncserver :9 # (Optional) open a vncserver if it's not running
./scripts/eval_rvt.sh # for RVT eval
./scripts/eval_racer.sh  # for RACER eval
```


### Model ckpt
Download the official [RVT](https://drive.google.com/drive/folders/1lf1znYM5I-_WSooR4VeJjzvydINWPj6B) model and place it into `racer/runs/rvt_ckpt`.  
Download our racer [model]() and place it into `racer/runs/racer_ckpt`.


## Trouble shooting
### CoppeliaSim issue


1. "Cannot load library /home/daiyp/manipulation/RACER/coppeliasim/CoppeliaSim_Edu_V4_5_1_rev4_Ubuntu22_04/libsimExtIM.so: libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by libopencv_core.so.406))"
 CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/libsimExtBlueZero.so: (libicui18n.so.66: cannot open shared object file: No such file or directory)"






# Acknowledgement

This code is adapted and modified upon the released  [RVT](https://github.com/NVlabs/RVT/tree/0b170d7f1e27a13299a5a06134eeb9f53d494e54) code.

We really appreciate their open-sourcing such high-quality code, which is very helpful to our research!
