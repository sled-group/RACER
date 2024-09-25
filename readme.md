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
pip install -e libs/RLBench 
pip install -e libs/YARR 
pip install -e libs/peract_colab
``` 











## Trouble shooting
### CoppeliaSim issue




# Acknowledgement

This code is adapted and modified upon the released  [RVT](https://github.com/NVlabs/RVT/tree/0b170d7f1e27a13299a5a06134eeb9f53d494e54) code.

We appreciate their open-sourcing such high-quality code, which is very helpful to our research.
