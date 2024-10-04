# Code for RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning

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
Download our language-guided failure-recovery [augmentation training data](https://huggingface.co/datasets/Yinpei/augmented_rlbench). put it under `RACER/racer/data/augmented_rlbench/xxx`, where `xxx` is either `train` or `val`.
For more information about our automatic data generation pipeline, please refer [this](https://github.com/rich-language-failure-recovery/RACER-DataGen).


### Data Processing
After set up the language encoder service following [this](https://github.com/rich-language-failure-recovery/Open-LLaVA-NeXT/tree/racer_llava?tab=readme-ov-file#51-set-up-language-encoder-service-ie-clip-and-t5-model-around-20gb-in-total),  you can process the data using 
```
python racer/utils/preprocess_data.py
```
The data will be processed into replay buffers by [YARR](https://github.com/stepjam/YARR).  

To save data processing time, you can also directly download our generated replay buffer data [racer_replay_public](https://huggingface.co/datasets/Yinpei/racer_replay_public) here without processing by yourself, and put place it under  `RACER/racer/replay_buffers`.   
This is useful only if you want to train RACER by yourself and not needed if you just want to evaluate the pre-trained model.




## Training RACER
### Training visuomotor policy
To train RACER on all RLBench tasks, use the following command:
```
./scripts/train_racer.sh
```
You can change the `rich/simple/task` yaml files for `--exp_cfg_path` to train different types of models, and change `exp_id` to name the trained model directory name.

### Training LLaVA model
Please refer to this [page](https://github.com/rich-language-failure-recovery/Open-LLaVA-NeXT/tree/racer_llava)



## Evaluating RACER
### Setup Service
We set a client-server framework for language encoder service and LLaVA model service that RACER/RVT needs, please refer this [page](https://github.com/rich-language-failure-recovery/Open-LLaVA-NeXT/tree/racer_llava?tab=readme-ov-file#5-set-up-online-service) for details.  

After the language encoder service is set up, you can test it with 
```
python racer/utils/lang_enc_utils_v2.py --lm-addr <lm service addr>
```

After the LLaVA service is set up, you can test it with 
```
python racer/evaluation/llava_api/api.py  --vlm-addr <vlm service addr>
```

### Model Checkpoints
Download the official [RVT](https://drive.google.com/drive/folders/1lf1znYM5I-_WSooR4VeJjzvydINWPj6B) model and place it into `racer/runs/rvt_ckpt`.   
Download our RACER visuomotor policy model trained wth [rich instructions]((https://huggingface.co/Yinpei/racer-visuomotor-policy-rich)) and place it under `racer/runs/racer-visuomotor-policy-rich`.
You can also download RACER visuomotor policy model trained with [simple instructions](https://huggingface.co/Yinpei/racer-visuomotor-policy-simple) or [no instructions](https://huggingface.co/Yinpei/racer-visuomotor-policy-taskgoal) for more evaluation. 

### Evaluate RVT
```
./scripts/eval_rvt.sh
```

### Evaluate RACER
```
./scripts/eval_racer.sh 
```
It takes around 5 hours to finish all tasks evaluation

Peak Memory: 19.2GB for langauge model service, 31.7 GB for llava service, 15.5 GB for visuomotor policy model. Using different GPUs or machines to allocate the memory usage is recommended. 

### Gradio Online Demo
First install `pip install gradio==4.36.1`, then run 
```
./scripts/demo.sh 
```
More detailed cases can be found [here](docs/gradio_interface_usage.md)

## Acknowledgement

This code is adapted and modified upon the released  [RVT](https://github.com/NVlabs/RVT/tree/0b170d7f1e27a13299a5a06134eeb9f53d494e54) code.

We really appreciate their open-sourcing such high-quality code, which is very helpful to our research!


## Citation
```
@article{dai2024racer,
  title={RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning},
  author={Dai, Yinpei and Lee, Jayjun and Fazeli, Nima and Chai, Joyce},
  journal={arXiv preprint arXiv:2409.14674},
  year={2024}
}
```