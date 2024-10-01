from copy import deepcopy
import os
from typing import Dict
import numpy as np
import torch
from yarr.agents.agent import ActResult

from racer.evaluation.utils import ROLLOUT_IMAGE_SIZE
from racer.utils.racer_utils import CAMERAS, load_agent, SCENE_BOUNDS
import racer.rvt.models.rvt_agent as rvt_agent
from racer.rvt.mvt.mvt_v2 import MVT


class Agent:
    r"""Abstract class for defining agents which act inside :ref:`core.env.Env`.

    This abstract class standardizes agents to allow seamless benchmarking.
    """
    def reset(self) -> None:
        r"""Called before starting a new episode in environment."""
        raise NotImplementedError

    def act(
        self, input_obs,
    ):
        # mainly for model-based agents
        raise NotImplementedError

class ModelRVTAgent(Agent):
    r"""Agent that acts according to a model trained with RVT."""
    def __init__(
        self, 
        model_path: str, 
        device: int,
        use_full_langlen: bool = False,
        lm_addr: str = None,

    ):
        self.model_path = model_path
        self.lm_addr = lm_addr
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

        self.use_full_langlen = use_full_langlen
        
        import racer.rvt.config as default_exp_cfg
        import racer.rvt.mvt.config as default_mvt_cfg
        model_folder = os.path.dirname(model_path)
        
        # load exp_cfg
        exp_cfg = default_exp_cfg.get_cfg_defaults()
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))
        # WARNING NOTE: a temporary hack to use place_with_mean in evaluation
        exp_cfg.rvt.place_with_mean = True
        exp_cfg.freeze()
        
        # load mvt_cfg
        mvt_cfg = default_mvt_cfg.get_cfg_defaults()
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))
        mvt_cfg.freeze()

        self.proprio_dim = mvt_cfg.proprio_dim
        self.image_size = exp_cfg.image_size
        
        print(f"build model from {model_path}")
        
        rvt = MVT(
            lang_model_name=exp_cfg.lang_model_name,
            renderer_device=self.device,
            **mvt_cfg,
        )
        
        self.agent = rvt_agent.RVTAgent(
            network=rvt.to(self.device),
            image_resolution=[self.image_size, self.image_size],
            add_lang=mvt_cfg.add_lang,
            scene_bounds=SCENE_BOUNDS,
            cameras=CAMERAS,
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
        self.agent._device = self.device

    def reset(self):
        self.agent.build(training=False, device=self.device)
        load_agent(self.model_path, self.agent)
        self.agent.eval()
        self.agent.load_lang_model(lm_addr=self.lm_addr)
        self.agent.reset()
        print("Agent Reset. Model loaded.")
    
    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype
    
    def _wrap_obs(self, obs: dict) -> Dict[str, torch.Tensor]:
        obs_history = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                obs_history[k] = [np.array(v, dtype=self._get_type(v))]
            else:
                obs_history[k] = [v]
        prepped_data = {}
        for k, v in obs_history.items():
            if isinstance(v[0], np.ndarray):
                prepped_data[k] = torch.tensor(np.array([v]), device=self.device)
            else:
                prepped_data[k] = v
        return prepped_data

    def act(self, input_obs: dict, input_lang_str: str) -> np.ndarray:
        down_sampled_obs = self.preprocess_obs_dict(input_obs)
        obs_tensor = self._wrap_obs(down_sampled_obs)
        if self.use_full_langlen: assert self.agent.lang_model_name == "clip" 
        # original rvt uses clip with full langlen 77
        act_result: ActResult = self.agent.act(
            step=0, observation=obs_tensor, input_lang_str=input_lang_str,
            use_full_langlen=self.use_full_langlen
        )
        return act_result.action
    

    def preprocess_obs_dict(self, obs_dict: dict):
        # 512 for llava -> image_size for rvt
        assert obs_dict["front_rgb"].shape[1] == ROLLOUT_IMAGE_SIZE
        obs_dict_copy = deepcopy(obs_dict)
        obs_dict_copy["low_dim_state"] = obs_dict_copy["low_dim_state"][:self.proprio_dim]
        factor = ROLLOUT_IMAGE_SIZE // self.image_size
        if factor == 1:
            return obs_dict_copy
        for cam in CAMERAS:
            intrinsic_key = f"{cam}_camera_intrinsics"
            rgb_key = f"{cam}_rgb"
            point_cloud_key = f"{cam}_point_cloud"

            # downsample rgb
            rgb = obs_dict[rgb_key]
            obs_dict_copy[rgb_key] = rgb[:, 1::factor, 1::factor]

            # downsample point cloud
            point_cloud = obs_dict[point_cloud_key]
            obs_dict_copy[point_cloud_key] = point_cloud[:, 1::factor, 1::factor]

            # change intrinsic matrix
            intrinsics = obs_dict[intrinsic_key]
            intrinsics[:2, :] /= factor
            obs_dict_copy[intrinsic_key] = intrinsics
        return obs_dict_copy            

        