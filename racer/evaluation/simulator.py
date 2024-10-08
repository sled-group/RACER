from numpy import ndarray
from PIL import Image
import numpy as np

from rlbench.backend.const import *
from rlbench.backend.utils import task_file_to_task_class

from racer.evaluation.utils import ROLLOUT_IMAGE_SIZE, CustomRLRenchEnv2
from racer.peract.helpers import utils
from racer.utils.racer_utils import CAMERAS
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from racer.rvt.utils.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
)
from yarr.agents.agent import ActResult
from yarr.utils.transition import Transition
from racer.evaluation.utils import STAND_POSE_ACTION



class RLBenchSim:
    def __init__(
        self,  
        task_name: str,
        dataset_root: str,
        episode_length: int=30,
        record_every_n: int = 5, # -1 means no recording
        resolution: int=ROLLOUT_IMAGE_SIZE,
        record_queue=None,
        never_terminal=False,
        unseen_task=False,
    ):
        self.task_name = task_name
        self.dataset_root = dataset_root
        self.episode_length = episode_length
        self.record_every_n = record_every_n
        self.record_queue = record_queue
        self.never_terminal = never_terminal
        self.unseen_task = unseen_task
        
        self.setup_env(resolution)

        self.last_action = None

        
    def reset(self, episode_num: int = 0, not_load_image: bool = True) -> dict:
        obs_dict, obs = self.env.reset_to_demo(episode_num, not_load_image)
        return obs_dict, obs
    
    def setup_env(self, resolution):
        camera_resolution = [resolution, resolution]
        obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")

        gripper_mode = Discrete()
        arm_action_mode = EndEffectorPoseViaPlanning()
        action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)
        self.env = CustomRLRenchEnv2(
            record_queue=self.record_queue,
            task_class=task_file_to_task_class(self.task_name),
            observation_config=obs_config,
            action_mode=action_mode,
            dataset_root=self.dataset_root,
            episode_length=self.episode_length,
            headless=True,
            time_in_state=True,
            include_lang_goal_in_obs=True,
            record_every_n=self.record_every_n,
            never_terminal=self.never_terminal,
            unseen_task=self.unseen_task,
        )
        self.env.eval = True
        self.env.launch()
    
    def set_new_task(self, task_name: str):
        self.env.set_new_task(task_name)
        self.task_name = task_name
    
    def set_new_dataset(self, dataset_root: str):
        self.env._rlbench_env._dataset_root = dataset_root
    
    @property
    def task_goal(self):
        return self.env._lang_goal

    def step(self, action: ndarray) -> Transition:
        # action is (9, ) array, 3 for pose, 4 for quaternion, 1 for gripper, 1 for ignore_collision
        wrap_action = ActResult(action=action)            
        transition = self.env.step(wrap_action) # get Transition(obs, reward, terminal, info, summaries)    
        if transition.info['error_status'] == "error": # sometimes RLbench throws strange error
            print(f"Error: action was {action}")
            if self.task_name in ["put_item_in_drawer"]: 
                transition = self.env.step(ActResult(action=STAND_POSE_ACTION))
            if self.task_name in ["open_drawer"] and self.last_action is not None: 
                # hot fix to avoid strange invalid error
                action[0] = (self.last_action[0] + action[0])/2
                action[2] = (self.last_action[2] + action[2])/2
                transition = self.env.step(ActResult(action=action))
        if isinstance(transition, tuple):
            transition = transition[0]
        self.transition = transition
        self.last_action = action
        return transition
    
    def is_success(self) -> bool:
        # always called when simulation ends
        score = self.transition.reward
        return True if score == 100.0 else False
        
    def close(self):
        self.env.shutdown()
    
    
    def get_video_frames(self, res=128, return_pil=True):
        ret = []
        for fra in self.env._recorded_images:
            if fra.shape[0] == 3:
                fra = fra.transpose(1, 2, 0)
            fra = Image.fromarray(fra).resize((res, res))
            if return_pil:
                ret.append(fra)
            else:
                ret.append(np.array(fra))
        self.env._recorded_images.clear()
        return ret
    