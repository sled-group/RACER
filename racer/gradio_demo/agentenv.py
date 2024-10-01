from copy import deepcopy
from dataclasses import dataclass
import json
from multiprocessing import Queue, sharedctypes
import os
import re
from typing import Dict
from PIL import Image

from racer.evaluation.simulator import RLBenchSim
from racer.evaluation.utils import START_ACTION
from racer.rvt.utils.peract_utils import IMAGE_RGB, CAMERA_FRONT, CAMERA_LS, CAMERA_RS, CAMERA_WRIST
from racer.utils.racer_utils import RLBENCH_TASKS
from .utils import *

from racer.evaluation.policy_agent import ModelRVTAgent
from racer.evaluation.rollout import Evaluator
from racer.evaluation.utils import get_robot_delta_state

@dataclass
class RobotState:
    robot_delta_state: str
    action_desc: str
    last_robot_delta_state: str = ""
    is_robot_state_changed: bool = False
    status: str = "ongoing" # success, failure, ongoing, error
    model_name: str = ""
    task_goal: str = ""
    last_insturction: str = ""
    instruction_role: str = ""
    input_text_str: str = ""

    def to_dict(self):
        return {
            "robot_delta_state": self.robot_delta_state,
            "last_robot_delta_state": self.last_robot_delta_state,
            "is_robot_state_changed": self.is_robot_state_changed,
            "action_desc": self.action_desc,
            "status": self.status,
            "input_text_str": self.input_text_str,
            "model_name": self.model_name,
            "last_insturction": self.last_insturction,
            "instruction_role": self.instruction_role,
            "task_goal": self.task_goal
        }
    

class AgentEnv:
    def __init__(
        self,
        vid_queue: Queue,
        front_rgb_queue: Queue, 
        left_shoulder_rgb_queue: Queue,
        right_shoulder_rgb_queue: Queue,
        wrist_rgb_queue: Queue, 
        inst_queue: Queue,  
        goal_queue: Queue,  
        state_queue: Queue,
        ep_value: sharedctypes.Synchronized,    
        ds_value: sharedctypes.Synchronized,    
        md_value: sharedctypes.Synchronized,   
        tn_value: sharedctypes.Synchronized, 
        step_value: sharedctypes.Synchronized,
        ctrl_value: sharedctypes.Synchronized,
        reset_value: sharedctypes.Synchronized,
        llava_talk_value: sharedctypes.Synchronized,
        model_dict: Dict[str, str],
        episode_length: int = 50, # TODO change it back to 30 later
        never_terminal: bool = False,
        unseen_task: bool = False,
        lm_address="http://localhost:8000",
        rlbench_dataroot="racer/data/rlbench"
    ):
        
        self.vid_queue = vid_queue
        self.front_rgb_queue = front_rgb_queue
        self.left_shoulder_rgb_queue = left_shoulder_rgb_queue
        self.right_shoulder_rgb_queue = right_shoulder_rgb_queue
        self.wrist_rgb_queue = wrist_rgb_queue
        self.inst_queue = inst_queue
        self.goal_queue = goal_queue
        self.state_queue = state_queue
        self.reset_value = reset_value
        self.ep_value = ep_value
        self.ds_value = ds_value
        self.md_value = md_value
        self.tn_value = tn_value
        self.step_value = step_value
        self.ctrl_value = ctrl_value
        self.llava_talk_value = llava_talk_value

        self.episode_length = episode_length
        self.never_terminal = never_terminal
        self.unseen_task = unseen_task
        
        self.rlbench_dataroot = rlbench_dataroot
        self.lm_addr = lm_address

        self.reset_env()
        self.step_count = 0

        self.reset_models(model_dict)
        self.eval_dataset = "test"

        self.last_model_id = self.md_value.value
        self.last_robot_delta_state = ""
    
    def reset_models(self, model_path_dict):
        # build two models: a baseline and our model
        # assert len(model_paths) == 2, "Two models are required"
        self.models: Dict[str, ModelRVTAgent] = {}
        for model_name, (model_path, device_id) in model_path_dict.items():
            if model_name == "RVT":
                self.models[model_name] = ModelRVTAgent(model_path=model_path, device=device_id,  use_full_langlen=True, lm_addr=self.lm_addr)
            else:
                self.models[model_name] = ModelRVTAgent(model_path=model_path, device=device_id, use_full_langlen=False, lm_addr=self.lm_addr)
            self.models[model_name].reset()


    def setup_env(self):
        logger.info(f"start to setup env")
        self.eval_dataset = DATASET[self.ds_value.value]
        self.task_name = RLBENCH_TASKS[self.tn_value.value]
        self.env = RLBenchSim(
            task_name=self.task_name,
            dataset_root=os.path.join(self.rlbench_dataroot, self.eval_dataset),
            episode_length=self.episode_length,
            record_queue=self.vid_queue,
            never_terminal=self.never_terminal,
            unseen_task=self.unseen_task
        )

    
    def reset_env(self):
        if not hasattr(self, "env"):
            self.setup_env()
        
        if self.eval_dataset != DATASET[self.ds_value.value]:
            self.eval_dataset = DATASET[self.ds_value.value]
            self.env.set_new_dataset(os.path.join(self.rlbench_dataroot, self.eval_dataset))
        
        self.env.set_new_task(RLBENCH_TASKS[self.tn_value.value])
        self.task_name = RLBENCH_TASKS[self.tn_value.value]

        self.obs_dict, _obs = self.env.reset(episode_num=self.ep_value.value)
        self.prev_obs = None
        self.cur_obs = _obs
        self.last_action = START_ACTION

        # # get video frame
        # frames = self.env.get_video_frames(res=128, return_pil=False)
        # print(len(frames), "frames")
        # frame_array = np.array(frames)
        # print(frame_array.shape)
        # self.vid_queue.put(frame_array)

        # get four camera images   
        self.front_rgb_queue.put(self.wrap_img(self.obs_dict[f"{CAMERA_FRONT}_{IMAGE_RGB}"], res=256))
        self.left_shoulder_rgb_queue.put(self.wrap_img(self.obs_dict[f"{CAMERA_LS}_{IMAGE_RGB}"], res=256))
        self.right_shoulder_rgb_queue.put(self.wrap_img(self.obs_dict[f"{CAMERA_RS}_{IMAGE_RGB}"], res=256))
        self.wrist_rgb_queue.put(self.wrap_img(self.obs_dict[f"{CAMERA_WRIST}_{IMAGE_RGB}"], res=256))
        
        self.task_goal = self.env.task_goal
        self.goal_queue.put(self.task_goal)

        logger.info(f"\n\n=== Set up Env on {DATASET[self.ds_value.value]} dataset, episode {self.ep_value.value}, task \"{RLBENCH_TASKS[self.tn_value.value]}\", task goal \"{self.task_goal}\", use model {MODELS[self.md_value.value]} ===\n")

        self.reset_value.value = 0
        self.step_count = 0

    
    @staticmethod
    def wrap_input_lang_str(task_goal, instruction, old_version=False):
        if old_version:  # old version of RVT input lang str e.g., "close the red jar"
            res = task_goal
            if instruction is not None and instruction.strip() != "":
                res += ". " + instruction                
            return res
        if (
            instruction is None or instruction.strip() == ""
        ):  # high-level goal only, e.g., "task goal: close the red jar."
            return f"Task goal: {task_goal}."
        else:  # blend with low-level instruction
            instruction = instruction.strip()
            if instruction.endswith("."):
                instruction = instruction[:-1]
            return f"Task goal: {task_goal}.\nCurrent instruction: {instruction.lower()}."

    def _action_desc(self, action):
        res = "step " + str(self.step_count+1) + ", "
        res += " position:" + str(action[:3]) + ","
        res += " rotation: " + str(action[3:7]) + ","
        res += " gripper open, " if action[-2] else " gripper close, "
        res += " ignore collision." if action[-1] else " avoid collision."
        return res
        

    def step(self, task_goal, instruction_tuple):
        # get image from env
        # make action prediction
        # step env
        role, instruction = instruction_tuple
        logger.info(f"=== Model {MODELS[self.md_value.value]} Predicting action for task goal \"{task_goal}\" and instruction {json.dumps(instruction)} from {role}===")

        if self.last_model_id != self.md_value.value: 
            self.last_model_id = self.md_value.value
            self.obs_dict["ignore_collisions"] = [self.last_action[-1]]
            self.obs_dict["low_dim_state"][0] = self.last_action[-2]
            self.obs_dict["gripper_pose"] = self.last_action[:7]
        
        model = self.models[MODELS[self.last_model_id]]
        input_lang_str = self.wrap_input_lang_str(task_goal, instruction, old_version=model.use_full_langlen) # only origianl rvt uses full langlen
        logger.info(f"wrap_input_lang_str: {json.dumps(input_lang_str)}")
        
        if self.env.unseen_task:
            # set up some initail pose for unseen tasks to migigate invalid action error
            if self.env.task_name == "close_drawer" and self.step_count ==0:
                action = [0.257, 0.2376, 1.0137, 0.7064, 0.0308, 0.0308, 0.7064, 1, 0]
            else:
                action = model.act(self.obs_dict, input_lang_str)
        else:            
            action = model.act(self.obs_dict, input_lang_str)
        action = Evaluator.action_check(action)
        action = Evaluator.postprocess(self.task_name, action, self.last_action, self.step_count)

        if instruction is not None:
            instruction = instruction.strip().lower()
            if re.search(r"^open (the )?gripper$", instruction):
                action = deepcopy(self.last_action)
                action[-2] = 1
            elif re.search(r"^close (the )?gripper$", instruction):
                action = deepcopy(self.last_action)
                action[-2] = 0
            elif instruction in ["move left a little bit", "move a little bit left"]:
                action = deepcopy(self.last_action)
                action[1] -= 0.01
                action[-1] = 1
            elif instruction in ["move left"]:
                action = deepcopy(self.last_action)
                action[1] -= 0.05
                action[-1] = 1
            elif instruction in ["move right a little bit", "move a little bit right"]:
                action = deepcopy(self.last_action)
                action[1] += 0.01
                action[-1] = 1
            elif instruction in ["move right"]:
                action = deepcopy(self.last_action)
                action[1] += 0.05
                action[-1] = 1
            elif instruction in ["move forward a little bit", "move a little bit forward"]:
                action = deepcopy(self.last_action)
                action[0] -= 0.01
                action[-1] = 1
            elif instruction in ["move forward"]:
                action = deepcopy(self.last_action)
                action[0] -= 0.05
                action[-1] = 1
            elif instruction in ["move backward a little bit", "move a little bit backward"]:
                action = deepcopy(self.last_action)
                action[0] += 0.01
                action[-1] = 1
            elif instruction in ["move backward"]:
                action = deepcopy(self.last_action)
                action[0] += 0.05
                action[-1] = 1
            elif instruction in ["move up a little bit", "move a little bit up"]:
                action = deepcopy(self.last_action)
                action[2] += 0.01
                action[-1] = 1
            elif instruction in ["move up"]:
                action = deepcopy(self.last_action)
                action[2] += 0.05
                action[-1] = 1
            elif instruction in ["move down a little bit", "move a little bit down"]:
                action = deepcopy(self.last_action)
                action[2] -= 0.01
                action[-1] = 1
            elif instruction in ["move down"]:
                action = deepcopy(self.last_action)
                action[2] -= 0.05
                action[-1] = 1
            elif instruction == "home":
                action = Evaluator.action_check(START_ACTION)
                action[-1] = 1
        
        
        logger.info(f"Action: {action}")

        transition = self.env.step(action)
        if self.obs_dict["low_dim_state"][0] == action[-2]:
            # hot fix for wrong gripper state occasionally returned by RLbench
            transition.observation["low_dim_state"][0] = action[-2]
        if "cupboard" in task_goal:
            # hot fix for put_groceries_in_cupboard task about wrong gripper state returned by RLbench
            transition.observation["low_dim_state"][0] = action[-2]
        self.obs_dict = transition.observation
        self.prev_obs = deepcopy(self.cur_obs)
        self.cur_obs = deepcopy(transition.info["obs"])

        # # stream the video to the video queue
        # frames = self.env.get_video_frames(res=128, return_pil=False)
        # print(len(frames), "frames")
        # frame_array = np.array(frames)
        # print(frame_array.shape)
        # self.vid_queue.put(frame_array)

        self.front_rgb_queue.put(self.wrap_img(self.obs_dict[f"{CAMERA_FRONT}_{IMAGE_RGB}"], res=256))
        self.left_shoulder_rgb_queue.put(self.wrap_img(self.obs_dict[f"{CAMERA_LS}_{IMAGE_RGB}"], res=256))
        self.right_shoulder_rgb_queue.put(self.wrap_img(self.obs_dict[f"{CAMERA_RS}_{IMAGE_RGB}"], res=256))
        self.wrist_rgb_queue.put(self.wrap_img(self.obs_dict[f"{CAMERA_WRIST}_{IMAGE_RGB}"], res=256))

        if self.env.is_success():
            status = "success"
        elif self.step_count == self.episode_length:
            status = "failure"
        else:
            if transition.info["error_status"] == "error":
                status = "InvalidActionError"
            else:
                status = "ongoing"
        
        logger.info(f"status: {status}")
        if status  == "InvalidActionError":
            robot_delta_state = "error"
        else:
            robot_delta_state, is_robot_state_changed = get_robot_delta_state(self.prev_obs, self.cur_obs)
            logger.info(f"robot_delta_state: {robot_delta_state}")
        robot_state = RobotState(
            robot_delta_state=robot_delta_state, 
            last_robot_delta_state=self.last_robot_delta_state,
            is_robot_state_changed=is_robot_state_changed,
            action_desc=self._action_desc(action), 
            status=status, 
            input_text_str=input_lang_str, 
            last_insturction=instruction,
            instruction_role=role,
            model_name=MODELS[self.md_value.value],
            task_goal=task_goal
            )        
        self.state_queue.put(robot_state)

        if is_robot_state_changed:  
            # only update when the gripper really changes state
            self.last_robot_delta_state = robot_delta_state
        
        logger.info(f"Agent action: {robot_state.action_desc}")
        self.last_action = action        
        
        self.step_value.value = 0
        self.step_count += 1

    
    def run(self):
        while True:
            if self.reset_value.value:
                self.reset_env()
            
            if self.step_value.value:
                inst = ("none", None)
                while not self.inst_queue.empty():
                    inst = self.inst_queue.get()
                if self.llava_talk_value.value == 0 and inst[0] == "llava": # llava talk is forbidden
                    inst = ("none", None)
                if not self.goal_queue.empty():
                    self.task_goal = self.goal_queue.get() # maybe a new goal
                
                self.step(self.task_goal, inst)
            
            if self.ctrl_value.value:
                if self.ctrl_value.value == 1: # gripper change
                    if self.last_action[-2] == 1:
                        self.step(self.task_goal, ("control", "close the gripper"))
                    else:
                        self.step(self.task_goal, ("control", "open the gripper"))
                elif self.ctrl_value.value == 2: #home
                    self.step(self.task_goal, ("control", "home"))
                elif self.ctrl_value.value == 3: # move left
                    self.step(self.task_goal, ("control", "move left a little bit"))
                elif self.ctrl_value.value == 4: # move right
                    self.step(self.task_goal, ("control", "move right a little bit"))
                elif self.ctrl_value.value == 5: # move forward
                    self.step(self.task_goal, ("control", "move forward a little bit"))
                elif self.ctrl_value.value == 6: # move backward
                    self.step(self.task_goal, ("control", "move backward a little bit"))
                elif self.ctrl_value.value == 7: # move up
                    self.step(self.task_goal, ("control", "move up a little bit"))
                elif self.ctrl_value.value == 8: # move down
                    self.step(self.task_goal, ("control", "move down a little bit"))
                self.ctrl_value.value = 0
    
            
    def wrap_img(self, rgb, res=512):
        if rgb.shape[0] == 3:  # (3, 512, 512) -> (512, 512, 3)
            rgb = rgb.transpose(1, 2, 0)
        # resize to resolution
        rgb = Image.fromarray(rgb).resize((res, res))
        return rgb
    

    def get_robot_delta_state(self):
        return "next"

