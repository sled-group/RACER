import copy
import pickle
import numpy as np
from pyrep.objects import VisionSensor, Dummy
from pyrep.const import RenderMode
from rlbench.backend.conditions import Condition
from racer.peract.helpers.custom_rlbench_env import CustomRLBenchEnv

from yarr.agents.agent import ActResult
from yarr.utils.transition import Transition

from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
from yarr.utils.process_str import change_case
from racer.utils.racer_utils import RLBENCH_TASKS
from rlbench.backend.utils import task_file_to_task_class
from PIL import Image
from dataclasses import dataclass, field
from numpy import ndarray
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation
from rlbench.backend.observation import Observation



START_ACTION = np.array([0.2785,-0.0082,1.4719,-0.0,0.9927,-0.0,0.1209,1.0,0.0])
STAND_POSE_ACTION = np.array(
    [0.29791760444641113, 
     0.08399009704589844, 
     1.3635880947113037, 
     -0.6755902076156602, 
     -0.7372773368101241, 
     4.514521653042017e-17, 
     4.1367969264590704e-17, 
     1.0, 
     0.0]
)
ROLLOUT_IMAGE_SIZE = 512


class NeverStop(Condition):
    def condition_met(self):
        return False, False
    
class CustomRLRenchEnv2(CustomRLBenchEnv):
    def __init__(self, record_queue=None, never_terminal=False, unseen_task=True, *args, **kwargs):
        super(CustomRLRenchEnv2, self).__init__(*args, **kwargs)
        self._task_classes = [task_file_to_task_class(task) for task in RLBENCH_TASKS]
        self._task_name_to_idx = {change_case(tc.__name__):i for i, tc in enumerate(self._task_classes)}
        self.record_queue = record_queue
        self.never_terminal = never_terminal
        self.unseen_task = unseen_task

    
    def reset(self) -> dict:
        super().reset()
        self._record_current_episode = False
        return self._previous_obs_dict
    
    def launch(self):
        super(CustomRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.set_position([-0.7, 0.0, 0.2])
            cam_base.rotate([0, np.pi/12, 0])
            self._record_cam = VisionSensor.create([384, 384])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def _my_callback(self):
        if self._record_current_episode and self.record_queue is not None:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self.record_queue.put(self.normalize_image(cap))
    
    def normalize_image(self, img):
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        return img
    
    def set_new_task(self, task_name: str):
        # Adapted from YARR/yarr/envs/rlbench_env.py MultiTaskRLBenchEnv class
        assert task_name in RLBENCH_TASKS, f"Task {task_name} not found in RLBENCH_TASKS"
        self.task_name = task_name
        self._active_task_id = self._task_name_to_idx[task_name]
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)
        descriptions, _ = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant
    
    def reset_to_demo(self, i, not_load_image=True):
        self._i = 0
        self._task.set_variation(-1)

        if self.unseen_task:
            # variation_count = self._task.variation_count()
            # self._task.set_variation(np.random.randint(variation_count))
            if self.task_name in ["close_drawer", "pick_up_cup", "reach_target"]:
                variation_count = self._task.variation_count()
                self._task.set_variation(i % variation_count)
            else:
                self._task.set_variation(0)
            with open(f"racer/gradio_demo/random_seeds/random_seed{i}.pkl", 'rb') as f:
                random_seed = pickle.load(f)
            np.random.set_state(random_seed)
            desc, obs = self._task.reset()
        else:
            d = self._task.get_demos(
                1, live_demos=False, 
                image_paths=not_load_image,  # not load image
                random_selection=False, from_episode_number=i)[0]

            self._task.set_variation(d.variation_number)
            desc, obs = self._task.reset_to_demo(d)


        obs_copy = copy.deepcopy(obs)
        self._lang_goal = desc[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = True if self.record_queue is not None else False
        self._episode_index += 1
        self._recorded_images.clear()
        if self.record_queue is not None:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self.record_queue.put(self.normalize_image(cap))
        # self.record_queue.put(self.normalize_image(self._previous_obs_dict['front_rgb']))

        return self._previous_obs_dict, obs_copy
    
    def step(self, act_result: ActResult) -> Transition:
        if self.never_terminal:
            self._task._task._success_conditions = [NeverStop()]
            
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.
        error_status = "success"
        info = {}

        try:
            obs, reward, terminal = self._task.step(action)
            if self.never_terminal:
                terminal = False
                reward = 0.0
            obs_copy = copy.deepcopy(obs)
            obs_copy.gripper_pose = action[:7]
            obs_copy.gripper_open = action[7]
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0
            obs_copy = None

            if isinstance(e, IKError):
                print("IKError")
                self._error_type_counts['IKError'] += 1
            elif isinstance(e, ConfigurationPathError):
                print("ConfigurationPathError")
                self._error_type_counts['ConfigurationPathError'] += 1
            elif isinstance(e, InvalidActionError):
                print("InvalidActionError")
                error_status = "error"
                self._error_type_counts['InvalidActionError'] += 1
            else:
                print("Unknown error")
            print(e)

            self._last_exception = e
        
        info.update({'error_status': error_status, 'obs': obs_copy})
        self._i += 1

        if self.record_queue is not None:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self.record_queue.put(self.normalize_image(cap))
        
        return Transition(obs, reward, terminal, info=info, summaries=[])
    

@dataclass
class Action:
    translation: ndarray    # (3,)
    rotation: quaternion.quaternion       # (4,) quaternion
    gripper_open: bool      
    ignore_collision: bool
    
    @property
    def T(self):
        return self.translation
    
    @T.setter
    def T(self, value):
        self.translation = value
    
    @property
    def R(self):
        return self.rotation
    
    @property
    def Rmat(self):
        return quaternion.as_rotation_matrix(self.rotation)
    
    @R.setter
    def R(self, value):
        self.rotation = value

    def to_numpy(self) -> ndarray:
        return np.concatenate((
            self.translation,
            self.quat_to_array(self.rotation, style='xyzw'),
            np.array([self.gripper_open, self.ignore_collision], dtype=float)
        ))
    
    @classmethod
    def from_numpy(cls, arr: ndarray):
        translation = arr[:3]
        rotation = cls.array_to_quat(arr[3:7], style='xyzw')
        gripper_open = bool(arr[7])
        ignore_collision = bool(arr[8])
        return cls(translation, rotation, gripper_open, ignore_collision)
    
    @staticmethod
    def quat_to_array(quat: quaternion.quaternion, style: str = 'xyzw'):
        # style is the output arr style
        a = quaternion.as_float_array(quat)
        if style == 'xyzw':
            return np.array([a[1], a[2], a[3], a[0]])
        elif style == 'wxyz':
            return a
        else:
            raise ValueError(f"Unknown style: {style}")
    
    @staticmethod
    def array_to_quat(arr: ndarray, style: str = 'xyzw'):
        # style is the input arr style
        if style == 'xyzw':
            return quaternion.quaternion(arr[3], arr[0], arr[1], arr[2])
        elif style == 'wxyz':
            return quaternion.quaternion(arr[0], arr[1], arr[2], arr[3])
        else:
            raise ValueError(f"Unknown style: {style}")
        
    @staticmethod
    def quat_to_euler(quat: quaternion.quaternion):
        return quaternion.as_euler_angles(quat)
    
    @staticmethod
    def delta_action(action_from: 'Action', action_to: 'Action'):
        delta_translation = action_to.T - action_from.T
        delta_rotation = Rotation.from_quat(action_from.quat_to_array(action_from.R.inverse() * action_to.R, 'xyzw')).as_euler('xyz', degrees=True)
        delta_gripper = int(action_to.gripper_open) - int(action_from.gripper_open)
        delta_ignore = int(action_to.ignore_collision) - int(action_from.ignore_collision)
        return {
            "translation": delta_translation,  
            "rotation": delta_rotation, 
            "gripper": delta_gripper,   # 0 = unchanged gripper state, 1 = open gripper, -1 = close gripper
            "collision": delta_ignore   # 0 = unchanged collision state, 1 = ignore collision, -1 = consider collision
        }

    def __str__(self):
        return_str = f"T: {self.T}\t"
        return_str += f"R: {self.quat_to_array(self.R, style='xyzw')}"
        if self.gripper_open:
            return_str += " gripper open "
        else:
            return_str += " gripper close"
        if self.ignore_collision:
            return_str += ", collision ignore"
        else:
            return_str += ", collision consider"
        return return_str




TEMPLATE_first_step = "<image>\nThe task goal is: {task_goal}. This is the first step and the robot is about to start the task. Based on the visual observation and the context, how does the robot fulfil that previous instruction and what's the next instruction for the robot arm?"
TEMPLATE_other_step = "<image>\nThe task goal is: {task_goal}. In the previous step, the robot arm was given the following instruction: \"{previous_instruction}\". {robot_delta_state} Based on the visual observation and the context, how does the robot fulfil that previous instruction and what's the next instruction for the robot arm?"
    
AXES = ["x", "y", "z"]
TRANSLATION_SMALL_THRES = 0.01
TRANSLATION_LARGE_THRES = 0.05
ROTATION_SMALL_THRES = 5
ROTATION_LARGE_THRES = 20
# Directions: (name, axis, +ve sign = 1)
# backward = closer to VLM's view & forward = further away from VLM's perspective
DIRECTIONS = [("backward", 0, 1), ("forward", 0, -1), ("right", 1, 1), ("left", 1, -1), ("down", 2, -1), ("up", 2, 1)]


def get_robot_delta_state(prev_obs:Observation, curr_obs:Observation):
        if prev_obs is None or curr_obs is None:
            return "The robot makes an invalid action."
        prev_action = Action.from_numpy(np.hstack((prev_obs.gripper_pose, prev_obs.gripper_open, prev_obs.ignore_collisions)))
        curr_action = Action.from_numpy(np.hstack((curr_obs.gripper_pose, curr_obs.gripper_open, curr_obs.ignore_collisions)))
        delta_action = curr_action.delta_action(prev_action, curr_action)

        sentence_parts = []

        is_translation_changed = True
        is_rotation_changed = True
        is_gripper_changed = True
        is_collision_changed = True
        # Position descriptions
        movements = []
        for direction, axis, sign in DIRECTIONS:
            translation_component = delta_action['translation'][axis]
            if sign * translation_component > TRANSLATION_SMALL_THRES:
                desc = f"moved {direction}"
                if abs(translation_component) < TRANSLATION_LARGE_THRES:
                    desc += " a little bit"
                movements.append(desc)
        if not movements:
            movements.append("didn't move its gripper")
            is_translation_changed = False

        sentence_parts.append(", ".join(movements))

        # Rotation description
        rotation = None
        if np.any(np.abs(delta_action['rotation']) > ROTATION_SMALL_THRES):
            if all(np.abs(delta_action['rotation']) > ROTATION_SMALL_THRES):
                rotation = "rotated the gripper"
            elif np.abs(delta_action['rotation'][2]) > ROTATION_LARGE_THRES:
                rotation = "rotated the gripper about z-axis"
        else:
            rotation = "didn't rotate the gripper"
            is_rotation_changed = False
        
        if rotation:
            sentence_parts.append(rotation)

        # Gripper change
        if delta_action['gripper'] != 0:
            gripper_change = "opened the gripper" if delta_action['gripper'] == 1 else "closed the gripper"
        else:
            gripper_change = "kept the gripper open" if curr_action.gripper_open == 1 else "kept the gripper closed"
            is_gripper_changed = False
        sentence_parts.append(gripper_change)

        # Collision plan
        collision_description = ""
        if delta_action['collision'] != 0:
            collision_description = "that can allow collisions" if delta_action['collision'] == 1 else "that avoids any collision"
            collision_description = f"by planning a motion path {collision_description}"
        else:
            is_collision_changed = False

        # Join parts with proper handling of "and"
        complete_sentence = "Then the robot " + sentence_parts[0]
        if len(sentence_parts) > 1:
            for part in sentence_parts[1:]:
                if part == gripper_change:
                    complete_sentence += f", and {part}"
                else:
                    complete_sentence += f", {part}"

        # Append collision description last if it exists
        if collision_description:
            complete_sentence += f" {collision_description}"

        complete_sentence += "."

        is_robot_state_changed = is_translation_changed or is_rotation_changed or is_gripper_changed or is_collision_changed
        return complete_sentence, is_robot_state_changed