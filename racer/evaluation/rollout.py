import argparse
from copy import deepcopy
import json
import os
import shutil
import textwrap
import logging
import json
from PIL import Image
import cv2
import numpy as np
import yaml

from rlbench.backend import task as rlbench_task
from racer.rvt.utils.peract_utils import CAMERAS, IMAGE_RGB
from racer.utils.racer_utils import RLBENCH_TASKS
from racer.evaluation.policy_agent import ModelRVTAgent
from racer.evaluation.simulator import RLBenchSim
from racer.evaluation.llava_api.api import LlavaAPI
from racer.evaluation.utils import START_ACTION, get_robot_delta_state, TEMPLATE_first_step, TEMPLATE_other_step



np.set_printoptions(precision=4, suppress=True)


def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--tasks", type=str, nargs="+", default=["close_jar"])
    parser.add_argument("--eval-datafolder", type=str, default="data/test")
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="start to evaluate from which episode",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=25,
        help="how many episodes to be evaluated for each task",
    )  # if only eval one episode, set this to 1

    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--log-name", type=str, default="test/xxx")
    parser.add_argument("--eval-log-dir", type=str, default="none")
    parser.add_argument("--prefix", type=str, default="", help="prefix for log dir, only works for slurm")


    # RVT policy agent
    parser.add_argument("--model-folder", type=str)
    parser.add_argument("--model-name", type=str, default="model_17.pth")
    parser.add_argument("--device", type=int, default=0)  # for poliy agent

    parser.add_argument("--retry-for-InvalidActionError", type=int, default=5)
    parser.add_argument("--append-text-to-gif", action="store_true")
    
    # LM service
    parser.add_argument("--lm-address", type=str, default="http://localhost:8000")

    # Llava VLM service
    parser.add_argument("--vlm-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--use-vlm", action="store_true"
    )  # use llava vlm agent to generate instruction. RVT does not need this

    # human cotrol
    parser.add_argument("--use-human-low-level", action="store_true")
    parser.add_argument("--use-manual-control", action="store_true")
    parser.add_argument("--use-human-high-level", action="store_true")
    parser.add_argument("--use-human-help", action="store_true")

    # old version lang input
    # The langauge input for RVT is full length (512) of the language encoder
    parser.add_argument("--use-full-langlen", action="store_true")


    args = parser.parse_args()

    if args.eval_log_dir == "none":
        args.eval_log_dir = os.path.join(args.model_folder, "new_eval")

    args.eval_log_dir = os.path.join(args.eval_log_dir, args.log_name)

    os.makedirs(args.eval_log_dir, exist_ok=True)

    # save the arguments for future reference
    with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)

    return args


class Evaluator:
    def __init__(self, args):
        self.args = args
        logging.basicConfig(filename=f"{args.eval_log_dir}/log", level=logging.INFO)

        model_path = os.path.join(args.model_folder, args.model_name)
        self.policy_executor = ModelRVTAgent(
            model_path=model_path, 
            device=args.device, 
            use_full_langlen=args.use_full_langlen,
            lm_addr=args.lm_address
        )

        self.use_vlm = args.use_vlm
        if self.use_vlm:
            logging.info("Using VLM for instruction generation")
            print("Using VLM for instruction generation")
            self.llava_api = LlavaAPI(args.vlm_address)

        self.use_human_low_level = args.use_human_low_level
        if self.use_human_low_level:
            logging.info("Using human control for low-level instruction")
            print("Using human control for low-level instruction")
        
        self.use_human_high_level = args.use_human_high_level
        if self.use_human_high_level:
            logging.info("Using human control for high-level task goal")
            print("Using human control for high-level task goal")
        
        self.use_human_help = args.use_human_help
        if self.use_human_help:
            logging.info("Using human help for catastrophic failure")
            print("Using human help for catastrophic failure")
        

        self.tasks = self.get_tasks(args.tasks)

        # initialize the environment for the first task
        self.env = RLBenchSim(
            task_name=self.tasks[0],
            dataset_root=args.eval_datafolder,
            episode_length=args.episode_length
        )
        self.debug = args.debug


        self.metric_dict = {}


    def record_image(self):
        if not self.debug:
            return
        front_rgb = self.get_img(self.obs_dict["front_rgb"], return_pil=False)
        left_shoulder_rgb = self.get_img(
            self.obs_dict["left_shoulder_rgb"], return_pil=False
        )
        right_shoulder_rgb = self.get_img(
            self.obs_dict["right_shoulder_rgb"], return_pil=False
        )
        wrist_rgb = self.get_img(self.obs_dict["wrist_rgb"], return_pil=False)
        # complie in a single image, 2x2
        rgb_up = np.concatenate([front_rgb, wrist_rgb], axis=1)
        rgb_down = np.concatenate([left_shoulder_rgb, right_shoulder_rgb], axis=1)
        rgb = np.concatenate([rgb_up, rgb_down], axis=0)
        Image.fromarray(rgb).save("current.png")

        if not self.args.use_human_low_level:
            input("Press Enter to continue...")

    def get_img(self, rgb, return_pil=True, res=512):
        if rgb.shape[0] == 3:  # (3, 512, 512) -> (512, 512, 3)
            rgb = rgb.transpose(1, 2, 0)
        rgb_pil = Image.fromarray(rgb).resize((res, res))
        if return_pil:
            return rgb_pil
        return np.array(rgb_pil)


    @staticmethod
    def get_tasks(tasks):
        task_files = [
            t.replace(".py", "")
            for t in os.listdir(rlbench_task.TASKS_PATH)
            if t != "__init__.py" and t.endswith(".py")
        ]
        if tasks[0] == "all":
            tasks = RLBENCH_TASKS
        else:
            tasks = tasks[0].split(",")
        ouput_tasks = []
        for task in tasks:
            if task in task_files:
                ouput_tasks.append(task)
            else:
                raise ValueError(f"Task {task} not found in RLBench tasks")
        logging.info(f"evaluate on {len(ouput_tasks)} tasks: {ouput_tasks}")
        return ouput_tasks


    def eval(self):
        self.policy_executor.reset()
        for task_name in self.tasks:
            self.metric_dict[task_name] = {}
            self.env.set_new_task(task_name)
            for episode_num in range(
                self.args.start_episode,
                self.args.start_episode + self.args.eval_episodes,
            ):
                self.metric_dict[task_name][episode_num] = {}
                success, error_status, step, episode_statistic, frames = (
                    self.eval_episode(task_name, episode_num)
                )
                trials = (
                    self.args.retry_for_InvalidActionError
                )  # RLbench sometimes throws weird internal InvalidActionError even the action looks very valid
                while error_status == "error" and trials > 0:
                    logging.info(
                        f"Retry {self.args.retry_for_InvalidActionError - trials} times for InvalidActionError"
                    )
                    success, error_status, step, episode_statistic, frames = (
                        self.eval_episode(task_name, episode_num)
                    )
                    trials -= 1

                ep_save_path = f"{self.args.eval_log_dir}/{task_name}/{episode_num}"
                if os.path.exists(ep_save_path):
                    shutil.rmtree(ep_save_path)
                os.makedirs(ep_save_path, exist_ok=True)
                with open(f"{ep_save_path}/episode_statistic.json", "w") as fp:
                    json.dump(episode_statistic, fp, indent=2)
                if success:
                    with open(f"{ep_save_path}/success_step{step}", "w") as fp:
                        fp.write("success")
                else:
                    with open(f"{ep_save_path}/failure_step{step}", "w") as fp:
                        fp.write("failure")
                
                # save gif
                for cam in CAMERAS:
                    frames[f"{cam}_{IMAGE_RGB}"][0].save(
                        f"{ep_save_path}/{cam}_{IMAGE_RGB}.gif",
                        save_all=True,
                        append_images=frames[f"{cam}_{IMAGE_RGB}"][1:],
                        duration=500,
                        loop=0,
                    )

                self.metric_dict[task_name][episode_num]["success"] = success
                self.metric_dict[task_name][episode_num]["episode_len"] = step
                self.metric_dict[task_name][episode_num]["retry_times"] = (
                    args.retry_for_InvalidActionError - trials
                )

                logging.info(
                    f"Task: {task_name} | Episode: {episode_num} | Success: {success} | LangGoal: {self.env.task_goal} |"
                    f" retry times: {args.retry_for_InvalidActionError - trials}\n\n"
                )
            success_rate = np.mean(
                [v["success"] for v in self.metric_dict[task_name].values()]
            )
            logging.info(
                f"Calculate... Task: {task_name}, Success Rate: {success_rate}\n"
            )

        # get overall success rate
        logging.info("Overall Success Rate:")
        avg_success_rate = []
        self.metric_dict["overall"] = {}
        for task_name in self.tasks:
            success_rate = np.mean(
                [v["success"] for v in self.metric_dict[task_name].values()]
            )
            self.metric_dict["overall"][task_name] = success_rate
            avg_success_rate.append(success_rate)
            logging.info(f"Task: {task_name}, Success Rate: {success_rate}")
            if self.debug:
                print(f"Task: {task_name}, Success Rate: {success_rate}")

        avg_success_rate = np.mean(avg_success_rate)
        self.metric_dict["overall"]["avg_success_rate"] = avg_success_rate
        logging.info(f"Avg Success Rate: {avg_success_rate}")
        with open(f"{self.args.eval_log_dir}/metrics.json", "w") as fp:
            json.dump(self.metric_dict, fp, indent=2)

    @staticmethod
    def get_ep_dict(action, task_goal, input_lang_str):
        return {
            "pose": (
                ",".join([str(a) for a in action[:3].tolist()])
                if action is not None
                else None
            ),
            "rotation": (
                ",".join([str(a) for a in action[3:7].tolist()])
                if action is not None
                else None
            ),
            "gripper_open": action[7] if action is not None else None,
            "ignore_collision": action[8] if action is not None else None,
            "task_goal": task_goal,
            "input_inst": input_lang_str,
        }


    def eval_episode(self, task_name, episode_num):
        episode_dict = {}
        frames = {f"{cam}_{IMAGE_RGB}": [] for cam in CAMERAS}
        # start the episode
        self.obs_dict, _obs = self.env.reset(episode_num)
        self.prev_obs = None
        self.cur_obs = _obs
        error_status = "success"
        step = 0
        for cam in CAMERAS:
            frames[f"{cam}_{IMAGE_RGB}"].append(
                self.get_img(self.obs_dict[f"{cam}_{IMAGE_RGB}"], res=256)
            )

        self.record_image()
        if self.debug: 
            print(f"task goal: {self.env.task_goal}.")
            print("\tstep 0, get started ... ")
        logging.info(f"\tstep 0, get started ... ")

        done = False
        success = False
        self.last_action = START_ACTION
        self.instruction = None
        self.ask_help = False

        while not done:
            task_goal = self.get_high_level_goal()
            self.task_goal = task_goal
            instruction = self.get_low_level_instruction()
            self.instruction = instruction
            if instruction and  instruction.strip() == "restart": # human control to restart the episode
                done = True
                error_status = "error"
                self.args.retry_for_InvalidActionError = 100
                continue
                
            if instruction and  instruction.strip() in ["exit", "stop", "end"]: # human control to exit the episode
                done = True
                error_status = ""
                continue

            if self.use_human_help:
                # figure out whether instruction tells catastrophic failure
                if self.ask_help:
                    done, success = True, True
                    continue

            input_lang_str = self.get_input_lang_str_for_policy(
                task_goal, instruction, old_version=self.args.use_full_langlen
            )

            episode_dict[step] = self.get_ep_dict(self.last_action, task_goal, input_lang_str)
            
            for k in frames: # add text to each frame
                frames[k][-1] = self.add_text_to_frame(frames[k][-1], input_lang_str)

            action = self.policy_executor.act(
                self.obs_dict, input_lang_str=input_lang_str)
            action = self.action_check(action)
            action = self.postprocess(task_name, action, self.last_action, step)

            ### human control simple rules:
            if self.use_human_low_level and self.args.use_manual_control:
                if instruction and len(instruction.split()) < 7:
                    action = self.manual_control(self.last_action, instruction)
   

            transition = self.env.step(action)
            for cam in CAMERAS:
                frames[f"{cam}_{IMAGE_RGB}"].append(
                    self.get_img(transition.observation[f"{cam}_{IMAGE_RGB}"], res=256)
                )
            if self.env.is_success():
                success = True

            self.obs_dict = transition.observation
            self.prev_obs = deepcopy(self.cur_obs)
            self.cur_obs = deepcopy(transition.info["obs"])
            done = transition.terminal
            error_status = transition.info["error_status"]
            logging.info(f'\tinput_lang_str: {json.dumps(input_lang_str)}')
            logging.info(
                f"\tstep {step+1}, pose: , {action[:7]}, gripper open: {action[7]==1}, ignore collision: {action[8]==1}\n"
            )
            if self.debug:
                print(f'\n\n\tinput_lang_str: {json.dumps(input_lang_str)}')
                print(
                    f"\tstep {step+1}, pose: , {action[:7]}, gripper open: {action[7]==1}, ignore collision: {action[8]==1}\n\n"
                )

            self.record_image()

            step += 1
            self.last_action = action
            if error_status != "error":
                self.cur_obs.gripper_open = self.last_action[-2]
                self.cur_obs.ignore_collisions = self.last_action[-1]
            if step >= self.args.episode_length:
                break
        
        final_comment = "Task success!!!" if success else "Task failure!!!"
        for k in frames: # add text to each frame
            frames[k][-1] = self.add_text_to_frame(frames[k][-1], final_comment)
        
        episode_dict[step] = self.get_ep_dict(self.last_action, task_goal, input_lang_str)

        return success, error_status, step, episode_dict, frames

    @staticmethod
    def postprocess(task_name, action, last_action, step):
        if task_name in ["place_shape_in_shape_sorter"]: # always avoid collision
            action[-1] = 0
        if task_name in ["stack_blocks"]:
            if last_action[-2] == 0 and action[-2] == 1: # lower a little bit to open gripper to place block, avoid unnecessary fell down 
                action[2] = last_action[2] - 0.06
        if task_name in ["meat_off_grill"]: # avoid unnecessary collision
            if last_action[-2] == 0 and action[-2] == 0:
                action[-1] = 0
        if task_name in ["close_jar"]: # avoid stuck in the table
            if np.linalg.norm(action[:3] - last_action[:3]) < 0.008 and last_action[-2]==0 and step <=3:
                action[2] += 0.18
            if last_action[-2]==0:
                action[-2] = 0
        if task_name in ["light_bulb_in"]:
            if last_action[-2]==0 and 4<step<7: # avoid to release the bulb too early
                action[-2] = 0
        return action

    @staticmethod
    def parse_vlm_instruction(vlm_output:str):
        if "The next instruction is:" in vlm_output:
            return vlm_output.split("The next instruction is:")[1].strip()
        elif "next instruction is" in vlm_output:
            return vlm_output.split("next instruction is")[1].strip()
        elif "next instruction:" in vlm_output:
            return vlm_output.split("next instruction:")[1].strip()
        else:
            return vlm_output

    def get_low_level_instruction(self):
        if self.use_vlm:
            user_msg = self.prepare_user_msg_for_vlm()
            logging.info(f"User message: {json.dumps(user_msg)}")
            instruction = self.llava_api.get_response(
                user_msg, image=self.obs_dict["front_rgb"]
            )
            logging.info(f"VLM output: {instruction}")
            if self.use_human_help and ("catastrophic" in instruction or "ask human for help" in instruction):
                self.ask_help = True
            instruction = self.parse_vlm_instruction(instruction)
        elif self.use_human_low_level:
            instruction = input("Human Control, current instruction: ")
            instruction = instruction.strip().lower()
        else:
            instruction = None
        return instruction

    def get_high_level_goal(self):
        if self.use_human_high_level:
            new_task_goal = input("Human Control, new task goal: ")
            # empty means no task change
            new_task_goal = new_task_goal.strip().lower()
            if new_task_goal:
                return new_task_goal
        return self.env.task_goal

    @staticmethod
    def get_input_lang_str_for_policy(task_goal, instruction, old_version=False):
        if old_version:  # old version of RVT input lang str e.g., "close the red jar"
            if instruction is None or instruction.strip() == "":
                return task_goal
            else:
                return task_goal + ". " + instruction
        if (
            instruction is None or instruction.strip() == ""
        ):  # high-level goal only, e.g., "task goal: close the red jar."
            return f"Task goal: {task_goal}."
        else:  # blend with low-level instruction
            if instruction.endswith("."):
                instruction = instruction[:-1]
            return f"Task goal: {task_goal}.\nCurrent instruction: {instruction.lower()}."


    def prepare_user_msg_for_vlm(self):
        if self.instruction is None: # first turn
            return TEMPLATE_first_step.format(task_goal=self.task_goal)
        else:
            robot_delta_state, _ = get_robot_delta_state(self.prev_obs, self.cur_obs)
            return TEMPLATE_other_step.format(task_goal=self.task_goal, previous_instruction=self.instruction, robot_delta_state=robot_delta_state)

    @staticmethod
    def action_check(action):
        action[3:7] = action[3:7] / np.linalg.norm(action[3:7])
        return action
    

    def manual_control(self, action, instruction):
        action = deepcopy(action)
        # match move up, down, left, right, forward, backward, open gripper, close gripper
        if "rotate" in instruction or "point" in instruction:
            if "down":
                action[3:7] = np.array([0, 1 , 0, 0])
            elif "left":
                action[3:7] = np.array([0.7071067811865475,0.0,0.0,0.7071067811865476])
            elif "right":
                action[3:7] = np.array([0.7071067811865475,0.0,0.0,-0.7071067811865476])

        else:
            if "a little" in instruction:
                d = 0.025
            elif "a lot" in instruction:
                d = 0.08
            else:
                d = 0.05
            if "up" in instruction:
                action[2] += d
            elif "down" in instruction:
                action[2] -= d
            elif "left" in instruction:
                action[1] -= d
            elif "right" in instruction:
                action[1] += d
            elif "forward" in instruction:
                action[0] -= d
            elif "backward" in instruction:
                action[0] += d
            
        action[8] = 0
        
        if "open gripper" in instruction:
            action[7] = 1
        if "close gripper" in instruction:
            action[7] = 0
        if "ignore collision" in instruction or "allow collision" in instruction:
            action[8] = 1
        return self.action_check(action)
    

    @staticmethod
    def wrap_text(text, font, max_width):
        lines = []
        if "\n" in text:
            lines.append(text.split("\n")[0])
            words = text.split("\n")[1].split()
        else:
            words = text.split()
        while words:
            line = ''
            while words and (font.getlength(line + words[0]) <= max_width):
                ch = words.pop(0) 
                line += (ch+ ' ')
            lines.append(line.strip())
        return lines
    

    @staticmethod
    def _add_text_beneath_frame(frame, text):
        # append insturction below the frame
        image = frame
        if isinstance(image, Image.Image):
            image = np.array(image)
        h, w, c = image.shape
        font_size = 0.3
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        blank_image = np.zeros((90,w,c), dtype=np.uint8)

        lines = text.split('\n')  # Split the text into lines based on newline characters
        wrapped_lines = []
        char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]

        for line in lines:
            wrapped_lines.extend(textwrap.wrap(line, width=int(w / char_size[0])+8))  # Wrap each line

        y = 0
        for line in wrapped_lines:
            textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
            y += textsize[1] + 2
            x = 5
            cv2.putText(
                blank_image,
                line,
                (x, y),
                font,
                font_size,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA,
            )
        # text_image = blank_image[0 : y + 20, 0:w]
        final = np.concatenate((image, blank_image), axis=0)
        return Image.fromarray(final)

    def add_text_to_frame(self, frame, text):
        return self._add_text_beneath_frame(frame, text)

if __name__ == "__main__":
    args = make_args()
    evaluator = Evaluator(args)
    evaluator.eval()