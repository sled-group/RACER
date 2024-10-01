from functools import partial
import json
import os
import shutil
import time
import uuid
import gradio as gr
from multiprocessing import Queue, sharedctypes
import imageio
from PIL import Image

from .agentenv import RobotState
from racer.evaluation.llava_api.api import LlavaAPI
from racer.evaluation.utils import TEMPLATE_first_step, TEMPLATE_other_step
from racer.evaluation.rollout import Evaluator

from .utils import *



class GradioInterface:
    def __init__(
        self,
        vid_queue: Queue,
        front_rgb_queue: Queue, 
        left_shoulder_rgb_queue: Queue,
        right_shoulder_rgb_queue,
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
        save_session: bool = False,
        save_prefix: str = "debug",
        vlm_address="http://localhost:21002"
    ):
        """
        Initialize the gradio interface with the necessary queues and shared values
        :param vid_queue: Queue to get the video frame
        :param front_rgb_queue: Queue to get the front rgb image
        :param left_shoulder_rgb_queue: Queue to get the left shoulder rgb image
        :param right_shoulder_rgb_queue: Queue to get the right shoulder rgb image
        :param wrist_rgb_queue: Queue to get the wrist rgb image
        :param inst_queue: Queue to get the instruction
        :param goal_queue: Queue to get the task goal
        :param state_queue: Queue to get the robot_delta_state, action description, etc,...
        :param ep_value: shared value to get the episode number, default is 0, ranging from 0-24 for val/test, 0-99 for train
        :param ds_value: shared value to get the dataset number, 0: train, 1: val, 2: test
        :param md_value: shared value to get the model number, 0: rvt, 1: ours
        :param tn_value: shared value to get the task name, the order is given in rvt.utils.rvt_utils.RLBENCH_TASKS
        :param reset_value: shared value to restart the episode, 0 means no restart, 1 means restart
        :param step_value: shared value to get the step action, 0 means no action, 1 means take action
        :param ctrl_value: shared value to get the manual control action, 0 means no action, 1 means gripper, 2 means home, 3 means move left, 4 means move right, 5 means move forward, 6 means move backward, 7 means move up, 8 means move down
        :param llava_talk_value: shared value to get the llava talk status, 0 means close, 1 means open
        """

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

        self.llava_api = LlavaAPI(addr=vlm_address)
        self.llava_talk_value = llava_talk_value

        self.last_frame =  Image.open("racer/gradio_demo/cover.png")
        self.last_front_rgb = Image.open("racer/gradio_demo/cover.png")
        self.last_left_shoulder_rgb = Image.open("racer/gradio_demo/cover.png")
        self.last_right_shoulder_rgb = Image.open("racer/gradio_demo/cover.png")
        self.last_wrist_rgb = Image.open("racer/gradio_demo/cover.png")
        self.last_inst = None
        self.last_task_goal = goal_queue.get()
        self.robot_state = RobotState(robot_delta_state="idle", action_desc="No action")

        self.step_lock = False # when llava gen, can not stepaction
        self.img_lock = False

        self.last_instruction = None
        self.save_session = save_session
        self.save_prefix = save_prefix
        self.session_id = uuid.uuid4().hex[:10]

        if save_session:
            self.record_vid = []
            self.record_obs = {key:[] for key in ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"]}
            self.record_ctx = { "metadata":
                {
                "dataset": DATASET[self.ds_value.value],
                "episode": self.ep_value.value,
                "task": RLBENCH_TASKS[self.tn_value.value],
                "task_goal": self.last_task_goal,
                "model": MODELS[self.md_value.value],
                },
                "context":[]
            }




    def get_img(self, key): # get front, left_shoulder, right_shoulder, wrist images
        assert key in ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"]
        queue = getattr(self, f"{key}_queue")
        if queue.empty() or self.img_lock:
            return getattr(self, f"last_{key}")
        else:
            img = queue.get()
            setattr(self, f"last_{key}", img)
            if self.save_session:
                self.record_obs[key].append(img)
            return img
    
    def get_vid(self):
        if self.vid_queue.empty():
            self.img_lock = False
            return self.last_frame
        else:
            self.img_lock = True
            self.last_frame = self.vid_queue.get()
            if self.save_session:
                self.record_vid.append(self.last_frame)
            return self.last_frame    

    def process_human_inst(self, message, chat_history):
        if not message.strip():
            return "", chat_history
        
        # use this when never_terminal is True
        # input "success" manually inside the chatbox to set the task status to success
        if message.strip() == "success": # used for task-goal change
            self.robot_state.status = "success"
        
        self.inst_queue.put(("human", message.strip()))
        chat_history.append(("(Human Talk) "+message, None))
        return "", chat_history
    
    def dump_session(self, chat_history):    
        time.sleep(1)
        current_date = datetime.now().strftime("%m-%d-%H")
        save_card = f"{MODELS[self.md_value.value]}/{DATASET[self.ds_value.value]}/{RLBENCH_TASKS[self.tn_value.value]}/ep{self.ep_value.value}"
        path = f"racer/gradio_demo/sessions/{self.save_prefix}/{save_card}/{current_date}-{self.session_id}-{self.robot_state.status}"
        if os.path.exists(path):
            shutil.rmtree(path)
        if not self.save_session or len(self.record_vid) < 10: 
            if self.robot_state.status == "InvalidActionError":
                os.makedirs(path, exist_ok=True)
            return
        os.makedirs(path, exist_ok=True)
        # save videos in video dir
        vid_dir = f"{path}/videos"
        os.makedirs(vid_dir, exist_ok=True)
        output_file = os.path.join(vid_dir, "complete_video.mp4")
        imageio.mimsave(output_file, self.record_vid, fps=20)

        for key in self.record_obs:
            for i, img in enumerate(self.record_obs[key]):
                if i < len(self.record_ctx["context"]):
                    task_goal = self.record_ctx["context"][i].get("task_goal", "none")
                    inst = self.record_ctx["context"][i].get("last_insturction", None)
                    role = self.record_ctx["context"][i].get("instruction_role", "")
                    if inst is not None:
                        self.record_obs[key][i] = Evaluator._add_text_beneath_frame(img, f"Task goal: {task_goal}\nInstruction: ({role}) {inst}")
                    else:
                        self.record_obs[key][i] = Evaluator._add_text_beneath_frame(img, f"Task goal: {task_goal}")
                else:
                    status = self.record_ctx["context"][-1].get("status", "none")
                    if status == "ongoing":
                        status = "failure"

                    # use this when never_terminal is True
                    if self.robot_state.status == "success":
                        self.record_ctx["context"][-1]["status"] = "success"
                        status = "success"
                    
                    self.record_obs[key][i] = Evaluator._add_text_beneath_frame(img, f"Task status: {status}")
            
            self.record_obs[key][0].save(
                f"{path}/{key}.gif",
                save_all=True,
                append_images=self.record_obs[key][1:],
                duration=500,
                loop=0,
            )
        
        with open(f"{path}/context.json", "w") as f:
            json.dump(self.record_ctx, f, indent=2)
        
        chat_history.append((None, f"--- Session saved ---"))
        return chat_history
            

    def restart(self):
        self.step_lock = False
        chat_history = []
        self.reset_value.value = 1
        self.last_task_goal = self.goal_queue.get()
        if self.save_session:
            self.record_vid = []
            self.record_obs = {key:[] for key in ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"]}
            self.record_ctx = { "metadata":
                {
                "dataset": DATASET[self.ds_value.value],
                "episode": self.ep_value.value,
                "task": RLBENCH_TASKS[self.tn_value.value],
                "task_goal": self.last_task_goal,
                "model": MODELS[self.md_value.value],
                },
                "context":[]
            }
            self.session_id = uuid.uuid4().hex[:10]
        chat_history.append((None, f"--- Start {DATASET[self.ds_value.value]} dataset, episode {self.ep_value.value}, task {RLBENCH_TASKS[self.tn_value.value]}, task goal \"{self.last_task_goal}\", use model {MODELS[self.md_value.value]} ---"))
        while self.reset_value.value == 1:
            pass
        self.robot_state = RobotState(robot_delta_state="idle", action_desc="No action")
        # clear last instruction
        while not self.inst_queue.empty():
            self.inst_queue.get()
        if self.llava_talk_value.value == 1:
            chat_history.append(["(Llava Talk) ", None])
            for resp in self.process_llava_inst():
                chat_history[-1][0] += resp
                yield self.last_task_goal, chat_history
        else:
            yield self.last_task_goal, chat_history
        # [ 0.0616 -0.1523 1.1663], rotation: [-0.98 0.199 -0. 0. ]
        # [0.1523 0.075 1.1713], rotation: [-0.98 0.199 -0. 0. ],

    def process_llava_inst(self):
        # self.step_lock = True
        # chat_history.append(["", None])
        # yield chat_history
        # for resp in self.llava_api.get_response_stream(self.wrap_prompt(), self.get_img()):
        #     chat_history[-1][0] += resp
        #     yield chat_history
        logger.info(f"RobotState: {json.dumps(self.robot_state.to_dict())}")
        if self.robot_state.robot_delta_state == "idle":
            user_msg = TEMPLATE_first_step.format(task_goal=self.last_task_goal)
        else: 
            if self.last_instruction is None:
                # chat_history.append([None, "--- Please manually input the instruction first! ---"])
                # return chat_history
                self.last_instruction = self.last_task_goal
            elif self.last_instruction == "home":
                self.last_instruction = "reset to the home position."
            robot_delta_state = self.robot_state.robot_delta_state
            # the gripper does not change state
            if not self.robot_state.is_robot_state_changed:
                robot_delta_state = self.robot_state.last_robot_delta_state
            self.last_robot_delta_state = robot_delta_state
            user_msg = TEMPLATE_other_step.format(task_goal=self.last_task_goal, previous_instruction=self.last_instruction, robot_delta_state=robot_delta_state)
        
        logger.info(f"query llava with {json.dumps(user_msg)}")
        # instruction = self.llava_api.get_response_stream(user_msg, image=self.last_front_rgb)
        instruction = ""
        for chuck in self.llava_api.get_response_stream(user_msg,  image=self.last_front_rgb):
            time.sleep(0.01)
            instruction += chuck
            yield chuck
        
        # # mimic llava streaming response
        # llava_message = "i am llava instruction. i am llava instruction. i am llava instruction. i am llava instruction. i am llava instruction"
        # yield chat_history
        # for ch in llava_message:
        #     chat_history[-1][0] += ch
        #     time.sleep(0.1)
        #     yield chat_history

        instruction = Evaluator.parse_vlm_instruction(instruction)
        self.inst_queue.put(("llava", instruction))

        # if self.robot_state.status == "ongoing":
        #     self.step_lock = False

                
    def wrap_prompt(self):
        # self.robot_state
        return "What is shown in the image?"
    
    def change_llava_talk(self, llava_dd, chat_history):
        logger.info(f"LLAVA talk changed to {llava_dd}")
        self.llava_talk_value.value = LLAVA_TALK_CHOICE.index(llava_dd)
        if llava_dd == "open llava":
            chat_history.append((None, f"--- Open LLAVA Talk ---"))
            yield chat_history
            chat_history.append(["(Llava Talk) ", None])
            for resp in self.process_llava_inst():
                chat_history[-1][0] += resp
                yield chat_history
        else:
            chat_history.append((None, f"--- Close LLAVA Talk (Any previous generated llava response will not be used)---"))
            yield chat_history
            


    def change_episode(self, episode_dd):
        self.ep_value.value = episode_dd
        logger.info(f"Episode changed to {self.ep_value.value}")

    def change_dataset(self, dataset_dd):
        self.ds_value.value = DATASET.index(dataset_dd)
        logger.info(f"Dataset changed to {DATASET[self.ds_value.value]}")
    
    def change_model(self, model_dd, chat_history):
        self.md_value.value = MODELS.index(model_dd)
        logger.info(f"Model changed to {MODELS[self.md_value.value]}")
        chat_history.append((None, f"--- Model changed to {MODELS[self.md_value.value]} ---"))
        return chat_history

    def change_task(self, taskname_dd):
        self.tn_value.value = RLBENCH_TASKS.index(taskname_dd)
        logger.info(f"Task changed to {RLBENCH_TASKS[self.tn_value.value]}")
    
    def change_goal(self, message, chat_history):
        self.last_task_goal = message
        self.goal_queue.put(message)
        logger.info(f"Task goal becomes: {message}")
        chat_history.append((None, f"--- Task goal changed to \"{message}\" ---"))
        return chat_history
    
    def step_action(self, chat_history):
        # if self.step_lock:
        #     return chat_history
        self.step_value.value = 1
        self.robot_state: RobotState = self.state_queue.get() # wait for policy finish and get the robot state
        self.last_instruction = self.robot_state.last_insturction
        if self.save_session:
            dic = {"frame_id": len(self.record_vid)}
            dic.update(self.robot_state.to_dict())
            self.record_ctx["context"].append(dic)
        step_result_str = f"Input_Text: {json.dumps(self.robot_state.input_text_str)}\n"\
                          f"Robot_State_Description: {json.dumps(self.robot_state.robot_delta_state)}\n"\
                          f"Prediction: {self.robot_state.action_desc}"
        chat_history.append([None, step_result_str])    
        yield chat_history
        while not self.vid_queue.empty():
            time.sleep(0.1)
        if self.robot_state.status == "success":
            chat_history.append((None, "Task completed! Consider restarting the episode."))
            # self.step_lock = True
            yield chat_history
        elif self.robot_state.status == "failure":
            chat_history.append((None, "Task failed due to maxmium steps! Consider restarting the episode."))
            # self.step_lock = True
            yield chat_history
        elif self.robot_state.status == "InvalidActionError":
            chat_history.append((None, "Invalid action! Consider restarting the episode."))
            # self.step_lock = True
            yield chat_history
        else:
            # use llava talk
            if self.llava_talk_value.value == 1:
                chat_history.append(["(Llava Talk) ", None])
                for resp in self.process_llava_inst():
                    chat_history[-1][0] += resp
                    yield chat_history
    

    def manual_control(self, type):
        dic = {"gripper_change": 1, "home":2, "move_left":3, "move_right":4, "move_forward":5, "move_backward":6, "move_up":7, "move_down":8}
        self.ctrl_value.value = dic[type]
        self.robot_state: RobotState = self.state_queue.get() # wait for policy finish and get the robot state
        if self.save_session:
            dic = {"frame_id": len(self.record_vid)}
            dic.update(self.robot_state.to_dict())
            self.record_ctx["context"].append(dic)
        self.last_instruction = self.robot_state.last_insturction
    

    def run(self):
        with gr.Blocks() as demo:
            with gr.Column():
                gr.Markdown("## 🔥Rich Language-Guided Failure Recovery Demo🚀")

            with gr.Row():
                with gr.Column(scale=1):
                    plot_video = gr.Image(self.last_frame)
                    with gr.Row():
                        task_goal = gr.Textbox(label="Task Goal (Press Enter)", value=self.last_task_goal)    
                        llava_dd = gr.Dropdown(choices=LLAVA_TALK_CHOICE, label="Use LLAVA Talk", value=self.llava_talk_value.value)            
                    with gr.Row():
                        home_button = gr.Button("Home")
                        gripper_change_button = gr.Button("Gripper")
                        move_up_button = gr.Button("Up")
                        move_down_button = gr.Button("Down")
                        move_left_button = gr.Button("Left")
                        move_right_button = gr.Button("Right")
                        move_forward_button = gr.Button("Forward")
                        move_backward_button = gr.Button("Backward")
                
                with gr.Column(scale=2):
                    with gr.Row(equal_height=True):
                        taskname_dd = gr.Dropdown(choices=RLBENCH_TASKS, label="Task Name", value=RLBENCH_TASKS[self.tn_value.value]) 
                        dataset_dd = gr.Dropdown(choices=DATASET, label="RLbench Dataset", value=DATASET[self.ds_value.value])
                        episode_dd = gr.Dropdown(choices=list(range(25)), label="Episode", value=self.ep_value.value)
                        model_dd = gr.Dropdown(choices=MODELS, label="Model", value=MODELS[self.md_value.value])
                        restart_btn = gr.Button("Restart Episode", elem_id="restart_btn")
                    chatbot = gr.Chatbot(value=[(None, f"--- Start the {DATASET[self.ds_value.value]} dataset, episode {self.ep_value.value}, task {RLBENCH_TASKS[self.tn_value.value]}, task goal \"{self.last_task_goal}\", use model {MODELS[self.md_value.value]}---")])
                    chatbot.height = 570
                    with gr.Row():
                        with gr.Column(scale=8):
                            msg = gr.Textbox(label="Human instruction (Press Enter)")
                        step_btn = gr.Button("Step Action")
                    save_btn = gr.Button("Save Session", scale=1)

                    msg.submit(self.process_human_inst, [msg, chatbot], [msg, chatbot])


            with gr.Row():
                with gr.Column(elem_id="centered-column"):
                    gr.Markdown("### Front RGB", elem_id="centered-text")
                    plot_front_rgb = gr.Image(value=self.last_front_rgb)
                with gr.Column(elem_id="centered-column"):
                    gr.Markdown("### Left Shoulder RGB", elem_id="centered-text")
                    plot_left_shoulder_rgb = gr.Image(value=self.last_left_shoulder_rgb)
                with gr.Column(elem_id="centered-column"):
                    gr.Markdown("### Right Shoulder RGB", elem_id="centered-text")
                    plot_right_shoulder_rgb = gr.Image(value=self.last_right_shoulder_rgb)
                with gr.Column(elem_id="centered-column"):
                    gr.Markdown("### Wrist RGB", elem_id="centered-text")
                    plot_wrist_rgb = gr.Image(value=self.last_wrist_rgb)

            demo.load(self.get_vid, None, plot_video, every=0.1)
            demo.load(partial(self.get_img, "front_rgb"), None, plot_front_rgb, every=0.1)
            demo.load(partial(self.get_img, "left_shoulder_rgb"), None, plot_left_shoulder_rgb, every=0.1)
            demo.load(partial(self.get_img, "right_shoulder_rgb"), None, plot_right_shoulder_rgb, every=0.1)
            demo.load(partial(self.get_img, "wrist_rgb"), None, plot_wrist_rgb, every=0.1)
                
            # # Interaction
            restart_btn.click(self.restart, outputs=[task_goal, chatbot])
            llava_dd.change(self.change_llava_talk, inputs=[llava_dd, chatbot], outputs=[chatbot])
            step_btn.click(self.step_action, [chatbot], [chatbot])
            taskname_dd.change(self.change_task, [taskname_dd])
            task_goal.submit(self.change_goal, [task_goal, chatbot], [chatbot])

            episode_dd.change(self.change_episode, [episode_dd])
            dataset_dd.change(self.change_dataset, [dataset_dd])
            model_dd.change(self.change_model, [model_dd, chatbot], [chatbot])

            home_button.click(partial(self.manual_control, "home"))
            gripper_change_button.click(partial(self.manual_control, "gripper_change"))
            move_up_button.click(partial(self.manual_control, "move_up"))
            move_down_button.click(partial(self.manual_control, "move_down"))
            move_left_button.click(partial(self.manual_control, "move_left"))
            move_right_button.click(partial(self.manual_control, "move_right"))
            move_forward_button.click(partial(self.manual_control, "move_forward"))
            move_backward_button.click(partial(self.manual_control, "move_backward"))

            save_btn.click(self.dump_session, [chatbot], [chatbot])

        demo.launch(server_name="0.0.0.0", server_port=7883)