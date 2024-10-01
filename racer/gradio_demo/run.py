from multiprocessing import Process, Queue, sharedctypes, Value

from racer.gradio_demo.interface import GradioInterface
from racer.gradio_demo.agentenv import AgentEnv
from racer.gradio_demo.utils import *



def run_gradio(
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
    vlm_address: str = "http://localhost:21002"
):
    # Initialize the gradio interface
    gradio_interface = GradioInterface(
        vid_queue=vid_queue,
        front_rgb_queue=front_rgb_queue,
        left_shoulder_rgb_queue=left_shoulder_rgb_queue,
        right_shoulder_rgb_queue=right_shoulder_rgb_queue,
        wrist_rgb_queue=wrist_rgb_queue,
        inst_queue=inst_queue,
        goal_queue=goal_queue,
        state_queue=state_queue,
        ep_value=ep_value,
        ds_value=ds_value,
        md_value=md_value,
        tn_value=tn_value,
        step_value=step_value,
        ctrl_value=ctrl_value,
        reset_value=reset_value,
        llava_talk_value=llava_talk_value,
        save_session=save_session,
        save_prefix=save_prefix,
        vlm_address=vlm_address
    )
    gradio_interface.run()

def demo(
    never_terminal=False, 
    save_session=False,
    unseen_task=False,
    save_prefix="debug", 
    lm_address="http://localhost:8000", 
    vlm_address="http://localhost:21002",
    rlbench_dataroot="racer/rlbench_data"
):
    # Initialize the shared values and queues
    vid_queue = Queue()
    front_rgb_queue = Queue()
    left_shoulder_rgb_queue = Queue()
    right_shoulder_rgb_queue = Queue()
    wrist_rgb_queue = Queue()
    inst_queue = Queue()
    goal_queue = Queue()
    state_queue = Queue()
    ep_value = Value('i', 16)
    ds_value = Value('i', 2)
    md_value = Value('i', 0)
    tn_value = Value('i', 13)
    step_value = Value('i', 0)
    ctrl_value = Value('i', 0)
    reset_value = Value('i', 0)
    llava_talk_value = Value('i', 0)

    # Start the gradio interface
    gradio_process = Process(target=run_gradio, args=(
        vid_queue, front_rgb_queue, left_shoulder_rgb_queue, right_shoulder_rgb_queue, wrist_rgb_queue, 
        inst_queue, goal_queue, state_queue, ep_value, ds_value, md_value, tn_value, step_value, ctrl_value,
        reset_value, llava_talk_value, save_session, save_prefix, vlm_address))
    
    # Initialize the agent env
    agent_env = AgentEnv(
        vid_queue=vid_queue,
        front_rgb_queue=front_rgb_queue,
        left_shoulder_rgb_queue=left_shoulder_rgb_queue,
        right_shoulder_rgb_queue=right_shoulder_rgb_queue,
        wrist_rgb_queue=wrist_rgb_queue,
        inst_queue=inst_queue,
        goal_queue=goal_queue,
        state_queue=state_queue,
        ep_value=ep_value,
        ds_value=ds_value,
        md_value=md_value,
        tn_value=tn_value,
        step_value=step_value,
        ctrl_value=ctrl_value,
        reset_value=reset_value,
        llava_talk_value=llava_talk_value,
        model_dict=MODEL_PATH_DICT,
        never_terminal=never_terminal,
        unseen_task=unseen_task,
        lm_address=lm_address,
        rlbench_dataroot=rlbench_dataroot
    )

    gradio_process.start()
    # Start the agent env
    agent_env.run()

    # Wait for the gradio process to finish
    gradio_process.join()


if __name__ == "__main__":
    # never_terminal: no any success criteria, good for task goal change
    # save_session: save the video and related context in the file, good for demo showcase
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--never_terminal", action="store_true")
    parser.add_argument("--save_session", action="store_true")
    parser.add_argument("--unseen_task", action="store_true")
    parser.add_argument("--save_prefix", type=str, default="debug")
    parser.add_argument("--lm-address", type=str, default="http://localhost:8000")    # LM service
    parser.add_argument("--vlm-address", type=str, default="http://localhost:21002")  # Llava VLM service
    parser.add_argument("--rlbench-dataroot", type=str, default="racer/data/rlbench")
    args = parser.parse_args()
    demo(
        never_terminal=args.never_terminal,
        save_session=args.save_session,
        unseen_task=args.unseen_task,
        save_prefix=args.save_prefix,
        lm_address=args.lm_address,
        vlm_address=args.vlm_address,
        rlbench_dataroot=args.rlbench_dataroot
        )
