import base64
from io import BytesIO
import json
import time
import numpy as np
import requests
from racer.evaluation.llava_api.utils import conv_llava_llama_3_racer
from PIL import Image
import requests



DEFAULT_IMAGE_TOKEN = "<image>"

HEADERS = {"User-Agent": "LLaVA Client"}



class LlavaAPI:
    def __init__(self, addr):
        self.addr = addr

    def wrap_image(self, image: Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8') 
        return image_base64

    def wrap_prompt(self, user_msg="What is shown in the legend?"):
        if DEFAULT_IMAGE_TOKEN not in user_msg:
            question = DEFAULT_IMAGE_TOKEN + f"\n{user_msg}"
        else:
            question = user_msg
        conv = conv_llava_llama_3_racer.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt
    

    def get_response(self, user_msg, image):
        if isinstance(image, np.ndarray):
            if image.shape[0] == 3: # (3, 512, 512) -> (512, 512, 3)
                image = image.transpose(1, 2, 0)
            image = Image.fromarray(image)
        pload = {
            "prompt": self.wrap_prompt(user_msg),
            "image": self.wrap_image(image)
        }        
        generated_text = ""
        while True:
            try:
                # Stream output
                response = requests.post(self.addr + "/worker_generate_stream",
                    headers=HEADERS, json=pload, stream=True, timeout=10)
                last_len = len(pload["prompt"])
                for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if chunk:
                        data = json.loads(chunk.decode())
                        if data["error_code"] == 0:
                            output = data["text"][last_len:]
                            generated_text += output
                            last_len = len(data["text"])
                        else:
                            output = data["text"] + f" (error_code: {data['error_code']})"
                break
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
            time.sleep(1)
        return generated_text
    

    def get_response_stream(self, user_msg, image):
        if isinstance(image, np.ndarray):
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            image = Image.fromarray(image)
        pload = {
            "prompt": self.wrap_prompt(user_msg),
            "image": self.wrap_image(image)
        }

        while True:
            try:
                response = requests.post(self.addr + "/worker_generate_stream",
                    headers=HEADERS, json=pload, stream=True, timeout=20)
                last_len = len(pload["prompt"])
                for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if chunk:
                        data = json.loads(chunk.decode())
                        if data["error_code"] == 0:
                            output = data["text"][last_len:]
                            yield output
                            last_len = len(data["text"])
                        else:
                            print(f" (error_code: {data['error_code']})")
                break
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
            time.sleep(0.1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-addr", type=str, default="http://141.212.110.118:21002")
    args = parser.parse_args()
    
    llava_api = LlavaAPI(args.vlm_addr)

    task_goal = "close the lime jar"
    

    TEMP_first_step= "<image>\nThe task goal is: {task_goal}. This is the first step and the robot is about to start the task. Based on the visual observation and the context, how does the robot fulfil that previous instruction and what's the next instruction for the robot arm?"

    TEMP_other_step = "<image>\nThe task goal is: {task_goal}. In the previous step, the robot arm was given the following instruction: \"{previous_instruction}\". Based on the visual observation and the context, how does the robot fulfil that previous instruction and what's the next instruction for the robot arm?"

    user_msg = TEMP_first_step.format(task_goal=task_goal)
    image = Image.open("racer/data/augmented_rlbench/train/close_jar/2/front_rgb/0_expert.png")
    instruction = llava_api.get_response(user_msg, image=image)
    print(instruction) 



    instruction = "Approach the lid by moving towards it and align the gripper above it."

    user_msg = TEMP_other_step.format(task_goal=task_goal, previous_instruction=instruction)
    image = Image.open("racer/data/augmented_rlbench/train/close_jar/2/front_rgb/52_expert.png")
    instruction = llava_api.get_response(user_msg, image=image)
    print(instruction) 