"""
Call model service instead of loading models in memory
"""

import threading
import time
from typing import List, Union
import numpy as np
import requests
import torch

ALL_MODELS = ["clip", "t5-11b"]

MAX_LEN = 77

# Define a class to hold the result and signal when it's ready
class QueryResult:
    def __init__(self):
        self.result = None
        self.event = threading.Event()

    def set_result(self, result):
        self.result = result
        self.event.set()

    def get_result(self):
        self.event.wait()  # Wait until the result is set
        return self.result
    
# Define the function to query the FastAPI service
def query_service(URL, text, model,  query_result):
    def run_query():
        while True:
            response = requests.post(URL, json={"text": text, "model": model})
            if response.status_code == 200:
                query_result.set_result(response.json())
                break
            else:
                print(f"Request failed with status code {response.status_code}")
            time.sleep(0.5)

    # Run the query in a separate thread
    thread = threading.Thread(target=run_query)
    thread.start()
    return thread


class LangModel:
    def __init__(self, model_name, lm_addr, *args, **kwargs):
        self.model_name = model_name
        self.lm_addr = lm_addr

    def make_model(self):
        pass
            
    def unmake_model(self):
        pass

    def _query(self, text):
        while True:
            try:
                if isinstance(text, list):
                    text = text[0]
                response = requests.post(self.lm_addr, json={"text": text, "model": self.model_name})
                if response.status_code == 200:
                    res = response.json()
                    self.is_queried = True
                    self.input_str = text
                    if "error" in res:
                        raise Exception(res["error"])
                    self.postprocess(res)
                    return
            except Exception as e:
                print(e)
                time.sleep(0.1)
        

    def postprocess(self, res):
        embeddings = torch.from_numpy(np.array(res["embeddings"])).float()
        token_len = res["token_len"]
        if token_len < MAX_LEN:
            shape = embeddings.shape
            padding = torch.zeros((shape[0], MAX_LEN - shape[1], shape[2]))
            embeddings = torch.cat([embeddings, padding], dim=1)
        else:
            embeddings = embeddings[:, :MAX_LEN, :]
            token_len = MAX_LEN
        self.embeddings = embeddings
        self.token_len = token_len
    
    
    def encode(self, texts: Union[str, List[str]], to_numpy=True):
        self._query(texts)
        if to_numpy:
            return self.embeddings.numpy(), self.token_len
        else:
            return self.embeddings, self.token_len


class LangModelZoo:
    def __init__(self, lm_addr, *args, **kwargs):
        self.lm_addr = lm_addr
        self.result_list = []
        self.thread_list = []
    
    def make_models(self):
        pass
    
    def unmake_models(self):
        pass

    def _query(self, text_id, text, model):
        query_result = QueryResult()
        self.thread_list.append(query_service(self.lm_addr, text, model, query_result))
        self.result_list.append(
            {"text_id":text_id, 
             "text":text, 
             "model":model, 
             "query_result": query_result
            })

    def send_task(self, text_id:str, text:str):
        for model in ALL_MODELS:
            self._query(text_id, text, model)

    def get_results(self):
        for thread in self.thread_list:
            thread.join()
        ret = [{"text_id":ret["text_id"], 
                 "text" : ret["text"], 
                 "model":ret["model"], 
                 "embeddings":ret["query_result"].get_result()["embeddings"], 
                 "token_len":ret["query_result"].get_result()["token_len"]} 
                for ret in self.result_list]

        self.result_list.clear()
        self.thread_list.clear()

        return ret
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm-addr", type=str, default="http://141.212.110.118:8000/encode/")
    args = parser.parse_args()
    
    lang_model_zoo = LangModelZoo(lm_addr=args.lm_addr)
    lang_model_zoo.make_models()

    task_goal = "task goal: close the jar."
    rich_inst = "task goal: close the jar. \n Current state: Move the gripper slightly back and to the right to position the mug over the next branch of the cup holder."
    simp_inst = "task goal: close the jar. \n Current state: Move to the cup."
    for text, text_id, sz in [
        (task_goal, "task_goal_embs", "task_goal_len"), 
        (rich_inst, "rich_inst_embs", "rich_inst_len"), 
        (simp_inst, "simp_inst_embs", "simp_inst_len")]:
        lang_model_zoo.send_task(text_id=text_id, text=text)
    
    results = lang_model_zoo.get_results()
    for item in results:
        print(item["text_id"], item["text"], item["model"], np.array(item["embeddings"]).shape, item["token_len"])
