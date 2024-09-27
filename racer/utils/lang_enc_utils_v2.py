"""
Call service instead of loading models in memory
"""

import threading
import time
from typing import Dict, List, Union
import numpy as np
import requests
import torch
from transformers import AutoTokenizer, T5EncoderModel, BartModel, RobertaModel
import clip
import os

# whistler 141.212.106.177
# aspen 141.212.110.118
URL = "http://141.212.106.177:8000/encode/"



ALL_MODELS = [
    "clip", 
        # "t5-3b", 
        "t5-11b", 
        # "bart-large", "roberta-large", 
        # "llama3"
        ]

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
def query_service(text, model,  query_result):
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
    def __init__(self, model_name="t5-3b", *args, **kwargs):
        self.model_name = model_name

    def make_model(self):
        pass
            
    def unmake_model(self):
        pass

    def _query(self, text):
        while True:
            try:
                if isinstance(text, list):
                    text = text[0]
                response = requests.post(URL, json={"text": text, "model": self.model_name})
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
    def __init__(self, *args, **kwargs):
        self.result_list = []
        self.thread_list = []
    
    def make_models(self):
        pass
    
    def unmake_models(self):
        pass

    def _query(self, text_id, text, model):
        query_result = QueryResult()
        self.thread_list.append(query_service(text, model, query_result))
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
    lang_model_zoo = LangModelZoo()
    lang_model_zoo.make_models()

    high_level_task_goal = "task goal: close the jar."
    fine_grained_gpt_lang = "task goal: close the jar. \n Current state: the jar is open.  the jar is open.  the jar is open.  the jar is open.  the jar is open.  the jar is open.  the jar is open.  the jar is open.  the jar is open.the jar is open.  the jar is open.  the jar is open.  the jar is open.  the jar is open.  the jar is open.  the jar is open.  the jar is open.  the jar is open."
    fine_grained_heuristic_lang = fine_grained_gpt_lang + " \n asdafsdf asdf"
    for text, text_id, sz in [
        (high_level_task_goal, "lang_goal_embs", "lang_len"), 
        (fine_grained_gpt_lang, "fine_gpt_lang_goal_embs", "fine_gpt_lang_len"), 
        (fine_grained_heuristic_lang, "fine_heuristic_lang_goal_embs", "fine_heuristic_lang_len")]:
        lang_model_zoo.send_task(text_id=text_id, text=text)
    
    # time.sleep(4)
    obs_dict = {}
    results = lang_model_zoo.get_results()
    for res in results:
        lang_model_name = res["model"]
        text_id = res["text_id"]
        obs_dict["%s_%s" % (text_id, lang_model_name)] = np.array(res["embeddings"][0], dtype=np.float32)
        obs_dict["%s_%s" % (text_id.replace("goal_embs", "len"), lang_model_name)] = np.array([res["token_len"]], dtype=np.int32)

        
    
    # for k, v in obs_dict.items():
    #     if "len" in k:
    #         print(k, v)
    #     else:
    #         print(k, v.shape, v[:3, :2])

        

    
    




# [[ 0.13851969]
#  [ 0.13851945]
#  [-0.6225466 ]
#  [-0.49029982]
#  [-0.20987208]
#  [ 0.7380806 ]
#  [-0.33663288]
#  [ 0.23175196]
#  [ 0.15370028]
#  [ 0.        ]]