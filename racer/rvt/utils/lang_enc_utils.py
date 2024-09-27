import time
from typing import Dict, List, Union
import torch
from transformers import AutoTokenizer, T5EncoderModel, BartModel, RobertaModel
import clip
import os


MODEL_PATHS = {
    "t5-3b": os.path.join(os.environ["LM_PATH"], "t5-3b"),
    "t5-11b": os.path.join(os.environ["LM_PATH"], "t5-11b"),
    "bart-large": os.path.join(os.environ["LM_PATH"], "bart-large"),
    "roberta-large": os.path.join(os.environ["LM_PATH"], "roberta-large"),
}

ALL_MODELS = list(MODEL_PATHS.keys()) + ["clip"]
MAX_LEN = 77

# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    return x


class LangModel:
    def __init__(self, model_name="t5-3b", device="cuda:0"):
        self.model_name = model_name
        self.device = device
        self.make_tokenizer()
    
    def make_tokenizer(self):
        if "clip" in self.model_name:
            pass
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[self.model_name])
    
    def make_model(self):
        tstart = time.time()
        print(f"Loading language model {self.model_name} ...")
        if "clip" in self.model_name:
            model, _ = clip.load("RN50", device=self.device)
            model = model.to(self.device)
        else:
            model_path = MODEL_PATHS[self.model_name]
            if "t5" in self.model_name:
                model = T5EncoderModel.from_pretrained(model_path, device_map='cuda')
            elif "bart" in self.model_name:
                model = BartModel.from_pretrained(model_path, device_map='cuda')
            elif "roberta" in self.model_name:
                model = RobertaModel.from_pretrained(model_path, device_map='cuda')
            else:
                raise ValueError(f"model_name={self.model_name} not supported")
        model.eval()
        print(f"Loaded language model in {time.time() - tstart:.2f}s")
        self.model = model
    
            
    def unmake_model(self):
        del self.model
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
    
    def tokenize(self, texts: Union[str, List[str]]):
        # return tokens and length
        if "clip" in self.model_name:
            token_tensor = clip.tokenize(texts, context_length=200)
            token_len = token_tensor.numpy()[0].tolist().index(49407) + 1
        else:
            token_tensor = self.tokenizer(texts, return_tensors='pt', padding=True).input_ids
            token_len = token_tensor.shape[1]
        
        token_tensor = token_tensor[:, :MAX_LEN]
        if token_len > MAX_LEN: token_len = MAX_LEN 

        return token_tensor, token_len
    

    def transformer_encode(self, texts: Union[str, List[str]], to_numpy=True, padding_length=MAX_LEN):    
        input_ids = self.tokenizer(texts, return_tensors='pt', padding=True)
        input_ids = input_ids.input_ids.to(self.device)

        with torch.no_grad():
            embeddings = self.model(input_ids=input_ids).last_hidden_state

        if embeddings.shape[1] < padding_length:
            shape = embeddings.shape
            padding = torch.zeros((shape[0], padding_length - shape[1], shape[2]), device=self.device)
            embeddings = torch.cat([embeddings, padding], dim=1)
        elif embeddings.shape[1] > padding_length:
            embeddings = embeddings[:, :padding_length, :]
        
        if to_numpy:
            return embeddings.float().detach().cpu().numpy()
        else:
            return embeddings.float()

    def clip_encode(self, texts: Union[str, List[str]], to_numpy=True, padding_length=MAX_LEN):
        token_tensor, _ = self.tokenize(texts)
        token_tensor = token_tensor.to(self.device)
        with torch.no_grad():
            lang_embs = _clip_encode_text(self.model, token_tensor)
        if to_numpy:
            return lang_embs.float().detach().cpu().numpy()
        else:
            return lang_embs.float()

    def encode(self, texts: Union[str, List[str]], to_numpy=True, padding_length=MAX_LEN):
        if "clip" in self.model_name:
            return self.clip_encode(texts, to_numpy, padding_length)
        else:
            return self.transformer_encode(texts, to_numpy, padding_length)


class LangModelZoo:
    def __init__(self, device) -> None:
        self.model_dict: Dict[str, LangModel] = {}
        self.device = device
    
    def make_models(self):
        for model_name in MODEL_PATHS:
            lang_enc = LangModel(model_name, self.device)
            lang_enc.make_model()
            self.model_dict[model_name] = lang_enc
            
        lang_enc = LangModel("clip", self.device)
        lang_enc.make_model()
        self.model_dict["clip"] = lang_enc
    
    def unmake_models(self):
        for lang_enc in self.model_dict.values():
            lang_enc.unmake_model()
        self.model_dict.clear()
        
    def encode(self, texts: Union[str, List[str]], to_numpy=True, padding_length=77):
        encodings = {}
        for model_name, lang_enc in self.model_dict.items():
            encodings[model_name] = lang_enc.encode(texts, to_numpy, padding_length)
        return encodings

    def token_len(self, texts: Union[str, List[str]]):
        token_len = {}
        for model_name, lang_enc in self.model_dict.items():
            token_len[model_name] = lang_enc.tokenize(texts)[1]
        return token_len
        

if __name__ == "__main__":
    string1 = "one"
    string2 = "task goal: close the jar. current instruction: Position the gripper directly above the lid by moving it slightly right and significantly forward from its current position, then lower it until it is just above the lid. Make sure the gripper remains open and avoid any collisions.  Position the gripper directly above the lid by moving it slightly right and significantly forward from its current position, then lower it until it is just above the lid. Make sure the gripper remains open and avoid any collisions."

    string3 = "task goal: close the jar. current instruction: Position the gripper directly above the lid by moving it slightly right and significantly forward from its current position, then lower it until it is just above the lid. Make sure the gripper remains open and avoid any collisions."
    
    lang_model_zoo = LangModelZoo("cuda:0")
    lang_model_zoo.make_models()
    for s in [string1, string2, string3]:
        encodings = lang_model_zoo.encode(s)
        for model_name, encoding in encodings.items():
            print(model_name, encoding.shape)
        token_len = lang_model_zoo.token_len(s)
        for model_name, length in token_len.items():
            print(model_name, length)

    lang_model_zoo.unmake_models()

    
    




