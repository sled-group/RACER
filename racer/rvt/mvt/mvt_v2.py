"""
Add Lang Encoder & failure head
"""

from torch import nn

from racer.rvt.mvt.mvt import MVT as MVTBase
from racer.rvt.mvt.attn import DenseBlock


class MVT(MVTBase):
    def __init__(
        self, 
        lang_model_name="t5-3b",
        add_failure_head=False,
        failure_head_dim=4,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        add_strong_lang=True if lang_model_name != "clip" else False
        if add_strong_lang:
            if lang_model_name == "llama3":
                self.lang_preprocess_raw = DenseBlock(
                    4096, 
                    kwargs["lang_dim"],
                    norm=None,
                    activation=kwargs["activation"],
                    use_dropout=kwargs["use_dropout"] if "use_dropout" in kwargs else False,
                )
            else:
                self.lang_preprocess_raw = DenseBlock(
                    1024,
                    kwargs["lang_dim"],
                    norm=None,
                    activation=kwargs["activation"],
                    use_dropout=kwargs["use_dropout"] if "use_dropout" in kwargs else False,
                )
        
        if add_failure_head:            
            self.failure_head_fc = nn.Sequential(
                nn.Linear(960, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, failure_head_dim),
            )
        
        self.lang_model_name = lang_model_name
        self.add_strong_lang = add_strong_lang
        self.add_failure_head = add_failure_head
        self.failure_head_dim = failure_head_dim
    
    def forward(
        self,
        pc,
        img_feat,
        proprio=None,
        lang_emb=None,
        lang_len=None,
        img_aug=0,
        **kwargs,
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape
            (bs, num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, max_lang_len, lang_dim)
        :param lang_len: (int) length of true lang len for each sample
        :param img_aug: (float) magnitude of augmentation in rgb image
        """
        if self.add_strong_lang:
            lang_emb_feat_dim = lang_emb.shape[-1]
            assert lang_emb_feat_dim in [1024, 4096]
            lang_emb = self.lang_preprocess_raw(lang_emb)
        
        self.verify_inp(pc, img_feat, proprio, lang_emb, img_aug)
        img = self.render(
            pc, # list of  torch.Size([45706, 3])
            img_feat, # list of torch.Size([45706, 3])
            img_aug,
            dyn_cam_info=None,
        )
        out = self.mvt1(img=img, proprio=proprio, lang_emb=lang_emb, lang_len=lang_len, **kwargs)
        
        if self.add_failure_head:
            out["failure_head"] = self.failure_head_fc(out["raw_feat"])
        return out
            
            