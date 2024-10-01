import torch

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
SCENE_BOUNDS = [
    -0.3,
    -0.5,
    0.6,
    0.7,
    0.5,
    1.6,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized

RLBENCH_TASKS = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
    # ##### unseen tasks #####
    # "slide_block_to_target", "close_drawer", "reach_target", "pick_up_cup", 
]


def load_agent(agent_path, agent):
    checkpoint = torch.load(agent_path, map_location="cpu")
    if hasattr(agent, "_q"):
        model = agent._q
    elif hasattr(agent, "_network"):
        model = agent._network

    if isinstance(model, DDP):
        model = model.module
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError:
        try:
            print(
                "WARNING: loading states in mvt1. "
                "Be cautious if you are using a two stage network."
            )
            model.mvt1.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            print(
                "WARNING: loading states with strick=False! "
                "KNOW WHAT YOU ARE DOING!!"
            )
            # print common keys
            
            model_state = model.state_dict()
            checkpoint_state = checkpoint["model_state"]
            for k in model_state.keys():
                if k not in checkpoint_state:
                    print(f"Key {k} not found in checkpoint")
            for k in checkpoint_state.keys():
                if k not in model_state:
                    print(f"Key {k} not found in model state")
            model.load_state_dict(checkpoint["model_state"], strict=False)



def print0(*args, **kwargs):
    if dist.is_initialized() and dist.get_rank() == 0:
        print(*args, **kwargs)
    else:
        print(*args, **kwargs)