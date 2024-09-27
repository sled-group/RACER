import torch

CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]

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


def load_agent(agent_path, agent=None, only_epoch=False, finetuning=False, evaluation=False, continue_epoch=False, initial_choice="all"):
    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if finetuning:
        new_model_state = {}
        for k in checkpoint["model_state"].keys():
            if initial_choice == "all":
                if any([x in k for x in ["lang_preprocess"]]):
                    continue
            else:
                pass
            # if initial_choice == "vision":
            #     if any([x in k for x in ["trans_decoder", "feat_fc", "lang_preprocess", "final", 
            #                              "fc_aft_attn", "layers.4", "layers.5", "layers.6", "layers.7", 
            #                              "up0.conv_up"]]):
            #         continue
            # elif initial_choice == "transformer":
            #     if any([x in k for x in ["trans_decoder", "feat_fc", "lang_preprocess", "final"]]):
            #         continue
            # else:
            #     # if any([x in k for x in ["lang_preprocess"]]):
            #     #     continue
            #     pass
            new_model_state[k] = checkpoint["model_state"][k]
        checkpoint["model_state"] = new_model_state

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network
        optimizer = agent._optimizer
        lr_sched = agent._lr_sched

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

        if "optimizer_state" in checkpoint and continue_epoch and not evaluation:
            print("continue train, loading optimizer state ... ")
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            print(
                "WARNING: No optimizer_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

        if "lr_sched_state" in checkpoint and continue_epoch and not evaluation:
            print("continue train, loading lr_sched state ... ")
            lr_sched.load_state_dict(checkpoint["lr_sched_state"])
        else:
            print(
                "WARNING: No lr_sched_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

    return epoch
