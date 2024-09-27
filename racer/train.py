import argparse
from collections import defaultdict
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from racer.trainer import DDPTrainer
import tqdm

from rvt.train import (
    save_agent, get_tasks, 
    get_logdir, dump_log, exp_cfg_mod, 
    short_name, get_dataset, DATA_FOLDER, mvt_cfg_mod,
    get_num_feat, rvt_agent, SCENE_BOUNDS, CAMERAS, 
    load_agent, TensorboardManager)

from rvt.mvt.mvt_v2 import MVT
from rvt.models.rvt_agent import print_loss_log

from rvt.utils.get_dataset_aug import get_dataset_v2 as get_dataset_aug

class ExampleTrainer(DDPTrainer):
    
    def make_cmd_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--refresh_replay", action="store_true", default=False)
        parser.add_argument("--device", type=str, default="0")
        parser.add_argument("--mvt_cfg_path", type=str, default="")
        parser.add_argument("--exp_cfg_path", type=str, default="")

        parser.add_argument("--mvt_cfg_opts", type=str, default="")
        parser.add_argument("--exp_cfg_opts", type=str, default="")

        parser.add_argument("--log-dir", type=str, default="runs")
        parser.add_argument("--with-eval", action="store_true", default=False)
        
        # parser.add_argument("--replay-dir", type=str, default="replay")
        parser.add_argument("--replay-dir-aug", type=str, default="replay_aug")
        parser.add_argument("--data-image-size", type=int, default=128)
        
        parser.add_argument("--initial-choice", type=str, choices=["vision", "transformer", "all"], default="all")
        parser.add_argument("--other-lang-path", type=str, default=None)

        
        cmd_args = parser.parse_args()
        return cmd_args
    
    def main(self, cmd_args, local_rank, world_rank, world_size):
        device = self.set_torch_device(local_rank)
        
        exp_cfg = exp_cfg_mod.get_cfg_defaults()
        if cmd_args.exp_cfg_path != "":
            exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
        if cmd_args.exp_cfg_opts != "":
            exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

        # Things to change
        BATCH_SIZE_TRAIN = exp_cfg.bs
        NUM_TRAIN = 100
        # to match peract, iterations per epoch
        # 127570 in total

        TRAINING_ITERATIONS = 125384 // (exp_cfg.bs * world_size) # TODO: to be determined. 
        EPOCHS = exp_cfg.epochs
        if exp_cfg.peract.warmup_proportion > 0:
            exp_cfg.peract.warmup_steps = int(TRAINING_ITERATIONS * exp_cfg.peract.warmup_proportion * EPOCHS)
        if world_rank == 0:
            print(f"dict(exp_cfg)={dict(exp_cfg)}")
        exp_cfg.freeze()
        
        log_dir = get_logdir(cmd_args, exp_cfg)
        tasks = get_tasks(exp_cfg) # list of task name
        print("Training on {} tasks: {}".format(len(tasks), tasks))

        t_start = time.time()
        get_dataset_func_aug = lambda: get_dataset_aug(
            tasks,
            BATCH_SIZE_TRAIN,
            None,
            None,
            f"{cmd_args.replay_dir_aug}/replay_train",
            None,
            None, 
            DATA_FOLDER,
            NUM_TRAIN,
            None,
            None,
            cmd_args.refresh_replay,
            device,
            num_workers=exp_cfg.num_workers,
            only_train=True,
            sample_distribution_mode=exp_cfg.sample_distribution_mode,
            only_data_gen=False,
            image_size=cmd_args.data_image_size,
            other_lang_path=cmd_args.other_lang_path
        )
        train_dataset_aug, _, _ = get_dataset_func_aug()
        train_dataset_iter_aug = iter(train_dataset_aug)
        t_end = time.time()
        print("Created Augment Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))


        if exp_cfg.agent == "our":
            mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
            if cmd_args.mvt_cfg_path != "":
                mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
            if cmd_args.mvt_cfg_opts != "":
                mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

            mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
            mvt_cfg.freeze()
            rvt = MVT(
                lang_model_name=exp_cfg.rvt.lang_model_name,
                add_failure_head=exp_cfg.rvt.add_failure_head,
                failure_head_dim=exp_cfg.rvt.failure_head_dim,
                renderer_device=device,
                **mvt_cfg,
            ).to(device)
            
            rvt = DDP(rvt, device_ids=[local_rank], output_device=local_rank)

            agent = rvt_agent.RVTAgent(
                network=rvt,
                image_resolution=[exp_cfg.image_size, exp_cfg.image_size],
                add_lang=mvt_cfg.add_lang,
                scene_bounds=SCENE_BOUNDS,
                cameras=CAMERAS,
                log_dir=f"{log_dir}/test_run/",
                cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
                image_size=exp_cfg.image_size,
                mix_ratio=exp_cfg.mix_ratio,
                **exp_cfg.peract,
                **exp_cfg.rvt,
            )
            agent.build(training=True, device=device)
        else:
            assert False, "Incorrect agent"
            
        
        start_epoch = 0
        end_epoch = EPOCHS
        if exp_cfg.resume != "":
            agent_path = exp_cfg.resume
            print(f"Recovering model and checkpoint from {exp_cfg.resume}")
            epoch = load_agent(agent_path, agent, only_epoch=False, finetuning=exp_cfg.finetuning, continue_epoch=exp_cfg.continue_epoch, initial_choice=cmd_args.initial_choice)
            if exp_cfg.continue_epoch:
                # continue to train from the last epoch of the loaded model
                start_epoch = epoch + 1
        dist.barrier()

        if world_rank == 0:
            dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
            tb = TensorboardManager(log_dir)

        print("Start training ...", flush=True)
        i = start_epoch
        while True:
            if i == end_epoch:
                break
            print(f"Rank [{world_rank}], Epoch [{i}]: Training on train dataset")
            out = train(agent, train_dataset_iter_aug, TRAINING_ITERATIONS, world_rank)

            if world_rank == 0:
                tb.update("train", i, out)
                if i > EPOCHS-5:
                    print(f"Saving model {log_dir}/model_{i}.pth ...")
                    save_agent(agent, f"{log_dir}/model_{i}.pth", i)
                    model_abspath = os.path.abspath(f"{log_dir}/model_{i}.pth")
                    os.system(f"ln -sf {model_abspath} {log_dir}/model_last.pth")
                    print("Model saved.")
            i += 1

        if world_rank == 0:
            tb.close()
            print("[Finish]")



# new train takes the dataset as input
def train(agent, data_iter_aug, training_iterations, rank=0):
    agent.train()
    log = defaultdict(list)

    iter_command = range(training_iterations)

    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):
        raw_batch = next(data_iter_aug)
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }          

        # # plot all the image
        # front_rgb = batch["front_rgb"] # [batch_size, 1, 3, 128, 128]
        # front_rgb = front_rgb.squeeze(1)
        # front_rgb = front_rgb.permute(0, 2, 3, 1)
        # front_rgb = front_rgb.int().cpu().numpy()
        # # turn to int8
        # front_rgb = front_rgb.astype('uint8')

        # # plot all the image in one polot
        # fig, axs = plt.subplots(1, front_rgb.shape[0], figsize=(20, 5))
        # for i in range(front_rgb.shape[0]):
        #     axs[i].imshow(front_rgb[i])
        #     axs[i].axis('off')
        # plt.savefig(f"front_rgb_runn.png")
        
        batch["tasks"] = raw_batch["tasks"]
        batch["demo"] = raw_batch["demo"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        update_args = {
            "step": iteration,
        }
        update_args.update(
            {
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
                "eval_log": False,
            }
        )
        agent.update(**update_args)

    if rank == 0:
        log = print_loss_log(agent)

    return log



if __name__ == "__main__":
    trainer = ExampleTrainer()
    trainer.ddp_run()