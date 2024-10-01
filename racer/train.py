import argparse
from collections import defaultdict
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from racer.trainer import DDPTrainer
import tqdm

from racer.rvt.train import (
    save_agent, get_tasks, 
    get_logdir, dump_log, exp_cfg_mod, 
    mvt_cfg_mod,
    get_num_feat, rvt_agent, SCENE_BOUNDS, CAMERAS, 
    load_agent, TensorboardManager)

from rvt.mvt.mvt_v2 import MVT
from rvt.models.rvt_agent import print_loss_log
from racer.utils.get_dataset_aug import get_dataset_v2 as get_dataset_aug
from racer.utils.racer_utils import print0

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
        parser.add_argument("--data-image-size", type=int, default=512)    

        
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
        
        TRAINING_ITERATIONS = 125384 // (exp_cfg.bs * world_size) # TODO: to be determined. 
        EPOCHS = exp_cfg.epochs
        if exp_cfg.peract.warmup_proportion > 0:
            exp_cfg.peract.warmup_steps = int(TRAINING_ITERATIONS * exp_cfg.peract.warmup_proportion * EPOCHS)
        if world_rank == 0:
            print0(f"dict(exp_cfg)={dict(exp_cfg)}")
        exp_cfg.freeze()
        
        log_dir = get_logdir(cmd_args, exp_cfg)
        tasks = get_tasks(exp_cfg) # list of task name
        print0("Training on {} tasks: {}".format(len(tasks), tasks))

        t_start = time.time()
        get_dataset_func_aug = lambda: get_dataset_aug(
            tasks,
            BATCH_SIZE_TRAIN,
            cmd_args.replay_dir_aug,
            num_workers=exp_cfg.num_workers,
            sample_distribution_mode=exp_cfg.sample_distribution_mode,
            image_size=cmd_args.data_image_size,
        )
        train_dataset_aug, _, _ = get_dataset_func_aug()
        train_dataset_iter_aug = iter(train_dataset_aug)
        t_end = time.time()
        print0("Created Augment Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))


        if exp_cfg.agent == "our":
            mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
            if cmd_args.mvt_cfg_path != "":
                mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
            if cmd_args.mvt_cfg_opts != "":
                mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

            mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
            mvt_cfg.freeze()
            rvt = MVT(
                lang_model_name=exp_cfg.lang_model_name,
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
                lang_model_name=exp_cfg.lang_model_name,
                image_size=exp_cfg.image_size,
                lang_level=exp_cfg.lang_level,
                **exp_cfg.peract,
                **exp_cfg.rvt,
            )
            agent.build(training=True, device=device)
        else:
            assert False, "Incorrect agent"
            
        
        start_epoch = 0
        end_epoch = EPOCHS
        if exp_cfg.resume != "":
            print0(f"Recovering model and checkpoint from {exp_cfg.resume}")
            load_agent(exp_cfg.resume, agent)
        dist.barrier()

        if world_rank == 0:
            dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
            tb = TensorboardManager(log_dir)

        print0("Start training ...", flush=True)
        i = start_epoch
        while True:
            if i == end_epoch:
                break
            print0(f"Rank [{world_rank}], Epoch [{i}]: Training on train dataset")
            out = train(agent, train_dataset_iter_aug, TRAINING_ITERATIONS, world_rank)

            if world_rank == 0:
                tb.update("train", i, out)
                if i > EPOCHS-5:
                    print0(f"Saving model {log_dir}/model_{i}.pth ...")
                    save_agent(agent, f"{log_dir}/model_{i}.pth", i)
                    model_abspath = os.path.abspath(f"{log_dir}/model_{i}.pth")
                    os.system(f"ln -sf {model_abspath} {log_dir}/model_last.pth")
                    print0("Model saved.")
            i += 1

        if world_rank == 0:
            tb.close()
            print0("[Finish]")



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
        batch["task_goal"] = raw_batch["task_goal"]
        batch["rich_inst"] = raw_batch["rich_inst"]
        batch["simp_inst"] = raw_batch["simp_inst"]
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