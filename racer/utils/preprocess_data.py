import multiprocessing
import os

from racer.utils.generate_replay_buffer import fill_replay
from racer.utils.dataset_aug import create_replay
from racer.rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
)
from racer.utils.lang_enc_utils_v2 import LangModelZoo


def process_task(
    task,
    lm_addr,
    augment_data_folder,
    rlbench_data_folder,
    replay_storage_folder
):
    # transform all datasets: train/val/test
    os.makedirs(replay_storage_folder, exist_ok=True)
    replay_buffer = create_replay(
        batch_size=1,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
        image_size=512
    )

    
    lang_model_zoo = LangModelZoo(lm_addr)


    EPISODES_FOLDER_TRAIN = f"train/{task}"
    EPISODES_FOLDER_VAL = f"val/{task}"
    data_path_train = os.path.join(augment_data_folder, EPISODES_FOLDER_TRAIN)
    data_path_val = os.path.join(augment_data_folder, EPISODES_FOLDER_VAL)

    reference_data_path_train = os.path.join(rlbench_data_folder, f"train/{task}/all_variations/episodes")
    reference_data_path_val = os.path.join(rlbench_data_folder, f"val/{task}/all_variations/episodes")


    # print("----- Train Buffer -----")
    fill_replay(
        replay=replay_buffer,
        task=task,
        task_replay_storage_folder=os.path.join(replay_storage_folder, task),
        rlbench_scene_bounds=SCENE_BOUNDS,
        voxel_sizes=VOXEL_SIZES,
        rotation_resolution=ROTATION_RESOLUTION,
        crop_augmentation=False,
        data_path_train=data_path_train,
        reference_data_path_train=reference_data_path_train,
        data_path_val=data_path_val,
        reference_data_path_val=reference_data_path_val,
        lang_model_zoo=lang_model_zoo)


if __name__ == "__main__":
    import argparse
    from racer.utils.racer_utils import RLBENCH_TASKS
    parser = argparse.ArgumentParser()  
    parser.add_argument("--lm_addr", type=str, default="http://141.212.110.118:8000/encode/")
    parser.add_argument("--augment_data_folder", type=str, default="racer/data/augmented_rlbench")
    parser.add_argument("--rlbench_data_folder", type=str, default="racer/data/rlbench")
    parser.add_argument("--replay_storage_folder", type=str, default="racer/replay_buffers/racer_replay_generated")
    args = parser.parse_args()
    
    with multiprocessing.Pool(processes=8) as pool:
        pool.starmap(
            process_task,
            [(
                task, 
                args.lm_addr, 
                args.augment_data_folder, 
                args.rlbench_data_folder, 
                args.replay_storage_folder
            ) for task in RLBENCH_TASKS]
        )