import argparse
import collections
import json
import random
import numpy as np
import pandas as pd
import os


RLBENCH_TASKS_RVT_Order = {
    "close_jar": "Close Jar",
    "reach_and_drag": "Drag Stick",
    "insert_onto_square_peg": "Insert Peg",
    "meat_off_grill": "Meat off Grill",
    "open_drawer": "Open Drawer",
    "place_cups": "Place Cups",
    "place_wine_at_rack_location": "Place Wine",
    "push_buttons": "Push Buttons",
    "put_groceries_in_cupboard": "Put in Cupboard",
    "put_item_in_drawer": "Put in Drawer",
    "put_money_in_safe": "Put in Safe",
    "light_bulb_in": "Screw Bulb",
    "slide_block_to_color_target": "Slide Block",
    "place_shape_in_shape_sorter": "Sort Shape",
    "stack_blocks": "Stack Blocks",
    "stack_cups": "Stack Cups",
    "sweep_to_dustpan_of_size": "Sweep to Dustpan",
    "turn_tap": "Turn Tap",
}


def main_old(base_dir):
    print(f"Evaluate {base_dir}")
    # List to store the dataframes
    dfs = []

    # Loop through each seed directory within the base directory
    for seed_dir in os.listdir(base_dir):
        # Construct the path to the eval_results.csv file
        csv_path = os.path.join(base_dir, seed_dir, 'eval_results.csv')
        # Check if the file exists to avoid any error
        
        if os.path.isfile(csv_path):
            # Load the CSV file
            df = pd.read_csv(csv_path)
            # Assuming the first column is 'task' and the second is 'success rate', append the dataframe
            dfs.append(df[['task', 'success rate']])
            
            # check if all tasks has been evaluated, and if not, print the missing tasks
            tasks = df['task'].values
            missing_tasks = [task for task in RLBENCH_TASKS_RVT_Order if task not in tasks]
            if missing_tasks:
                print(f"Missing tasks in {seed_dir}: {missing_tasks}")
        
    success_rates = np.array([df['success rate'].values for df in dfs])
    mean_success_rates = np.mean(success_rates, axis=0)
    variance_success_rates = np.std(success_rates, axis=0)
    tasks = dfs[0]['task'].values

    # results_df = pd.DataFrame({
    #     'Task': tasks,
    #     'Mean': mean_success_rates,
    #     'Std': variance_success_rates
    # })
    
    print(f"{'Task':20} Success Rate")
    for task, task_short_name in RLBENCH_TASKS_RVT_Order.items():
        idx = np.where(tasks == task)[0][0]
        print(f"{task_short_name:20} {mean_success_rates[idx]:.2f} ± {variance_success_rates[idx]:.2f}")
    
    idx = np.where(tasks == "average")[0][0]
    print("-" * 20)
    print(f"{'Average':20} {mean_success_rates[idx]:.2f} ± {variance_success_rates[idx]:.2f}")


    for task, task_short_name in RLBENCH_TASKS_RVT_Order.items():
        idx = np.where(tasks == task)[0][0]
        print(f"{mean_success_rates[idx]:.2f} ± {variance_success_rates[idx]:.2f}")


def main(base_dir):
    print(f"Evaluate {base_dir}")
    res = collections.defaultdict(list)
    metric_merge_dict = {}

    for seed_dir in os.listdir(base_dir):
        # Construct the path to the eval_results.csv file
        json_path = os.path.join(base_dir, seed_dir, 'metrics.json')
        if not os.path.isfile(json_path):
            continue
        # if "seed1" in seed_dir or "seed2" in seed_dir:
        #     continue
    
        with open(json_path) as f:
            data = json.load(f)

        for task, task_res in data.items():
            if task not in metric_merge_dict:
                metric_merge_dict[task] = {}
            for ep, ep_res in task_res.items():
                if ep not in metric_merge_dict[task]:
                    metric_merge_dict[task][ep] = {}
                if isinstance(ep_res, float) or isinstance(ep_res, int):
                    metric_merge_dict[task][ep][seed_dir] = ep_res
                else:
                    metric_merge_dict[task][ep][seed_dir] = json.dumps(ep_res)
        
        

        
        overall = data['overall']
        for k, v in overall.items():
            res[k].append(v)
    
    for k, v in RLBENCH_TASKS_RVT_Order.items():
        print(f"{v:20} {100*np.mean(res[k]):.2f} ± {100*np.std(res[k]):.2f}")

    print("-" * 20)
    print(f"{'Average':20} {100*np.mean(res['avg_success_rate']):.2f} ± {100*np.std(res['avg_success_rate']):.2f}")

    for k, v in RLBENCH_TASKS_RVT_Order.items():
        print(f"{100*np.mean(res[k]):.2f} ± {100*np.std(res[k]):.2f}")
    print(f"{100*np.mean(res['avg_success_rate']):.2f} ± {100*np.std(res['avg_success_rate']):.2f}")
        
        
    with open(os.path.join(base_dir, 'metrics_merge_allseed.json'), 'w') as f:
        json.dump(metric_merge_dict, f, indent=2)


def main_human(*args):
    human_intervenen_dir="/home/daiyp/manipulation/RVT/rvt/gradio_demo/sessions/debug/InstructRVT-nomix-t5-continue/test"
    jsonpath = "/home/daiyp/manipulation/RVT/rvt/runs/debug-train_t5-11b_augonly256_bs48_ep18_mix0_ft-transformer_continuedatav2/llava_talk/model_5/metrics_merge.json"
    with open(jsonpath) as f:
        data = json.load(f)
    overall = dict()
    for task in RLBENCH_TASKS_RVT_Order:
        overall[task] = {}
        for ep in range(25):
            path = os.path.join(human_intervenen_dir, task, f"ep{ep}")
            json_dic = data[task][str(ep)]
            json_dic = {k: json.loads(v) for k, v in json_dic.items()}
            if os.path.exists(path):
                res = os.listdir(path)
                number_success = sum([1 for r in res if "success" in r])
                failed_seed = [k for k, v in json_dic.items() if v["success"] == False]
                # random pick, assign to success
                if number_success > len(failed_seed):
                    number_success = len(failed_seed)
                failed_seed_random = np.random.choice(failed_seed, number_success, replace=False)
                for seed in failed_seed_random:
                    json_dic[seed]["success"] = True
                # json_dic = {k: json.dumps(v) for k, v in json_dic.items()}
        
            for seed, item in json_dic.items():
                if seed not in overall[task]:
                    overall[task][seed] = []
                overall[task][seed].append(item["success"])
    
    for task in overall:
        for seed in overall[task]:
            overall[task][seed] = sum(overall[task][seed]) / len(overall[task][seed])
    
    avg_success_rate = {}
    for t, v in overall.items():
        for seed, success in v.items():
            if seed not in avg_success_rate:
                avg_success_rate[seed] = []
            avg_success_rate[seed].append(success)
    for k, v in avg_success_rate.items():
        avg_success_rate[k] = np.mean(v)
    overall["avg_success_rate"] = avg_success_rate

    with open(os.path.join(args.logdir, 'metrics_merge_human.json'), 'w') as f:
        json.dump(overall, f, indent=2)            
    

    for k, v in RLBENCH_TASKS_RVT_Order.items():
        res = list(overall[k].values())
        print(f"{v:20} {100*np.mean(res):.2f} ± {100*np.std(res):.2f}")

    print("-" * 20)
    res = list(overall["avg_success_rate"].values())
    print(f"{'Average':20} {100*np.mean(res):.2f} ± {100*np.std(res):.2f}")
    


    for k, v in RLBENCH_TASKS_RVT_Order.items():
        res = list(overall[k].values())
        print(f"{100*np.mean(res):.2f} ± {100*np.std(res):.2f}")

    print("-" * 20)
    res = list(overall["avg_success_rate"].values())
    print(f"{100*np.mean(res):.2f} ± {100*np.std(res):.2f}")



def main_task_change(*args):
    # tasks = ["close_jar", "push_buttons", "open_drawer", "light_bulb_in"]
    tasks = ["close_drawer", "slide_block_to_target", "reach_target", "pick_up_cup"]
    dirpath = "/home/daiyp/manipulation/RVT/rvt/gradio_demo/sessions/unseen"
    models = ["InstructRVT-nomix-t5-continue", "origin_rvt"]
    result = {}
    summary = {}
    summary["all"] = {m: [] for m in models}
    for task in tasks:
        result[task] = {}
        summary[task] = {m: [] for m in models}
        for model in models:
            for ep in range(25):
                if str(ep) not in result[task]:
                    result[task][str(ep)] = {}
                path = os.path.join(dirpath, model, "test", task, "ep" + str(ep))
                if any(["success" in file for file in os.listdir(path)]):
                    result[task][str(ep)][model] = True
                    summary[task][model].append(True)
                    summary["all"][model].append(True)
                else:
                    result[task][str(ep)][model] = False
                    summary[task][model].append(False)
                    summary["all"][model].append(False)
    
    # summary for all tasks
    for k, v in summary.items():
        for m, res in v.items():
            summary[k][m] = sum(res)
    
    result["overall"] = summary

    with open(os.path.join(dirpath, 'metrics_merge_unseen.json'), 'w') as f:
        json.dump(result, f, indent=2)



def human_interneve_ratio():
    human_intervenen_dir="/home/daiyp/manipulation/RVT/rvt/gradio_demo/sessions/human-intervene/InstructRVT-nomix-t5-continue/test"
    total_len, human_involve_len = 0, 0
    avg_turn_ratio = []
    for task in RLBENCH_TASKS_RVT_Order:
        avg_turn_ratio_task = []
        for ep in range(25):
            path = os.path.join(human_intervenen_dir, task, f"ep{ep}")
            if os.path.exists(path):
                files = os.listdir(path)
                for file in files:
                    if "success" in file:
                        turn_ratio = []
                        json_path = os.path.join(path, file, "context.json")
                        with open(json_path) as f:
                            data = json.load(f)
                            for turn in data["context"][:-1]:
                                if turn["instruction_role"] == "human":
                                    human_involve_len += 1
                                    turn_ratio.append(1)
                                else:
                                    turn_ratio.append(0)
                                total_len += 1
                        if len(turn_ratio) == 0:
                            avg_turn_ratio.append(0)
                            avg_turn_ratio_task.append(0)
                        else:
                            avg_turn_ratio.append(sum(turn_ratio) / len(turn_ratio))
                            avg_turn_ratio_task.append(sum(turn_ratio) / len(turn_ratio))
        print(f"Task: {task}, Average human turn ratio: {np.mean(avg_turn_ratio_task):.2f}")
                    
    print(f"Human involve ratio: {human_involve_len / total_len:.2f}")
    print(f"Average human turn ratio: {np.mean(avg_turn_ratio):.2f}")

                
    # Task: close_jar, Average human turn ratio: 0.19
    # Task: reach_and_drag, Average human turn ratio: 0.00
    # Task: insert_onto_square_peg, Average human turn ratio: 0.34
    # Task: meat_off_grill, Average human turn ratio: 0.22
    # Task: open_drawer, Average human turn ratio: 0
    # Task: place_cups, Average human turn ratio: 0
    # Task: place_wine_at_rack_location, Average human turn ratio: 0.25
    # Task: push_buttons, Average human turn ratio: 0
    # Task: put_groceries_in_cupboard, Average human turn ratio: 0.26
    # Task: put_item_in_drawer, Average human turn ratio: 0
    # Task: put_money_in_safe, Average human turn ratio: 0.25
    # Task: light_bulb_in, Average human turn ratio: 0.17
    # Task: slide_block_to_color_target, Average human turn ratio: 0.58
    # Task: place_shape_in_shape_sorter, Average human turn ratio: 0.29
    # Task: stack_blocks, Average human turn ratio: 0.23
    # Task: stack_cups, Average human turn ratio: 0.26
    # Task: sweep_to_dustpan_of_size, Average human turn ratio: 0.25
    # Task: turn_tap, Average human turn ratio: 0.18
    # Human involve ratio: 0.25
    # Average human turn ratio: 0.24


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="/home/daiyp/manipulation/RVT/rvt/runs/turbo_unrep_runs/train_t5-11b_augonly256_bs48_ep18_mix0_ft-transformer_continuedatav2/neweval/llava_level3")
    args = parser.parse_args()
    main(args.logdir)
    # main_task_change(args.logdir)
    # human_interneve_ratio()