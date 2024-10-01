import logging
from datetime import datetime
import os
from racer.utils.racer_utils import RLBENCH_TASKS


# Get the current date and time for the log file name
current_date = datetime.now().strftime("%m-%d_%H-%M-%S")


log_dir = "racer/gradio_demo/logs"
os.makedirs(log_dir, exist_ok=True)

# Create the log file name with the current date
log_file_name = f"racer/gradio_demo/logs/log_{current_date}.log"

# Configure the logging
logging.basicConfig(
    filename=log_file_name,
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger
logger = logging.getLogger('my_logger')

DATASET = ["train", "val", "test"]
MODEL_PATH_DICT = {
    # name: (path, cuda_device)
    "RVT": ("racer/runs/rvt_ckpt/model_14.pth", 0), # original rvt model
    "RACER": ("racer/runs/racer_ckpt/model_17.pth", 1), # our racer model
}
MODELS = list(MODEL_PATH_DICT.keys())
LLAVA_TALK_CHOICE = ["close llava", "open llava"]