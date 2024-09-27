"""
set basic class for distributed training program
handling GPU communication
"""

import os
import socket
import torch.multiprocessing as mp
import torch.distributed as dist

import torch

class DDPTrainer:
    
    def __init__(self):
        self.SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
    
    @property
    def tag(self) -> str:
        pid = os.getpid()
        hostname = socket.gethostname()
        return hostname + "-" + str(pid)
    
    @property
    def is_slurm_job(self) -> bool:
        return self.SLURM_JOBID is not None
    
    @property
    def is_slurm_hetero(self) -> bool:
        return "NODE_GPU_COUNTS" in os.environ
    
    def make_cmd_args(self):
        raise NotImplementedError()
    
    def make_env_args(self):        
        # Default port to initialized the TCP store on
        DEFAULT_PORT = 8739 # (random.randint(0, 3000) % 3000) + 27000
        DEFAULT_MAIN_ADDR = "127.0.0.1" # Default address of world rank 0
        DEFAULT_PORT_RANGE = 127
        main_port = int(os.environ.get("MAIN_PORT", DEFAULT_PORT))
        if self.is_slurm_job:
            main_port += int(self.SLURM_JOBID) % int(
                os.environ.get("MAIN_PORT_RANGE", DEFAULT_PORT_RANGE)
            )
        main_addr = os.environ.get("MAIN_ADDR", DEFAULT_MAIN_ADDR)
        
        local_rank = int(os.environ["SLURM_LOCALID"]) if "SLURM_LOCALID" in os.environ else int(os.environ["LOCAL_RANK"])
        world_rank = int(os.environ["SLURM_PROCID"]) if "SLURM_PROCID" in os.environ else int(os.environ["RANK"])
        world_size = int(os.environ["SLURM_NTASKS"]) if "SLURM_NTASKS" in os.environ else int(os.environ["WORLD_SIZE"])
        node_rank = int(os.environ['SLURM_NODEID']) if "SLURM_NODEID" in os.environ else 0
        env_args = {
            "world_size": world_size,
            "local_rank": local_rank,
            "world_rank": world_rank,
            "node_rank": node_rank,
            "main_port": main_port,
            "main_addr": main_addr,
        }
        if self.is_slurm_job:
            print(f"{self.tag} Slurm-cluster LOCAL_RANK: {local_rank} WOLRD_RANK:  {world_rank} WORLD_SIZE: {world_size}")
        else:
            print(f"{self.tag} single-node LOCAL_RANK: {local_rank} WOLRD_RANK: {world_rank} WORLD_SIZE: {world_size}")
        return env_args
    
    
    def ddp_setup(self, env_args):
        local_rank, world_rank, world_size, node_rank = \
            env_args["local_rank"], env_args["world_rank"], env_args["world_size"], env_args["node_rank"]
        main_port, main_addr = env_args["main_port"], env_args["main_addr"]
        
        # initialize the process group
        tcp_store = dist.TCPStore(main_addr, main_port, world_size, world_rank == 0)
        dist.init_process_group(backend='nccl', store=tcp_store, rank=world_rank, world_size=world_size)

        if dist.is_initialized():
            print(self.tag, "Torch distributed initialized successfully", end="\n\t")
            print(f"main_addr: {main_addr}, main_port: {main_port}, world rank: {world_rank}, "
                f"local rank: {local_rank}, node rank: {node_rank}, world size: {world_size}")
        else:
            print(self.tag, "Torch distributed initialization failed")
        
        return local_rank, world_rank, world_size
    
    def ddp_cleanup(self):
        dist.destroy_process_group()
        print(self.tag, "Torch distributed destroyed successfully")
    
    def experiment(self, cmd_args, env_args):
        local_rank, world_rank, world_size = self.ddp_setup(env_args)
        self.main(cmd_args, local_rank, world_rank, world_size)
        self.ddp_cleanup()    
    
    def ddp_run(self):
        env_args = self.make_env_args()
        cmd_args = self.make_cmd_args()
        self.experiment(cmd_args, env_args)
    
    def main(self, cmd_args, local_rank, world_rank, world_size):
        # main logic for any training program
        raise NotImplementedError()
    
    def set_torch_device(self, local_rank):
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        device = torch.device('cuda', local_rank)
        return device

    
    
    
    
    
    

    
    
    
    
    