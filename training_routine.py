import os
import torch

root="./configs/S3NET_S1_2/"   ######Folder path to config file (fold-wise)
for folder in sorted(os.listdir(root)):
        fn = folder[-1]
        for epochs in sorted(os.listdir(os.path.join(root,folder))):
                path_to_config = os.path.join(os.path.join(root,folder)+"/"+epochs)
                #######For one GPU
                #print("Command:","CUDA_VISIBLE_DEVICES=0 python tools/train.py "+path_to_config)
                #os.system("CUDA_VISIBLE_DEVICES=0 python tools/train.py "+path_to_config)
                #######For two GPUs
                print("Command:", "CUDA_VISIBLE_DEVICES=0,1, python -W ignore -m torch.distributed.launch --nproc_per_node=2 tools/train.py "+path_to_config+" --gpus 2 --launcher pytorch")
                os.system("CUDA_VISIBLE_DEVICES=0,1, python -W ignore -m torch.distributed.launch --nproc_per_node=2 tools/train.py "+path_to_config+" --gpus 2 --launcher pytorch")
                
               
