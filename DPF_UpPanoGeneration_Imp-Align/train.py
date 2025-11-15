import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import time
import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import argparse
from DualProjectionFusionUp_ConvNext_ViT import DualProjectionFusionUp 


torch.manual_seed(100)
torch.cuda.manual_seed(100)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--data_path", default="H:/DPF_UpPanoGeneration_Imp-Align/", type=str, help="path to dataset")
    parser.add_argument("--dataset", default="3d60", choices=["3d60", "matterport3d", "stanford2d3d"], type=str, help="which dataset to train")
    parser.add_argument("--transformer_path", type=str, help="path to load pertrained CViT")
    
    #network settings
    parser.add_argument("--net", type=str, default="TwoBranch", choices=["SphereNet", "NormalNet", "TwoBranch"], help="choose branch")
    parser.add_argument("--model_name", type=str, default="GLPanoUpright", help="model name")
    parser.add_argument("--height", type=int, default=256, help="input image height")
    parser.add_argument("--width", type=int, default=512, help="input image width")

    #loss settings
    parser.add_argument("--berhuloss", type=float, default=0.2, help="berhu loss threhold")
    parser.add_argument("--learning_rate", type=float, default=1*1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=6, help="batch size")
    parser.add_argument("--batch_size_test", type=int, default=1, help="batch size") ##
    parser.add_argument("--num_epochs", type=int, default=110, help="number of epochs")
    
    #system settings
    parser.add_argument("--num_workers", type=int, default=6, help="number of dataloader workers")
    parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0], help="available gpus")

    # loading and logging settings
    parser.add_argument("--load_weights_dir", type=str, help="path to trained model")
    parser.add_argument("--log_dir", type=str, default="H:/DPF_UpPanoGeneration_Imp-Align//experiments_Upright_LOG/", help="path to log")
    
    parser.add_argument("--log_frequency", type=int, default=200, help="number of batches between each tensorboard log")
    parser.add_argument("--save_frequency", type=int, default=1, help="number of epochs between each save")
    
    parser.add_argument("--num_layers", type=int, default=18, choices=[2, 18, 34, 50, 101],
                        help="number of resnet layers; if 2, use mobilenetv2")
    parser.add_argument("--imagenet_pretrained",default=True#action="store_true"#
                        , help="if set, use imagenet pretrained parameters")
    parser.add_argument("--fusion", type=str, default="cee", choices=["cee", "cat", "biproj"],#cee
                        help="the method to fuse cubemap features to equirectangular features")
    parser.add_argument("--se_in_fusion", action="store_true",
                        help="if set, use the squeeze-and-excitation module in the fusion")

    
    
    # network setting
    args = parser.parse_args()
    model = DualProjectionFusionUp(args)
    model.train() ############