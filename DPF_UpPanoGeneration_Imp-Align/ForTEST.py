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
import datasets
from models import TwoBranchConvNextViT, TransformerNet#, ModelLoadTransformer
from utils.losses import BerhuLoss
import torch.nn.functional as F

from utils.metrics import Evaluator
from saverTestResult import Saver
import lpips
import json
import math


class DualProjectionFusionUp:
    def __init__(self, args):
        self.settings = args 

        self.device = torch.device("cuda" if len(self.settings.gpu_devices) else "cpu")
        self.gpu_devices = ','.join([str(id) for id in self.settings.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)

        # data
        self.dataset = datasets.loadDataLUT
        ## SUN360toy
        train_file_list = './datasets/SUN360_RGBLUT_PitchRoll_trainlabel.txt' ## SUN360_RGBLUT_PitchRoll_trainlabel
        val_file_list = './datasets/SUN360_RGBLUT_PitchRoll_vallabel.txt'  ## SUN360_RGBLUT_PitchRoll_vallabel
        test_file_list = './datasets/SUN360_RGBLUT_PitchRoll_testlabel.txt' ## SUN360_RGBLUT_PitchRoll_testlabel
        
        ## 训练集
        train_dataset = self.dataset(self.settings.data_path, train_file_list) 
        self.train_loader = DataLoader(train_dataset, self.settings.batch_size, True,
                                       num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)       
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs
        ## 验证集
        val_dataset = self.dataset(self.settings.data_path, val_file_list)
        
        self.val_loader = DataLoader(val_dataset, self.settings.batch_size_test, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=False)
        ## 测试集
        test_dataset = self.dataset(self.settings.data_path, test_file_list)
        
        self.test_loader = DataLoader(test_dataset, self.settings.batch_size_test, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=False)
                
        self.settings.cube_w = self.settings.height//2
               
        #BranchLoader
        Model_dict = {"TwoBranchConvNextViT":TwoBranchConvNextViT, "TransformerNet":TransformerNet}
        # Net = Model_dict[self.settings.net] ## TwoBranch
        Net = Model_dict["TwoBranchConvNextViT"]

        self.model = Net(image_height=self.settings.height, image_width=self.settings.width)  ## TwoBranch
        
        
        ############################################################################################################################################
        model_loadpath='.\\Pretrainned_weights\\model.pth' 
        optimizer_load_path='.\\Pretrainned_weights\\adam.pth'
        ############################################################################################################################################         
        # 加载模型
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(model_loadpath)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        ############################################################################################################################################
        self.model.to(self.device)
        self.parameters_to_train = list(self.model.parameters())
        self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)
        ############################################################################################################################################
        # 加载优化器
        optimizer_dict = torch.load(optimizer_load_path)
        self.optimizer.load_state_dict(optimizer_dict)
        ############################################################################################################################################
        self.compute_loss = BerhuLoss(threshold=self.settings.berhuloss)
        self.evaluator = Evaluator()
        self.L1_loss = torch.nn.L1Loss() # L1
        self.MSE_loss = torch.nn.MSELoss() # L2
        
        lpips_model = lpips.LPIPS(net='vgg', version=0.1) # 可以选择'vgg'或'squeezenet'作为backbone
        self.lpips_model = lpips_model.cuda()
        
 ########################################################################################################
    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()
        self.epoch=0
        saver = Saver('H:\\DPF_UpPanoGeneration_Imp-Align\\Test_Result\\')
        self.evaluator.reset_eval_metrics()
        pbar = tqdm.tqdm(self.test_loader) ######################################## test set
        pbar.set_description("testing Epoch_{}".format(self.epoch))
         
        Err=[] #### 验证集角度误差   

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):

                for key, ipt in inputs.items():
                    if key not in ["rgb", "cube_rgb"]:
                        inputs[key] = ipt.to(self.device)
                        
                
                equi_inputs = inputs["normalized_rgb_Rotate"] 
                cube_input = inputs["normalized_cube_rgb"]
                gt_Upright = inputs["normalized_LUT_Up"] 
                gtUpIMG = inputs["normalized_rgb_Upright"]
                input_up_gt = inputs["normalized_rgb_Upright"]
                
                B, C, H, W = equi_inputs.shape
                
                ############### 训练好的骨干网预测 ##############
                fus_result = self.model(equi_inputs, cube_input)
                pred_Up = fus_result["pred_Upright"].detach()
                pdLUT = pred_Up.permute(0,2,3,1)
                NeedOutputs=self.grid_sampleLUT(pdLUT)
                pre_UpIMG = F.grid_sample(equi_inputs, NeedOutputs["needLUT"], mode='bilinear', padding_mode='border', align_corners=True)
                self.evaluator.compute_eval_metrics(gtUpIMG, pre_UpIMG, self.lpips_model)
                ################################################
  
                ############################################################################### 
                errNow=(fus_result["PitchRoll_norm_pred"]-inputs["PitchRollAng"]).detach().cpu().numpy() ####
                Err.append(abs(errNow)) ####
                ###############################################################################
                
  
                for i in range(gt_Upright.shape[0]):
                    self.evaluator.compute_eval_metrics(gtUpIMG[i:i + 1], pre_UpIMG[i:i + 1], self.lpips_model)
                    
                    
                    
                if batch_idx%1 ==0:
                    p_r_Pred = fus_result["PitchRoll_norm_pred"]
                    
                    UpIMGbyPreAng=[]
                    
                    # p_r_Pred=torch.round(p_r_Pred * 100) / 100
                    p_r_gt = inputs["PitchRollAng"]
                    saver.save_samples(equi_inputs, pre_UpIMG, p_r_Pred, p_r_gt, gtUpIMG, batch_idx, pred_Up, UpIMGbyPreAng) ## ????
                    
        # self.evaluator.print(self.epoch, 'H:\\DPF_UpPanoGeneration_Imp-Align\\Test_Result\\')
                    
        ############################################################################### 验证集角度误差
        Err=np.array(Err) 
        meanErr=str(np.round(np.mean(Err,0),2)).replace("[[","").replace("]]","")
        minErr=str(np.round(np.min(Err,0),2)).replace("[[","").replace("]]","") 
        maxErr=str(np.round(np.max(Err,0),2)).replace("[[","").replace("]]","")
        medErr=str(np.round(np.median(Err,0),2)).replace("[[","").replace("]]","")
        with open('H:\\DPF_UpPanoGeneration_Imp-Align\\Test_Result\\results\\TestAngErrSAVE.txt', 'a') as file:
            file.write(minErr+'----'+medErr+'----'+maxErr+'----'+meanErr+ "\n")
        ###############################################################################

        
        del inputs, fus_result 
        
        
    ########################################################################################################
    def calculate_psnr(self, target, prediction):
        assert target.shape == prediction.shape, "The shape of target and prediction must be the same."
        target = target.type(torch.float32)
        prediction = prediction.type(torch.float32)
        diff = target - prediction
        diff = torch.abs(diff) 
        mse = torch.mean(diff ** 2)
        max_pixel_value = 1.0 
        psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
        psnr = torch.tensor(psnr)   
        return psnr
    ########################################################################################################
    def pre_rota(self, B,h,w):
        a = torch.ones((1, w)).cuda()
        i = torch.arange(1, h+1).reshape((1, h)).cuda()  #(1 256)
        j = torch.arange(1, w+1).reshape((1, w)).cuda()   #(1 512)
        theta = i * 180 / h * (3.1416 / 180)  #(1 256)
        fai = j * 360 / w * (3.1416 / 180)  #(1 512)
        X = (torch.sin(theta.T) * torch.cos(fai)) #(256 512)
        X = X.unsqueeze(0)
        Y = (torch.sin(theta.T) * torch.sin(fai))
        Y = Y.unsqueeze(0)
        Z = (torch.cos(theta.T)) * a
        Z = Z.unsqueeze(0)
        xs = torch.cat([X, Y, Z], dim = 0)
        xs = xs.unsqueeze(0).repeat(B, 1, 1, 1) #(B, 3, 256 512)
        return xs
    ########################################################################################################
    def rotationMatrix(self, angle, flowupdate, flag):
        p = angle[:, 0] * (3.1416 / 180)
        r = angle[:, 1] * (3.1416 / 180)
        CosP = torch.cos(p)
        SinP = torch.sin(p)
        CosR = torch.cos(r)
        SinR = torch.sin(r)

        Rx = torch.zeros((flowupdate.size(0), 3, 3)).cuda()
        Rx[:, 0, 0] = 1
        Rx[:, 1, 1] = CosR
        Rx[:, 1, 2] = -SinR
        Rx[:, 2, 1] = SinR
        Rx[:, 2, 2] = CosR

        Ry = torch.zeros((flowupdate.size(0), 3, 3)).cuda()
        Ry[:, 0, 0] = CosP
        Ry[:, 0, 2] = SinP
        Ry[:, 1, 1] = 1
        Ry[:, 2, 0] = -SinP
        Ry[:, 2, 2] = CosP

        Rz = torch.zeros((flowupdate.size(0), 3, 3)).cuda()
        Rz[:, 0, 0] = 1
        Rz[:, 1, 1] = 1
        Rz[:, 2, 2] = 1

        R = torch.matmul(torch.matmul(Rx, Ry), Rz)
        if flag == False:
            R = torch.inverse(R)
        flowupdate = flowupdate.view(flowupdate.size(0), flowupdate.size(1), -1)
        flowupdate = torch.matmul(R, flowupdate)
        flowupdate = flowupdate.view(flowupdate.size(0), flowupdate.size(1), 256, 512)
        return flowupdate
    ########################################################################################################
    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        return torch.nn.functional.grid_sample(inp, deformation,align_corners=True)
    ########################################################################################################
    def warp3Dflow(self,flowupdate): # 输入# (B, 3, 256, 512)
        flownorm = torch.norm(flowupdate, dim=1).unsqueeze(1) 
        flownorm = flowupdate / (flownorm + 1e-5)
        theta_new = torch.arccos(flownorm[:, 2, :, :]).unsqueeze(1)  # (4, 1, 256, 512)

        fai_new = torch.atan2(flownorm[:, 1, :, :], flownorm[:, 0, :, :]).unsqueeze(1)  # (4, 1, 256, 512)
        fai_2 = fai_new.clone()
        fai_2[fai_2 > 0] = 0
        fai_2[fai_2 < 0] = 2 * 3.1416
        fai_new = fai_new + fai_2

        x_new = fai_new * 512 / 2. / 3.1416  # (4, 1, 256, 512)
        x_new[x_new < 1] = 1
        y_new = theta_new * 256 / 3.1416  # (4, 1, 256, 512)
        y_new[y_new < 1] = 1

        lut = torch.cat([y_new, x_new], dim=1)  # (4, 2, 256, 512)
        rota_lut1 = torch.zeros((flownorm.size(0), 256, 512, 2)).cuda()
        rota_lut1[:, :, :, 1] = (lut[:, 0, :, :] - 128) / 128.
        rota_lut1[:, :, :, 0] = (lut[:, 1, :, :] - 256) / 256.
        return rota_lut1
    ########################################################################################################    
    def grid_sampleLUT(self, predLUT): # bs*256*512*3
        predLUT=predLUT.detach().cpu()
        Lshape=predLUT.shape
        pingfangheALL=np.empty((Lshape[0],Lshape[1],Lshape[2]))
        for bs in range(Lshape[0]):
            NowLUT=predLUT[bs,:,:,:]
            pingfangheALL[bs,:,:]=NowLUT[:,:,0]**2+NowLUT[:,:,1]**2+NowLUT[:,:,2]**2
        needLUT=self.warp3Dflow(predLUT.permute(0, 3, 1, 2))
        pingfangheALL=pingfangheALL.astype(np.float32)
        pingfangheALL=torch.from_numpy(pingfangheALL)
        pingfangheALL=pingfangheALL.cuda()
        needLUT=needLUT.cuda() 
        NeedOutputs={}
        NeedOutputs["needLUT"]=needLUT
        NeedOutputs["pingfangheALL"]=pingfangheALL
        
        return NeedOutputs
    ########################################################################################################