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
from models import TwoBranchConvNextViT, TransformerNet
from utils.losses import BerhuLoss
import torch.nn.functional as F

from utils.metrics import Evaluator
#from lossesV3 import BerhuLoss, SSIM, MS_SSIM
from saverLUT import Saver
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
        train_file_list = './datasets/SUN360_RGBLUT_PitchRoll_toylabel.txt' ## SUN360_RGBLUT_PitchRoll_trainlabel
        val_file_list = './datasets/SUN360_RGBLUT_PitchRoll_toylabel.txt'  ## SUN360_RGBLUT_PitchRoll_vallabel
        test_file_list = './datasets/SUN360_RGBLUT_PitchRoll_toylabel.txt' ## SUN360_RGBLUT_PitchRoll_testlabel
        
        
        train_dataset = self.dataset(self.settings.data_path, train_file_list) 
        self.train_loader = DataLoader(train_dataset, self.settings.batch_size, True,
                                       num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)       
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs

        val_dataset = self.dataset(self.settings.data_path, val_file_list)
        
        self.val_loader = DataLoader(val_dataset, self.settings.batch_size_test, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=False)

        test_dataset = self.dataset(self.settings.data_path, test_file_list)
        
        self.test_loader = DataLoader(test_dataset, self.settings.batch_size_test, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=False)
                
        self.settings.cube_w = self.settings.height//2
               
        Model_dict = {"TwoBranchConvNextViT":TwoBranchConvNextViT, "TransformerNet":TransformerNet}
        Net = Model_dict["TwoBranchConvNextViT"]

        self.model = Net(image_height=self.settings.height, image_width=self.settings.width)
         
        self.model.to(self.device)
        self.parameters_to_train = list(self.model.parameters())
        self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)
        
        if self.settings.load_weights_dir is not None:
            self.load_model()

        self.compute_loss = BerhuLoss(threshold=self.settings.berhuloss)
        self.evaluator = Evaluator()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        print("Init successfully! Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)
        
        # self.SSIM_loss = SSIM() #
        # self.MS_SSIM = MS_SSIM() #
        self.L1_loss = torch.nn.L1Loss() # L1
        self.MSE_loss = torch.nn.MSELoss() # L2
        
        lpips_model = lpips.LPIPS(net='vgg', version=0.1) 
        self.lpips_model = lpips_model.cuda()
        
        self.save_settings()

    ########################################################################################################
    def train(self):
        """Run the entire training pipeline
        """
        self.step = 0
        for self.epoch in range(0, self.settings.num_epochs): 
            self.train_one_epoch() ####
            self.validate()
            if (self.epoch + 1) % self.settings.save_frequency == 0:
                self.save_model()

    ########################################################################################################    
    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train() ####
        
        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))
        
        for batch_idx, inputs in enumerate(pbar):
            outputs, losses = self.forward_batch(inputs)
            self.optimizer.zero_grad()
            losses["losses"].backward()
            self.optimizer.step()
            self.step += 1 
        print('当前LOSS输出： ', losses)
        
        #################################################################################################### 
        with open('H:\\GLPanoDepth_ConvNext_ViT_TFC_ANG_vitCUB2ERP_SUN360\\experiments_Upright_LOG\\LossesSAVE.txt', 'a') as file:
            loss_Ang=np.round(losses["loss_Ang"].detach().cpu().numpy(),2)
            LossesLUT=np.round(losses["loss_LUT"].detach().cpu().numpy(),2)
            LossesFeel=np.round(losses["IMG_feel"].detach().cpu().numpy(),2)
            LossesALL=np.round(losses["losses"].detach().cpu().numpy(),2)
            file.write(str(loss_Ang)+'----'+str(LossesLUT)+'----'+str(LossesFeel)+'----'+str(LossesALL) + "\n")

    ########################################################################################################
    def forward_batch(self, inputs):
        for key, ipt in inputs.items():
            if key not in ["rgb", "cube_rgb"]:
                inputs[key] = ipt.to(self.device) 

        losses = {}

        equi_inputs = inputs["normalized_rgb_Rotate"] 
        cube_input = inputs["normalized_cube_rgb"] 
        gt_Upright = inputs["normalized_LUT_Up"] 
        
        fus_result = self.model(equi_inputs, cube_input)
        
        
        ###################################################################################################
        criterionSmoothL1 = torch.nn.SmoothL1Loss()
        flowSrc = self.pre_rota(6,256,512) 
        p_r_Pred = fus_result["PitchRoll_norm_pred"]
        p_r_gt = inputs["PitchRollAng"]
        ################################################################# 
        flowupdate = self.rotationMatrix(p_r_Pred, flowSrc, False)
        rota_lut_pre = self.warp3Dflow(flowupdate)
        input_up = self.deform_input(equi_inputs, rota_lut_pre)        
        ################################################################# 
        criterionL1 = torch.nn.L1Loss()
        input_up_gt = inputs["normalized_rgb_Upright"]
        ################################################################# 
        loss_angle = criterionSmoothL1(p_r_Pred.float(), p_r_gt.float())
        loss_L1 = criterionL1(input_up.float(), input_up_gt.float())   
        losses["loss_Ang"] = loss_angle.float() + loss_L1.float()
        #################################################################
        pdLUT=fus_result["pred_Upright"]
        pdLUT = pdLUT.permute(0,2,3,1)
           
        with torch.no_grad():
            NeedOutputs=self.grid_sampleLUT(pdLUT)
            pdUPIMG = F.grid_sample(equi_inputs, NeedOutputs["needLUT"], mode='bilinear', padding_mode='border', align_corners=True)
            gtUpIMG=inputs["normalized_rgb_Upright"]
            pingfanghe=NeedOutputs["pingfangheALL"]
            B, C, H, W = equi_inputs.shape
            ones_tensor = torch.ones(B, H, W).cuda()

        LUT_L2_loss = self.MSE_loss(gt_Upright.float(), fus_result["pred_Upright"].float())
        UnitSphereLoss = self.L1_loss(pingfanghe.float(),ones_tensor.float())
        IMG_L1_loss = self.L1_loss(pdUPIMG.float(), gtUpIMG.float())
        
        LPIPSs = self.lpips_model(gtUpIMG.float(), pdUPIMG.float())
        LPI=LPIPSs.squeeze()
        LPI_Loss = torch.mean(LPI) ## ????
        PSNR=self.calculate_psnr(gtUpIMG.float(), pdUPIMG.float())
        PSNR_Loss=10*torch.reciprocal(PSNR) ## ????
        
        losses["loss_LUT"] = (LUT_L2_loss.float() + UnitSphereLoss.float() + IMG_L1_loss.float() +IMG_L1_loss.float())*10
        losses["IMG_feel"] = (LPI_Loss + PSNR_Loss)*10       
        
        
        ###################################################################################################
        losses["losses"] = losses["loss_Ang"].float() + losses["loss_LUT"].float() + losses["IMG_feel"].float()
        
        return fus_result, losses

   
 ########################################################################################################
    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()
        
        saver = Saver('H:\\GLPanoDepth_ConvNext_ViT_TFC_ANG_vitCUB2ERP_SUN360\\experiments_Upright_LOG\\IMGcheckBackbone\\')
        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
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
                input_up_gt = inputs["normalized_rgb_Upright"] 
                
                B, C, H, W = equi_inputs.shape
                
                fus_result = self.model(equi_inputs, cube_input)
                pred_Up = fus_result["pred_Upright"].detach()

                self.evaluator.compute_eval_metrics(gt_Upright, pred_Up, self.lpips_model)
                    
                errNow=(fus_result["PitchRoll_norm_pred"]-inputs["PitchRollAng"]).detach().cpu().numpy()
                Err.append(abs(errNow)) 
                
                for i in range(gt_Upright.shape[0]):
                    self.evaluator.compute_eval_metrics(gt_Upright[i:i + 1], pred_Up[i:i + 1], self.lpips_model)
                    
                if batch_idx%100 ==0:
                    pdLUT=fus_result["pred_Upright"]
                    pdLUT = pdLUT.permute(0,2,3,1)
                    NeedOutputs=self.grid_sampleLUT(pdLUT)
                    pre_UpIMG = F.grid_sample(equi_inputs, NeedOutputs["needLUT"], mode='bilinear', padding_mode='border', align_corners=True)
                    
                    saver.save_samples(equi_inputs, gt_Upright, pred_Up, pre_UpIMG, input_up_gt, self.epoch) 
                    
                    print("角度真值：", inputs["PitchRollAng"])
                    print("FUS预测：", fus_result["PitchRoll_norm_pred"])
                    
                    
        Err=np.array(Err) 
        meanErr=str(np.round(np.mean(Err,0),2)).replace("[[","").replace("]]","")
        minErr=str(np.round(np.min(Err,0),2)).replace("[[","").replace("]]","") 
        maxErr=str(np.round(np.max(Err,0),2)).replace("[[","").replace("]]","")
        medErr=str(np.round(np.median(Err,0),2)).replace("[[","").replace("]]","")
        with open('H:\\GLPanoDepth_ConvNext_ViT_TFC_ANG_vitCUB2ERP_SUN360\\experiments_Upright_LOG\\ValAngErrSAVE.txt', 'a') as file:
            file.write(minErr+'----'+medErr+'----'+maxErr+'----'+meanErr+ "\n")
     
        self.evaluator.print(self.epoch, 'H:\\GLPanoDepth_ConvNext_ViT_TFC_ANG_vitCUB2ERP_SUN360\\experiments_Upright_LOG\\') 
        
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
        i = torch.arange(1, h+1).reshape((1, h)).cuda() 
        j = torch.arange(1, w+1).reshape((1, w)).cuda() 
        theta = i * 180 / h * (3.1416 / 180) 
        fai = j * 360 / w * (3.1416 / 180) 
        X = (torch.sin(theta.T) * torch.cos(fai))
        X = X.unsqueeze(0)
        Y = (torch.sin(theta.T) * torch.sin(fai))
        Y = Y.unsqueeze(0)
        Z = (torch.cos(theta.T)) * a
        Z = Z.unsqueeze(0)
        xs = torch.cat([X, Y, Z], dim = 0)
        xs = xs.unsqueeze(0).repeat(B, 1, 1, 1) 
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
    def warp3Dflow(self,flowupdate):
        flownorm = torch.norm(flowupdate, dim=1).unsqueeze(1) 
        flownorm = flowupdate / (flownorm + 1e-5)
        theta_new = torch.arccos(flownorm[:, 2, :, :]).unsqueeze(1)  

        fai_new = torch.atan2(flownorm[:, 1, :, :], flownorm[:, 0, :, :]).unsqueeze(1) 
        fai_2 = fai_new.clone()
        fai_2[fai_2 > 0] = 0
        fai_2[fai_2 < 0] = 2 * 3.1416
        fai_new = fai_new + fai_2

        x_new = fai_new * 512 / 2. / 3.1416  
        x_new[x_new < 1] = 1
        y_new = theta_new * 256 / 3.1416 
        y_new[y_new < 1] = 1

        lut = torch.cat([y_new, x_new], dim=1) 
        rota_lut1 = torch.zeros((flownorm.size(0), 256, 512, 2)).cuda()
        rota_lut1[:, :, :, 1] = (lut[:, 0, :, :] - 128) / 128.
        rota_lut1[:, :, :, 0] = (lut[:, 1, :, :] - 256) / 256.
        return rota_lut1
    ########################################################################################################    
    def grid_sampleLUT(self, predLUT):
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
    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
    ########################################################################################################        
    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model")) 
        to_save = self.model.state_dict()
        # save the input sizes
        to_save['height'] = self.settings.height
        to_save['width'] = self.settings.width
        # save the dataset to train on
        to_save['dataset'] = self.settings.dataset
        to_save['net'] = self.settings.net

        torch.save(to_save, save_path)
        save_path = os.path.join(save_folder, "{}.pth".format("adam")) 
        torch.save(self.optimizer.state_dict(), save_path)