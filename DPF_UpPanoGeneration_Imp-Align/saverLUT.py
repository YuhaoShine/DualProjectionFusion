import os

import numpy as np
import matplotlib.pyplot as plt

import cv2

from scipy.io import savemat



def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass


class Saver(object):

    def __init__(self, save_dir):
        self.idx = 0
        self.save_dir = os.path.join(save_dir, "results")
        if not os.path.exists(self.save_dir):
            mkdirs(self.save_dir)

    def save_as_point_cloud(self, depth, rgb, path, mask=None):
        h, w = depth.shape
        Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        Phi = -np.repeat(Phi, h, axis=0)

        X = depth * np.sin(Theta) * np.sin(Phi)
        Y = depth * np.cos(Theta)
        Z = depth * np.sin(Theta) * np.cos(Phi)

        if mask is None:
            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()
            R = rgb[:, :, 0].flatten()
            G = rgb[:, :, 1].flatten()
            B = rgb[:, :, 2].flatten()
        else:
            X = X[mask]
            Y = Y[mask]
            Z = Z[mask]
            R = rgb[:, :, 0][mask]
            G = rgb[:, :, 1][mask]
            B = rgb[:, :, 2][mask]

        XYZ = np.stack([X, Y, Z], axis=1)
        RGB = np.stack([R, G, B], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(XYZ)
        pcd.colors = o3d.utility.Vector3dVector(RGB)
        o3d.io.write_point_cloud(path, pcd)
    def save_feature(self, feature):
        feature = feature.data.cpu().numpy()
        cmap = plt.get_cmap("rainbow_r")
        feature = cmap(feature.astype(np.float32) / 10)
        feature = np.delete(feature, 3, 2)
        path = os.path.join(r'D:\project\feature_sph.png')
        cv2.imwrite(path, (feature * 255).astype(np.uint8))

    def save_feature1(self, feature):
        feature = feature.data.cpu().numpy()
        cmap = plt.get_cmap("rainbow_r")

        feature = cmap(feature.astype(np.float32) / 10)
        feature = np.delete(feature, 3, 2)
        path = os.path.join(r'D:\project\feature_equi.png')
        cv2.imwrite(path, (feature * 255).astype(np.uint8))

    def save_samples(self, rgbs, gt_depths, pred_depths, pre_UpIMG, input_up_gts, E): 
        """
        Saves samples
        """
        rgbs = rgbs.cpu().numpy().transpose(0, 2, 3, 1)
        input_up_gts = input_up_gts.cpu().numpy().transpose(0, 2, 3, 1)
        gt_depths = gt_depths.cpu().numpy().transpose(0, 2, 3, 1) 
        gt_depths=np.squeeze(gt_depths, 0)
        mat_dict_gtLUT = {'gtLUT': gt_depths}
        
        pred_depths = pred_depths.cpu().numpy().transpose(0, 2, 3, 1)
        pred_depths=np.squeeze(pred_depths, 0)
        mat_dict_pdLUTbb = {'pdLUTbb': pred_depths}
        
        pre_UpIMG = pre_UpIMG.cpu().numpy().transpose(0, 2, 3, 1)
        
        
        for i in range(rgbs.shape[0]):
            self.idx = self.idx+1
            mkdirs(os.path.join(self.save_dir, '%04d'%(self.idx)))
            
            path = os.path.join(self.save_dir, '%04d' % (self.idx), 'Epoch'+str(E)+'_pdLUTbb.mat')
            savemat(path, mat_dict_pdLUTbb)
            
            pre_UpIMG = (pre_UpIMG[i] * 255).astype(np.uint8)
            path = os.path.join(self.save_dir, '%04d' % (self.idx), 'Epoch'+str(E)+'_pre_UpIMG.jpg')
            cv2.imwrite(path, pre_UpIMG[:,:,::-1])
            
                        
            if E==0:
                path = os.path.join(self.save_dir, '%04d' % (self.idx), 'Epoch'+str(E)+'_gtLUT.mat')
                savemat(path, mat_dict_gtLUT)
            
            if E==0: 
               rgb = (rgbs[i] * 255).astype(np.uint8)
               path = os.path.join(self.save_dir, '%04d'%(self.idx), 'Epoch'+str(E)+'_Ori_IMG.jpg')
               cv2.imwrite(path, rgb[:,:,::-1])
               input_up_gt = (input_up_gts[i] * 255).astype(np.uint8)
               path = os.path.join(self.save_dir, '%04d'%(self.idx), 'Epoch'+str(E)+'_gt_UpIMG.jpg')
               cv2.imwrite(path, input_up_gt[:,:,::-1])
            

