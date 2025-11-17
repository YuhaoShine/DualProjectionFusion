# DualProjectionFusion

# This repository contains the official implementation of the following paper:

**Dual-Projection Fusion for Accurate Upright Panorama Generation in Robotic Vision**
*Yuhao Shan, Qianyi Yuan, Jingguo Liu, Shigang Li, Jianfeng Li, Tong Chen*  
**Corresponding Author:** Yuhao Shan (shanyuhao@swu.edu.cn)  

**Status:** Submitted to *The Visual Computer* (Under Review)  

Abstract: Panoramic cameras, capable of capturing a 360-degree field of view, are crucial in robotic vision, particularly in environments with sparse features. However, non-upright panoramas due to unstable robot postures hinder downstream tasks. Traditional IMU-based correction methods suffer from drift and external disturbances, while vision-based approaches offer a promising alternative. This study presents a dual-stream angle-aware generation network that jointly estimates camera inclination angles and reconstructs upright panoramic images. The network comprises a CNN branch that extracts local geometric structures from equirectangular projections and a ViT branch that captures global contextual cues from cubemap projections. These are integrated through a dual-projection adaptive fusion module that aligns spatial features across both domains. To further enhance performance, we introduce a high-frequency enhancement block, circular padding, and channel attention mechanisms to preserve 360° continuity and improve geometric sensitivity. Experiments on the SUN360 and M3D datasets demonstrate that our method outperforms existing approaches in both inclination estimation and upright panorama generation. Ablation studies further validate the contribution of each module and highlight the synergy between the two tasks. 

# This work is currently under review. The code is provided to support the review process and ensure reproducibility.


<img width="840" height="470" alt="image" src="https://github.com/user-attachments/assets/eedf7aeb-d09d-4115-8ae6-7fa984c261e7" />

pip install -r requirements.txt

Code for SUN360 Dataset：
1) DPF_UpPanoGeneration_Imp-Align: Implicit Data-Driven Alignment
2) DPF_UpPanoGeneration_Exp-Align: Explicit Geometric Alignment

For train, run "DualProjectionFusionUp_ConvNext_ViT.py".
For test, please following the step in ".\DPF_UpPanoGeneration_Imp-Align\Test_Result\results\introduction.txt"

Datasets and pretrainned models for SUN360 can be find in：
Link: https://pan.baidu.com/s/14qgkAhhq9zJXTE5pUj9lGA?pwd=p9az Code: p9az 

1)	Inclination angle Estimation Task
<img width="510" height="285" alt="image" src="https://github.com/user-attachments/assets/07d7995b-bb91-4964-9818-86c8f5f8e495" />

2)	Upright Panoramic Images Generation Task
<img width="493" height="173" alt="image" src="https://github.com/user-attachments/assets/99f7dc1b-16da-497d-93e3-d30e28893aba" />

If you find this project useful in your research, please consider citing our paper:

```bibtex
@article{shan2025dualprojection,
  title={Dual-Projection Fusion for Accurate Upright Panorama Generation in Robotic Vision},
  author={Shan, Yuhao and Yuan, Qianyi and Liu, Jingguo and Li, Shigang and Li, Jianfeng and Chen, Tong},
  journal={Submitted to The Visual Computer},
  year={2025},
  note={Under Review}
}

(We will update the citation information with the official details upon acceptance.)


