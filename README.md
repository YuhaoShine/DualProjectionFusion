# DualProjectionFusion

# This repository contains the official implementation of the following paper:

**Learning Projection-Aware 360-Degree Image Rectification via Dual-Projection Fusion**
*Yuhao Shan, Qianyi Yuan, Jingguo Liu, Shigang Li, Jianfeng Li, Tong Chen*  
**Corresponding Author:** Yuhao Shan (shanyuhao@swu.edu.cn)  

**Status:** Submitted to *The Visual Computer* (Under Review)  

Abstract:  Panoramic cameras provide a 360° field of view and are widely used in panoramic vision, immersive visual computing, and robotic perception. However, changes in camera orientation can produce non-upright panoramas, introducing geometric variations that can complicate downstream visual analysis. Existing vision-based rectification methods usually operate within a single projection domain, limiting their ability to jointly exploit local geometric structures and global contextual information. To address this, we formulate 360° image rectification as a projection-aware representation learning problem and propose a dual-projection framework for upright panoramic rectification. A convolutional neural network branch captures local geometric structures from equirectangular projection (ERP) inputs, while a vision transformer branch models global contextual cues from cubemap projections. Cross-projection feature transformation and multi-level feature fusion enable effective interaction between these complementary representations. The learned representation supports collaborative inclination estimation and upright panorama generation, with the two tasks providing complementary geometric and appearance supervision. Experiments on SUN360 and M3D show consistent improvements over existing methods, achieving accuracies within a 1° error threshold of 65.9% and 85.2% and Fréchet Inception Distance scores of 5.87 and 3.26, respectively. Ablation studies verify the contributions of dual-projection representation, cross-projection feature transformation and fusion, and collaborative multi-task learning. The proposed framework provides a projection-aware visual computing approach for panoramic rectification. 

# This work is currently under review. The code is provided to support the review process and ensure reproducibility.

<img width="1365" height="873" alt="image" src="https://github.com/user-attachments/assets/1fa6735d-4bb8-4e36-b52c-b108dde22c4e" />

pip install -r requirements.txt

Code for SUN360 Dataset：
1) DPF_UpPanoGeneration_Imp-Align: Implicit Data-Driven Alignment
2) DPF_UpPanoGeneration_Exp-Align: Explicit Geometric Alignment

For train, run "DualProjectionFusionUp_ConvNext_ViT.py".
For test, please following the step in ".\DPF_UpPanoGeneration_Imp-Align\Test_Result\results\introduction.txt"

Datasets and pretrainned models for SUN360 can be find in：
Link: https://pan.baidu.com/s/14qgkAhhq9zJXTE5pUj9lGA?pwd=p9az Code: p9az 

1)	Inclination angle Estimation Task
<img width="512" height="323" alt="image" src="https://github.com/user-attachments/assets/c87e19d3-955d-4696-b98a-2ec3d8d6e574" />
<img width="518" height="261" alt="image" src="https://github.com/user-attachments/assets/633cc7aa-112f-4e60-820e-b2d98a74e500" />

2)	Upright Panoramic Images Generation Task
<img width="1055" height="393" alt="image" src="https://github.com/user-attachments/assets/96e1459c-b5a8-4d4e-b216-9d37d049d108" />

3) Representative qualitative rectification results on real-world RICOH THETA panoramas.
<img width="1015" height="554" alt="image" src="https://github.com/user-attachments/assets/eba65d6a-4005-450d-9877-0fa6c929b3a3" />

4) Runtime and Deployment Analysis
<img width="507" height="233" alt="image" src="https://github.com/user-attachments/assets/c340d566-b43d-4b28-81b7-622807a33f27" />


If you find this project useful in your research, please consider citing our paper:

```bibtex
@article{shan2025dualprojection,
  title={Learning Projection-Aware 360-Degree Image Rectification via Dual-Projection Fusion},
  author={Shan, Yuhao and Yuan, Qianyi and Liu, Jingguo and Li, Shigang and Li, Jianfeng and Chen, Tong},
  journal={Submitted to The Visual Computer},
  year={2026},
  note={Under Review}
}

(We will update the citation information with the official details upon acceptance.)


