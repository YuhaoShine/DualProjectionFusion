# DualProjectionFusion

This repository contains the official implementation of the following paper:

**Learning Projection-Aware 360-Degree Image Rectification via Dual-Projection Fusion**
*Yuhao Shan, Qianyi Yuan, Jingguo Liu, Shigang Li, Jianfeng Li, Tong Chen*  
**Corresponding Author:** Yuhao Shan (shanyuhao@swu.edu.cn)  

**Status:** Revised manuscript submitted to The Visual Computer (Under Review) 

Abstract:  Panoramic cameras provide a 360° field of view and are widely used in panoramic vision, immersive visual computing, and robotic perception. However, changes in camera orientation can produce non-upright panoramas, introducing geometric variations that can complicate downstream visual analysis. Existing vision-based rectification methods usually operate within a single projection domain, limiting their ability to jointly exploit local geometric structures and global contextual information. To address this, we formulate 360° image rectification as a projection-aware representation learning problem and propose a dual-projection framework for upright panoramic rectification. A convolutional neural network branch captures local geometric structures from equirectangular projection (ERP) inputs, while a vision transformer branch models global contextual cues from cubemap projections. Cross-projection feature transformation and multi-level feature fusion enable effective interaction between these complementary representations. The learned representation supports collaborative inclination estimation and upright panorama generation, with the two tasks providing complementary geometric and appearance supervision. Experiments on SUN360 and M3D show consistent improvements over existing methods, achieving accuracies within a 1° error threshold of 65.9% and 85.2% and Fréchet Inception Distance scores of 5.87 and 3.26, respectively. Ablation studies verify the contributions of dual-projection representation, cross-projection feature transformation and fusion, and collaborative multi-task learning. The proposed framework provides a projection-aware visual computing approach for panoramic rectification. 

# This work is currently under review. The code is provided to support the review process and ensure reproducibility. The repository is being actively maintained and will continue to be updated as additional documentation and resources are organized and verified.

<img width="1365" height="873" alt="image" src="https://github.com/user-attachments/assets/1fa6735d-4bb8-4e36-b52c-b108dde22c4e" />

**Environment / Installation:**
The experiments were conducted using Python and PyTorch on an NVIDIA RTX 3090 GPU.
pip install -r requirements.txt

## Reproducibility

The random seed used for the experiments reported in the paper is: 100

**Code for SUN360 Dataset：**
1) DPF_UpPanoGeneration_Imp-Align: Implicit Data-Driven Alignment
2) DPF_UpPanoGeneration_Exp-Align: Explicit Geometric Alignment

**For train, run "train.py".**

**For test, please following steps in ".\DPF_UpPanoGeneration_Imp-Align\Test_Result\results\introduction.txt"**


## Extended Evaluation Metrics

The extended angle and image evaluation scripts should be placed in:

`./DPF_UpPanoGeneration_Imp-Align/Test_Result/results/`

After generating the model predictions, run the scripts **from this directory**:

```bat
cd DPF_UpPanoGeneration_Imp-Align\Test_Result\results
python AngleMetrics_extended_v3.py
python ImageMetrics_extended_v3.py
```

### Angle Evaluation

`AngleMetrics_extended_v3.py` reads:

- `PitchRollAngGT.txt`: ground-truth pitch and roll angles in degrees.
- `PitchRollAngPRED.txt`: predicted pitch and roll angles in degrees.

Both files must contain the same number of samples, with corresponding samples in the same row order and pitch followed by roll.

The script reports threshold accuracies and MAE, RMSE, median, and P95 of the scalar angular error. It also reports component-wise pitch/roll errors and optional SO(3) geodesic error statistics with yaw fixed to zero. These error definitions are reported separately and should not be treated as interchangeable.

Per-sample results are saved to `angle_metrics_per_sample.csv`. Predictions are evaluated without additional rounding.

### Image Evaluation

`ImageMetrics_extended_v3.py` reads ground-truth upright images from `./gt_UpIMG/` and generated upright images from `./pre_UpIMG/`.

Image pairs are matched using the numeric sample ID at the beginning of filenames, such as `IMG0_gt_UpIMG.jpg` and `IMG0_pre_UpIMG.jpg`. Corresponding images must have identical dimensions. Check the reported matched-pair and unmatched-file counts to confirm that the intended evaluation set is included.

The script reports PSNR, SSIM, LPIPS (AlexNet, version 0.1), NRMSE, and NMAE, including their mean, standard deviation, and median. Per-sample results are saved to `image_metrics_per_sample.csv`.

FID is evaluated separately using the existing `pytorch-fid` evaluation script.



## Pretrained Models and Additional Resources

Pretrained model weights and additional reproduction resources are available through [GitHub Releases](https://github.com/YuhaoShine/DualProjectionFusion/releases), providing an alternative download source to Baidu Netdisk.

| Model / experiment | Download and instructions |
|---|---|
| Implicit data-driven alignment | [Pretrained weights — Ours-Imp](https://github.com/YuhaoShine/DualProjectionFusion/releases/tag/DPF_UpPanoGeneration_Imp-Align) |
| Explicit geometric alignment | [Pretrained weights — Ours-Exp](https://github.com/YuhaoShine/DualProjectionFusion/releases/tag/DPF_UpPanoGeneration) |
| Noise-aware fine-tuning experiment reported in Table 13 | [Noise-aware fine-tuned model](https://github.com/YuhaoShine/DualProjectionFusion/releases/tag/NA_FINETUNED_MODEL_for_Table13) |
| Noise-aware model for real-world RICOH THETA evaluation | [Model and additional testing/training resources](https://github.com/YuhaoShine/DualProjectionFusion/releases/tag/NA_Model_for_RealThetaIMG) |

Please consult the corresponding release notes for the available files, model-specific instructions, and SHA-256 checksums where provided. Checksums must be matched to the exact downloaded filename.

`model.pth` contains the model parameters, while `adam.pth`, where provided, contains the optimizer state for resuming training. The current evaluation scripts also load the optimizer state; follow the corresponding script requirements when preparing the files.

Before running the scripts, configure the dataset, checkpoint, and output paths for your local environment. For installation and environment details, see [INSTALL.md](INSTALL.md).

## Contact and Reproduction Support

If you encounter difficulties downloading the resources or reproducing the results, please contact **Yuhao Shan** at [shanyuhao@swu.edu.cn](mailto:shanyuhao@swu.edu.cn).

To help diagnose the issue, please include the release or model variant used, your operating system and Python/PyTorch versions, the command or script you ran, and the complete error message or traceback. Please also describe the dataset and checkpoint configuration relevant to the issue.


**Datasets and pretrainned models for SUN360 can be found in：**

Link: https://pan.baidu.com/s/14qgkAhhq9zJXTE5pUj9lGA?pwd=p9az Code: p9az 

**200-image real-world panoramic evaluation set can be downloaded from the following link:**

Link: https://pan.baidu.com/s/1TCpFJ5pdT2z2Ck2a1bLcUA Code: v7k8

Link: https://drive.google.com/file/d/1mBvogvBe1JXJ4ZSsUssOUos69FDmGkqP/view?usp=sharing

**1)	Inclination angle Estimation Task**

<img width="512" height="323" alt="image" src="https://github.com/user-attachments/assets/c87e19d3-955d-4696-b98a-2ec3d8d6e574" />  <img width="518" height="261" alt="image" src="https://github.com/user-attachments/assets/633cc7aa-112f-4e60-820e-b2d98a74e500" />

**2)	Upright Panoramic Images Generation Task**

<img width="1055" height="393" alt="image" src="https://github.com/user-attachments/assets/96e1459c-b5a8-4d4e-b216-9d37d049d108" />

**3) Representative qualitative rectification results on real-world RICOH THETA panoramas**

<img width="1015" height="554" alt="image" src="https://github.com/user-attachments/assets/eba65d6a-4005-450d-9877-0fa6c929b3a3" />

**4) Runtime and Deployment Analysis**

<img width="507" height="233" alt="image" src="https://github.com/user-attachments/assets/c340d566-b43d-4b28-81b7-622807a33f27" />

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

If you find this project useful in your research, please consider citing our paper:

```bibtex
@article{shan2026dualprojection,
  title={Learning Projection-Aware 360-Degree Image Rectification via Dual-Projection Fusion},
  author={Shan, Yuhao and Yuan, Qianyi and Liu, Jingguo and Li, Shigang and Li, Jianfeng and Chen, Tong},
  journal={Submitted to The Visual Computer},
  year={2026},
  note={Under Review}
}
```
*The citation information will be updated with the official publication details upon acceptance.*

