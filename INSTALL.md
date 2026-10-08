# Environment and Installation

## Reference Environment

The following configuration records the authors' current development and execution environment.

| Component | Version / configuration |
|---|---|
| Operating system | Windows 10 Education, version 22H2, 64-bit |
| OS build | 19045.6456 |
| CPU | Intel Core i9-11900K @ 3.50 GHz |
| System memory | 64 GB |
| GPU | NVIDIA GeForce RTX 3090, 24 GB VRAM |
| Python | 3.9.25 |
| PyTorch | 1.12.0+cu113 |
| torchvision | 0.13.0+cu113 |
| PyTorch CUDA runtime | 11.3 |
| cuDNN | 8.3.2 (`torch.backends.cudnn.version()` returns `8302`) |
| Environment management | Anaconda3 / Conda |
| Development IDE | Spyder 6.1.0 |
| Spyder kernels | 3.1.1 |
| MATLAB | R2020a |

The CUDA version above refers to the runtime associated with the installed PyTorch build. GPU execution requires a compatible NVIDIA driver. Spyder is the authors' development IDE and is not required for command-line execution. MATLAB is used for the MATLAB-based evaluation scripts.

## Dependency File

`requirements.txt` retains the full Python package inventory from the reference environment, including development tools and their dependencies. It is an environment record rather than a minimal list of model dependencies.

The local wheel references for PyTorch and torchvision must be replaced with publicly installable specifications:

```text
--extra-index-url https://download.pytorch.org/whl/cu113

torch==1.12.0+cu113
torchvision==0.13.0+cu113
```

These entries belong in `requirements.txt`; the remaining recorded package versions may be retained. The extra index provides the CUDA 11.3 PyTorch wheels.

## Installation on Windows

Open Anaconda Prompt and create a separate environment:

```bat
conda create -n dpf-repro python=3.9.25 pip=25.2 -y
conda activate dpf-repro
```

From the repository root, install the recorded dependencies:

```bat
python -m pip install -r requirements.txt
```

The full dependency inventory contains Windows-specific packages. This installation procedure targets Windows; Linux and macOS installation have not been validated.

### Known Dependency Conflict

The reference environment contains:

```text
numpy==1.26.4
opencv-python==4.12.0.88
```

For Python 3.9 and later, the published dependency metadata of `opencv-python==4.12.0.88` requires NumPy `>=2,<2.3`. Consequently, these two recorded pins conflict, and a normal pip installation of the unchanged full inventory may fail. The original versions are documented here for transparency; clean-environment installation of the full inventory has not yet been verified.

A candidate resolution is to retain NumPy 1.26.4 and change the OpenCV pin to `opencv-python==4.11.0.86` in a separate reproduction environment. This is a proposed compatibility adjustment, not the recorded OpenCV version. Its effect on image preprocessing, model outputs, and evaluation results must be checked before describing the adjusted environment as validated. Other dependencies in the full inventory also require installation checks.

## Environment Checks

After dependency installation succeeds, check package consistency:

```bat
python -m pip check
```

Expected output for a dependency-consistent environment:

```text
No broken requirements found.
```

Check the main imports and installed versions:

```bat
python -c "import torch, torchvision, numpy, cv2, scipy, skimage, PIL, matplotlib, tqdm, tensorboardX, einops, timm, lpips; print('Core imports OK'); print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected PyTorch and torchvision versions are `1.12.0+cu113` and `0.13.0+cu113`. `CUDA available` should be `True` for a correctly configured NVIDIA GPU environment.

Check the timm imports used by the repository:

```bat
python -c "from timm.models.layers import trunc_normal_, DropPath; from timm.models.registry import register_model; print('Repository timm imports OK')"
```

Check a basic GPU operation:

```bat
python -c "import torch; x=torch.ones(2, device='cuda'); print(x+x)"
```

The result should contain two values of `2.` on a CUDA device. These checks verify imports and basic GPU execution; they do not by themselves establish reproduction of the paper's results.

## Repository Setup

For each alignment variant, extract the contents of the source archives into the following directories under that variant's project directory:

| Archive | Target directory |
|---|---|
| `models.zip` | `models/` |
| `datasets.zip` | `datasets/` |
| `utils.zip` | `utils/` |
| `Test_Result.zip` | `Test_Result/` |
| `experiments_Upright_LOG.zip` | `experiments_Upright_LOG/` |

Avoid creating an extra nested directory when extracting archives. Run scripts from the corresponding variant's project directory so relative module and dataset-list paths resolve correctly. Configure dataset, checkpoint, and output paths for the local machine before training or evaluation; some released scripts retain machine-specific paths.

The training entry point is `train.py`; it uses the class defined in `DualProjectionFusionUp_ConvNext_ViT.py`. The evaluation entry point is `RunForPingjiaShiYan.py`. Additional metric scripts are included in `Test_Result/results/`. The bundled `pytorch_fid` source is used for FID evaluation, and `.m` scripts require MATLAB R2020a.

After setup, verify a small inference batch, a training iteration, and the required evaluation scripts. Installation and import checks should be followed by comparisons against the reference model outputs and metrics.

## References

- [Official PyTorch installation instructions for previous versions](https://pytorch.org/get-started/previous-versions/)
- [OpenCV Python dependency discussion](https://github.com/opencv/opencv-python/issues/1122)
