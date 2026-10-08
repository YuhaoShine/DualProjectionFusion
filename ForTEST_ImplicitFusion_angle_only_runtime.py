import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from thop import profile

import datasets
from models import TwoBranchConvNextViT


# ==================================================================================================
# Angle-only deployment path extracted from the trained Full Model (Imp-Trans)
# ==================================================================================================
class AngleOnlyFromFullImpTrans(torch.nn.Module):
    """
    Deployment-oriented angle-only path extracted from the jointly trained
    Full Model (Imp-Trans).

    IMPORTANT:
    According to the actual Full Model implementation:

        ConVNext_Results = ConvNext_branch(img)
        Cf5 = ConVNext_Results["equi_enc_feat4"]

    and inside Fuse_module.forward():

        p_and_R_Angle = avgpool8(Cf5)
        p_and_R_Angle = fc(p_and_R_Angle)
        p_and_R_Angle = sigmoid1(p_and_R_Angle)

    Therefore, the angle prediction uses the RAW deepest ERP ConvNeXt feature
    Cf5 directly. It does NOT use:
      1) the ViT / cubemap branch,
      2) the cross-projection fusion blocks,
      3) the channel-attention blocks in Fuse_module,
      4) the LUT-generation decoder.

    Retained deployment path:
        ERP input
        -> Full Model ConvNeXt encoder
        -> deepest feature Cf5
        -> Full Model avgpool8
        -> Full Model fc
        -> Full Model sigmoid1
        -> Pitch/Roll

    The angle-regression head is taken from full_model.fuse_branch because these
    are exactly the parameters used for PitchRoll_norm_pred in the trained
    Full Model (Imp-Trans).
    """

    def __init__(self, full_model):
        super().__init__()

        # ConvNeXt ERP encoder from the trained Full Model.
        self.equi_encoder = full_model.ConvNext_branch.equi_encoder

        # Exact angle-regression head used by Fuse_module.forward().
        self.avgpool8 = full_model.fuse_branch.avgpool8
        self.fc = full_model.fuse_branch.fc
        self.sigmoid1 = full_model.fuse_branch.sigmoid1

    def forward(self, input_equi_image):
        # Full ConvNeXt encoder. Only the deepest feature Cf5 is needed.
        _, _, _, _, feat4 = self.equi_encoder(input_equi_image)

        # IMPORTANT: no att4 here.
        # Fuse_module.forward() regresses Pitch/Roll directly from raw Cf5.
        angle = self.avgpool8(feat4)
        angle = angle.view(angle.size(0), -1)
        angle = self.fc(angle)
        angle = self.sigmoid1(angle)

        return angle * 180.0 - 90.0


# ==================================================================================================
# Test / benchmark class
# ==================================================================================================
class GLPanoDepth:
    def __init__(self, args):
        self.settings = args
        self.epoch = 99

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and len(self.settings.gpu_devices) else "cpu"
        )
        self.gpu_devices = ','.join([str(i) for i in self.settings.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)

        # ------------------------------------------------------------------------------------------
        # Test data
        # ------------------------------------------------------------------------------------------
        self.dataset = datasets.loadDataLUT

        # M3D
        test_file_list = './datasets/M3D_RGBLUT_PitchRoll_testlabel.txt'

        test_dataset = self.dataset(self.settings.data_path, test_file_list)
        self.test_loader = DataLoader(
            test_dataset,
            self.settings.batch_size_test,
            False,
            num_workers=self.settings.num_workers,
            pin_memory=True,
            drop_last=False
        )

        self.settings.cube_w = self.settings.height // 2

        # ------------------------------------------------------------------------------------------
        # Load the trained FULL Imp-Trans model first.
        # We need it only to load the jointly trained weights and to verify that
        # the extracted angle-only path reproduces the full-model angle output.
        # ------------------------------------------------------------------------------------------
        self.model = TwoBranchConvNextViT(
            image_height=self.settings.height,
            image_width=self.settings.width
        )

        model_loadpath = (
            'I:\\pingjiashiyan4paper\\M3D\\ConvNeXt_P_H_C_V\\'
            'experiments_Upright_LOG\\GLPanoUpright\\models\\weights_98\\model.pth'
        )

        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(model_loadpath, map_location=self.device)

        # Compatible with checkpoints optionally wrapped in state_dict.
        if isinstance(pretrained_dict, dict) and "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]

        clean_state = {}
        for k, v in pretrained_dict.items():
            if k.startswith("module."):
                k = k[7:]
            clean_state[k] = v

        pretrained_dict = {
            k: v for k, v in clean_state.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded full-model tensors: {len(pretrained_dict)}/{len(model_dict)}")

        # Extract the angle-only deployment path FROM THE TRAINED FULL MODEL.
        self.angle_model = AngleOnlyFromFullImpTrans(self.model).to(self.device).eval()

        self.save_settings()

    # ==============================================================================================
    def validate(self):
        """
        Benchmark the angle-only deployment path extracted from the Full Model (Imp-Trans).

        Timing scope:
            ERP ConvNeXt encoder + full-model angle-regression head.

        Explicitly excluded:
            - ViT / cubemap branch
            - cross-projection feature fusion
            - LUT-generation decoder
            - ERP-to-cubemap preprocessing
            - geometric image resampling
            - disk I/O
        """
        self.model.eval()
        self.angle_model.eval()

        pbar = tqdm.tqdm(self.test_loader)
        pbar.set_description("testing Full Imp-Trans angle-only deployment")

        abs_component_errors = []
        benchmark_done = False

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):

                # Only ERP input is needed by the actual angle-only deployment path.
                equi_inputs = inputs["normalized_rgb_Rotate"].to(self.device, non_blocking=True)
                gt_angle = inputs["PitchRollAng"].to(self.device, non_blocking=True)

                # ==================================================================================
                # One-time correctness check + runtime / complexity benchmark
                # ==================================================================================
                if not benchmark_done:
                    # ------------------------------------------------------------------------------
                    # 1) Verify that the extracted path reproduces the angle output of the FULL model.
                    #    This full-model forward is ONLY a correctness check and is NOT timed.
                    # ------------------------------------------------------------------------------
                    cube_input = inputs["normalized_cube_rgb"].to(self.device, non_blocking=True)

                    full_outputs_check = self.model(equi_inputs, cube_input)
                    angle_full = full_outputs_check["PitchRoll_norm_pred"]
                    angle_only_check = self.angle_model(equi_inputs)

                    max_angle_diff = (angle_full - angle_only_check).abs().max().item()
                    mean_angle_diff = (angle_full - angle_only_check).abs().mean().item()

                    print("\n========== Full-model vs angle-only equivalence check ==========")
                    print(f"max  |full-angle - angle-only| : {max_angle_diff:.10f} deg")
                    print(f"mean |full-angle - angle-only| : {mean_angle_diff:.10f} deg")
                    print("Expected: numerical zero (or only floating-point-level difference).")
                    print("================================================================")

                    # A non-negligible difference means that the extracted Cf5 path does not exactly
                    # match the feature used inside the full-model forward and must be inspected.
                    if max_angle_diff > 1e-4:
                        print(
                            "WARNING: angle-only output does not exactly match the Full Model. "
                            "Please verify that Fuse_module.forward() regresses Pitch/Roll directly "
                            "from Cf5 and that the loaded checkpoint matches this model definition."
                        )

                    # Free full-forward temporary tensors before benchmarking.
                    del full_outputs_check, angle_full, angle_only_check, cube_input
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                    # ------------------------------------------------------------------------------
                    # 2) FLOPs / Params of the RETAINED angle-only path only.
                    # ------------------------------------------------------------------------------
                    flops, params = profile(
                        self.angle_model,
                        inputs=(equi_inputs,),
                        verbose=False
                    )
                    flops_g = flops / 1e9
                    params_m = params / 1e6

                    # Approximate FP32 deployment weight size of retained modules only.
                    weight_bytes = sum(
                        p.numel() * p.element_size()
                        for p in self.angle_model.parameters()
                    )
                    buffer_bytes = sum(
                        b.numel() * b.element_size()
                        for b in self.angle_model.buffers()
                    )
                    size_mib = (weight_bytes + buffer_bytes) / (1024 ** 2)

                    # ------------------------------------------------------------------------------
                    # 3) Warm-up + CUDA-event latency.
                    # ------------------------------------------------------------------------------
                    warmup_runs = 50
                    timing_runs = 200

                    for _ in range(warmup_runs):
                        _ = self.angle_model(equi_inputs)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                        starter = torch.cuda.Event(enable_timing=True)
                        ender = torch.cuda.Event(enable_timing=True)
                        elapsed_ms = []

                        for _ in range(timing_runs):
                            starter.record()
                            _ = self.angle_model(equi_inputs)
                            ender.record()
                            torch.cuda.synchronize()
                            elapsed_ms.append(starter.elapsed_time(ender))
                    else:
                        # CPU fallback. GPU/CUDA-event results should be used for the paper.
                        import time
                        elapsed_ms = []
                        for _ in range(timing_runs):
                            t0 = time.perf_counter()
                            _ = self.angle_model(equi_inputs)
                            elapsed_ms.append((time.perf_counter() - t0) * 1000.0)

                    avg_time = float(np.mean(elapsed_ms))
                    std_time = float(np.std(elapsed_ms))
                    fps = 1000.0 / avg_time

                    print("\n========== Full Imp-Trans: Angle-Only deployment benchmark ==========")
                    print(f"ERP input        : {equi_inputs.shape[-2]}x{equi_inputs.shape[-1]}")
                    print(f"Batch size       : {equi_inputs.shape[0]}")
                    print(f"FLOPs            : {flops_g:.2f} G")
                    print(f"Params           : {params_m:.2f} M")
                    print(f"FP32 weight size : {size_mib:.2f} MiB")
                    print(f"Latency          : {avg_time:.2f} +/- {std_time:.2f} ms")
                    print(f"FPS              : {fps:.2f}")
                    print(
                        "Timing scope      : ERP ConvNeXt encoder + Full-Model angle-regression head"
                    )
                    print(
                        "Excluded          : ViT/cubemap branch, cross-projection fusion, "
                        "LUT decoder, ERP-to-cubemap preprocessing, geometric resampling"
                    )
                    print("=======================================================================\n")

                    benchmark_done = True

                # ==================================================================================
                # Angle-only prediction for the whole test set (optional sanity check).
                # No ViT branch and no decoder are executed here.
                # ==================================================================================
                pred_angle = self.angle_model(equi_inputs)

                component_abs_err = (pred_angle - gt_angle).abs().detach().cpu().numpy()
                abs_component_errors.append(component_abs_err)

        # --------------------------------------------------------------------------------------------------
        # Component-wise statistics are only a sanity check here; keep your manuscript's established
        # angular-error definition for the official accuracy tables.
        # --------------------------------------------------------------------------------------------------
        if len(abs_component_errors) > 0:
            Err = np.concatenate(abs_component_errors, axis=0)
            print("========== Angle-only component-error sanity check ==========")
            print("Pitch/Roll min    :", np.round(np.min(Err, axis=0), 4))
            print("Pitch/Roll median :", np.round(np.median(Err, axis=0), 4))
            print("Pitch/Roll max    :", np.round(np.max(Err, axis=0), 4))
            print("Pitch/Roll mean   :", np.round(np.mean(Err, axis=0), 4))
            print("==============================================================")

    # ==============================================================================================
    def save_settings(self):
        """Save run settings."""
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        to_save = self.settings.__dict__.copy()
        with open(os.path.join(models_dir, "settings.json"), "w") as f:
            json.dump(to_save, f, indent=2)
