# Frozen RT-DETRv4 Ego-Path Regression Report

## Result

All four local RT-DETRv4 HGNet checkpoints were used as frozen feature extractors and trained with the same ego-path regression head recipe. Artifacts are under:

`weights/rt_detrv4_weights/`

The best held-out test IoU is from the X backbone with flip TTA:

`weights/rt_detrv4_weights/rtdetrv4-x-640-tta`

| Run | Backbone | Method | Val IoU | Test IoU | Repo-Style Latency | Detector Mean |
|---|:---:|:---:|---:|---:|---:|---:|
| `rtdetrv4-s-640-deep-stage1` | RT-DETRv4-S | Regression | 0.9401 | 0.9390 | - | - |
| `rtdetrv4-m-640-deep-stage1` | RT-DETRv4-M | Regression | 0.9408 | 0.9388 | - | - |
| `rtdetrv4-l-640-deep-stage1` | RT-DETRv4-L | Regression | 0.9393 | 0.9380 | - | - |
| `rtdetrv4-x-640-deep-stage1` | RT-DETRv4-X | Regression | 0.9429 | 0.9416 | - | - |
| `rtdetrv4-s-640-tta` | RT-DETRv4-S | Regression + flip TTA | 0.9439 | 0.9430 | 3.07 ms | 15.138 ms |
| `rtdetrv4-m-640-tta` | RT-DETRv4-M | Regression + flip TTA | 0.9441 | 0.9437 | 4.08 ms | 17.648 ms |
| `rtdetrv4-l-640-tta` | RT-DETRv4-L | Regression + flip TTA | 0.9452 | 0.9425 | 4.90 ms | 18.516 ms |
| `rtdetrv4-x-640-tta` | RT-DETRv4-X | Regression + flip TTA | 0.9471 | 0.9458 | 7.85 ms | 21.625 ms |

The latency uses the same fields as the DINOv3 report. Repo-style latency is one PyTorch model forward on a dummy tensor and excludes preprocessing/postprocessing/TTA orchestration. Detector mean includes image tensor conversion, resize, host-to-device copy, flip TTA, model forward, and regression postprocessing on one `1280x720` RailSem image.

TensorRT was not profiled for these RT-DETRv4 runs in this pass; the table above is PyTorch CUDA latency.

## Approach

The RT-DETRv4 repository was cloned into `external/RT-DETRv4`, and the local checkpoints in `rtdetrv4_models/` were loaded:

- `RTv4-S-hgnet.pth`
- `RTv4-M-hgnet.pth`
- `RTv4-L-hgnet.pth`
- `RTv4-X-hgnet.pth`

Each model uses the detector's frozen HGNet backbone plus HybridEncoder feature pyramid. The detector decoder is not used for ego-path prediction. The selected feature maps are resized to the stride-16 level, concatenated, and passed through a trainable regression adapter/head.

The backbone and encoder stay frozen:

- `requires_grad=False` for every RT-DETRv4 feature parameter.
- The feature extractor is kept in eval mode.
- RT-DETRv4 deploy conversion is used to fuse supported inference blocks.
- Only the ego-path adapter and regression MLP are optimized.

The regression target/loss is the same TEP-Net-style setup used for the DINOv3 experiments:

- 64 anchors.
- Two x-coordinate rail predictions per anchor.
- One y-limit prediction.
- Perspective-weighted SmoothL1 trajectory loss plus y-limit loss.

## Training Recipe

All four trained runs used:

- Input resolution: `640x640`.
- Batch size: `32`.
- Epochs: `250`.
- Optimizer: AdamW.
- Scheduler: OneCycleLR.
- Max LR: `1e-3`.
- Weight decay: `1e-3`.
- Gradient clip: `1.0`.
- AMP: CUDA bf16.
- `torch.compile`: disabled for these RT-DETRv4 runs.
- Head: adapter channels `512`, adapter depth `3`, pool channels `16`, FC hidden size `2048`.

I did not cache backbone features for training. The active training dataset uses stochastic rail-centered crop, horizontal flip, and color augmentation. Caching one feature tensor per original image would not match the augmented crop geometry or color distribution seen by the head. Caching could help deterministic validation/profiling, but that is not the training bottleneck here. The practical speed choices were frozen no-grad feature extraction, deployed RT-DETRv4 blocks, bf16 AMP, batch size 32, pinned dataloaders, and persistent workers.

## Commands

Set up the optional RT-DETRv4 dependencies:

```bash
.venv/bin/pip install -r requirements-rtdetrv4.txt
git clone https://github.com/RT-DETRs/RT-DETRv4 external/RT-DETRv4
```

Train the four frozen-backbone heads:

```bash
.venv/bin/python train.py regression rtdetrv4-s --device cuda --run-name rtdetrv4-s-640 --save-run-name rtdetrv4-s-640-deep-stage1 --weights-subdir rt_detrv4_weights --epochs 250 --batch-size 32 --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
.venv/bin/python train.py regression rtdetrv4-m --device cuda --run-name rtdetrv4-m-640 --save-run-name rtdetrv4-m-640-deep-stage1 --weights-subdir rt_detrv4_weights --epochs 250 --batch-size 32 --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
.venv/bin/python train.py regression rtdetrv4-l --device cuda --run-name rtdetrv4-l-640 --save-run-name rtdetrv4-l-640-deep-stage1 --weights-subdir rt_detrv4_weights --epochs 250 --batch-size 32 --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
.venv/bin/python train.py regression rtdetrv4-x --device cuda --run-name rtdetrv4-x-640 --save-run-name rtdetrv4-x-640-deep-stage1 --weights-subdir rt_detrv4_weights --epochs 250 --batch-size 32 --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
```

Create the calibrated flip-TTA artifacts:

```bash
.venv/bin/python train.py regression rtdetrv4-s --device cuda --run-name rtdetrv4-s-640 --resume-from weights/rt_detrv4_weights/rtdetrv4-s-640-deep-stage1 --calibrate-postprocess --save-run-name rtdetrv4-s-640-tta --weights-subdir rt_detrv4_weights --calibration-iterations 3 --inference-tta-flip --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
.venv/bin/python train.py regression rtdetrv4-m --device cuda --run-name rtdetrv4-m-640 --resume-from weights/rt_detrv4_weights/rtdetrv4-m-640-deep-stage1 --calibrate-postprocess --save-run-name rtdetrv4-m-640-tta --weights-subdir rt_detrv4_weights --calibration-iterations 3 --inference-tta-flip --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
.venv/bin/python train.py regression rtdetrv4-l --device cuda --run-name rtdetrv4-l-640 --resume-from weights/rt_detrv4_weights/rtdetrv4-l-640-deep-stage1 --calibrate-postprocess --save-run-name rtdetrv4-l-640-tta --weights-subdir rt_detrv4_weights --calibration-iterations 3 --inference-tta-flip --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
.venv/bin/python train.py regression rtdetrv4-x --device cuda --run-name rtdetrv4-x-640 --resume-from weights/rt_detrv4_weights/rtdetrv4-x-640-deep-stage1 --calibrate-postprocess --save-run-name rtdetrv4-x-640-tta --weights-subdir rt_detrv4_weights --calibration-iterations 3 --inference-tta-flip --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
```

Benchmark the calibrated TTA artifacts:

```bash
.venv/bin/python train.py regression rtdetrv4-s --device cuda --run-name rtdetrv4-s-640 --resume-from weights/rt_detrv4_weights/rtdetrv4-s-640-tta --benchmark-latency --save-run-name rtdetrv4-s-640-tta --weights-subdir rt_detrv4_weights --latency-warmup 50 --latency-runs 1000 --inference-tta-flip --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
.venv/bin/python train.py regression rtdetrv4-m --device cuda --run-name rtdetrv4-m-640 --resume-from weights/rt_detrv4_weights/rtdetrv4-m-640-tta --benchmark-latency --save-run-name rtdetrv4-m-640-tta --weights-subdir rt_detrv4_weights --latency-warmup 50 --latency-runs 1000 --inference-tta-flip --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
.venv/bin/python train.py regression rtdetrv4-l --device cuda --run-name rtdetrv4-l-640 --resume-from weights/rt_detrv4_weights/rtdetrv4-l-640-tta --benchmark-latency --save-run-name rtdetrv4-l-640-tta --weights-subdir rt_detrv4_weights --latency-warmup 50 --latency-runs 1000 --inference-tta-flip --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
.venv/bin/python train.py regression rtdetrv4-x --device cuda --run-name rtdetrv4-x-640 --resume-from weights/rt_detrv4_weights/rtdetrv4-x-640-tta --benchmark-latency --save-run-name rtdetrv4-x-640-tta --weights-subdir rt_detrv4_weights --latency-warmup 50 --latency-runs 1000 --inference-tta-flip --pool-channels 16 --fc-hidden-size 2048 --rtdetrv4-adapter-channels 512 --rtdetrv4-adapter-depth 3 --no-compile
```

## Artifacts

- Best RT-DETRv4-S: `weights/rt_detrv4_weights/rtdetrv4-s-640-tta/best.pt`
- Best RT-DETRv4-M: `weights/rt_detrv4_weights/rtdetrv4-m-640-tta/best.pt`
- Best RT-DETRv4-L: `weights/rt_detrv4_weights/rtdetrv4-l-640-tta/best.pt`
- Best RT-DETRv4-X: `weights/rt_detrv4_weights/rtdetrv4-x-640-tta/best.pt`
