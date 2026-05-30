# Fixed-DINOv3 Ego-Path Regression Report

## Result

The accepted model is:

`weights/dinov3-vits16plus-512-tta`

It uses a frozen `dinov3-vits16plus` ViT backbone at `512x512`, a trainable regression adapter/head, and horizontal flip test-time augmentation (TTA) at inference. The held-out test IoU is above the requested threshold.

| Run | Backbone | Method | Latency PyTorch / TensorRT | IoU |
|---|:---:|:---:|:---:|:---:|
| `dinov3-vits16plus-512-tta` | DINOv3 ViT-S+ | Regression + flip TTA | 4.97 / 1.16 ms | 0.9608 |
| `dinov3-vits16plus-512-deep-stage1` | DINOv3 ViT-S+ | Regression | 4.91 / 1.15 ms | 0.9589 |
| `dinov3-vits16plus-512-lr1e-5-constant` | DINOv3 ViT-S+ | Regression | not rerun / same architecture | 0.9588 |
| `dinov3-vits16-512-stage1` | DINOv3 ViT-S | Regression | 4.18 / 0.94 ms | 0.9317 |

The ViT-S+ rows use the deeper `512` channel, depth `3` adapter/head. The ViT-S baseline row is the original smaller adapter/head run (`256` channel, depth `1`), included because it was the first completed smallest-backbone baseline.

The latency column above is intentionally formatted like the original README table. It follows the original repo's PyTorch/TensorRT latency method from `eval.py` and `LatencyEvaluator`: model forward on a dummy tensor, reported in milliseconds. That measurement excludes image preprocessing, postprocessing, TTA orchestration, and disk I/O. For the TTA model, the listed TensorRT latency is one engine forward; deployment with flip TTA performs two forwards.

Latency was measured on the local RTX 4090 with driver `580.126.09`, PyTorch `2.12.0+cu130`, torchvision `0.27.0+cu130`, TensorRT `11.0.0.114`, pycuda `2026.1`, and ONNX `1.21.0`. TensorRT 11 removes the old weak-typing precision flags such as `BuilderFlag.FP16`; NVIDIA's [Python migration docs](https://docs.nvidia.com/deeplearning/tensorrt/11.0.0/api/migration/tensorrt-10x-to-11x-python-api.html) and [trtexec migration docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html) describe that networks are strongly typed in TensorRT 11. To get a reduced-precision engine comparable to the original repo's TensorRT path, I exported an explicit FP16 ONNX graph and built `best.fp16.trt`.

For operational context, I also measured full detector latency on one `1280x720` RailSem image (`rs00444.jpg`). This includes image tensor conversion, resize/normalization, host-to-device copy, forward pass, optional flip TTA, and regression postprocessing:

| Run | TTA | Detector Mean | Detector P95 | FPS |
|---|:---:|---:|---:|---:|
| `dinov3-vits16plus-512-tta` | yes | 15.528 ms | 16.231 ms | 64.4 |
| `dinov3-vits16plus-512-deep-stage1` | no | 8.317 ms | 8.760 ms | 120.2 |
| `dinov3-vits16-512-stage1` | no | 7.575 ms | 7.996 ms | 132.0 |

## Approach

The supervised target was the ego-path JSON labels in `egopath/rs19_egopath.json`, not the COCO boxes. The Roboflow images are `1280x720`, while the ego-path labels are in original `1920x1080` coordinates, so the dataset code scales annotations before generating regression targets and masks.

The model follows the TEP-Net style regression target:

- 64 vertical anchors.
- Two x-coordinate rail predictions per anchor.
- One sigmoid y-limit prediction.
- Perspective-weighted SmoothL1 trajectory loss plus y-limit loss.

The backbone is fixed DINOv3:

- `requires_grad=False` for every DINOv3 parameter.
- The backbone is kept in eval mode.
- Features are extracted from the last 4 transformer blocks using DINOv3 `get_intermediate_layers(..., reshape=True, norm=True)`.
- The head concatenates the 4 patch feature maps, applies a small trainable convolutional adapter, pools with a 1x1 projection, and predicts `2 * anchors + 1` regression values.

The training recipe was tuned for this machine:

- GPU: NVIDIA RTX 4090, 24 GB.
- CPU: Intel Core i7-14700KF, 28 logical CPUs.
- RAM: 62 GiB.
- Batch size 32 at `512x512`.
- CUDA bf16 autocast, TF32 matmul, pinned dataloaders, persistent workers, and `torch.compile` during training.
- AdamW over trainable adapter/head parameters only.
- OneCycle schedule with max LR `1e-3`, weight decay `1e-3`, gradient clip `1.0`, 250 epochs.

## Findings

The smallest ViT-S backbone trained cleanly but plateaued far below the target: `0.931716` test IoU.

The default ViT-S+ head was better but still not close enough early in training. Increasing input size to `768x768` with batch size 16 did not help enough for the time cost; that run reached about `0.873` validation IoU by epoch 50 and was stopped.

A deeper trainable adapter was the key architecture change. With `dinov3-vits16plus`, adapter channels `512`, adapter depth `3`, pool channels `16`, and FC hidden size `2048`, the frozen-backbone run reached `0.959603` validation IoU and `0.958918` test IoU without TTA.

Postprocess calibration was checked as a possible source of the remaining gap. Grid search over y-limit offset and rail-width scale selected the unmodified decoder: `ylimit_offset=0.0`, `rail_width_scale=1.0`, `center_offset=0.0`. That means the miss was not caused by an obvious global width or y-limit bias.

A constant `1e-5` LR continuation found a validation checkpoint above 0.96 (`0.960242`) but did not improve held-out test IoU (`0.958782`), so it was not accepted.

Flip TTA was the decisive final step. It averages the original prediction with a horizontally flipped prediction transformed back into the original coordinate frame. This kept the same trained frozen-DINOv3 model and improved the held-out test IoU to `0.960845`.

## Recreate The Accepted Model

Set up the environment and DINOv3 repo:

```bash
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-tensorrt.txt --extra-index-url https://pypi.nvidia.com
git clone https://github.com/facebookresearch/dinov3 external/dinov3
```

Train the accepted checkpoint weights from scratch:

```bash
.venv/bin/python train.py regression dinov3-vits16plus --device cuda --run-name dinov3-vits16plus-512 --save-run-name dinov3-vits16plus-512-deep-stage1 --dinov3-adapter-channels 512 --dinov3-adapter-depth 3 --pool-channels 16 --fc-hidden-size 2048
```

Create the final accepted TTA artifact from that checkpoint:

```bash
.venv/bin/python train.py regression dinov3-vits16plus --device cuda --run-name dinov3-vits16plus-512 --resume-from weights/dinov3-vits16plus-512-deep-stage1 --calibrate-postprocess --save-run-name dinov3-vits16plus-512-tta --calibration-iterations 3 --inference-tta-flip --dinov3-adapter-channels 512 --dinov3-adapter-depth 3 --pool-channels 16 --fc-hidden-size 2048
```

Benchmark the accepted artifact. This writes both the repo-style PyTorch forward latency and the full detector latency:

```bash
.venv/bin/python train.py regression dinov3-vits16plus --device cuda --run-name dinov3-vits16plus-512 --resume-from weights/dinov3-vits16plus-512-tta --benchmark-latency --benchmark-tensorrt --trt-precision fp16 --save-run-name dinov3-vits16plus-512-tta --latency-warmup 50 --latency-runs 1000 --inference-tta-flip --dinov3-adapter-channels 512 --dinov3-adapter-depth 3 --pool-channels 16 --fc-hidden-size 2048
```

The benchmark writes:

`weights/dinov3-vits16plus-512-tta/latency.yaml`

The repo-style latency field in that file is:

```yaml
repo_style_pytorch_forward:
  runtime: pytorch
  precision: fp32
  runs: 1000
repo_style_tensorrt_forward:
  runtime: tensorrt
  precision: fp16
  runs: 1000
```

## Variant Commands

Same recipe with the smaller ViT-S backbone:

```bash
.venv/bin/python train.py regression dinov3-vits16 --device cuda --run-name dinov3-vits16-512 --dinov3-adapter-channels 512 --dinov3-adapter-depth 3 --pool-channels 16 --fc-hidden-size 2048
```

Same ViT-S+ recipe at `768x768`:

```bash
.venv/bin/python train.py regression dinov3-vits16plus --device cuda --run-name dinov3-vits16plus-768 --input-size 768 --batch-size 16 --epochs 125 --dinov3-adapter-channels 512 --dinov3-adapter-depth 3 --pool-channels 16 --fc-hidden-size 2048
```

Same ViT-S+ recipe without flip TTA:

```bash
.venv/bin/python train.py regression dinov3-vits16plus --device cuda --run-name dinov3-vits16plus-512 --save-run-name dinov3-vits16plus-512-no-tta --dinov3-adapter-channels 512 --dinov3-adapter-depth 3 --pool-channels 16 --fc-hidden-size 2048
```

Benchmark a non-TTA ViT-S+ checkpoint:

```bash
.venv/bin/python train.py regression dinov3-vits16plus --device cuda --run-name dinov3-vits16plus-512 --resume-from weights/dinov3-vits16plus-512-deep-stage1 --benchmark-latency --benchmark-tensorrt --trt-precision fp16 --save-run-name dinov3-vits16plus-512-deep-stage1 --latency-warmup 50 --latency-runs 1000 --dinov3-adapter-channels 512 --dinov3-adapter-depth 3 --pool-channels 16 --fc-hidden-size 2048
```

Benchmark the ViT-S baseline:

```bash
.venv/bin/python train.py regression dinov3-vits16 --device cuda --run-name dinov3-vits16-512 --resume-from weights/dinov3-vits16-512-stage1 --benchmark-latency --benchmark-tensorrt --trt-precision fp16 --save-run-name dinov3-vits16-512-stage1 --latency-warmup 50 --latency-runs 1000
```

## Artifacts

- Accepted model: `weights/dinov3-vits16plus-512-tta/best.pt`
- Accepted TensorRT FP16 engine: `weights/dinov3-vits16plus-512-tta/best.fp16.trt`
- Accepted ONNX FP16 export: `weights/dinov3-vits16plus-512-tta/best.fp16.onnx`
- Accepted config: `weights/dinov3-vits16plus-512-tta/config.yaml`
- Accepted metrics: `weights/dinov3-vits16plus-512-tta/metrics.yaml`
- Accepted latency: `weights/dinov3-vits16plus-512-tta/latency.yaml`
- Best non-TTA source checkpoint: `weights/dinov3-vits16plus-512-deep-stage1/best.pt`
