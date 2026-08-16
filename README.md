      
# EdGE: Streaming Endoscopy Geometry Estimation

**Generalizable Zero-shot** _Depth Estimation_ tool for endoscopic RGB images and videos.

As the inference code from 

<div align="center">
  
---
  
<var>
<h2> EdGE: A Foundation Model for Streaming Geometry Estimation from Monocular Endoscopic Observation </h2>
<h4> <a href='https://baymax-shao.netlify.app/'>Liangjing Shao</a><sup>1,2</sup>, Jinsong Lin<sup>1</sup>, Mingwu Su<sup>1</sup>, Rulin Zhou<sup>1,2</sup>, Haoxuan Wu<sup>1</sup>, Hongliang Ren<sup>1,2</sup> </h4>
<h5> <sup>1</sup>The Chinese University of Hong Kong  <sup>2</sup>Shenzhen Loop Area Institute</h5>
<h3> Under Review </h3>
</var> 
  
| **[[arXiv](<https://arxiv.org/abs/2307.05182>)]** |
|:-------------------:|
---
</div>

## Examples
#### Zero-shot example for Cholecystectomy
![](assets/exp1.mp4)

## Install

```bash
cd /path/to/EdGE-infer
conda create -n edge-infer python=3.11 -y
conda activate edge-infer
pip install -r requirements.txt

# video decode (AV1 / H264, …)
conda install -c conda-forge ffmpeg -y
```

Dowload pretrained weight from [Google Drive](https://drive.google.com/file/d/1F0m2Q0rTPo58yQz2EPVAqVHqY4nTyowW/view?usp=sharing) and Place weights at `weights/model.safetensors` (or pass `--weights`).

## Quick start

```bash
# single image (default resize 280×224)
python infer.py --input path/to/frame.png --weights weights/model.safetensors

# higher quality: larger model input (depth still saved at original resolution)
python infer.py --input path/to/frame.png --width 518 --height 392

# custom saved size (otherwise = original image/video size)
python infer.py --input path/to/frame.png --out_width 1280 --out_height 720

# image folder (temporal streaming across files, sorted by name)
python infer.py --input path/to/frames/ --out_dir outputs/seq

# video (every frame; use --sample_stride 5 to keep 0,5,10,…)
python infer.py --input path/to/video.mp4 --out_dir outputs/vid \
  --window_size 5 --bank_size 32 --sample_stride 1

# mask near-black pixels in depth (optional)
python infer.py --input video.mp4 --dark_thresh 25 --sample_stride 5
```

Outputs under `--out_dir` (default `outputs/<input_stem>/`):

| File                    | Meaning                                     |
| ----------------------- | ------------------------------------------- |
| `depth_XXXX.png`        | Grayscale depth                             |
| `depth_color_XXXX.png`  | Inferno colorized depth                     |
| `depth_masked_XXXX.png` | Dark-RGB masked gray (if `--dark_thresh>0`) |
| `depth_XXXX.npy`        | Raw float depth (if `--save_npy`)           |

## Python API

```python
import torch
from safetensors.torch import load_file
from edge import Edge, EdgeSession
from infer import preprocess_rgb_np  # or copy the helper

device = "cuda"
model = Edge()
model.load_state_dict(load_file("weights/model.safetensors"), strict=False)
model = model.to(device).eval()

session = EdgeSession(model, window_size=5, bank_size=32)

# rgb: HxWx3 uint8 → default 280×224
x = preprocess_rgb_np(rgb, width=280, height=224).to(device)  # [1,3,H,W]
with torch.no_grad():
    pred = session.forward_stream(x)
depth = pred["depth"][0, -1].float().cpu().numpy()  # HxWx1 or HxW
session.clear()
```

`EdgeSession` uses **bank** mode: GPU memory bank + short working KV (`window_size`).

## Notes

- Model input defaults to **280×224** (multiples of 14). Larger `--width`/`--height` usually improves depth quality at the cost of speed.
- After inference, depth is **resized back to the original image/video resolution** by default. Override with `--out_width` / `--out_height`.
- OpenCV cannot decode some AV1 files; `infer.py` uses **ffmpeg** pipes instead.

