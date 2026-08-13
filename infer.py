"""EdGE depth inference CLI for images and videos.

Examples
--------
  # single image / image folder
  python infer.py --input path/to/img.jpg --weights weights/model.safetensors
  python infer.py --input path/to/frames/ --weights weights/model.safetensors --out_dir outs/

  # video (AV1/H264/...)
  python infer.py --input path/to/video.mp4 --weights weights/model.safetensors \\
      --sample_stride 1 --out_dir outs/video_name
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge.models.edge import Edge
from edge.session import EdgeSession

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def resolve_bin(name: str) -> str:
    for c in (Path(sys.executable).resolve().parent / name, Path(sys.prefix) / "bin" / name):
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"'{name}' not found. Install ffmpeg: conda install -c conda-forge ffmpeg"
    )


def load_model(weights: str, device: str) -> Edge:
    weights = os.path.abspath(weights)
    model = Edge()
    if weights.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(weights)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded {weights}: missing={len(missing)} unexpected={len(unexpected)}")
    elif os.path.isdir(weights):
        model = Edge.from_pretrained(weights)
    else:
        raise ValueError(f"Unsupported weights: {weights}")
    model = model.to(device).eval()
    # Speed knobs for H100 / Ampere+
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return model


PATCH_SIZE = 14
DEFAULT_WIDTH = 280
DEFAULT_HEIGHT = 224


def snap_to_patch(size: int, patch: int = PATCH_SIZE) -> int:
    """Round to nearest multiple of ``patch`` (ViT patch size)."""
    return max(patch, int(round(size / patch)) * patch)


def resolve_model_hw(width: int, height: int) -> tuple[int, int]:
    """Return model input (in_w, in_h) snapped to multiples of 14."""
    in_w, in_h = snap_to_patch(width), snap_to_patch(height)
    if (in_w, in_h) != (width, height):
        print(f"Note: model input {width}x{height} → {in_w}x{in_h} (multiples of {PATCH_SIZE})")
    return in_w, in_h


def resolve_out_hw(
    src_w: int,
    src_h: int,
    out_width: int | None = None,
    out_height: int | None = None,
) -> tuple[int, int]:
    """Default output size = original source resolution; optional overrides."""
    ow = int(src_w if out_width is None else out_width)
    oh = int(src_h if out_height is None else out_height)
    return ow, oh


def preprocess_rgb_np(
    rgb: np.ndarray,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> torch.Tensor:
    """RGB HxWx3 uint8 → [1,3,H,W] float in [0,1], resized to width×height."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    w = snap_to_patch(width)
    h = snap_to_patch(height)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    img = torch.tensor(np.ascontiguousarray(resized), dtype=torch.float32).permute(2, 0, 1) / 255.0
    return img.unsqueeze(0)


def resize_hw(arr: np.ndarray, width: int, height: int, nearest: bool = False) -> np.ndarray:
    """Resize HxW or HxWxC array to height×width."""
    if arr.shape[0] == height and arr.shape[1] == width:
        return arr
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(arr, (width, height), interpolation=interp)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return out
    lo, hi = np.percentile(depth[valid], [2, 98])
    dn = np.clip((depth - lo) / (hi - lo + 1e-8), 0, 1)
    colored = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colored[~valid] = 0
    return colored


def depth_to_gray_u8(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros(depth.shape, dtype=np.uint8)
    if not valid.any():
        return out
    lo, hi = np.percentile(depth[valid], [2, 98])
    dn = np.clip((depth - lo) / (hi - lo + 1e-8), 0, 1)
    out[valid] = (dn[valid] * 255.0).astype(np.uint8)
    return out


def dark_mask(rgb: np.ndarray, thresh: float) -> np.ndarray:
    return rgb.astype(np.float32).mean(axis=-1) >= float(thresh)


def trim_predictions(session: EdgeSession, keep: int = 8) -> None:
    if not hasattr(session, "predictions"):
        return
    for k, v in list(session.predictions.items()):
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] > keep:
            session.predictions[k] = v[:, -keep:]


def probe_video(path: Path) -> dict:
    try:
        ffprobe = resolve_bin("ffprobe")
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
        st = json.loads(out)["streams"][0]
        w, h = int(st["width"]), int(st["height"])
        rate = st.get("r_frame_rate", "30/1")
        if "/" in rate:
            a, b = rate.split("/")
            fps = float(a) / max(float(b), 1e-8)
        else:
            fps = float(rate) if rate else 30.0
        return {"w": w, "h": h, "fps": fps}
    except FileNotFoundError:
        pass
    ffmpeg = resolve_bin("ffmpeg")
    err = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr or ""
    import re

    m = re.search(r"(\d{2,5})x(\d{2,5})", err)
    if not m:
        raise RuntimeError(f"Cannot probe {path}")
    return {"w": int(m.group(1)), "h": int(m.group(2)), "fps": 30.0}


def iter_video_rgb(path: Path, w: int, h: int):
    ffmpeg = resolve_bin("ffmpeg")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    nbytes = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    finally:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS])


def save_depth_outputs(
    out_dir: Path,
    stem: str,
    depth: np.ndarray,
    rgb: np.ndarray | None = None,
    dark_thresh: float = 0.0,
    save_color: bool = True,
    save_gray: bool = True,
    save_npy: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = depth.astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if rgb is not None and dark_thresh > 0:
        m = dark_mask(rgb, dark_thresh)
        if m.shape != d.shape:
            m = cv2.resize(m.astype(np.uint8), (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        d_masked = d.copy()
        d_masked[~m] = 0.0
    else:
        d_masked = None

    if save_gray:
        cv2.imwrite(str(out_dir / f"depth_{stem}.png"), depth_to_gray_u8(d))
        if d_masked is not None:
            cv2.imwrite(str(out_dir / f"depth_masked_{stem}.png"), depth_to_gray_u8(d_masked))
    if save_color:
        cv2.imwrite(str(out_dir / f"depth_color_{stem}.png"), colorize_depth(d))
        if d_masked is not None:
            cv2.imwrite(str(out_dir / f"depth_masked_color_{stem}.png"), colorize_depth(d_masked))
    if save_npy:
        np.save(out_dir / f"depth_{stem}.npy", d)
        if d_masked is not None:
            np.save(out_dir / f"depth_masked_{stem}.npy", d_masked)


@torch.inference_mode()
def infer_images(
    paths: list[Path],
    model: Edge,
    device: str,
    out_dir: Path,
    window_size: int = 5,
    bank_size: int = 32,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    out_width: int | None = None,
    out_height: int | None = None,
    dark_thresh: float = 0.0,
    save_npy: bool = False,
    amp: bool = True,
    cuda_graphs: bool = True,
) -> None:
    in_w, in_h = resolve_model_hw(width, height)
    session = EdgeSession(
        model,
        window_size=window_size,
        bank_size=bank_size,
        depth_only=True,
        pad_working_set=cuda_graphs,
    )
    use_amp = bool(amp and str(device).startswith("cuda") and torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    graphs_ready = False
    for i, p in enumerate(tqdm(paths, desc="images")):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip unreadable: {p}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        src_h, src_w = rgb.shape[:2]
        ow, oh = resolve_out_hw(src_w, src_h, out_width, out_height)
        x = preprocess_rgb_np(rgb, width=in_w, height=in_h).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp, cache_enabled=False):
            pred = session.forward_stream(x)
        if (
            cuda_graphs
            and not graphs_ready
            and str(device).startswith("cuda")
            and torch.cuda.is_available()
            and session.num_bank_frames >= 1
        ):
            from edge.cuda_graphs import install_cuda_graphs

            install_cuda_graphs(
                model, in_h, in_w, window_size=window_size, dtype=amp_dtype, sample_image=x
            )
            graphs_ready = True
            session.pad_working_set = True
        depth = pred["depth"][0, -1].detach().float().cpu().numpy()
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = resize_hw(depth.astype(np.float32), ow, oh)
        # Use original RGB for dark-mask / alignment (not the model-resized tensor).
        rgb_m = resize_hw(rgb, ow, oh)
        stem = f"{i:04d}_{p.stem}"
        save_depth_outputs(out_dir, stem, depth, rgb=rgb_m, dark_thresh=dark_thresh, save_npy=save_npy)
    session.clear()


@torch.inference_mode()
def infer_video(
    path: Path,
    model: Edge,
    device: str,
    out_dir: Path,
    window_size: int = 5,
    bank_size: int = 32,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    out_width: int | None = None,
    out_height: int | None = None,
    sample_stride: int = 1,
    dark_thresh: float = 0.0,
    save_npy: bool = False,
    amp: bool = True,
    cuda_graphs: bool = True,
) -> None:
    in_w, in_h = resolve_model_hw(width, height)
    meta = probe_video(path)
    src_w, src_h = meta["w"], meta["h"]
    ow, oh = resolve_out_hw(src_w, src_h, out_width, out_height)
    print(f"Video {src_w}x{src_h} → model {in_w}x{in_h} → output {ow}x{oh}")
    session = EdgeSession(
        model,
        window_size=window_size,
        bank_size=bank_size,
        depth_only=True,
        pad_working_set=cuda_graphs,
    )
    use_amp = bool(amp and str(device).startswith("cuda") and torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    graphs_ready = False
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    for i, rgb in enumerate(tqdm(iter_video_rgb(path, src_w, src_h), desc=path.name)):
        if i % max(1, sample_stride) != 0:
            continue
        x = preprocess_rgb_np(rgb, width=in_w, height=in_h).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp, cache_enabled=False):
            pred = session.forward_stream(x)
        if (
            cuda_graphs
            and not graphs_ready
            and use_cuda
            and session.num_bank_frames >= 1
        ):
            from edge.cuda_graphs import install_cuda_graphs

            install_cuda_graphs(
                model, in_h, in_w, window_size=window_size, dtype=amp_dtype, sample_image=x
            )
            graphs_ready = True
            session.pad_working_set = True
        depth = pred["depth"][0, -1].detach().float().cpu().numpy()
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = resize_hw(depth.astype(np.float32), ow, oh)
        rgb_out = resize_hw(rgb, ow, oh)
        save_depth_outputs(
            out_dir,
            f"{i:04d}",
            depth,
            rgb=rgb_out,
            dark_thresh=dark_thresh,
            save_npy=save_npy,
        )
    if use_cuda:
        torch.cuda.synchronize()
    session.clear()


def main():
    ap = argparse.ArgumentParser(description="EdGE depth inference (image / video)")
    ap.add_argument("--input", type=str, required=True, help="Image, image folder, or video")
    ap.add_argument("--weights", type=str, default=str(ROOT / "weights/model.safetensors"))
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--window_size", type=int, default=5, help="GPU working-set size (anchor + recent)")
    ap.add_argument("--bank_size", type=int, default=32, help="Max frames in GPU memory bank")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Model input width (default 280)")
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Model input height (default 224)")
    ap.add_argument(
        "--out_width",
        type=int,
        default=None,
        help="Output width (default: original source width)",
    )
    ap.add_argument(
        "--out_height",
        type=int,
        default=None,
        help="Output height (default: original source height)",
    )
    ap.add_argument("--sample_stride", type=int, default=1, help="Video: keep frames 0,S,2S,...")
    ap.add_argument("--dark_thresh", type=float, default=0.0, help="Mask dark RGB (0=off)")
    ap.add_argument("--save_npy", action="store_true")
    ap.add_argument("--no_amp", action="store_true", help="Disable bf16/fp16 autocast")
    ap.add_argument(
        "--no_cuda_graphs",
        action="store_true",
        help="Disable CUDA-graph fast path (on by default)",
    )
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / inp.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights, device)
    common = dict(
        window_size=args.window_size,
        bank_size=args.bank_size,
        width=args.width,
        height=args.height,
        out_width=args.out_width,
        out_height=args.out_height,
        dark_thresh=args.dark_thresh,
        save_npy=args.save_npy,
        amp=not args.no_amp,
        cuda_graphs=not args.no_cuda_graphs,
    )
    suf = inp.suffix.lower()
    if inp.is_dir() or suf in IMG_EXTS:
        paths = list_images(inp)
        if not paths:
            raise RuntimeError(f"No images under {inp}")
        infer_images(paths, model, device, out_dir, **common)
    elif suf in VID_EXTS:
        infer_video(
            inp,
            model,
            device,
            out_dir,
            sample_stride=args.sample_stride,
            **common,
        )
    else:
        raise ValueError(f"Unsupported input: {inp}")

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
