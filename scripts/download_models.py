#!/usr/bin/env python3
"""Download and prepare the local model assets.

Everything is stored under ``models/`` so the application can run offline:

* YuNet face detector (OpenCV Zoo ONNX)
* YOLO-World v2-s checkpoint with the ``fish`` class baked in
* ConvNeXt-Tiny ImageNet-1k verifier weights
* BioCLIP-2 ViT-L/14 Hugging Face cache

Run from the project root with the project virtualenv:

    .venv/bin/python scripts/download_models.py

Use ``--verify`` to check existing assets without downloading.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
HF_CACHE = MODELS_DIR / "hf_cache"

YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

BASE_YOLO_PATH = MODELS_DIR / "yolov8s-worldv2.pt"
FISH_YOLO_PATH = MODELS_DIR / "fish_detector.pt"
IMAGENET_PATH = MODELS_DIR / "convnext_tiny_imagenet.pth"
BIOCLIP_DIR = HF_CACHE / "hub" / "models--imageomics--bioclip-2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"  downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "AquaVision-setup/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        with temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                if total:
                    percent = min(100, int(received * 100 / total))
                    print(f"\r    {percent:3d}%  {received / 1024 / 1024:.1f} MiB", end="")
        print()
    temporary.replace(destination)


def ensure_yunet(force: bool) -> None:
    print("[1/4] YuNet face detector")
    if YUNET_PATH.is_file() and not force:
        actual = sha256(YUNET_PATH)
        status = "ok" if actual == YUNET_SHA256 else "checksum mismatch"
        print(f"  present · {status} · {actual[:16]}…")
        if actual == YUNET_SHA256:
            return
        print("  replacing invalid file")
    download(YUNET_URL, YUNET_PATH)
    actual = sha256(YUNET_PATH)
    if actual != YUNET_SHA256:
        raise SystemExit(
            f"YuNet checksum mismatch: expected {YUNET_SHA256}, got {actual}. "
            "The upstream file may have changed; delete it and inspect manually."
        )
    print(f"  ready · {actual[:16]}…")


def ensure_fish_detector(force: bool) -> None:
    print("[2/4] YOLO-World fish detector")
    if FISH_YOLO_PATH.is_file() and not force:
        try:
            from ultralytics import YOLO

            model = YOLO(str(FISH_YOLO_PATH), task="detect")
            names = {str(value).lower() for value in (model.names or {}).values()}
            if "fish" in names:
                print("  present · 'fish' class verified")
                return
            print("  present but missing the 'fish' class; rebuilding")
        except Exception as exc:
            print(f"  present but unreadable ({exc}); rebuilding")

    if not BASE_YOLO_PATH.is_file():
        print("  downloading the YOLO-World v2-s base checkpoint …")
        from ultralytics import YOLO

        previous = Path.cwd()
        try:
            os.chdir(MODELS_DIR)
            YOLO("yolov8s-worldv2.pt")
        finally:
            os.chdir(previous)
        downloaded = MODELS_DIR / "yolov8s-worldv2.pt"
        if not downloaded.is_file():
            raise SystemExit("YOLO-World base checkpoint download failed")

    from ultralytics import YOLO

    model = YOLO(str(BASE_YOLO_PATH), task="detect")
    model.set_classes(["fish"])
    temporary = FISH_YOLO_PATH.with_suffix(".tmp.pt")
    model.save(str(temporary))
    temporary.replace(FISH_YOLO_PATH)

    check = YOLO(str(FISH_YOLO_PATH), task="detect")
    if "fish" not in {str(value).lower() for value in (check.names or {}).values()}:
        raise SystemExit("Saved fish detector does not contain the 'fish' class")
    print(f"  ready · saved with baked 'fish' class · {sha256(FISH_YOLO_PATH)[:16]}…")


def ensure_imagenet(force: bool) -> None:
    print("[3/4] ConvNeXt-Tiny ImageNet verifier")
    if IMAGENET_PATH.is_file() and not force:
        try:
            import torch
            import torchvision

            state = torch.load(str(IMAGENET_PATH), map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model = torchvision.models.convnext_tiny(weights=None)
            model.load_state_dict(state)
            print(f"  present · state_dict verified · {sha256(IMAGENET_PATH)[:16]}…")
            return
        except Exception as exc:
            print(f"  present but invalid ({exc}); rebuilding")

    import torch
    import torchvision

    weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
    model = torchvision.models.convnext_tiny(weights=weights)
    model.eval()
    torch.save(model.state_dict(), IMAGENET_PATH)
    print(f"  ready · {sha256(IMAGENET_PATH)[:16]}…")


def ensure_bioclip(force: bool) -> None:
    print("[4/4] BioCLIP-2 species model")
    if force and BIOCLIP_DIR.is_dir():
        print("  --force: removing cached BioCLIP-2 model")
        shutil.rmtree(BIOCLIP_DIR)

    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if BIOCLIP_DIR.is_dir() and not force:
        snapshots = BIOCLIP_DIR / "snapshots"
        if snapshots.is_dir() and any(snapshots.iterdir()):
            print("  present · project-local Hugging Face cache")
            return

    print("  downloading BioCLIP-2 (about 1.6 GiB on first setup) …")
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2",
        device="cpu",
    )
    del model
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
    tokenizer(["a photo of a fish."])
    if not BIOCLIP_DIR.is_dir():
        raise SystemExit("BioCLIP-2 download did not populate the expected cache directory")
    print("  ready · cached for offline use")


def write_checksums() -> None:
    lines = []
    for path in (YUNET_PATH, FISH_YOLO_PATH, IMAGENET_PATH):
        if path.is_file():
            lines.append(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}")
    if lines:
        (MODELS_DIR / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\nUpdated models/CHECKSUMS.txt")


def verify() -> bool:
    print("\nVerifying model assets")
    checks = {
        "YuNet face detector": YUNET_PATH.is_file(),
        "YOLO-World fish detector": FISH_YOLO_PATH.is_file(),
        "ConvNeXt ImageNet verifier": IMAGENET_PATH.is_file(),
        "BioCLIP-2 cache": BIOCLIP_DIR.is_dir(),
    }
    for name, ok in checks.items():
        print(f"  {'ok     ' if ok else 'MISSING'}  {name}")
    if YUNET_PATH.is_file() and sha256(YUNET_PATH) != YUNET_SHA256:
        print("  WARNING YuNet checksum does not match the pinned value")
        return False
    return all(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare AquaVision model assets")
    parser.add_argument("--force", action="store_true", help="re-download/rebuild every asset")
    parser.add_argument("--verify", action="store_true", help="only verify existing assets")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    if args.verify:
        return 0 if verify() else 1

    print("AquaVision model setup")
    print(f"  project: {PROJECT_ROOT}")
    print(f"  models:  {MODELS_DIR}\n")

    ensure_yunet(args.force)
    ensure_fish_detector(args.force)
    ensure_imagenet(args.force)
    ensure_bioclip(args.force)
    write_checksums()
    ok = verify()
    if not ok:
        print("\nModel setup is incomplete.", file=sys.stderr)
        return 1
    print("\nAll models are ready. Start the app with ./run.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
