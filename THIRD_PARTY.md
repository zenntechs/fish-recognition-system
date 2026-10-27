# Third-Party Components and Model Attribution

AquaVision's original source code lives in this repository. The models and
runtime libraries below are **not authored by this project** and retain their
own licences. No model weights are committed to this repository; they are
fetched by `scripts/download_models.py` or installed from PyPI by `setup.sh`.

## Models

| Component | Upstream | Purpose | Licence |
| --- | --- | --- | --- |
| YuNet face detector (`face_detection_yunet_2023mar.onnx`) | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Face / human detection | Apache-2.0 |
| YOLO-World v2-s (`yolov8s-worldv2.pt`, rebuilt as `fish_detector.pt`) | [Ultralytics](https://github.com/ultralytics/ultralytics) | Fish localisation with a baked `fish` class | **AGPL-3.0** |
| ConvNeXt-Tiny ImageNet-1k (`convnext_tiny_imagenet.pth`) | [PyTorch Vision](https://github.com/pytorch/vision) | Fish verifier and ImageNet tie-break | BSD-3-Clause |
| BioCLIP-2 ViT-L/14 (`imageomics/bioclip-2`) | [Hugging Face model card](https://huggingface.co/imageomics/bioclip-2) | Zero-shot species ranking | MIT |
| OpenCLIP runtime | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | BioCLIP loading and inference | MIT |

The YOLO-World research code is also published by
[AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) under GPL-3.0.
The checkpoint distributed through Ultralytics is AGPL-3.0. **Review those
terms before redistributing this application, its weights, or a derivative
work, especially for commercial or network-service use.**

## Runtime and development dependencies

| Package | Licence |
| --- | --- |
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| OpenCV (`opencv-python`) | Apache-2.0 |
| NumPy | BSD-3-Clause |
| Pillow | HPND / MIT-CMU |
| PyTorch | BSD-3-Clause |
| TorchVision | BSD-3-Clause |
| Ultralytics | AGPL-3.0 |
| open-clip-torch | MIT |
| Hugging Face Hub | Apache-2.0 |
| pytest (dev) | MIT |
| HTTPX (dev) | BSD-3-Clause |

Dependency versions are constrained in `requirements.txt` and
`requirements-dev.txt`. Licence texts for installed packages are available from
their upstream repositories and distributions.

## Species data

The 80-species list in `app/species.py` is a curated set of common and
scientific names assembled for this project. It is not a copy of a proprietary
taxonomy database. Scientific names should be checked against an authoritative
taxonomic source before scientific use.

## Attribution in publications

If you describe or demonstrate this project, please credit the upstream model
authors (OpenCV Zoo, Ultralytics / YOLO-World, PyTorch Vision / ConvNeXt,
Imageomics / BioCLIP-2) alongside this repository.
