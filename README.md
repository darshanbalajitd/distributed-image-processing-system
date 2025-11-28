# Distributed Image Processing System (Map-Reduce for Face Embeddings)

This repository implements a map-reduce style, distributed image processing system for extracting 512‑dimensional face feature embeddings using ONNX Runtime and the VGGFace2 ResNet50 feature model. It supports:

* Local processing on a single device (CPU/GPU workers)

* Distributed processing across devices via TCP (master + workers)

* Per‑image preprocessing and ONNX inference

* Real‑time CSV streaming (each image appends a row on completion)

## Overview

* Map‑Reduce Schema:

  * Map: workers independently preprocess images and run inference to produce embeddings

  * Reduce: the master collects results and streams rows into a single CSV

* Model: VGGFace2 ResNet50 feature extractor converted to ONNX

  * Input: `float32[1,224,224,3]` (NHWC RGB)

  * Output: `float32[1,512]` (L2‑normalized embedding)

* Dataset: celebrity images (e.g., Pins Face Recognition)

  * Kaggle dataset link: <https://www.kaggle.com/datasets/hereisburak/pins-face-recognition/data>

## Folder Structure

```
Distributed-Image-Processing-System/
├─ include/
│  ├─ common/
│  │  ├─ base64.h
│  │  └─ csv_writer.h
│  ├─ inference/
│  │  ├─ backend.h
│  │  ├─ factory.h
│  │  └─ onnx_backend.h
│  ├─ master/
│  │  └─ net_master.h
│  └─ networking/
│     ├─ protocol.h
│     ├─ tcp_client.h
│     └─ tcp_server.h
├─ src/
│  ├─ common/
│  │  ├─ base64.cpp
│  │  └─ csv_writer.cpp
│  ├─ inference/
│  │  └─ onnx_backend.cpp
│  ├─ master/
│  │  ├─ main.cpp
│  │  └─ net_master.cpp
│  ├─ networking/
│  │  ├─ protocol.cpp
│  │  ├─ tcp_client.cpp
│  │  └─ tcp_server.cpp
│  └─ worker/
│     └─ main.cpp
├─ onnx converter/
│  ├─ convert_to_onnx.py
│  ├─ model.py, resnet.py, toolkits.py, utils.py
│  ├─ weights.h5
│  └─ vggface2_resnet50.onnx
├─ data/
│  └─ images/
│     ├─ pins_<Celebrity A>/...
│     └─ pins_<Celebrity B>/...
├─ build/
│  └─ src/Release/
│     ├─ master.exe
│     ├─ worker.exe
│     └─ required DLLs (OpenCV, ONNX Runtime, CUDA, cuDNN)
├─ output/
│  └─ embeddings.csv (created at runtime)
├─ CMakeLists.txt
└─ README.md
```

## Installation

Prerequisites (Windows x64):

* Visual Studio 2022 (MSVC), CMake

* vcpkg (for OpenCV)

* OpenCV via vcpkg

* ONNX Runtime (Windows GPU or CPU package)

* CUDA Toolkit 12.x (e.g., 12.9) and cuDNN 9 (for GPU workers)

### vcpkg and OpenCV

* Install vcpkg and integrate with VS/CMakelists.

* Install OpenCV: `vcpkg install opencv[core,imgcodecs,imgproc]:x64-windows`

### ONNX Runtime (Windows)

* Download ONNX Runtime for Windows (GPU package for CUDA support, or CPU‑only if you don’t need GPU).

* Set env or CMake variables to include and lib/bin paths.

### CUDA/cuDNN (GPU)

* Install CUDA 12.x (e.g., 12.9) and cuDNN 9 matching CUDA version.

* Ensure their `bin` folders are available to the process (either on `PATH` or next to the executable).

## Build

PowerShell command (one line):

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/Users/SESA837120/vcpkg/scripts/buildsystems/vcpkg.cmake -DUSE_ONNXRUNTIME=ON -DUSE_OPENCV=ON -DONNXRUNTIME_INCLUDE=C:/onnxruntime/include -DONNXRUNTIME_LIB=C:/onnxruntime/lib -DONNXRUNTIME_BIN_DIR=C:/onnxruntime/bin
cmake --build build --config Release
```

Multi‑line (PowerShell, escaping as needed):

```
cmake -S . -B build \
  -G "Visual Studio 17 2022" \
  -A x64 \
  -DCMAKE_TOOLCHAIN_FILE=C:/Users/SESA837120/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DUSE_ONNXRUNTIME=ON \
  -DUSE_OPENCV=ON \
  -DONNXRUNTIME_INCLUDE=C:/onnxruntime/include \
  -DONNXRUNTIME_LIB=C:/onnxruntime/lib \
  -DONNXRUNTIME_BIN_DIR=C:/onnxruntime/bin
cmake --build build --config Release
```

## GPU Troubleshooting

If GPU provider fails to load (e.g., missing DLLs), copy the required DLLs next to the executable (`build/src/Release`):

* ONNX Runtime GPU: `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll`

* CUDA 12.x: `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`, `curand64_12.dll` (version names may vary)

* cuDNN 9: `cudnn64_9.dll`
  Alternatively, add their folders to `PATH` before launching.

## Models

* You can use the provided ONNX model, or convert your own.

* Visualize model with Netron: <https://netron.app/>

* Converter code and weights:

  * Folder: `onnx converter/`

  * Files: `convert_to_onnx.py`, `weights.h5`, model definition scripts, and `vggface2_resnet50.onnx`

  * To convert: run `python convert_to_onnx.py`

* Original resources: <https://drive.google.com/file/d/1AHVpuB24lKAqNyRRjhX7ABlEor6ByZlS/view>

## How It Works

* Master (local‑thread mode):

  * Enumerates images under `data/images/pin_<celebrity>/...`

  * Creates local CPU/GPU worker threads

  * Each worker preprocesses per image (resize 224×224, BGR→RGB, float32 NHWC), runs ONNX inference, and returns a 512‑dim embedding

  * Master writes rows to CSV immediately after each job finishes

* Master (TCP mode):

  * Starts a TCP server and builds a job queue

  * Can spawn local embedded workers, and accepts remote workers over the network

  * Dispatches tasks (path‑based) to workers; receives results and streams CSV per job

* Worker (TCP client):

  * Connects to master and announces provider capability (CUDA or CPU)

  * Receives tasks with `path` to the image; opens and preprocesses locally, runs ONNX, sends back embeddings

  * Can run multiple connections per device via `--gpu-workers N --cpu-workers M`

* CSV Streaming:

  * CSV file: `output/embeddings.csv`

  * Header: `label,path,e0..e511`

  * Each image appends a row when its embedding is ready (truncate/pad to 512 dims if necessary)

## Executable Commands

* Local Master (no TCP):

```
.\build\src\Release\master.exe --gpu-workers 1 --cpu-workers 4
```

* Master (TCP) + Local Embedded Workers:

```
.\build\src\Release\master.exe --mode master --bind 0.0.0.0 --port 5555 --local-gpu-workers 1 --local-cpu-workers 4
```

* Master (TCP) only (accepts separate worker processes):

```
.\build\src\Release\master.exe --mode master --bind 0.0.0.0 --port 5555
```

* Worker (same device):

```
.\build\src\Release\worker.exe --master 127.0.0.1:5555 --gpu-workers 1 --cpu-workers 4
```

* Worker (remote device):

```
.\build\src\Release\worker.exe --master <MASTER_IP>:5555 --gpu-workers 1 --cpu-workers 4
```