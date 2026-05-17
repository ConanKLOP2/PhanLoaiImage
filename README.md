# PhanLoaiImage

Python tool for scanning an image folder and sorting images into three subfolders:

- `nude`
- `sexy`
- `normal`

By default, the tool creates an `_classified` folder inside the source folder and **copies** images into the category folders so the original files are preserved. You can switch to `move` mode to avoid duplicating disk usage.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If this project was previously installed with `nudenet==3.0.8`, upgrade it so the fallback `nudenet` engine has `detect_batch` support:

```powershell
pip install --upgrade "nudenet>=3.4.2"
```

## NVIDIA GPU Setup

Install GPU dependencies:

```powershell
pip install -r requirements-gpu.txt
python check_gpu.py
```

If `check_gpu.py` prints `CUDAExecutionProvider`, ONNX Runtime can see the GPU.

If you get an error about missing `cublasLt64_12.dll`, the CUDA provider was requested but CUDA 12 runtime, cuDNN 9, or the MSVC runtime is missing from `PATH`. A quick fix to try first:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python check_gpu.py
```

If it still fails, install NVIDIA CUDA Toolkit 12.x and cuDNN 9.x for Windows, then open a new terminal so `PATH` is refreshed.

Do not keep increasing `batch-size` if the GPU is not active. Verify first:

```powershell
python check_gpu.py
python classify_images.py "D:\Path\To\Images" --mode move --device gpu --engine onnx --batch-size 64 --limit 100
```

The final `providers=` output should include `CUDAExecutionProvider`. If it only shows `CPUExecutionProvider`, the run is still using CPU.

## GUI

```powershell
python app.py
```

The GUI supports multiple folders. Use `Add...` to add folders one by one, or drag and drop folders into the folder list if `tkinterdnd2` is installed.
Use `Output root folder` to collect all results under the selected folder's top-level `_classified` directory. Use `Output per subfolder` to create a separate `_classified` directory in each folder that contains source images.

Current GUI defaults:

- `Mode`: `copy`
- `Device`: `gpu`
- `Engine`: `onnx`
- `Batch size`: `256`
- `Preprocess workers`: CPU workers for image reading, decoding, and preprocessing
- `Transfer workers`: `0` means auto; `copy` uses 2 workers, `move` uses 1 worker

The GUI includes quick presets:

- `HDD`: conservative disk-friendly settings
- `SSD`: higher CPU preprocessing throughput
- `CPU`: CPU-only fallback

If the source drive is almost full, choose `Move file into _classified` instead of `Copy`.

## CLI

Recommended command for large folders when disk space matters:

```powershell
python classify_images.py "D:\Path\To\Images" --mode move --device gpu --engine onnx --batch-size 128 --preprocess-workers 8 --transfer-workers 1
```

You can pass multiple folders. They are processed sequentially, each with its own `_classified` output folder:

```powershell
python classify_images.py "D:\Set1" "E:\Set2" "F:\Set3" --mode move --device gpu --engine onnx
```

To keep classification output beside each subfolder that contains images:

```powershell
python classify_images.py "D:\Set1" --mode move --device gpu --engine onnx --output-strategy per-folder
```

Safer command that preserves originals:

```powershell
python classify_images.py "D:\Path\To\Images" --mode copy --device gpu --engine onnx --batch-size 128 --preprocess-workers 8 --transfer-workers 2
```

## Engines

`onnx` is the default and fastest engine. It bypasses the `NudeDetector` wrapper and:

- reads Unicode paths with `np.fromfile + cv2.imdecode`
- decodes and preprocesses images with CPU worker threads
- builds real numpy batches
- runs `onnxruntime.InferenceSession.run()` directly
- keeps copy/move work on background transfer workers

`nudenet` is the compatibility engine. It calls the `NudeDetector` wrapper and should mainly be used for comparison or fallback.

## Performance Tuning

Start with:

```powershell
--batch-size 128 --preprocess-workers 8 --transfer-workers 1
```

For an RTX 3060 12GB, try:

```text
batch-size: 128, 256, 384
preprocess-workers: 4, 6, 8
```

Choose the combination with the highest `img/s`. If Task Manager shows the HDD at `100% active time`, the drive is the bottleneck. In that case, increasing CPU workers or batch size may not help. Moving the source folder to an SSD is the biggest improvement.

For HDD-based runs:

```powershell
python classify_images.py "D:\Path\To\Images" --mode move --device gpu --engine onnx --batch-size 256 --preprocess-workers 4 --transfer-workers 1
```

For SSD-based runs:

```powershell
python classify_images.py "D:\Path\To\Images" --mode move --device gpu --engine onnx --batch-size 256 --preprocess-workers 8 --transfer-workers 1
```

Avoid running multiple full classifier processes on a single GPU. Prefer one process with larger batches.

## Resume

Each successfully processed file is written to:

```text
<image folder>\_classified\manifest.csv
```

If the run is stopped midway, run the same command again. Files already recorded in the manifest are skipped.

## Logging

By default, no log file is created. A log file is created only when there is an error or batch warning:

```text
<image folder>\_classified\debug.log
```

For a small debug run:

```powershell
python classify_images.py "D:\Path\To\Images" --mode copy --device cpu --engine onnx --batch-size 8 --limit 20
```

To force verbose per-image logging, add `--debug-log`. This is slower and should not be used for full 50,000-image runs.

## Thresholds

The classifier checks `nude` first, then `sexy`, then falls back to `normal`.

- Lower thresholds catch more images but increase false positives.
- Higher thresholds reduce false positives but may miss borderline images.

Common starting points:

```text
nude-threshold: 0.50 to 0.60
sexy-threshold: 0.45 to 0.60
```

## Notes

- The model can classify images incorrectly. Test with `--limit 200` before running the full folder.
- This is a local sorting tool, not a legal, safety, or policy decision system.
- Unicode file names are supported by the default `onnx` engine.
- If you use the compatibility `nudenet` engine, Unicode paths may require temporary ASCII staging under `_classified\_tmp_ascii_paths`.
