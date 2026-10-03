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

The `nudenet` package supplies the `320n.onnx` model file. If an old version is installed, upgrade it:

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
python classify_images.py "D:\Path\To\Images" --mode move --device gpu --batch-size 64 --limit 100
```

The final `providers=` output should include `CUDAExecutionProvider`. If it only shows `CPUExecutionProvider`, the run is still using CPU.

## GUI

```powershell
python app.py
```

The GUI supports multiple folders. Use `Add folder...` repeatedly, or drag and drop multiple folders into the folder list if `tkinterdnd2` is installed.
Use `Output root folder` to collect all results under the selected folder's top-level `_classified` directory. Use `Output per subfolder` to create a separate `_classified` directory in each folder that contains source images.

Current GUI defaults:

- `Mode`: `copy`
- `Device`: `gpu`
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
python classify_images.py "D:\Path\To\Images" --mode move --device gpu --batch-size 128 --preprocess-workers 8 --transfer-workers 1
```

You can pass multiple folders. They are processed sequentially, each with its own `_classified` output folder:

```powershell
python classify_images.py "D:\Set1" "E:\Set2" "F:\Set3" --mode move --device gpu
```

To keep classification output beside each subfolder that contains images:

```powershell
python classify_images.py "D:\Set1" --mode move --device gpu --output-strategy per-folder
```

Safer command that preserves originals:

```powershell
python classify_images.py "D:\Path\To\Images" --mode copy --device gpu --batch-size 128 --preprocess-workers 8 --transfer-workers 2
```

## How it works

Detection runs the NudeNet `320n.onnx` model directly through ONNX Runtime (the old `nudenet` wrapper engine was removed; `--engine onnx` is still accepted so existing command lines keep working):

- reads Unicode paths with `np.fromfile + cv2.imdecode`
- decodes and preprocesses images on CPU worker threads, **one batch ahead** of the GPU
- an unreadable image goes to `errors/` on its own; the rest of its batch is still classified together
- post-processes detections with vectorised numpy
- loads the model **once** for all folders in a run
- keeps copy/move work on background transfer workers

`--fast-decode` (GUI: *Fast JPEG decode*) decodes large JPEGs at 1/2, 1/4 or 1/8 size, keeping the long side at least 320 px. It is roughly twice as fast for big photos but about 2-5% of images change category (measured on 300 large JPEGs: 94.7% to 97.7% agreement depending on thresholds), so it is off by default. Check with `bench.py --compare-golden` or `evaluate.py` before relying on it.

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
python classify_images.py "D:\Path\To\Images" --mode move --device gpu --batch-size 256 --preprocess-workers 4 --transfer-workers 1
```

For SSD-based runs:

```powershell
python classify_images.py "D:\Path\To\Images" --mode move --device gpu --batch-size 256 --preprocess-workers 8 --transfer-workers 1
```

Avoid running multiple full classifier processes on a single GPU. Prefer one process with larger batches.

## Benchmark and tests

```powershell
python bench.py --dir "D:\Path\To\Images" --limit 2000 --device gpu --batch-size 64
python bench.py --dir "D:\Path\To\Images" --save-golden golden.json
python bench.py --dir "D:\Path\To\Images" --fast-decode --compare-golden golden.json
```

`bench.py` prints img/s for preprocess, inference and postprocess separately, so you can see which stage limits your hardware. `--compare-golden` fails (exit code 2) if fewer than 99% of categories match the saved run. `python bench.py --synthetic 200` needs no images.

```powershell
pip install -r requirements.txt
python -m pytest
```

## Measuring and tuning accuracy

Labels: **nude** = sensitive parts clearly exposed, **sexy** = suggestive but nothing clearly exposed, **normal** = everything else.

```powershell
# 1. run the model once and keep the raw scores (nothing is copied or moved)
python score_images.py "D:\Path\To\Images" --out scores.jsonl --device gpu

# 2. pick images to label, stratified by the model's prediction
python make_sample.py --scores scores.jsonl --per-class 100 --out sample.csv

# 3. label them by hand (keys 1=nude 2=sexy 3=normal, Space=skip, u=undo)
#    the model's prediction is hidden so it cannot bias you; progress is saved
python label_tool.py --sample sample.csv --labels labels.csv

# 4. report and threshold search
python evaluate.py --scores scores.jsonl --labels labels.csv --sample sample.csv --errors 20
python evaluate.py --scores scores.jsonl --labels labels.csv --sample sample.csv --objective cost --miss-cost 3
```

`evaluate.py` prints precision, recall, F1 and a confusion matrix for the current thresholds (0.55 and 0.8 by default), then sweeps every `nude` x `sexy` threshold pair. The best pair is optimistic because it is chosen on the same images, so the k-fold cross-validation line is the honest estimate. `--objective cost` weights under-rating an image (for example nude rated normal) by `--miss-cost` per step. With `--sample`, the numbers are weighted to describe the whole folder rather than just the labelled subset. About 300-500 labelled images is a reasonable start. Keep your images and label files local, not in the repository.

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
python classify_images.py "D:\Path\To\Images" --mode copy --device cpu --batch-size 8 --limit 20
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
- Unicode file names are supported.
- Files that cannot be read or decoded are moved/copied to `_classified\errors` and recorded in `manifest.csv` with status `error`; they are retried on the next run.
