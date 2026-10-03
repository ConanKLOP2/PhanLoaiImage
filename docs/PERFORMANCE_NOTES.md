# Performance notes (measured 2026-10-03)

Record of the benchmarks and experiments already run, so they are not repeated.
All numbers are from one machine; treat them as relative, not absolute.

## Environment

| Item | Value |
|---|---|
| OS / Python | Windows 11, Python 3.14.7 |
| CPU | 6 physical cores / 12 threads |
| GPU | RTX 2080 Max-Q, 8 GB, driver 595.79 (CUDA 13.2) |
| Disk (images) | NVMe Kingston SNV2S500G, measured read 716 MB/s |
| onnxruntime-gpu | 1.30.0 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 (installed only for the nvJPEG test) |

Test venv: `%TEMP%\pl_venv` (safe to delete).

## Datasets

- `C:\CASTROICE`: 13 JPEG, 6.7-12.2 MB each.
- `C:\Xiuren\[[WALLPAPER]`: 2625 images; the first 2000 (sorted order) were used. Typical size 4422x6633 (~29 MP, ~12 MB).

## GPU setup (done, no driver update needed)

`onnxruntime-gpu` first reported missing `cublasLt64_13.dll`, `cudart64_13.dll`, ... The driver was fine; the CUDA runtime libraries were missing from the venv. Fix ("option A"):

```powershell
pip install nvidia-cuda-runtime nvidia-cublas nvidia-cudnn-cu13 nvidia-cufft nvidia-curand
python check_gpu.py     # now lists CUDAExecutionProvider
```

Installed versions: cublas 13.8.0.4, cudnn-cu13 9.27.0.42, cufft 12.4.0.43, cuda-runtime 13.4.92, curand 10.4.4.72.

## Results

Columns are img/s; time in seconds in brackets. "sequential" = stages one after another; "overlapped" = prepare batch N+1 while running batch N.

### Post-processing (vectorised vs old loop)

64 random predictions, 640x640 image: **6.11 ms/img -> 0.337 ms/img (18x)**. Equivalence to the old loop is covered by `tests/test_fast_onnx_detector.py`.

### Synthetic images, CPU (200 images, batch 32, 4 workers)

preprocess 112.2, inference 77.4, postprocess 48247, sequential 44.9, overlapped 59.0.

### `C:\CASTROICE`, CPU (13 large JPEG)

| Config | preprocess | inference | postprocess | sequential | overlapped |
|---|---|---|---|---|---|
| batch 4 | 2.8 | 45.1 | 1934 | 2.6 | 3.0 |
| batch 13 | 3.3 | 29.5 | 4596 | 3.0 | 2.9 |
| batch 13, `--fast-decode` | 13.3 | 42.2 | 4968 | 10.0 | 9.8 |

`--fast-decode` golden agreement: 92.31% (1 of 13 changed: `31.jpg` normal -> sexy). Sample too small to conclude.

### `C:\Xiuren\[[WALLPAPER]`, 2000 images, 6 preprocess workers

| Run | preprocess | inference | postprocess | sequential | overlapped |
|---|---|---|---|---|---|
| CPU, batch 32 (**contaminated**: pip install ran concurrently) | 4.5 (443s) | 29.2 (69s) | 2542 | 3.9 (514s) | 6.2 (321s) |
| CPU, batch 64 | 8.7 (229s) | 67.7 (30s) | 6236 | 7.7 (260s) | 8.4 (238s) |
| **GPU**, batch 64, full decode | 3.7 (546s) | 190.2 (10.5s) | 2352 | 3.6 (560s) | 3.4 (580s) |

- GPU inference is ~2.8x faster than CPU inference (190 vs 68 img/s), but it is a tiny share of total time.
- The GPU run's decode (3.7 img/s) was much slower than the CPU run's (8.7). Cause not determined (possible background load, cache state or thermals). Do not read it as "GPU slows decode"; decode is CPU-side in both.
- CPU `--fast-decode` on 2000 images flipped many labels (normal->sexy, normal->nude, nude->normal, sexy->nude). The exact agreement percentage was **lost** (output was cut by `tail -9`). The GPU `--fast-decode` run was killed by the 25-minute limit; **no data**.

### Decode micro-benchmarks (29 MP JPEG)

- Single-thread OpenCV decode: **461 ms/img**. File read (NVMe) is not a bottleneck.
- 24 images, img/s by thread count:

| Decoder | 1 | 3 | 6 | 12 |
|---|---|---|---|---|
| CPU OpenCV | 2.1 | 6.5 | 8.5 | 9.4 |
| GPU nvJPEG (torchvision `decode_jpeg(device="cuda")`) | 2.6 | 2.2 | 1.8 | 2.1 |
| CPU reduced 1/8 (`IMREAD_REDUCED_COLOR_8`), 6 threads | - | - | 15.8 | - |

- nvJPEG per image incl. resize: chunk 4/8/16 = 362/266/354 ms, peak VRAM 1.4/1.8/2.5 GB.
- Resize fidelity check: plain bilinear without antialias on GPU matches the OpenCV blob (mean abs diff 0.0013 after channel reversal); antialiased resize differs more (0.0096). The existing pipeline feeds the model BGR order, same as NudeNet.

## Conclusions and decisions

1. **Bottleneck is CPU JPEG decode**, not disk, model or post-processing. It saturates at ~8-9 img/s on 6 cores for ~29 MP images.
2. **GPU decode rejected**: ~2-3 img/s and does not scale with threads (RTX 2080 has no hardware JPEG engine). Not implemented.
3. **Accepted**: ~4 minutes per 2000 large images with default settings. `--fast-decode` stays off.
4. No driver update needed; install the `nvidia-*` wheels instead.

Recommended command:

```powershell
python classify_images.py "C:\Xiuren\[[WALLPAPER]" --mode copy --device gpu --batch-size 64 --preprocess-workers 6
```

## Observations worth remembering

- The default path downscales 29 MP -> 320 px with non-antialiased bilinear, which aliases heavily. `--fast-decode` averages first (DCT scaling), so label flips do not by themselves prove it is worse. Only a hand-labelled set can say which is more accurate.
- Decode threads stop helping beyond the physical core count (6 -> 12 gave ~10%).

## Not tried yet

- Hand-labelled sample (100-200 images) to compare full decode vs `--fast-decode`.
- Intermediate reduction (1/2 or 1/4) followed by `INTER_AREA` resize.
- GPU batch sizes 32 and 128; GPU `--fast-decode` (killed run).
- TensorRT / FP16, IOBinding, reading each file once when copying, SQLite manifest.

## Reproduce

```powershell
python bench.py --dir "<folder>" --limit 2000 --device gpu --batch-size 64 --preprocess-workers 6 --save-golden golden.json
python bench.py --dir "<folder>" --limit 2000 --device gpu --batch-size 64 --preprocess-workers 6 --fast-decode --compare-golden golden.json
```

Run long benchmarks in the background without other heavy jobs, and do not truncate the output with `tail` (keep the summary lines).
