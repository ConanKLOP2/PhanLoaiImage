# PhanLoaiImage

Cong cu Python de quet mot thu muc anh va chia anh thanh 3 thu muc con:

- `nude`
- `sexy`
- `normal`

Mac dinh chuong trinh se tao thu muc `_classified` ben trong thu muc nguon va **copy** anh vao cac thu muc con de giu file goc. Co the doi sang che do move.

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Neu da cai project truoc do voi `nudenet==3.0.8`, nen nang cap lai de co `detect_batch`:

```powershell
pip install --upgrade "nudenet>=3.4.2"
```

Neu co NVIDIA GPU va muon chay CUDA:

```powershell
pip install -r requirements-gpu.txt
python check_gpu.py
```

Neu `check_gpu.py` in ra `CUDAExecutionProvider` thi ONNX Runtime da nhan GPU.

Neu gap loi thieu `cublasLt64_12.dll`, CUDA provider da duoc goi nhung may thieu CUDA 12 runtime/cuDNN 9/MSVC runtime trong PATH. Cach xu ly:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python check_gpu.py
```

Neu van loi, cai NVIDIA CUDA Toolkit 12.x va cuDNN 9.x cho Windows, sau do mo terminal moi de PATH duoc cap nhat.

Neu GPU khong chay, dung tang batch tiep. Kiem tra truoc:

```powershell
python check_gpu.py
python classify_images.py "D:\DuongDan\ThuMucAnh" --mode move --device gpu --batch-size 64 --limit 100
```

Sau khi chay, dong `providers=` phai co `CUDAExecutionProvider`. Neu khong co, chuong trinh van dang chay CPU.

## Chay bang giao dien

```powershell
python app.py
```

GUI mac dinh dung `copy`, `gpu` va `batch size = 128`. Neu CUDA chua cai dung, doi `Device` ve `cpu` hoac sua loi GPU truoc khi chay full.
GUI mac dinh dung engine `onnx`, tuc la chay truc tiep ONNX Runtime thay vi wrapper NudeDetector.

## Chay bang CLI cho thu muc lon

Voi khoang 50.000 files, nen chay CLI de de theo doi log:

```powershell
python classify_images.py "D:\DuongDan\ThuMucAnh" --mode move --device auto --batch-size 64
```

Neu muon an toan hon va giu lai file goc:

```powershell
python classify_images.py "D:\DuongDan\ThuMucAnh" --mode copy --device auto --batch-size 64
```

## Toc do

Phan cham nhat la model AI detect tung anh, nhat la khi chay CPU. Mot vai cach tang toc:

- Dung `move` thay vi `copy` neu khong can giu file goc.
- Tang `--batch-size` len `32` hoac `64`. Neu may bi day RAM thi giam lai `16`.
- Neu co GPU NVIDIA, cai `requirements-gpu.txt`, kiem tra `CUDAExecutionProvider`, roi chay `--device gpu`.
- Dung engine `onnx` mac dinh de doc Unicode bang `np.fromfile + cv2.imdecode`, preprocess song song CPU va chay batch ONNX truc tiep.
- Tang `--preprocess-workers` neu CPU/disk con ranh, vi du `4`, `8`, `12`.
- Khong bat `--debug-log` khi chay full 50.000 anh.
- Copy/move duoc chay nen bang `--transfer-workers`. Mac dinh `0` la tu dong: copy dung 2 worker, move dung 1 worker.
- Chay CLI se nhe hon GUI mot chut:

```powershell
python classify_images.py "D:\DuongDan\ThuMucAnh" --mode move --device gpu --engine onnx --batch-size 128 --preprocess-workers 8 --transfer-workers 1
```

Voi 1 GPU, thuong khong nen chay nhieu process song song cung luc vi cac process se tranh VRAM. Nen uu tien tang `--batch-size` truoc: `64`, `128`, neu du VRAM thi thu `256`.

## Resume

Ket qua moi file duoc ghi vao `_classified\manifest.csv`. Neu lan chay bi dung giua chung, chay lai cung lenh, chuong trinh se bo qua cac file da xu ly trong manifest.

## Debug loi

Mac dinh chuong trinh khong ghi log. File log chi duoc tao/ghi khi co loi:

```text
<thu muc anh>\_classified\debug.log
```

Neu toan bo file bi vao `errors`, thu chay batch nho de xem loi ro hon:

```powershell
python classify_images.py "D:\DuongDan\ThuMucAnh" --mode copy --batch-size 1 --limit 20
```

Code da co fallback: neu batch loi, chuong trinh se thu detect tung file rieng. Neu file nao van loi, stack trace se duoc ghi vao `debug.log`.

Neu can log tung anh, them `--debug-log`; tuy nhien che do nay cham hon:

```powershell
python classify_images.py "D:\DuongDan\ThuMucAnh" --mode copy --batch-size 1 --limit 20 --debug-log
```

## Luu y

- Model co the phan loai sai. Nen kiem tra thu mot tap nho bang `--limit 200` truoc khi chay het.
- Khong nen dung cong cu nay de ra quyet dinh nhay cam/phap ly; day chi la cong cu sap xep anh cuc bo.
- Neu may yeu, giam `--batch-size` xuong `8` hoac `16`.
- Ten file Unicode duoc xu ly bang cach tao path tam ASCII trong `_classified\_tmp_ascii_paths` khi detect. Neu cung o NTFS, chuong trinh uu tien hardlink nen khong nhan doi dung luong.
