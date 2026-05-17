# PhanLoaiImage

Cong cu Python de quet mot thu muc anh va chia anh thanh 3 thu muc con:

- `nude`
- `sexy`
- `normal`

Mac dinh chuong trinh se tao thu muc `_classified` ben trong thu muc nguon va **move** anh vao cac thu muc con. Co the doi sang che do copy.

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chay bang giao dien

```powershell
python app.py
```

## Chay bang CLI cho thu muc lon

Voi khoang 50.000 files, nen chay CLI de de theo doi log:

```powershell
python classify_images.py "D:\DuongDan\ThuMucAnh" --mode move --batch-size 16
```

Neu muon an toan hon va giu lai file goc:

```powershell
python classify_images.py "D:\DuongDan\ThuMucAnh" --mode copy --batch-size 16
```

## Resume

Ket qua moi file duoc ghi vao `_classified\manifest.csv`. Neu lan chay bi dung giua chung, chay lai cung lenh, chuong trinh se bo qua cac file da xu ly trong manifest.

## Debug loi

Log chi tiet nam tai:

```text
<thu muc anh>\_classified\debug.log
```

Neu toan bo file bi vao `errors`, thu chay batch nho de xem loi ro hon:

```powershell
python classify_images.py "D:\DuongDan\ThuMucAnh" --mode copy --batch-size 1 --limit 20
```

Code da co fallback: neu batch loi, chuong trinh se thu detect tung file rieng va ghi stack trace vao `debug.log`.

## Luu y

- Model co the phan loai sai. Nen kiem tra thu mot tap nho bang `--limit 200` truoc khi chay het.
- Khong nen dung cong cu nay de ra quyet dinh nhay cam/phap ly; day chi la cong cu sap xep anh cuc bo.
- Neu may yeu, giam `--batch-size` xuong `4` hoac `8`.
