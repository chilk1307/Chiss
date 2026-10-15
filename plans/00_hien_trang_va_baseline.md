# 00. Hiện trạng và baseline

Cập nhật: 16/10/2026.
Nguồn số liệu: `outputs/REPORT.md`, `outputs/REPORT_results.md`, `outputs/logs/`, `outputs/phi_models/phi_B32/validation_metrics.csv`, `outputs/prev_results_oct5/`, `outputs/submissions/`.

## Những gì đã làm

- **Tái lập toàn bộ pipeline SEARLE** với CLIP ViT-B/32, dùng code gốc commit `a9c314b`:
  `image_concepts_association` → OTI → `train_phi` → `validate.py`.
- **OTI** trên 100.000 ảnh ImageNet1K test, chia 3 shard liên tiếp theo tên file (seed 42/43/44, mỗi shard 33.334 ảnh), rồi gộp thành `oti_B32` bằng `outputs/merge_oti.py`.
- **Train phi** 100 epoch với đúng siêu tham số của paper, được `phi_B32`.
- **Đánh giá** CIRR val, CIRCO val, FashionIQ val (tập con) cho phi_B32 và SEARLE gốc.
- **File nộp test của phi_B32** (`outputs/submissions/`):
  - `phi_B32_cirr.json`: CIRR test1, 4.148 truy vấn × top-50, `metric = recall`;
  - `phi_B32_circo.json`: CIRCO test, 800 truy vấn × top-50.
  - Còn thiếu file subset của CIRR (`metric = recall_subset`, để server tính Rs@K); chưa có điểm từ server.
- **Lần chạy cũ quy mô nhỏ** `phi_b32` (OTI trên 6.824 ảnh) nằm trong `SEARLE-main/data/`.
- **Demo web Flask** [demo/app.py](../SEARLE-main/demo/app.py): gallery, 3 chế độ truy vấn, đổi checkpoint, xem từ CLIP gần pseudo-word nhất, trang tự đánh giá tiến độ đọc kết quả thật trong `outputs/` (tự nhận diện file nộp test).

## Môi trường

- Google Colab, GPU A100-SXM4-40GB.
- Python 3.8, torch 1.11.0+cu113, torchvision 0.12.0+cu113, transformers 4.24.0, pandas 1.4.2, numpy 1.23.5, CLIP commit `d05afc43`.
- `setuptools==69.5.1` (cần cho `comet_ml`; Comet không dùng).
- Danh sách đầy đủ: `outputs/pip_freeze.txt`.

## Siêu tham số đã dùng

| Bước | Tham số |
|---|---|
| OTI | 350 bước, lr 2e-2, batch 32, top-k 15, λ_gpt 0.5, λ_cos 1, EMA 0.99, targetpad 1.25 |
| phi | 100 epoch, batch 256, lr 1e-4 cố định (không có lịch LR), weight decay 0.01, λ_distil 1, λ_gpt 0.75, temperature 0.25, dropout 0.5, seed 42 |

## Kết quả trên tập val

> Dòng "Paper" lấy từ hằng số trong `demo/app.py` và **chưa được đối chiếu với bảng trong bài báo**. Đối chiếu ở [01](01_nen_tang_danh_gia.md) mục 1.6 trước khi đưa vào báo cáo.

### CIRR val

| Mô hình | R@1 | R@5 | R@10 | R@50 | Rs@1 | Rs@2 | Rs@3 | Nguồn |
|---|---|---|---|---|---|---|---|---|
| Paper (SEARLE B/32, chưa đối chiếu) | 25.09 | 55.18 | 68.79 | 90.82 | 54.84 | 76.63 | 87.95 | `demo/app.py` |
| SEARLE gốc (torch.hub) | 25.66 | 55.47 | 69.24 | 90.58 | 55.08 | 76.58 | 87.85 | `outputs/logs/eval_searle_cirr.log` |
| **phi_B32** (OTI 100.000 ảnh) | 25.47 | 55.23 | 69.24 | 90.43 | 55.20 | 76.11 | 87.83 | `outputs/logs/eval_phi_cirr.log` |
| phi_b32 cũ (OTI 6.824 ảnh) | 22.60 | 52.45 | 66.23 | 88.97 | 55.58 | 77.16 | 88.18 | `outputs/prev_results_oct5/cirr_val.txt` |

Kết luận: tái lập thành công; phi_B32 gần như trùng SEARLE gốc (chênh ≤ 0.5 điểm).

### CIRCO val (220 truy vấn)

| Mô hình | mAP@5 | mAP@10 | mAP@25 | mAP@50 | R@5 | R@10 | R@25 | R@50 |
|---|---|---|---|---|---|---|---|---|
| SEARLE gốc | 6.82 | 7.78 | 9.14 | 9.76 | 15.91 | 23.18 | 36.36 | 47.27 |
| **phi_B32** | 6.42 | 7.15 | 8.48 | 9.08 | 17.73 | 23.18 | 35.45 | 47.27 |

Nguồn: `outputs/logs/eval3_searle_circo.log`, `outputs/logs/eval_phi_circo.log`.
Chưa chắc gallery CIRCO đủ ảnh (các lần chạy trước từng báo `FileNotFoundError` trong `COCO2017_unlabeled`). Kiểm tra ở [01](01_nen_tang_danh_gia.md) mục 1.2.

### FashionIQ val: TẬP CON, không so được với paper

Khoảng 2.431 ảnh là link chết nên bị bỏ. Số truy vấn còn lại: dress 1.861, shirt 239, toptee 72 (tập val chính thức có khoảng 2.000 truy vấn mỗi loại).

| Mô hình | Dress R@10 / R@50 | Shirt R@10 / R@50 | Toptee R@10 / R@50 | Trung bình R@10 / R@50 |
|---|---|---|---|---|
| Paper (tập đầy đủ, chưa đối chiếu) | 18.54 / 39.51 | 24.44 / 41.61 | 25.70 / 46.46 | 22.89 / 42.53 |
| SEARLE gốc (tập con) | 19.67 / 38.53 | 42.68 / 65.27 | 48.61 / 66.67 | 36.99 / 56.82 |
| **phi_B32** (tập con) | 19.08 / 38.53 | 41.84 / 63.18 | 48.61 / 69.44 | 36.51 / 57.05 |

Nguồn: `outputs/logs/eval5_searle_fashioniq.log`, `outputs/logs/eval5_phi_fashioniq.log`.
Shirt và toptee cao gần gấp đôi paper vì gallery nhỏ đi; chỉ dress (gần đủ truy vấn) là gần paper. Tập con này chỉ dùng để so sánh tương đối giữa các phương pháp.

### Tập test

| Mô hình | CIRR test1 | CIRCO test |
|---|---|---|
| phi_B32 | đã có file chính; thiếu file subset; chưa có điểm | đã có file; chưa có điểm |
| SEARLE gốc | chưa có file | chưa có file |

## Phát hiện từ đường cong train phi

Từ `outputs/phi_models/phi_B32/validation_metrics.csv` (CIRR val sau mỗi epoch):

| Epoch | 1 | 10 | 30 | 50 | 70 | 90 | 92 (tốt nhất) | 100 |
|---|---|---|---|---|---|---|---|---|
| R@1 | 22.05 | 23.44 | 24.49 | 24.80 | 24.99 | 25.38 | 25.54 | 25.47 |
| R@10 | 65.70 | 66.73 | 67.85 | 68.45 | 68.91 | 69.05 | 69.17 | 69.24 |
| Rs@1 | 56.85 | 55.18 | 54.92 | 54.65 | 54.84 | 55.01 | 55.11 | 55.20 |

- Đến epoch 100 vẫn còn tăng nhẹ; chọn checkpoint khác `phi_last` gần như không có lợi.
- Lượng dữ liệu OTI ảnh hưởng mạnh: 6.824 ảnh cho R@1 22.60; 100.000 ảnh cho 25.47 (phân tích thêm ở [01](01_nen_tang_danh_gia.md) mục 1.14).

## Chi phí tính toán đã đo

| Việc | Thời gian | Ghi chú |
|---|---|---|
| OTI 1 shard (33.334 ảnh, 1.042 batch) | khoảng 14 giờ 49 phút | ~61 giây/batch khi chạy 3 shard song song trên một A100 |
| OTI toàn bộ 100.000 ảnh | khoảng 15 giờ | 3 tiến trình song song |
| Train phi, 1 epoch (390 batch) | khoảng 3 phút 46 giây + validation CIRR | 100 epoch khoảng 6–7 giờ |
| Đánh giá CIRR val | vài phút | |
| Đánh giá CIRCO val | lâu hơn nhiều | gallery khoảng 123 nghìn ảnh; đọc từ Drive rất chậm, nên giải nén ra `/content` |

## Thay đổi so với code gốc

| File | Thay đổi | Ghi chú |
|---|---|---|
| `src/datasets.py` (repo) | `ImageFile.LOAD_TRUNCATED_IMAGES = True`; FashionIQ lọc bộ ba thiếu ảnh; `except: pass` thay cho in lỗi ([dòng 133](../SEARLE-main/src/datasets.py#L133), [dòng 256](../SEARLE-main/src/datasets.py#L256)); ImageNet đọc mọi đuôi `.jpg/.jpeg/.png` và sắp xếp tên | `except: pass` sửa ở C3 |
| `src/data_utils.py` (repo) | `collate_fn` trả `None` khi cả batch rỗng | Bên gọi chưa xử lý `None` |
| `outputs/scripts/datasets_patched.py`, `patch_fiq.py` | Bản vá dùng trên Colab: lọc ảnh thiếu, thử mở ảnh 6 lần | **Khác** bản trong repo; kết quả ở trên chạy bằng bản này |
| `demo/` | Toàn bộ là phần nhóm tự viết | |

## Điểm yếu đã biết

- `train_metrics.csv` chỉ giữ epoch cuối, vì `train_one_epoch` nối DataFrame nhưng không trả về. Loss đầy đủ nằm ở `train_metrics_full.csv` (lấy lại từ log).
- `train_phi.py` tra token đích bằng `names_list.index(...)` ([train_phi.py:175](../SEARLE-main/src/train_phi.py#L175)): tìm tuyến tính trong 100.000 tên cho từng ảnh, có thể làm chậm train; và luôn train trên **mọi** ảnh ImageNet test, nên chưa train được trên tập con.
- `validate.py` chỉ in số ra màn hình, không lưu file.
- Logic tạo truy vấn (`a photo of $ that ...`) viết lặp ở `validate.py` và `generate_test_submission.py`.

## Những gì còn thiếu và nơi xử lý

| Còn thiếu | Xử lý ở |
|---|---|
| Hai bản `datasets.py`; lỗi bị nuốt; FIQ thiếu ảnh; CIRCO chưa xác nhận đủ ảnh | [01](01_nen_tang_danh_gia.md) mục 1.1–1.2 |
| Quy trình chung, lưu kết quả, khoảng tin cậy | [01](01_nen_tang_danh_gia.md) mục 1.3–1.4 |
| File subset CIRR, file nộp của SEARLE gốc, điểm từ server | [01](01_nen_tang_danh_gia.md) mục 1.5 |
| Số "Paper" chưa đối chiếu | [01](01_nen_tang_danh_gia.md) mục 1.6 |
| Baseline đơn giản, baseline xử lý ảnh cổ điển, phân tích lỗi | [01](01_nen_tang_danh_gia.md) mục 1.8–1.13 |
| Đóng góp riêng của nhóm | [02](02_kiem_chung_thuoc_tinh.md), [03](03_do_ben_va_phuc_hoi_anh.md), [04](04_tien_xu_ly_va_truy_van.md) |
| README ở gốc repo, ghi nguồn, phân định phần nhóm làm, tổng quan tài liệu | [05](05_demo_va_bao_cao.md) |
| File nhỏ quan trọng bị `.gitignore` bỏ qua (REPORT, pip_freeze, merge_oti, csv, log, submissions) | [01](01_nen_tang_danh_gia.md) mục 1.1 |
