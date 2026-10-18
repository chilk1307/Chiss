# Nhật ký thí nghiệm

## Cách ghi

- **Tự động:** mỗi lần chạy có `--output-dir` tự thêm một dòng vào `SEARLE/results/runs_<người chạy>.csv` trên Drive (thời điểm, mã thí nghiệm, lệnh, chỉ số chính, thời gian chạy, loại GPU, ghi chú `--note`). Không cần chép tay từng lần chạy.
- **Bảng dưới đây:** chỉ ghi các kết quả **dùng trong báo cáo**: baseline, điểm test, kết luận chính của từng phần. Chí cập nhật dựa trên báo cáo cuối ngày của Hòa và Doanh.
- Mã thí nghiệm theo quy ước trong [README](README.md): `E<số file>.<mục><biến thể>-<mô tả>-<tập>`. Mã `B…` dành cho các kết quả có từ trước kế hoạch 2 tuần và các file nộp test của baseline.

## Bảng kết quả dùng trong báo cáo

| Mã | Ngày | Mô tả | Mô hình / cấu hình | Tập | Chỉ số chính | So với | Nguồn | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| B0 | 17/10 | Số của paper (đã đối chiếu) | SEARLE B/32 | CIRR val | R@1 25.09, R@10 68.79 | — | arXiv 2303.15247v2, Table 9 | Các bảng khác (FIQ val, CIRCO val, test) ở [01](01_nen_tang_danh_gia.md) mục 1.6 |
| B0c | 17/10 | Số của paper | SEARLE B/32 | CIRCO val | mAP@5 6.82, mAP@10 7.83 | — | arXiv 2303.15247v2, Table 10 | B4 của nhóm gần như trùng: gallery CIRCO nhiều khả năng đủ ảnh |
| B1 | trước 16/10 | SEARLE gốc | torch.hub B/32 | CIRR val | R@1 25.66, R@10 69.24 | — | `outputs/logs/eval_searle_cirr.log` | |
| B2 | trước 16/10 | Tái lập | phi_B32, `phi_last.pt` | CIRR val | R@1 25.47, R@10 69.24 | −0.19 / 0.00 so với B1 | `outputs/logs/eval_phi_cirr.log` | |
| B3 | trước 16/10 | Tái lập quy mô nhỏ | phi_b32 (OTI 6.824 ảnh) | CIRR val | R@1 22.60, R@10 66.23 | −3.06 / −3.01 so với B1 | `outputs/prev_results_oct5/cirr_val.txt` | |
| B4 | trước 16/10 | SEARLE gốc | torch.hub B/32 | CIRCO val | mAP@5 6.82, mAP@10 7.78 | — | `outputs/logs/eval3_searle_circo.log` | Chưa chắc gallery đủ ảnh |
| B5 | trước 16/10 | Tái lập | phi_B32, `phi_last.pt` | CIRCO val | mAP@5 6.42, mAP@10 7.15 | −0.40 / −0.63 so với B4 | `outputs/logs/eval_phi_circo.log` | Chưa chắc gallery đủ ảnh |
| B6 | trước 16/10 | SEARLE gốc | torch.hub B/32 | FIQ val (tập con) | TB R@10 36.99, R@50 56.82 | — | `outputs/logs/eval5_searle_fashioniq.log` | Thiếu khoảng 2.431 ảnh |
| B7 | trước 16/10 | Tái lập | phi_B32, `phi_last.pt` | FIQ val (tập con) | TB R@10 36.51, R@50 57.05 | −0.48 / +0.23 so với B6 | `outputs/logs/eval5_phi_fashioniq.log` | Thiếu khoảng 2.431 ảnh |
| B8 | 16/10 | File nộp test | phi_B32 | CIRR test1 | _(chờ điểm server)_ | — | `outputs/submissions/phi_B32_cirr.json` | 4.148 truy vấn, mỗi truy vấn 50 kết quả; đã có `subset_phi_B32_cirr.json` (4.148 truy vấn, 3 kết quả) |
| B9 | 16/10 | File nộp test | phi_B32 | CIRCO test | _(chờ điểm server)_ | — | `outputs/submissions/phi_B32_circo.json` | 800 truy vấn |
| B10 | | File nộp test | SEARLE gốc | CIRR test1 | _(chờ)_ | — | sinh ở H4 | |
| B11 | | File nộp test | SEARLE gốc | CIRCO test | _(chờ)_ | — | sinh ở H4 | |
| E2.0s | 18/10 | Tỉ lệ truy vấn có thuộc tính đo được (**sơ bộ**) | `attributes/parser.py` | CIRR val (4.181), FIQ val (6.016, đủ câu), CIRCO val (220) | 36,3% / 90,7% / 25,5% | Ngưỡng M3: 10% | file chú thích công khai trên GitHub của từng tập | Chưa phải lượt chính thức (H9, trên tập con FIQ có ảnh) |
| E2.1 | 18/10 | Kiểm thử bộ đo trên ảnh tổng hợp | `attributes/meters.py` | 46 sắc độ màu, 20 ảnh sọc/không sọc, 6 ảnh mẫu | Màu 100%; sọc 19/20 (đúng hướng 8/8); độ sáng, độ rực màu, diện tích đơn điệu; mặt nạ Otsu, GrabCut IoU 1,00, spectral residual 0,05–0,08 | — | `tools/test_meters.py` | Sai duy nhất: đốm thưa (ít cạnh) không được coi là có hoa văn |
| C11-chấm | 18/10 | Độ chính xác parser và bộ phân loại | `parser.py`, `taxonomy.py` | 100 câu thật ngẫu nhiên (50 CIRR + 50 FIQ, seed 1) | Ràng buộc đúng 97%; nhóm đúng 78% | — | `tools/grading/parser_grading_seed1.csv` | Chấm bởi trợ lý AI thay Chí: Chí nên rà lại phiếu |
| C11-CIRCO | 18/10 | Bộ phân loại so với nhãn người gán `semantic_aspects` | `taxonomy.py` | CIRCO val (220) | Phủ định P/R 1,00/1,00; góc nhìn 0,90/0,98; số lượng 0,71/0,97; vị trí–nền 0,85/0,77; thêm đối tượng 0,53/0,60 | — | `annotations/val.json` của CIRCO | |
| E3.3c | 18/10 | Hiệu chỉnh luật `--restore auto` (**tạm**) | `degradation_estimation.py` | 17 ảnh mẫu × (sạch + 8 loại × 2 mức) | Nhận đúng loại 86,9%; sai số σ nhiễu trung bình 1,1; 9/17 ảnh sạch được giữ nguyên | — | `tools/calibrate_restore_rules.py` | Phải hiệu chỉnh lại trên 300 ảnh CIRR train trước E3.3 |

## Mẫu ghi chi tiết

Copy khối dưới đây cho mỗi kết quả dùng trong báo cáo:

````markdown
### <Mã thí nghiệm>: <tên ngắn>

- **Ngày:** DD/MM/2026   **Người chạy:** Hòa / Doanh / Chí
- **Phần kế hoạch:** (ví dụ 02 → mục 2.3)
- **Giả thuyết:**
- **Lệnh chạy:**
  ```sh
  $PY src/validate.py ...
  ```
- **Commit git:** `abc1234`   **GPU:** A100 / L4 / T4
- **Dữ liệu:** tập, split, số truy vấn, số ảnh gallery
- **Thời gian chạy:**
- **Kết quả:** (bảng chỉ số, kèm khoảng tin cậy 95%)
- **So với baseline:** (hiệu số + khoảng tin cậy 95% của hiệu số)
- **Kết luận:** giữ / bỏ / cần thêm thí nghiệm
- **Vấn đề gặp phải:**
````

## Ghi chi tiết

_(Thêm các mục chi tiết bên dưới, mới nhất ở trên cùng.)_
