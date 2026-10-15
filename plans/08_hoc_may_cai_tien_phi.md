# 08. Học máy: cải tiến mạng phi

**Mục tiêu:** train lại phi trên token OTI đã có (không chạy lại OTI) để (a) phân tích ảnh hưởng của lượng dữ liệu, (b) tìm cấu hình train tốt hơn cấu hình của paper, (c) đo vai trò của từng thành phần loss.

**Vai trò:** đóng góp **học máy**, chạy song song với các đóng góp xử lý ảnh.
**Ước lượng:** 4–6 ngày công, 30–50 giờ GPU (tuỳ số thí nghiệm chọn làm).
**Phụ thuộc:** [01](01_nen_tang_danh_gia.md).
**Liên quan:** các bộ câu GPT dùng cho GPT loss (P0 gốc, P1 làm sạch, P2–P3 sinh lại) được tạo ở [06](06_sinh_lai_cum_tu_gpt.md); mục 8.3d–e dùng chung với 06.

## Chỗ trùng với kế hoạch 2 tuần (dùng chung, không làm hai lần)

| Mục ở file này | Việc tương ứng trong kế hoạch 2 tuần | Cách xử lý |
|---|---|---|
| 8.0 Hạ tầng train | C10 "code train mới" (Chí): đã gồm `--lr-scheduler`, `--warmup-epochs`, `--oti-token-type`; [01](01_nen_tang_danh_gia.md) đã thêm vào giao kèo lệnh chạy | Chỉ bổ sung các cờ còn thiếu trong bảng 8.0 |
| 8.1 Lượng dữ liệu OTI | T1 (Hòa, ngày 6, `phi_scale_s0`) và T2 (Hòa, ngày 8) = E1.14a/b ở [01](01_nen_tang_danh_gia.md) mục 1.14 = **E8.1a/b** | Chạy một lần; dẫn chiếu ở cả hai nơi |
| 8.2a Lịch LR cosine | H16 (Hòa, ngày 10, chạy nền A100), exp `phi_B32_cosine` | Chạy một lần |
| 8.5 Nhiễu đặc trưng | `phi_s0_noise` σ = 0.1 trong E3.8a (Doanh) = **một điểm của E8.5**; [03](03_do_ben_va_phuc_hoi_anh.md) mục 3.4 "Phi bền với ảnh xấu" | Các giá trị σ còn lại của E8.5 chạy sau. 03 làm xấu **ảnh** khi train, 8.5 thêm nhiễu vào **đặc trưng**; so cả hai trên cùng bộ suy giảm của 03 mục 3.1 |
| 8.3e Bộ câu GPT | D17 (Doanh, ngày 13) = E6.4 ở [06](06_sinh_lai_cum_tu_gpt.md) | Baseline = T1 |

## 8.0. Hạ tầng (làm trước)

| Việc | Vị trí | Lý do |
|---|---|---|
| Tra token đích bằng `dict` thay vì `names_list.index(...)` | [train_phi.py:174-175](../SEARLE-main/src/train_phi.py#L174-L175) | Tìm tuyến tính trong 100.000 tên cho từng ảnh; có thể là nút thắt tốc độ. Đo thời gian/epoch trước và sau |
| Sửa lỗi `train_metrics.csv` chỉ giữ epoch cuối | [train_phi.py:224-229](../SEARLE-main/src/train_phi.py#L224-L229) | `train_one_epoch` nối DataFrame nhưng không trả về |
| Cho phép train trên tập con | [train_phi.py:244-247](../SEARLE-main/src/train_phi.py#L244-L247) | Hiện train trên **mọi** ảnh ImageNet test; ảnh không có token OTI sẽ gây lỗi. Thêm bước lọc theo `names_list` |
| Tổng quát hoá `merge_oti.py` | `outputs/merge_oti.py` | Gộp các shard bất kỳ (ví dụ s0 + s1), không cố định 3 shard |
| Thêm `--lr-scheduler {none,cosine}` và `--warmup-epochs` | [train_phi.py:269](../SEARLE-main/src/train_phi.py#L269) | Hiện lr cố định |
| Thêm `--oti-token-type {ema,raw}` | [train_phi.py:132-134](../SEARLE-main/src/train_phi.py#L132-L134) | Hiện luôn dùng token EMA |
| Thêm `--feature-noise <σ>` | [train_phi.py:170](../SEARLE-main/src/train_phi.py#L170) | Cho mục 8.5 |
| Thêm `--phrase-match`, `--concept-filter` | [train_phi.py:117-125](../SEARLE-main/src/train_phi.py#L117-L125), [:181-182](../SEARLE-main/src/train_phi.py#L181-L182) | Cho mục 8.3e; đặc tả ở [06](06_sinh_lai_cum_tu_gpt.md) mục 6.2. Bộ câu khác chọn bằng `--gpt-exp-name` sẵn có |
| Lưu checkpoint lên Drive, chạy tiếp bằng `--resume` | `train_phi.py` | Colab hay ngắt giữa chừng (README mục 6) |

Kiểm tra không hồi quy: sau khi sửa, chạy 2 epoch với cấu hình gốc; chỉ số CIRR val phải trùng 2 dòng đầu của `outputs/phi_models/phi_B32/validation_metrics.csv` (train đã đặt `torch.use_deterministic_algorithms(True)`).

## 8.1. Ảnh hưởng của lượng dữ liệu OTI

Đã có 2 điểm: 6.824 ảnh cho R@1 22.60; 100.000 ảnh cho 25.47. Thêm 2 điểm từ các shard sẵn có:

| Mã | Dữ liệu | Số ảnh | Thời gian ước tính (100 epoch) |
|---|---|---|---|
| (đã có) | oti_b32 cũ | 6.824 | — |
| E8.1a | oti_B32_s0 | 33.334 | 2–3 giờ |
| E8.1b | oti_B32_s0 + s1 | 66.668 | 4–5 giờ |
| (đã có) | oti_B32 | 100.000 | — |

- Giữ nguyên 100 epoch như paper (ít ảnh hơn thì ít bước cập nhật hơn; ghi rõ điều này khi báo cáo).
- Kết quả: biểu đồ R@1 và R@10 theo số ảnh (thang log). Câu hỏi: đã bão hoà ở 100.000 ảnh chưa? Nếu chưa thì có lý do tạo thêm token (dùng warm-start ở [07](07_hoc_may_oti_warm_start.md) E7.6 cho rẻ).
- Lưu ý: chưa rõ 6.824 ảnh của lần chạy cũ có trùng với shard s0 không (khác cách chọn ảnh); ghi chú khi vẽ biểu đồ.

## 8.2. Train lâu hơn và lịch learning rate

Đường cong validation của phi_B32 vẫn còn tăng ở epoch 100 (xem [00](00_hien_trang_va_baseline.md)).

| Mã | Cấu hình | Thời gian ước tính |
|---|---|---|
| E8.2a | 100 epoch, cosine + warmup 5 epoch | 6–7 giờ |
| E8.2b | 200 epoch, lr cố định (như gốc) | 12–14 giờ |
| E8.2c | 200 epoch, cosine + warmup | 12–14 giờ, chỉ làm nếu E8.2a có lợi |

## 8.3. Siêu tham số và thành phần của loss (sàng lọc ở quy mô nhỏ)

Để tiết kiệm GPU: **sàng lọc** trên một shard (33.334 ảnh), 50 epoch (khoảng 1–1,5 giờ/lần). Chỉ cấu hình tốt nhất mới chạy lại ở quy mô đầy đủ.

| Mã | Biến | Giá trị |
|---|---|---|
| E8.3-base | Cấu hình gốc ở quy mô nhỏ | mốc so sánh cho mọi thí nghiệm sàng lọc |
| E8.3a | `--lambda-gpt` | 0.5, 0.75 (gốc), 1.0 |
| E8.3b | `--temperature` | 0.1, 0.25 (gốc), 0.5 |
| E8.3c | Token đích | EMA (gốc), raw |
| E8.3d | **Vai trò câu GPT:** `--lambda-gpt 0` | Bỏ hẳn regularization bằng câu GPT; cho biết câu GPT đóng góp bao nhiêu |
| E8.3e | **Bộ câu GPT khác** | P1 (làm sạch), P2, P3 (sinh lại) từ [06](06_sinh_lai_cum_tu_gpt.md). Đây chính là E6.4; ghi kết quả một lần, dẫn chiếu ở cả hai nơi |

So sánh sàng lọc luôn với E8.3-base (cùng quy mô), không so với phi_B32.

## 8.4. Kiến trúc phi (sàng lọc ở quy mô nhỏ)

phi hiện là MLP 3 lớp: 512 → 2048 → 2048 → 512, GELU, dropout 0.5 ([phi.py](../SEARLE-main/src/phi.py)).

| Mã | Biến thể |
|---|---|
| E8.4a | Độ rộng lớp ẩn: 1024 / 2048 (gốc) / 4096 |
| E8.4b | Dropout: 0.3 / 0.5 (gốc) |
| E8.4c | MLP có kết nối residual + LayerNorm |

- Giữ nguyên class `Phi` cũ để các checkpoint cũ vẫn nạp được; kiến trúc mới đặt tên class khác và ghi tên kiến trúc vào `hyperparameters.json`.
- `validate.py` ([validate.py:448-450](../SEARLE-main/src/validate.py#L448-L450)), module truy xuất chung và demo phải đọc tên kiến trúc từ `hyperparameters.json` khi nạp checkpoint. Demo hiện ghi cứng `Phi(input_dim=512, hidden_dim=2048, ...)` ([app.py:125](../SEARLE-main/demo/app.py#L125)).

## 8.5. Thêm nhiễu vào đặc trưng ảnh khi train

**Ý tưởng** (tương tự CapDec): thêm nhiễu Gaussian vào đặc trưng ảnh đầu vào của phi khi train, ví dụ `f' = f + σ · ‖f‖ · ε / √d` với `ε ~ N(0, I)` và `d = 512`, để phi bớt nhạy với những thay đổi nhỏ của đặc trưng.

| Mã | σ (tương đối so với độ lớn đặc trưng) |
|---|---|
| E8.5 | 0 (gốc), 0.05, 0.1, 0.2, sàng lọc ở quy mô nhỏ |

Kiểm tra bằng bộ suy giảm ảnh của [03](03_do_ben_va_phuc_hoi_anh.md): phi train có nhiễu đặc trưng có giảm điểm ít hơn khi ảnh truy vấn bị xấu không, và so với phi bền train trên ảnh xấu của 03 thì thế nào? Bảng so sánh chung:

| Mô hình | Ảnh sạch | Nhiễu | Mờ | JPEG | Thiếu sáng |
|---|---|---|---|---|---|
| phi_B32 | | | | | |
| + nhiễu đặc trưng (E8.5, σ tốt nhất) | | | | | |
| phi bền với ảnh xấu (03) | | | | | |

## 8.6. Xác nhận cấu hình tốt nhất

- [ ] Gộp các thay đổi có lợi từ 8.2–8.5, train đầy đủ trên 100.000 ảnh.
- [ ] Train thêm 2 seed (43, 44) cho cả cấu hình gốc và cấu hình mới (6–7 giờ/lần). Chỉ khẳng định cải thiện khi chênh lệch lớn hơn độ dao động giữa các seed.
- [ ] Đánh giá trên CIRR val, CIRCO val, FIQ val (tập con); nộp test.

## Ưu tiên nếu thiếu GPU

1. 8.0 (bắt buộc trước mọi lần train)
2. 8.1 (đã có trong T1–T2) và E8.3d (một lần train nhỏ, giá trị phân tích cao)
3. 8.2a
4. 8.5 (liên kết với phần xử lý ảnh)
5. 8.3a–c, 8.4, 8.3e
6. 8.6 với đủ 3 seed

## Tiêu chí hoàn thành

- [ ] Hạ tầng 8.0 xong và qua kiểm tra không hồi quy.
- [ ] Biểu đồ theo lượng dữ liệu (4 điểm).
- [ ] Bảng sàng lọc 8.3–8.5, trong đó có E8.3d (có/không câu GPT).
- [ ] Cấu hình cuối có ≥ 3 seed và có số test.

## Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Hết quota GPU Colab | Làm theo thứ tự ưu tiên ở trên; 8.3–8.5 chỉ sàng lọc ở quy mô nhỏ |
| Kết quả sàng lọc ở quy mô nhỏ không đúng ở quy mô lớn | Luôn xác nhận cấu hình được chọn ở quy mô đầy đủ (8.6) |
| Chọn mô hình theo CIRR val (train tự validate trên CIRR val) | Báo cáo thêm CIRCO val, FIQ val và test |
| Tranh GPU với train phi bền của 03 | Dùng chung hạ tầng 8.0; xếp lịch train nối tiếp, không chạy trùng tài khoản Colab |

## Lịch sử thay đổi

- 2026-10-16: tạo kế hoạch (bản cũ, số 04).
- 2026-10-16: chuyển thành file 08 theo yêu cầu giữ phần học máy; cập nhật liên kết sang cấu trúc kế hoạch 2 tuần; đổi mã thí nghiệm E4 → E8; thêm bảng chỗ trùng, E8.3d–e (vai trò câu GPT), cờ `--resume`, thứ tự ưu tiên.
- 2026-10-16: gộp kế hoạch: E8.3e dùng các bộ câu của 06 (= E6.4); thay cờ `--gpt-phrases-file` bằng `--phrase-match`, `--concept-filter` (dùng `--gpt-exp-name` sẵn có); bỏ liên kết tới bản lưu trữ đã xoá.
