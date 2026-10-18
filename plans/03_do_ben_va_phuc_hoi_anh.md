# 03. Đóng góp thứ hai: độ bền với ảnh xấu, phục hồi ảnh và phi bền

**Tóm tắt:** ảnh truy vấn thực tế hay bị nhiễu, mờ, nén mạnh, thiếu sáng, lệch màu. Phần này (1) đo SEARLE giảm điểm bao nhiêu với từng loại suy giảm; (2) phục hồi ảnh bằng kỹ thuật xử lý ảnh cổ điển, tự chọn phép phục hồi theo loại suy giảm ước lượng được; (3) train trên Colab một phi "bền" bằng cách cho phi học cả đặc trưng của ảnh đã làm xấu.

**Thời gian:** code Ngày 4, 5, 8; thí nghiệm Ngày 5–12.
**Người làm:** Chí viết code (C9, C10, C14, C16); Doanh chạy thí nghiệm và train (D5–D13).

## Định vị so với công trình đã có

- Đã có benchmark độ bền cho bài toán CIR (Sun et al., 2023, "Benchmarking Robustness of Text-Image Composed Retrieval": CIRR-C, FashionIQ-C), chủ yếu cho các phương pháp có giám sát. Chỉ đo độ bền thì không còn mới.
- Điểm khác của nhóm:
  1. đo cho phương pháp **zero-shot** SEARLE;
  2. **phục hồi thích ứng:** ước lượng loại suy giảm bằng xử lý ảnh cổ điển rồi chọn phép phục hồi;
  3. so sánh "phục hồi tốt nhất theo PSNR/SSIM" với "phục hồi tốt nhất cho truy xuất";
  4. **phi bền:** train phi với đặc trưng của ảnh đã làm xấu, kiểm tra trên cả loại suy giảm đã thấy và chưa thấy khi train;
  5. phân tích miền tần số và độ bền của bộ đo thuộc tính ([02](02_kiem_chung_thuoc_tinh.md)).
- Liên quan: Liu et al., "When Image Denoising Meets High-Level Vision Tasks" (IJCAI 2018).

## Thiết lập chung

- Chỉ làm xấu **ảnh tham chiếu**; gallery giữ nguyên (ảnh trong kho sạch, ảnh người dùng chụp thì xấu).
- Seed cố định theo tên ảnh để kết quả lặp lại được.
- Đánh giá chính trên CIRR val và CIRCO val; FIQ val (tập con) tuỳ chọn.
- Kết quả trên ảnh sạch dùng lại E1.7d, E1.7e của Hòa.

## 3.1. Bộ suy giảm (Chí: C9, Ngày 4)

File mới `src/degradations.py`. Tham số: `--degrade <loại>:<mức>`, mức 1 = nhẹ, 2 = nặng; `--degrade mixed` = mỗi ảnh một suy giảm ngẫu nhiên (seed cố định).

| Loại | Mức 1 / mức 2 | Mô hình suy giảm |
|---|---|---|
| `gauss_noise` | σ = 10 / 25 (thang 0–255) | nhiễu cộng |
| `salt_pepper` | 2% / 10% điểm ảnh | nhiễu xung |
| `gauss_blur` | σ = 1 / 2 px | tích chập |
| `motion_blur` | nhân dài 9 / 21 px, góc ngẫu nhiên | tích chập với nhân đã biết |
| `jpeg` | chất lượng 30 / 10 | lượng tử hoá DCT |
| `low_light` | gamma 2.0 + nhiễu σ = 10 / gamma 3.0 + nhiễu σ = 20 | biến đổi cường độ |
| `low_res` | thu nhỏ ×2 / ×4 rồi phóng lại bằng bicubic | lấy mẫu lại |
| `color_cast` | kênh R × 1.2, B × 0.8 / R × 1.4, B × 0.6 | sai cân bằng trắng |

## 3.2. Bộ phục hồi (Chí: C9, Ngày 4)

File mới `src/restorations.py`. Tham số: `--restore <phương pháp>`; `--restore oracle` = phương pháp tốt nhất cho loại suy giảm đã biết (bảng ánh xạ điền sau E3.2).

| Phương pháp | Dùng cho | Kiến thức xử lý ảnh |
|---|---|---|
| `median` | nhiễu muối tiêu | lọc phi tuyến |
| `gaussian`, `bilateral` | nhiễu, JPEG | lọc không gian |
| `nlm` (non-local means) | nhiễu Gaussian | khử nhiễu |
| `wiener`, `richardson_lucy` (biết nhân mờ) | mờ chuyển động, mờ Gaussian | khôi phục ảnh, giải chập |
| `unsharp` | mờ, độ phân giải thấp | làm sắc nét |
| `histeq`, `clahe`, `gamma`, `retinex` (Retinex đơn thang) | thiếu sáng | tăng cường ảnh |
| `gray_world`, `white_patch` | lệch màu | cân bằng màu |
| `deblock` (làm mượt dọc biên khối 8 × 8) | JPEG | nén ảnh |

## 3.3. Ước lượng suy giảm và phục hồi tự động (Chí: C14, Ngày 8)

File mới `src/degradation_estimation.py`.

| Đại lượng | Cách ước lượng | Kiến thức xử lý ảnh |
|---|---|---|
| Mức nhiễu σ | trung vị trị tuyệt đối hệ số wavelet chi tiết chia 0.6745 (`skimage.restoration.estimate_sigma`) | Biến đổi wavelet |
| Nhiễu xung | tỉ lệ điểm ảnh bằng 0 hoặc 255 | Histogram |
| Độ mờ | phương sai Laplacian; tỉ lệ năng lượng tần số cao trong phổ Fourier | Toán tử đạo hàm, miền tần số |
| Mức nén JPEG | độ "khối": chênh lệch cường độ qua biên khối 8 × 8 so với bên trong khối | Nén ảnh |
| Độ sáng | trung bình và phân vị của L* | Histogram |
| Lệch màu | trung bình a*, b* trong CIELAB | Không gian màu |

- `--restore auto`: luật if-then trên các đại lượng trên; ngưỡng chọn trên ảnh CIRR **train** (không dùng tập đánh giá).
- `--restore auto_iqa`: thử vài phép phục hồi, chọn ảnh có điểm BRISQUE (đánh giá chất lượng không cần ảnh gốc) tốt nhất. Thư viện: `piq` hoặc module quality của OpenCV contrib (Chí chọn khi cài).
- Kiểm chứng bộ ước lượng: độ chính xác nhận dạng loại suy giảm, sai số ước lượng σ trên ảnh đã làm xấu có nhãn.

## 3.4. Phi bền với ảnh xấu (Chí: C10 viết code; Doanh: train trên Colab)

**Ý tưởng:** phi học ánh xạ đặc trưng ảnh → pseudo-token. Khi train, với xác suất p thay đặc trưng của ảnh sạch bằng đặc trưng của **chính ảnh đó đã bị làm xấu**, nhưng vẫn chưng cất về token OTI của ảnh sạch. Phi sẽ học cách "nhìn xuyên" qua nhiễu, mờ, nén.

- Đặc trưng ảnh xấu được tính trước một lần bằng `tools/precompute_degraded_features.py` (đi qua đúng tiền xử lý targetpad 1.25 như khi train), nên train vẫn nhanh.
- **Suy giảm dùng khi train (đã thấy):** `gauss_noise`, `gauss_blur`, `jpeg`, `low_light` (cả mức 1 và 2).
- **Suy giảm chỉ dùng khi kiểm tra (chưa thấy):** `salt_pepper`, `motion_blur`, `low_res`, `color_cast`. Nhờ vậy biết phi bền có tổng quát được không.
- So sánh thêm với cách rẻ hơn: cộng nhiễu Gaussian thẳng vào đặc trưng (`--feature-noise`, ý tưởng giống CapDec). Bản σ = 0.1 ở đây cũng là một điểm của E8.5 trong [08](08_hoc_may_cai_tien_phi.md).

| Bước | Dữ liệu | Thời gian ước tính (A100) |
|---|---|---|
| Tính đặc trưng ảnh xấu, shard s0 (33.334 ảnh × 9 phiên bản) | ImageNet test | khoảng 1 giờ |
| Sàng lọc: 3 bản phi, 50 epoch trên shard s0 | `phi_s0_clean`, `phi_s0_degaug`, `phi_s0_noise` | 1–1,5 giờ mỗi bản |
| Tính đặc trưng ảnh xấu cho 100.000 ảnh | ImageNet test | khoảng 2–3 giờ |
| Train bản đầy đủ: 100 epoch, 100.000 ảnh, biến thể tốt nhất | `phi_B32_robust` | 7–9 giờ |

## Thí nghiệm

| Mã | Nội dung | Người chạy | Ngày |
|---|---|---|---|
| E3.1 | Mức sụt giảm theo từng loại × từng mức (không phục hồi), CIRR val và CIRCO val | Doanh | 5 |
| E3.2 | Từng phép phục hồi cố định cho từng loại suy giảm (biết trước loại suy giảm) | Doanh | 7 |
| E3.3 | Phục hồi tự động (`auto`, `auto_iqa`) so với không phục hồi và `oracle`, trên bộ `mixed`; và trên **ảnh sạch** (kiểm tra không làm hỏng ảnh tốt) | Doanh | 9 |
| E3.4 | Tương quan PSNR/SSIM với mức hồi phục điểm truy xuất: biểu đồ phân tán, hệ số Spearman | Doanh | 10 |
| E3.5 | Phân tích miền tần số trên ảnh sạch: lọc thông thấp / thông cao ảnh tham chiếu với tần số cắt 0.05, 0.1, 0.2, 0.3, 0.5 | Doanh | 5 |
| E3.6 | Độ bền của bộ đo thuộc tính ([02](02_kiem_chung_thuoc_tinh.md)) dưới suy giảm; gray-world có cứu được bộ đo màu khi lệch màu không | Doanh | 11 |
| E3.7 | Hình minh hoạ: ảnh gốc / ảnh xấu / ảnh phục hồi, kèm top-5 kết quả | Doanh | 11 |
| E3.8a | Sàng lọc phi bền: 3 bản × (ảnh sạch + 8 loại suy giảm mức 2), CIRR val | Doanh | 6–8 |
| E3.8b | Train phi bền bản đầy đủ | Doanh | 9–10 |
| E3.8c | Đánh giá phi bền: 3 tập val sạch + bộ suy giảm trên CIRR val, CIRCO val; so với phi_B32 | Doanh | 10 |
| E3.9 | Độ trôi vector: với mỗi cấu hình của E3.1–E3.3, E3.5, vector ảnh, pseudo-token và vector truy vấn lệch bao nhiêu so với ảnh sạch; độ trôi hay PSNR/SSIM đi cùng mức thiệt hại truy xuất hơn; phép phục hồi có kéo vector ảnh về gần ảnh sạch không; phi bền có ít trôi token hơn phi_B32 không | Doanh | 10 |

## Kết quả cần có

1. Bảng mức sụt giảm (E3.1) và bảng phục hồi (E3.2, E3.3).
2. Biểu đồ PSNR/SSIM so với điểm truy xuất (E3.4).
3. Đường cong điểm truy xuất theo tần số cắt (E3.5).
4. Bảng phi bền: suy giảm đã thấy và chưa thấy (E3.8).
5. Hình minh hoạ (E3.7).
6. Bảng độ trôi vector và bảng "phục hồi tốt cho mắt hay cho CLIP" (E3.9): giải thích vì sao có phép phục hồi tăng PSNR mà không tăng điểm truy xuất. Công cụ `tools/vector_analysis.py drift` (C24) đọc cấu hình từ chính các thư mục kết quả E3.x và lấy lại đặc trưng ảnh từ cache, nên chỉ tốn GPU cho việc mã hoá lại câu truy vấn (khoảng 15 giây mỗi cấu hình trên CIRR val).

## Tiêu chí hoàn thành

- [ ] Bộ suy giảm, phục hồi, ước lượng có kiểm thử trên vài ảnh (lưu hình trước/sau).
- [ ] Bảng E3.1–E3.3 cho CIRR val và CIRCO val, kèm khoảng tin cậy.
- [ ] Trả lời: phục hồi tự động có hơn không phục hồi không, và có làm hỏng ảnh sạch không.
- [ ] Phi bền: biết rõ được gì, mất gì trên ảnh sạch, và có tổng quát sang suy giảm chưa thấy không.

## Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Phục hồi sinh thêm nhiễu giả, làm điểm giảm | Luôn đo cả trên ảnh sạch; bộ chọn được phép chọn "không phục hồi" |
| Wiener cần biết nhân mờ | Ghi rõ là khôi phục không mù (non-blind) |
| Phi bền làm giảm điểm trên ảnh sạch | Báo cáo trung thực; thử p nhỏ hơn (0.3) nếu còn thời gian |
| Colab ngắt khi train bản đầy đủ | `--resume` từ checkpoint trên Drive |
| Bị xem là không mới | Định vị rõ so với Sun et al. (2023) như ở đầu file |

## Lịch sử thay đổi

- 2026-10-16: tạo kế hoạch (mở rộng từ mục độ bền của bản cũ; thêm phục hồi tự động và phi bền).
- 2026-10-19: thêm E3.9 (độ trôi vector của ảnh tham chiếu đã xử lý).
