# Kế hoạch 2 tuần: cải tiến SEARLE bằng xử lý ảnh

**Thời gian:** 14 ngày, từ **Thứ Bảy 17/10/2026** đến **Thứ Sáu 30/10/2026**. Nếu bắt đầu ngày khác, chỉ cần dời cột ngày trong lịch.

**Nhóm và phân công:**
- **Chí:** viết toàn bộ code, đưa ra ý tưởng, thiết kế thí nghiệm, tích hợp demo, viết báo cáo và slide.
- **Hòa:** chạy trên Colab: dựng môi trường, kiểm tra dữ liệu, baseline, nộp test, thí nghiệm của đóng góp chính, train phân tích lượng dữ liệu.
- **Doanh:** chạy trên Colab: thí nghiệm tiền xử lý ảnh, độ bền với ảnh xấu và phục hồi ảnh, train phi bền với ảnh xấu.

Kế hoạch gồm hai luồng chạy song song: **xử lý ảnh** (01–05, ưu tiên trong lịch 14 ngày) và **học máy** (06–09, chạy vào thời gian GPU còn rảnh, xem mục 4).

## 1. Mục tiêu

Sau 2 tuần, nhóm có:

1. **Quy trình đánh giá đáng tin:** một code duy nhất cho tập val và file nộp test; có baseline (cả CLIP lẫn xử lý ảnh cổ điển), khoảng tin cậy và điểm test chính thức. Xem [01](01_nen_tang_danh_gia.md).
2. **Đóng góp chính (xử lý ảnh):** đo thuộc tính ảnh (màu, độ sáng, độ rực màu, hoa văn, mức phóng to) bằng kỹ thuật xử lý ảnh cổ điển để xếp hạng lại kết quả SEARLE. Xem [02](02_kiem_chung_thuoc_tinh.md).
3. **Đóng góp thứ hai (xử lý ảnh):** đo độ bền của SEARLE khi ảnh truy vấn kém chất lượng, phục hồi ảnh thích ứng, và train phi bền với ảnh xấu trên Colab. Xem [03](03_do_ben_va_phuc_hoi_anh.md).
4. **Phân tích tiền xử lý ảnh:** đệm ảnh, phép nội suy, lật ảnh khi truy vấn. Xem [04](04_tien_xu_ly_va_truy_van.md).
5. **Demo, báo cáo, slide, video** trình bày được các đóng góp trên. Xem [05](05_demo_va_bao_cao.md).
6. **Luồng học máy (song song):** sinh lại cụm từ GPT ([06](06_sinh_lai_cum_tu_gpt.md)), OTI khởi tạo từ phi ([07](07_hoc_may_oti_warm_start.md)), cải tiến phi ([08](08_hoc_may_cai_tien_phi.md)), các hướng mở rộng ([09](09_hoc_may_huong_mo_rong.md)).

## 2. Các file trong thư mục

| File | Nội dung | Ai đọc |
|---|---|---|
| [README.md](README.md) | Tổng quan, phân công, lịch 14 ngày, mốc kiểm tra, quy ước | Cả nhóm |
| [00_hien_trang_va_baseline.md](00_hien_trang_va_baseline.md) | Những gì đã làm, số liệu hiện có, vấn đề đã biết | Cả nhóm |
| [01_nen_tang_danh_gia.md](01_nen_tang_danh_gia.md) | Quy trình đánh giá chung, **giao kèo lệnh chạy**, baseline, phân tích lỗi | Cả nhóm |
| [02_kiem_chung_thuoc_tinh.md](02_kiem_chung_thuoc_tinh.md) | **Đóng góp chính:** kiểm chứng thuộc tính bằng xử lý ảnh cổ điển | Chí, Hòa |
| [03_do_ben_va_phuc_hoi_anh.md](03_do_ben_va_phuc_hoi_anh.md) | **Đóng góp thứ hai:** độ bền, phục hồi ảnh, train phi bền | Chí, Doanh |
| [04_tien_xu_ly_va_truy_van.md](04_tien_xu_ly_va_truy_van.md) | Đệm ảnh, nội suy, TTA; kết hợp điểm, ensemble prompt | Chí, Doanh, Hòa |
| [05_demo_va_bao_cao.md](05_demo_va_bao_cao.md) | Demo, cấu trúc báo cáo, slide, video, kiểm tra trước khi nộp | Chí |
| [06_sinh_lai_cum_tu_gpt.md](06_sinh_lai_cum_tu_gpt.md) | **Học máy:** sinh lại cụm từ GPT (`concept_to_phrases.pkl`), so sánh các bộ câu | Chí, Hòa, Doanh |
| [07_hoc_may_oti_warm_start.md](07_hoc_may_oti_warm_start.md) | **Học máy:** OTI khởi tạo từ phi (warm-start), chế độ chính xác cao cho demo | Chí, Hòa |
| [08_hoc_may_cai_tien_phi.md](08_hoc_may_cai_tien_phi.md) | **Học máy:** lượng dữ liệu, lịch LR, loss, kiến trúc, nhiễu đặc trưng | Chí, Hòa, Doanh |
| [09_hoc_may_huong_mo_rong.md](09_hoc_may_huong_mo_rong.md) | **Học máy:** LLM, SEARLE-XL, nhiều token, thích nghi miền, iSEARLE, ứng dụng thực tế | Chí |
| [nhiem_vu_Chi.md](nhiem_vu_Chi.md) | Việc của Chí theo từng ngày | Chí |
| [nhiem_vu_Hoa.md](nhiem_vu_Hoa.md) | Việc của Hòa theo từng ngày, kèm lệnh chạy | Hòa |
| [nhiem_vu_Doanh.md](nhiem_vu_Doanh.md) | Việc của Doanh theo từng ngày, kèm lệnh chạy | Doanh |
| [nhat_ky_thi_nghiem.md](nhat_ky_thi_nghiem.md) | Nhật ký thí nghiệm | Cả nhóm |
| [tien_do.md](tien_do.md) | Bảng tiến độ có ô đánh dấu cho từng việc và từng mốc | Cả nhóm |

## 3. Phân công và cách phối hợp

| Người | Làm | Không làm | File nhiệm vụ |
|---|---|---|---|
| Chí | Toàn bộ code (`src/`, `tools/`, `colab/`, demo); ý tưởng; thiết kế thí nghiệm; chọn cấu hình cuối; báo cáo, slide, video | Không ngồi canh các lượt chạy dài trên Colab | [nhiem_vu_Chi.md](nhiem_vu_Chi.md) |
| Hòa | Chạy trên Colab: môi trường, dữ liệu, baseline, nộp test, thí nghiệm thuộc tính, ví dụ định tính, train phân tích lượng dữ liệu | Không sửa code; gặp lỗi thì gửi log cho Chí | [nhiem_vu_Hoa.md](nhiem_vu_Hoa.md) |
| Doanh | Chạy trên Colab: tiền xử lý, độ bền và phục hồi ảnh, train phi bền với ảnh xấu | Không sửa code; gặp lỗi thì gửi log cho Chí | [nhiem_vu_Doanh.md](nhiem_vu_Doanh.md) |

**Cách phối hợp:**
1. Chí đẩy code lên GitHub `chilk1307/XLA_SEARLE` và nhắn nhóm: "Xong C…, Hòa chạy được H…, Doanh chạy được D…".
2. Hòa và Doanh chạy đúng lệnh trong file nhiệm vụ của mình. Đầu mỗi phiên Colab đều lấy code mới nhất.
3. Mỗi mã thí nghiệm chỉ do **một người** chạy, kết quả nằm trong thư mục riêng `results/<mã>/`, nên không ai ghi đè kết quả của ai.
4. Cuối mỗi ngày, mỗi người gửi báo cáo ngắn vào nhóm (mẫu ở cuối mỗi file nhiệm vụ).

## 4. Lịch 14 ngày

Mã việc: **C** = Chí, **H** = Hòa, **D** = Doanh, **T** = lượt train tuỳ chọn của Hòa. Chi tiết ở từng file nhiệm vụ.

| Ngày | Ngày tháng | Chí (code, ý tưởng) | Hòa (Colab) | Doanh (Colab) | Mốc |
|---|---|---|---|---|---|
| 1 | T7 17/10 | C1 chia sẻ Drive; C2 notebook và script Colab; C3 gộp `datasets.py` | H1 dựng Colab; H2 kiểm tra dữ liệu | D1 dựng Colab; chuẩn bị ImageNet | |
| 2 | CN 18/10 | C4 module truy xuất chung; C5 baseline CLIP | H4 sinh file test của SEARLE gốc, nộp test (H3 đã xong: file subset CIRR đã có) | D2 chạy thử train 2 epoch | **M1** |
| 3 | T2 19/10 | C6 cờ tiền xử lý; C7 công cụ đánh giá; C8 CBIR cổ điển | H5 kiểm tra không hồi quy, chạy lại baseline | D3 đệm ảnh, nội suy (chiều) | |
| 4 | T3 20/10 | C9 suy giảm, phục hồi, lọc tần số | H6 baseline CLIP và CBIR | D4 lật ảnh (TTA) | |
| 5 | T4 21/10 | C10 code train mới; C11 tách ràng buộc, phân loại truy vấn | H7 khoảng tin cậy, gom kết quả | D5 mức sụt giảm; D6 phân tích tần số | **M2** |
| 6 | T5 22/10 | C12 bộ đo thuộc tính, kiểm thử ảnh tổng hợp | H8 phân loại truy vấn, giữ–sửa; T1 train nền | D7 đặc trưng ảnh xấu (shard s0); bắt đầu sàng lọc phi bền | |
| 7 | T6 23/10 | C13 xếp hạng lại; **họp mốc M3** | H9 thống kê thuộc tính, đồng thuận bộ đo; H17 E7.0, E6.2-Pλ0 (06) | D8 phục hồi cố định; sàng lọc (tiếp) | **M3** |
| 8 | T7 24/10 | C14 phục hồi tự động; C15 kết hợp điểm, ensemble prompt | H10 quét xếp hạng lại; T2 train nền | D9 đánh giá bản sàng lọc, chọn biến thể | |
| 9 | CN 25/10 | C16 công cụ hình ảnh, ví dụ; C21 OTI warm-start (07); C22 công cụ bộ câu GPT (06) | H10 (tiếp) mặt nạ, phủ định; đánh giá T1, T2 | D10 train phi bền bản đầy đủ; phục hồi thích ứng | |
| 10 | T2 26/10 | C17 demo | H11 kết hợp, tổng quát hoá; H16 E8.2a chạy nền (08) | D11 PSNR/SSIM; đánh giá phi bền | |
| 11 | T3 27/10 | C17 demo (tiếp); C18 chọn cấu hình cuối; C23 notebook sinh câu (06) | H12 ví dụ định tính, bản đồ che khuất; H15 E7.1 CIRCO val (07) | D12 độ bền bộ đo thuộc tính; hình minh hoạ | **M4** |
| 12 | T4 28/10 | C19 README, CHANGES.md, báo cáo | H13 nộp test cấu hình cuối | D13 chạy bổ sung; nộp test phi bền; D15 E7.2 nếu còn GPU (07); D16 sinh câu quy mô nhỏ (06) | |
| 13 | T5 29/10 | C19 báo cáo, slide | H14 tổng hợp, bàn giao; H18 E6.2 bộ câu mới (06) | D14 tổng hợp, bàn giao; D17 E6.4 train phi với bộ câu mới (06) | |
| 14 | T6 30/10 | C20 video, kiểm tra cuối, nộp bài | Dự phòng, đối chiếu số | Dự phòng, đối chiếu số | **M5** |

### Luồng học máy (song song)

Các việc học máy đã xếp vào bảng trên (số file ghi trong ngoặc), dùng chỗ trống của luồng xử lý ảnh và không chặn các mốc M1–M5:
- **Train phi:** T1 (ngày 6) và T2 (ngày 8) = E8.1a/b; H16 (ngày 10) = E8.2a ([08](08_hoc_may_cai_tien_phi.md)).
- **Sinh lại cụm từ GPT:** H17, C22, C23, D16, H18, D17 (bảng phân công trong [06](06_sinh_lai_cum_tu_gpt.md)).
- **OTI warm-start:** C21, H15, D15 ([07](07_hoc_may_oti_warm_start.md)).
- **Phần còn lại** (E6.3, E6.5, E7.3–E7.6, 8.3–8.6, các hướng của [09](09_hoc_may_huong_mo_rong.md)) làm khi còn GPU, theo thứ tự ưu tiên trong từng file. Phần chưa xong ghi vào mục "Hướng phát triển" của báo cáo.

## 5. Mốc kiểm tra

| Mốc | Ngày | Phải đạt |
|---|---|---|
| M1 | Ngày 2 (18/10) | Colab chạy được cho cả Hòa và Doanh; dữ liệu đã kiểm tra; code cũ chạy lại ra đúng số; đã nộp test của SEARLE gốc và phi_B32 |
| M2 | Ngày 5 (21/10) | Quy trình mới ra đúng số cũ; có bảng baseline (CLIP và cổ điển) kèm khoảng tin cậy; code suy giảm, phục hồi, code train mới chạy được |
| M3 | Ngày 7 (23/10) | Có E2.0 và E2.2: **quyết định** giữ hướng chính hay đổi (mục 6); có kết quả tiền xử lý và mức sụt giảm; sàng lọc phi bền đang chạy |
| M4 | Ngày 11 (27/10) | Đủ kết quả chính của 02, 03, 04; phi bền bản đầy đủ đã train xong; demo có 2 tab mới |
| M5 | Ngày 14 (30/10) | Báo cáo, slide, video, README hoàn chỉnh; nộp bài |

## 6. Kế hoạch dự phòng

| Tình huống | Xử lý |
|---|---|
| Mốc M3: hướng chính không khả thi (ít truy vấn có thuộc tính, hoặc bộ đo không khớp câu mô tả) | 03 thành đóng góp chính; Hòa chuyển sang chạy thêm thí nghiệm của 03 (FIQ val, phục hồi chọn theo BRISQUE); 02 giữ lại như một phân tích |
| Hết GPU hoặc quota Colab | Đánh giá chạy trên T4/L4; bỏ T1–T2 của Hòa; phi bền chỉ giữ bản sàng lọc |
| Colab ngắt khi đang train | Train lưu checkpoint lên Drive và chạy tiếp bằng `--resume` (Chí làm trong C10) |
| Code của Chí trễ | Hòa và Doanh làm trước các việc không cần code mới: kiểm tra dữ liệu, chạy lại bằng code cũ, nộp test, chạy thử train |
| Thiếu thời gian cuối kỳ | Cắt theo thứ tự: tiếng Việt trong demo → ensemble prompt → bản đồ che khuất → T1–T2 |
| Thiếu GPU cho luồng học máy | Giữ theo thứ tự: E6.1–E6.2 và E7.1 (rẻ) → E8.3d → 8.1, 8.2a → phần còn lại của 06–09; luồng học máy không được làm trễ các mốc của luồng xử lý ảnh |

## 7. Kiến thức Xử lý ảnh dùng trong đề tài

Bảng này cũng là khung cho phần "cơ sở lý thuyết" trong báo cáo.

| Chủ đề của môn | Kỹ thuật dùng trong đề tài | File |
|---|---|---|
| Không gian màu | RGB → HSV, CIELAB; khoảng cách màu CIEDE2000 | 02, 03 |
| Biến đổi hình học, nội suy | đệm ảnh, đổi kích thước, lật ảnh; nội suy nearest, bilinear, bicubic, Lanczos | 04 |
| Tăng cường ảnh | cân bằng histogram, CLAHE, hiệu chỉnh gamma, Retinex đơn thang, làm sắc nét unsharp masking | 03 |
| Lọc không gian | lọc trung bình, Gaussian, trung vị, bilateral, non-local means | 03 |
| Miền tần số | biến đổi Fourier (FFT); lọc thông thấp, thông cao; bản đồ nổi bật spectral residual; phát hiện sọc bằng đỉnh phổ | 02, 03 |
| Mô hình nhiễu và khôi phục ảnh | nhiễu Gaussian, muối tiêu; mờ Gaussian, mờ chuyển động; giải chập Wiener, Richardson–Lucy | 03 |
| Nén ảnh | JPEG (DCT, lượng tử hoá), hiện tượng khối, khử khối | 03 |
| Cân bằng màu | gray-world, white-patch | 03 |
| Hình thái học | phép mở, phép đóng, thành phần liên thông | 02 |
| Phân vùng | ngưỡng Otsu, phân cụm k-means, GrabCut | 02 |
| Cạnh và kết cấu | Sobel, Canny, LBP, bộ lọc Gabor, HOG | 01, 02 |
| Wavelet | ước lượng mức nhiễu từ hệ số wavelet | 03 |
| Đánh giá chất lượng ảnh | PSNR, SSIM; BRISQUE (không cần ảnh gốc) | 03 |
| Truy xuất ảnh theo nội dung (CBIR) | histogram màu, LBP, HOG, Gabor làm baseline cổ điển | 01 |
| Phân tích, trực quan hoá | bản đồ che khuất (occlusion map) | 01 |

## 8. Ngân sách GPU (ước lượng)

| Người | Việc dùng GPU | Ước lượng | Nên dùng |
|---|---|---|---|
| Hòa | Đánh giá, nộp test, thí nghiệm thuộc tính | 12–18 giờ (thêm 7–8 giờ nếu chạy T1–T2) | T4/L4 cho đánh giá; A100 cho T1–T2 |
| Doanh | Tiền xử lý, độ bền, phục hồi; tính đặc trưng ảnh xấu; train phi bền (3 bản sàng lọc + 1 bản đầy đủ) | 25–35 giờ | A100 cho train và tính đặc trưng; T4/L4 cho đánh giá |
| Chí | Chạy thử ngắn | 2–4 giờ | Bất kỳ |
| Hòa, thêm cho học máy | H16 E8.2a khoảng 6–7 giờ; H15 E7.1 khoảng 1–2 giờ; H17, H18 dưới 1 giờ | 8–10 giờ | A100 cho H16; T4/L4 đủ cho H15, H17, H18 |
| Doanh, thêm cho học máy | D15 E7.2 khoảng 1–2 giờ; D16 sinh câu quy mô nhỏ 1–3 giờ; D17 train phi 2 lần × 2–3 giờ | 6–11 giờ | A100 |
| Phần học máy còn lại (06–09) | E6.5, E7.6, 8.3–8.6, các hướng của 09 | 30–50 giờ nếu làm hết | Chỉ khi còn quota |

Đánh giá thường bị chậm vì đọc ảnh hơn là vì GPU, nên GPU rẻ vẫn ổn. Luôn giải nén dữ liệu ra `/content/data` thay vì đọc thẳng từ Drive.

## 9. Quy ước chung

1. **Chọn trên val, báo cáo trên test.** Mọi tham số chỉ chọn trên tập val; server test của CIRR, CIRCO chỉ dùng cho cấu hình cuối.
2. **Luôn có baseline cùng điều kiện:** SEARLE gốc, phi_B32 và các baseline ở [01](01_nen_tang_danh_gia.md), chạy trên đúng cùng dữ liệu và tiền xử lý.
3. **Chỉ gọi là "cải thiện"** khi khoảng tin cậy 95% của hiệu số không chứa 0 (bootstrap, [01](01_nen_tang_danh_gia.md) mục 1.4).
4. **Mã thí nghiệm:** `E<số file>.<mục><biến thể>-<mô tả>-<tập>`, ví dụ `E2.3-color-w0.1-cirr`, `E3.1-jpeg2-circo`.
5. **Vị trí:**
   - code trong repo, thư mục `SEARLE-main/`: `src/`, `tools/`, `colab/`;
   - kết quả trên Drive: `SEARLE/results/<mã>/`; cache đặc trưng: `SEARLE/cache/`;
   - mỗi lần chạy tự ghi một dòng vào `SEARLE/results/runs_<tên người chạy>.csv`.
6. **Một logic truy xuất duy nhất:** mọi cải tiến cài vào module chung ([01](01_nen_tang_danh_gia.md) mục 1.3), để số val và file nộp test luôn chạy cùng một logic.
7. **Ghi nhật ký:** `runs_*.csv` ghi tự động; kết luận quan trọng ghi vào [nhat_ky_thi_nghiem.md](nhat_ky_thi_nghiem.md).
8. **Kiểm tra không hồi quy:** sau mỗi thay đổi code, baseline CIRR val phải ra đúng số cũ (sai lệch ≤ 0.05).
9. **Ghi rõ phần nhóm làm:** mọi file sửa hoặc thêm đều ghi vào `CHANGES.md` ([05](05_demo_va_bao_cao.md) mục 5.5).

## 10. Lịch sử thay đổi

- 2026-10-16: tạo kế hoạch 6 phần.
- 2026-10-16: viết lại cho 2 tuần, 3 người; đóng góp chính chuyển sang xử lý ảnh; thêm file nhiệm vụ cá nhân.
- 2026-10-16: gộp kế hoạch: giữ luồng học máy song song (07–09, chuyển từ bản 6 phần); "Sinh lại cụm từ GPT" vào kế hoạch chính ([06](06_sinh_lai_cum_tu_gpt.md)); bỏ file "đề xuất sau BTL" và thư mục lưu trữ bản cũ.
