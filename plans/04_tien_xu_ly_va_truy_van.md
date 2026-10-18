# 04. Tiền xử lý ảnh và cách tạo truy vấn

**Tóm tắt:** đo xem các bước tiền xử lý ảnh (đệm ảnh, phép nội suy, lật ảnh) ảnh hưởng thế nào tới SEARLE, và thử hai cải tiến rẻ ở phía truy vấn (kết hợp điểm ảnh + văn bản, ensemble prompt). Không cần train.

**Thời gian:** Ngày 3–4 (tiền xử lý), Ngày 7 (phân tích left/right), Ngày 10 (kết hợp điểm, prompt).
**Người làm:** Chí viết code (C6, C15); Doanh chạy E4.1–E4.3; Hòa chạy E4.4–E4.5.

## 4.1. Đệm ảnh trước khi đổi kích thước (Doanh, Ngày 3)

CLIP cần ảnh vuông 224 × 224. SEARLE dùng **targetpad**: nếu ảnh quá dài hoặc quá cao (tỉ lệ cạnh dài / cạnh ngắn ≥ 1.25), đệm thêm viền đen cho tỉ lệ về 1.25, rồi mới đổi kích thước và cắt giữa ([data_utils.py:28-68](../SEARLE-main/src/data_utils.py#L28-L68)). Cách này giữ được nhiều nội dung ảnh hơn so với cắt thẳng.

| Mã | Cấu hình |
|---|---|
| E4.1-pad1.0 | đệm mọi ảnh không vuông thành vuông |
| E4.1-pad1.25 | mặc định |
| E4.1-pad1.5, E4.1-pad2.0 | đệm ít dần (2.0 gần như không đệm) |
| E4.1-clipcrop | tiền xử lý gốc của CLIP: đổi kích thước rồi cắt giữa, không đệm |

Lưu ý: phi được train với targetpad 1.25, nên đổi tiền xử lý có thể làm lệch đầu vào của phi; đó cũng là điều cần đo.

## 4.2. Phép nội suy (Doanh, Ngày 3)

Đổi kích thước ảnh cần nội suy giá trị điểm ảnh mới. So sánh 4 phép nội suy kinh điển (mặc định là bicubic):

| Mã | Phép nội suy | Đặc điểm |
|---|---|---|
| E4.2-nearest | láng giềng gần nhất | nhanh, răng cưa |
| E4.2-bilinear | song tuyến tính | mượt, hơi mờ |
| E4.2-bicubic | song lập phương (mặc định) | sắc nét hơn bilinear |
| E4.2-lanczos | Lanczos | giữ chi tiết tốt nhất, chậm nhất |

## 4.3. Lật ảnh khi truy vấn (TTA) (Doanh, Ngày 4 và 7)

Lấy trung bình đặc trưng của ảnh gốc và ảnh lật ngang:

| Mã | Lật ảnh nào |
|---|---|
| E4.3-flip-ref | chỉ ảnh tham chiếu |
| E4.3-flip-gallery | chỉ ảnh trong gallery |
| E4.3-flip-both | cả hai |

- Với ảnh tham chiếu: lấy trung bình đặc trưng rồi mới qua phi.
- CIRR có nhiều mô tả chứa "left" / "right"; lật ảnh có thể làm hỏng các truy vấn này. **E4.3b (Ngày 7):** tách nhóm truy vấn có từ chỉ hướng bằng `tools/query_taxonomy.py`, so sánh có và không lật.
- Cấu hình tiền xử lý tốt nhất trên CIRR val (E4.1–E4.3) được chạy lại trên CIRCO val và FIQ val **không chỉnh thêm** (Ngày 4).

## 4.4. Kết hợp điểm ảnh + văn bản (Hòa, Ngày 10)

`điểm(g) = cos(t, g) + α · cos(v, g)`, với `t` là đặc trưng truy vấn của SEARLE và `v` là đặc trưng ảnh tham chiếu. Quét α = 0, 0.05, 0.1, 0.2, 0.3, 0.5 trên CIRR val (`--fusion-weight`). Với CIRCO và FIQ phải dùng `--exclude-reference`, vì khi α > 0 chính ảnh tham chiếu sẽ có điểm cao nhất.

## 4.5. Ensemble prompt (Hòa, Ngày 10)

Mã hoá cùng một truy vấn bằng nhiều mẫu câu rồi lấy trung bình (`--prompt-set ensemble`):
`a photo of $ that {c}`, `an image of $ that {c}`, `a picture of $ that {c}`, `$ that {c}`, `a photo of $ , {c}`, `a photo of $ which {c}`.

Mẫu thứ năm phải có dấu cách trước dấu phẩy: bộ tách từ của CLIP gộp `$,` thành một token khác, khi đó không còn token `$` để thay bằng pseudo-token (phát hiện khi viết C15).

## Đã kiểm tra: chọn checkpoint khác `phi_last`

Đường cong validation cho thấy `phi_last` (R@1 25.47) gần bằng checkpoint tốt nhất (25.54 ở epoch 92). Không cần thí nghiệm này.

## Tiêu chí hoàn thành

- [ ] Bảng E4.1–E4.3 trên CIRR val, kèm khoảng tin cậy; cấu hình tốt nhất đã chạy trên CIRCO val và FIQ val.
- [ ] Bảng nhóm truy vấn left/right (E4.3b).
- [ ] Bảng E4.4–E4.5; nếu có lợi thì đưa vào cấu hình cuối cùng với [02](02_kiem_chung_thuoc_tinh.md) (E2.6).

## Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Khác biệt rất nhỏ giữa các phép nội suy | Vẫn là kết quả đáng báo cáo; kèm khoảng tin cậy để thấy rõ không khác biệt |
| Mỗi cấu hình tiền xử lý phải trích lại đặc trưng gallery | CIRR val nhỏ nên nhanh; với CIRCO chỉ chạy cấu hình tốt nhất |

## Lịch sử thay đổi

- 2026-10-16: tạo kế hoạch.
