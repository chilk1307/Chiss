# 09. Học máy: hướng mở rộng

Các hướng dưới đây thuộc kế hoạch chính, nhưng tốn công hoặc tốn GPU hơn nên xếp theo thứ tự ưu tiên ở cột "Thời điểm". Hướng rẻ làm ngay trong 2 tuần; hướng đắt làm từ mốc M4 trong [README](README.md) khi còn GPU, nếu chưa xong thì ghi phần đã làm và phần còn lại vào mục "Hướng phát triển" của báo cáo. Mỗi hướng ghi rõ chi phí để nhóm cân nhắc.

Sinh lại cụm từ GPT có file riêng: [06](06_sinh_lai_cum_tu_gpt.md).

## Bảng tóm tắt

| Hướng | Công sức | GPU | Rủi ro | Giá trị cho báo cáo | Thời điểm |
|---|---|---|---|---|---|
| 9.1 Mô tả ảnh + LLM | Trung bình | Thấp–trung bình | Trung bình | Cao | Từ M4 |
| 9.2 SEARLE-XL (ViT-L/14) | Thấp | Thấp | Thấp | Trung bình | **Ngay trong 2 tuần** (rẻ) |
| 9.3 Nhiều token theo vùng | Cao | Thấp | Cao | Cao nếu thành công | Từ M4, dùng mặt nạ từ [02](02_kiem_chung_thuoc_tinh.md) |
| 9.4 Thích nghi miền thời trang | Cao | Cao | Trung bình | Cao | Từ M4, chỉ khi có đủ ảnh FashionIQ |
| 9.5 So sánh với iSEARLE | Thấp–trung bình | Tuỳ | Thấp | Trung bình | Đọc tài liệu: **ngay trong 2 tuần** |
| 9.6 Backbone mới | Rất cao | Rất cao | Cao | Thấp so với chi phí | Ưu tiên thấp nhất: chỉ khảo sát (mục 9.6) |
| 9.7 Ứng dụng thực tế, demo mở rộng | Trung bình | Không đáng kể | Thấp | Trung bình–cao | Song song với demo ở [05](05_demo_va_bao_cao.md) |
| 9.8 Xếp hạng lại bằng mở rộng truy vấn | Thấp | Không đáng kể | Trung bình | Thấp–trung bình | Ngay trong 2 tuần |

## 9.1. Mô tả ảnh + LLM viết lại truy vấn (kiểu CIReVL)

**Ý tưởng.** Không dùng pseudo-token:
1. Một mô hình captioning (BLIP hoặc BLIP-2) mô tả ảnh tham chiếu.
2. Một LLM ghép mô tả đó với câu sửa đổi thành mô tả của ảnh đích.
3. CLIP mã hoá mô tả ảnh đích và tìm ảnh.

Tham khảo: CIReVL, "Vision-by-Language for Training-Free Compositional Image Retrieval" (ICLR 2024).

**Kết hợp với SEARLE:** `score = cos(t_SEARLE, g) + γ · cos(t_LLM, g)`. Hai nguồn bổ sung cho nhau: pseudo-token giữ chi tiết thị giác khó diễn đạt bằng lời; LLM xử lý câu sửa đổi phức tạp tốt hơn.

**Liên kết với xử lý ảnh:** điểm `γ · cos(t_LLM, g)` cộng được với điểm kiểm chứng thuộc tính của [02](02_kiem_chung_thuoc_tinh.md); so xem LLM hay bộ đo xử lý ảnh xử lý tốt hơn các câu có thuộc tính (màu, độ sáng, hoa văn).

**Chi phí.** Captioning + LLM cho 220 truy vấn CIRCO val là nhỏ; CIRR val (vài nghìn truy vấn) vẫn chạy được trên Colab với LLM cỡ nhỏ (1–3B tham số) hoặc qua API.

**Rủi ro.** LLM bịa chi tiết; kết quả phụ thuộc prompt của LLM, nên cố định prompt và ghi vào nhật ký.

## 9.2. Backbone lớn hơn: SEARLE-XL (ViT-L/14)

- Đánh giá miễn phí bằng mô hình phát hành sẵn:
  ```sh
  python src/validate.py --eval-type searle-xl --dataset cirr --dataset-path <CIRR>
  ```
- Các cải tiến xử lý ảnh ([02](02_kiem_chung_thuoc_tinh.md), [03](03_do_ben_va_phuc_hoi_anh.md), [04](04_tien_xu_ly_va_truy_van.md)) không phụ thuộc backbone: áp dụng lại cho SEARLE-XL để xem cải tiến có còn giữ khi mô hình mạnh hơn.
- Lưu ý: ViT-L/14 dùng ảnh 224×224 nhưng đặc trưng 768 chiều; cache đặc trưng gallery phải tách riêng theo backbone.
- Tự tái lập SEARLE-XL (OTI + phi với ViT-L/14) **không khuyến nghị**: chậm hơn ViT-B/32 nhiều lần.

## 9.3. Nhiều pseudo-token theo vùng

**Ý tưởng.** Ảnh có nhiều đối tượng thì một token cho cả ảnh dễ bỏ sót. Cắt vùng đối tượng, cho từng vùng qua CLIP + phi (phi áp dụng được cho bất kỳ ảnh nào), rồi ghép vào một prompt nhiều token, ví dụ `a photo of $ with $`.

**Chọn vùng bằng xử lý ảnh:** dùng lại mặt nạ đối tượng của [02](02_kiem_chung_thuoc_tinh.md) (Otsu, GrabCut, hình thái học, thành phần liên thông) và bản đồ nổi bật spectral residual thay vì detector học sâu; cách đơn giản nhất để so sánh là lưới 2×2 + toàn ảnh.

`encode_with_pseudo_tokens` đã hỗ trợ nhiều `$` trong một prompt ([encode_with_pseudo_tokens.py:15-29](../SEARLE-main/src/encode_with_pseudo_tokens.py#L15-L29)).

**Cần nghiên cứu thêm:** cách chọn vùng; cách viết prompt nhiều token; phi chưa từng thấy câu nhiều token khi train.

**Biến thể cho demo: ghép hai ảnh tham chiếu.** Cùng cơ chế nhiều `$` (tensor B × k × 512), ví dụ `a photo of $ and $`, `$ in the style of $`. Chất lượng khi ghép 2 token chưa được kiểm chứng; thử định tính trước khi làm giao diện.

## 9.4. Thích nghi miền thời trang

**Ý tưởng.** FashionIQ là tập SEARLE yếu nhất. Chạy OTI trên **ảnh train** của FashionIQ (không dùng caption, không dùng ảnh val/test), rồi fine-tune phi trên token ImageNet + token thời trang.

- Dùng OTI warm-start ([07](07_hoc_may_oti_warm_start.md)) để giảm chi phí OTI.
- Cần bộ ảnh FashionIQ đầy đủ (hiện thiếu khoảng 2.431 ảnh, xem [00](00_hien_trang_va_baseline.md) và [01](01_nen_tang_danh_gia.md)).
- Ghi rõ trong báo cáo: dùng ảnh **không nhãn** cùng miền. Vẫn không dùng bộ ba có nhãn, nhưng không còn hoàn toàn "zero-shot theo miền".

## 9.5. So sánh với iSEARLE

Bản mở rộng của chính nhóm tác giả: "iSEARLE: Improving Textual Inversion for Zero-Shot Composed Image Retrieval" (arXiv 2405.02951).

- [ ] Đọc và tóm tắt các thay đổi so với SEARLE (đưa vào phần tổng quan tài liệu của báo cáo, [05](05_demo_va_bao_cao.md)).
- [ ] Kiểm tra code đã được công bố chưa (README gốc ghi sẽ công bố khi bài được chấp nhận).
- [ ] Nếu có code: so sánh trực tiếp. Nếu không: chọn 1–2 ý tưởng dễ cài đặt nhất để thử trong [08](08_hoc_may_cai_tien_phi.md).

## 9.6. Backbone mới (OpenCLIP, SigLIP): ưu tiên thấp nhất

Phải viết lại `encode_with_pseudo_tokens` cho bộ mã hoá văn bản mới, chạy lại toàn bộ OTI (khoảng 15 giờ A100 cho 100.000 ảnh với ViT-B/32, lâu hơn với backbone lớn) và train lại phi. Toàn bộ không vừa 2 tuần, nên chỉ làm phần khảo sát:
- [ ] Đánh giá baseline **không cần pseudo-token** (chỉ ảnh, chỉ văn bản, ảnh + văn bản, như [01](01_nen_tang_danh_gia.md) mục 1.8) với backbone mới, để biết backbone mới mạnh hơn bao nhiêu.
- [ ] Ghi lại các chỗ phải sửa trong `encode_with_pseudo_tokens` (vị trí token `$` 259, chiều embedding, cách mã hoá văn bản).

## 9.7. Ứng dụng thực tế và demo mở rộng

**Kịch bản ứng dụng:** "trợ lý tìm ảnh" cho thư viện ảnh (ảnh cá nhân hoặc kho ảnh của tổ chức): ảnh mẫu + mô tả tiếng Việt.

- **Đánh giá trên ảnh thật:** nhóm tự chụp 300–500 ảnh, viết khoảng 50 truy vấn tiếng Việt, chấm tay số ảnh đúng trong top-5 (precision@5).
- **Nhập tiếng Việt:** dịch sang tiếng Anh trước khi đưa vào CLIP. Phương án offline: MarianMT `Helsinki-NLP/opus-mt-vi-en` (kiểm tra tương thích transformers 4.24, có thể cần môi trường riêng cho demo); phương án online: API của một LLM. Đánh giá nhỏ: viết lại 100 mô tả của CIRR val bằng tiếng Việt, so R@1 và R@10 với mô tả tiếng Anh gốc trên cùng 100 truy vấn.
- **Truy vấn nhiều lượt:** nút "Dùng làm ảnh tham chiếu" trên mỗi kết quả; thanh lịch sử các lượt; ảnh trong gallery đã có đặc trưng nên không phải mã hoá lại.
- **Đổi phong cách:** mẫu câu dựng sẵn `a sketch of $`, `a painting of $`, `a cartoon of $`, `an origami $`, `a toy $`; cần gallery có ảnh nhiều phong cách.
- **Gallery lớn:** FAISS (`faiss-cpu`, `IndexFlatIP`) khi gallery vượt khoảng 50 nghìn ảnh; có sẵn gallery mẫu (ví dụ COCO val2017, 5 nghìn ảnh).
- **Đóng gói, triển khai:** ghim phiên bản thư viện trong [demo/requirements.txt](../SEARLE-main/demo/requirements.txt); Dockerfile hoặc Hugging Face Spaces (CPU miễn phí đủ cho ViT-B/32 với gallery nhỏ). Khi public: tắt `/api/gallery/folder`, giảm giới hạn upload (`MAX_CONTENT_LENGTH` hiện 512 MB), như [05](05_demo_va_bao_cao.md) đã ghi.

## 9.8. Xếp hạng lại bằng mở rộng truy vấn (query expansion)

- `q' = normalize(q + Σ_{i≤k} w_i · g_i)`, với `g_i` là đặc trưng của top-k kết quả; k = 1, 3, 5. Cài trong module truy xuất chung ([01](01_nen_tang_danh_gia.md) mục 1.3), mặc định tắt.
- Rủi ro: có thể làm giảm R@1 khi top-1 sai. Chỉ giữ nếu bootstrap cho thấy cải thiện rõ.
- So sánh với xếp hạng lại bằng thuộc tính xử lý ảnh ở [02](02_kiem_chung_thuoc_tinh.md): hai cách có bổ sung cho nhau không.

## Lịch sử thay đổi

- 2026-10-16: tạo kế hoạch (bản cũ, số 06).
- 2026-10-16: chuyển thành file 09 theo yêu cầu giữ phần học máy; cập nhật liên kết sang cấu trúc kế hoạch 2 tuần; đổi số mục 6.x → 9.x; thêm cột thời điểm và liên kết với phần xử lý ảnh (02, 03, 04).
- 2026-10-16: gộp kế hoạch: mọi hướng thuộc kế hoạch chính (không còn "sau BTL"); 9.6 chỉ làm phần khảo sát; thêm 9.7 (ứng dụng thực tế, demo mở rộng: chuyển từ file 06 cũ và kế hoạch demo cũ) và 9.8 (mở rộng truy vấn, chuyển từ kế hoạch cũ); bỏ liên kết tới bản lưu trữ đã xoá.
