# 05. Demo, báo cáo và trình bày

**Tóm tắt:** demo trình diễn hai đóng góp xử lý ảnh; repo GitHub rõ ràng, ghi nguồn đầy đủ; báo cáo, slide, video hoàn chỉnh.

**Thời gian:** demo Ngày 10–11; báo cáo, slide Ngày 12–13; video và kiểm tra cuối Ngày 14.
**Người làm:** Chí (C17, C19, C20). Hòa và Doanh bàn giao bảng số và hình ở Ngày 13.

## 5.1. Demo hiện có

[demo/app.py](../SEARLE-main/demo/app.py) (Flask, chạy ở 127.0.0.1:5050):
- nạp CLIP ViT-B/32 + phi; đổi checkpoint ngay trên giao diện;
- gallery: tải ảnh lên hoặc quét thư mục trên máy; đặc trưng lưu ở `demo/cache/gallery.pt`;
- 3 chế độ truy vấn ([app.py:221-227](../SEARLE-main/demo/app.py#L221-L227)): chỉ ảnh, chỉ văn bản, ảnh + mô tả;
- hiển thị các từ CLIP gần pseudo-word nhất;
- trang tự đánh giá tiến độ, tự nhận diện file nộp test (hàm `scan_submissions`, [app.py:295](../SEARLE-main/demo/app.py#L295)).

## 5.2. Tính năng demo mới (theo thứ tự ưu tiên, C17)

| # | Tính năng | Lấy từ |
|---|---|---|
| 1 | **Tab thuộc tính:** bật/tắt xếp hạng lại; hiện các ràng buộc tách được từ câu mô tả; mỗi kết quả kèm nhãn thuộc tính đo được (ô màu chủ đạo, độ sáng, độ rực, có sọc hay không), mặt nạ đối tượng và lời giải thích vì sao được đẩy lên/xuống | [02](02_kiem_chung_thuoc_tinh.md) |
| 2 | **Tab ảnh xấu → phục hồi:** chọn loại suy giảm để làm xấu ảnh truy vấn (hoặc tải ảnh thật chụp bằng điện thoại); hiện kết quả ước lượng suy giảm, phép phục hồi được chọn, ảnh trước/sau, kết quả tìm kiếm trước/sau | [03](03_do_ben_va_phuc_hoi_anh.md) |
| 3 | **So sánh phương pháp:** thêm chế độ `sum` và CBIR cổ điển; xem kết quả các phương pháp cạnh nhau | [01](01_nen_tang_danh_gia.md) |
| 4 | **Nút "Vì sao?":** bản đồ che khuất của ảnh tham chiếu đối với một kết quả | [01](01_nen_tang_danh_gia.md) |
| 5 | (Nếu kịp) nhập mô tả bằng tiếng Việt: dịch sang tiếng Anh trước khi đưa vào CLIP, hiển thị câu đã dịch | — |

## 5.3. Bảo mật khi deploy công khai

Hiện demo chỉ chạy trên máy cá nhân (127.0.0.1) nên chưa nguy hiểm. Nếu đưa lên mạng:
- tắt API `/api/gallery/folder` ([app.py:494-515](../SEARLE-main/demo/app.py#L494-L515)): API này nhận đường dẫn thư mục bất kỳ trên máy chủ và đưa ảnh trong đó vào gallery;
- giảm giới hạn upload 512 MB (`MAX_CONTENT_LENGTH`, [app.py:65](../SEARLE-main/demo/app.py#L65)).

## 5.4. Tổng quan tài liệu (C19)

Danh sách tối thiểu cần đọc và trích dẫn:

| Nhóm | Công trình |
|---|---|
| Nền tảng | CLIP (Radford et al., ICML 2021) |
| CIR có giám sát | CLIP4Cir (Baldrati et al., CVPR 2022; trích dẫn có trong `src/data_utils.py`) |
| CIR zero-shot | Pic2Word (Saito et al., CVPR 2023); SEARLE (Baldrati et al., ICCV 2023); iSEARLE (arXiv 2405.02951); LinCIR (Gu et al., CVPR 2024); Context-I2W (Tang et al., AAAI 2024); CIReVL (Karthik et al., ICLR 2024) |
| Giới hạn của mô hình thị giác – ngôn ngữ | ARO (Yuksekgonul et al., ICLR 2023) |
| Độ bền của CIR | Sun et al., 2023, "Benchmarking Robustness of Text-Image Composed Retrieval" |
| Xử lý ảnh cổ điển cho truy xuất | Color indexing (Swain & Ballard, IJCV 1991); LBP (Ojala et al., TPAMI 2002); colorfulness (Hasler & Süsstrunk, 2003); spectral residual (Hou & Zhang, CVPR 2007) |
| Phục hồi ảnh và tác vụ cấp cao | Liu et al., IJCAI 2018 |

- Bảng so sánh với tài liệu: lấy số từ bài báo, ghi rõ backbone (B/32, L/14) và tập (val/test); không trộn val với test; nói rõ SEARLE (2023) không còn là phương pháp tốt nhất.
- Tra cứu kỹ trước khi viết rằng đóng góp ở [02](02_kiem_chung_thuoc_tinh.md) là "mới".

## 5.5. README ở gốc repo và CHANGES.md (C19)

- **README.md** ở gốc repo: giới thiệu đề tài, thành viên, môn học; kết quả chính (bảng ngắn) và ảnh demo; cách chạy (môi trường, dữ liệu theo [SETUP_MISSING_FILES.md](../SEARLE-main/SETUP_MISSING_FILES.md), lệnh đánh giá, demo, notebook Colab); cấu trúc thư mục; liên kết tới `plans/`.
- **Ghi nguồn:** dự án dựa trên SEARLE (Baldrati, Agnolucci, Bertini, Del Bimbo, ICCV 2023), repo gốc `miccunifi/SEARLE`, giấy phép CC BY-NC 4.0. Giấy phép này yêu cầu ghi nguồn và **ghi rõ các thay đổi**.
- **CHANGES.md:** liệt kê mọi file sửa (kèm lý do) và mọi file mới của nhóm. Nguyên tắc: hạn chế sửa file gốc; code mới đặt trong file mới.

## 5.6. Cấu trúc báo cáo (C19)

| Chương | Nội dung | Lấy từ |
|---|---|---|
| 1. Giới thiệu | Bài toán truy xuất ảnh kết hợp (CIR), zero-shot, ứng dụng; đóng góp của nhóm | — |
| 2. Tổng quan | Mục 5.4 | — |
| 3. Cơ sở lý thuyết | CLIP; textual inversion; OTI (cosine loss + GPT loss, có công thức); phi và chưng cất tương phản; targetpad; chỉ số Recall@K, mAP@K; các kỹ thuật xử lý ảnh dùng trong đề tài (bảng ở README mục 7) | `src/`, README |
| 4. Tái lập | Môi trường, quy trình, chi phí, kết quả, sai khác so với paper | [00](00_hien_trang_va_baseline.md), [01](01_nen_tang_danh_gia.md) |
| 5. Phân tích | Baseline CLIP và CBIR, theo nhóm sửa đổi, giữ – sửa, ví dụ, bản đồ che khuất | [01](01_nen_tang_danh_gia.md) |
| 6. Đóng góp chính | Kiểm chứng thuộc tính bằng xử lý ảnh | [02](02_kiem_chung_thuoc_tinh.md) |
| 7. Đóng góp thứ hai | Độ bền, phục hồi ảnh, phi bền | [03](03_do_ben_va_phuc_hoi_anh.md) |
| 8. Tiền xử lý | Đệm ảnh, nội suy, TTA; kết hợp điểm, prompt | [04](04_tien_xu_ly_va_truy_van.md) |
| 9. Demo | Ảnh chụp các tab | mục 5.2 |
| 10. Học máy | Bộ câu GPT, OTI warm-start, cải tiến phi, các hướng mở rộng đã làm | [06](06_sinh_lai_cum_tu_gpt.md)–[09](09_hoc_may_huong_mo_rong.md) |
| 11. Kết luận | Kết quả chính, hạn chế, hướng phát triển (phần chưa làm xong của 06–09) | [09](09_hoc_may_huong_mo_rong.md) |

## 5.7. Danh sách hình và bảng

1. Sơ đồ pipeline SEARLE (OTI → phi → truy xuất).
2. Sơ đồ phương pháp kiểm chứng thuộc tính.
3. Bảng tái lập so với paper và SEARLE gốc.
4. Bảng baseline (CLIP và CBIR).
5. Bảng theo nhóm kiểu sửa đổi.
6. Đường cong train phi (có số trong `validation_metrics.csv`).
7. Bảng kết quả chính của đóng góp chính + ablation.
8. Bảng và biểu đồ độ bền, phục hồi, phân tích tần số, PSNR so với điểm truy xuất.
9. Bảng phi bền: suy giảm đã thấy / chưa thấy.
10. Hình ví dụ định tính, bản đồ che khuất, ví dụ có giải thích.
11. Ảnh chụp demo.

## 5.8. Slide và video (C19, C20)

- **Slide** 15–20 trang: bài toán (2), phương pháp gốc (3), tái lập (2), phân tích (2), đóng góp chính (4), độ bền và phục hồi (3), demo (1), kết luận (1).
- **Video demo** 2–3 phút: một truy vấn thường; một truy vấn có thuộc tính (xếp hạng lại kèm giải thích); một ảnh xấu được phục hồi rồi tìm lại.
- **Chuẩn bị câu hỏi phản biện:** vì sao zero-shot? vì sao không fine-tune? phần nào nhóm tự làm? tính mới so với công trình đã có? hạn chế là gì?

## 5.9. Kiểm tra trước khi nộp (C20)

- [ ] Mọi số trong báo cáo truy được về `results/<mã>/metrics.json` hoặc log.
- [ ] Số "Paper" đã đối chiếu với bài báo.
- [ ] FashionIQ ghi rõ là tập con.
- [ ] Mọi khẳng định "cải thiện" có khoảng tin cậy.
- [ ] Clone sạch repo và chạy lại được theo README.
- [ ] Ghi nguồn đầy đủ; CHANGES.md cập nhật.
- [ ] Không commit dữ liệu có bản quyền hoặc file lớn.

## Lịch sử thay đổi

- 2026-10-16: tạo kế hoạch.
