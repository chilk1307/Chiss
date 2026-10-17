# Tiến độ dự án

**Cập nhật:** 18/10/2026 (Ngày 2): Chí làm xong C9, C11, C12, C13, C15 (Ngày 4–8); C10, C14, C21, C22 làm một phần; C16 chưa làm. Code từ 18/10 chưa commit, chưa push.
**Cách dùng:** làm xong việc nào thì đổi `- [ ]` thành `- [x]`, ghi ngày xong ở cuối dòng, rồi cập nhật bảng tóm tắt bên dưới. Việc làm được một phần thì giữ `- [ ]` và ghi "(một phần: …)".

## Tóm tắt

| Phần | Xong | Tổng | Tiến độ | Ghi chú |
|---|---|---|---|---|
| **Giai đoạn 0: tái lập SEARLE** (trước kế hoạch) | 13 | 15 | **87%** | Còn thiếu: điểm test từ server; FashionIQ chỉ có tập con |
| Kế hoạch 2 tuần: Chí (C1–C23) | 10 | 23 | 43% | C3, C5–C9, C11–C13, C15 xong; C2, C4 chờ nghiệm thu trên Colab; C10, C14, C21, C22 một phần; C1 cần làm tay |
| Kế hoạch 2 tuần: Hòa (H1–H18, T1–T2) | 1 | 20 | 5% | H3 xong (file subset CIRR đã có) |
| Kế hoạch 2 tuần: Doanh (D1–D17) | 0 | 17 | 0% | |
| **Kế hoạch 2 tuần: cả nhóm** | 11 | 60 | **18%** | Code của Chí đi trước lịch khoảng 5 ngày |
| Mốc kiểm tra M1–M5 | 0 | 5 | 0% | M1 hạn 18/10 |

Trang tự đánh giá của demo (`/api/assessment`, chỉ tính phần tái lập): lõi **93%**, tính cả phần mở rộng (SEARLE-XL, SEARLE-OTI) **75%**.

## Giai đoạn 0: tái lập SEARLE (đã làm trước kế hoạch)

- [x] Cụm từ GPT (`concept_to_phrases.pkl`, bản tác giả phát hành)
- [x] Liên kết khái niệm cho 100.000 ảnh ImageNet test
- [x] OTI trên 100.000 ảnh (3 shard, gộp thành `oti_B32`)
- [x] Train phi 100 epoch (`phi_B32`)
- [x] Đánh giá CIRR val: R@1 25.47 (SEARLE gốc 25.66)
- [x] Đánh giá CIRCO val: mAP@5 6.42 (SEARLE gốc 6.82, trùng số paper Table 10 nên gallery nhiều khả năng đủ ảnh; H2 kiểm tra lại)
- [ ] Đánh giá FashionIQ val trên tập **đầy đủ** (một phần: đã chạy trên tập con, thiếu khoảng 2.431 ảnh)
- [x] File nộp test của phi_B32: CIRR (`phi_B32_cirr.json` + `subset_phi_B32_cirr.json`) và CIRCO (`phi_B32_circo.json`)
- [ ] Điểm test từ server CIRR và CIRCO
- [x] Demo web Flask (gallery, 3 chế độ truy vấn, trang tự đánh giá, tự nhận diện file nộp test)
- [x] Repo GitHub `chilk1307/XLA_SEARLE`, loại file lớn khỏi git
- [x] `SETUP_MISSING_FILES.md` và link Google Drive cho file lớn
- [x] Ghi chép tái lập: `outputs/REPORT.md`, `outputs/REPORT_results.md`
- [x] Kế hoạch 2 tuần trong `plans/` (đã commit `70328e1`)
- [x] Phân công Chí, Hòa, Doanh, mỗi người một file nhiệm vụ

## Kế hoạch 2 tuần: Chí (code, ý tưởng)

Chi tiết: [nhiem_vu_Chi.md](nhiem_vu_Chi.md).

- [ ] **C1** (Ngày 1) Chia sẻ Drive cho Hòa, Doanh; xác nhận vị trí dữ liệu và dung lượng Drive (một phần: vị trí dữ liệu đã ghi vào `colab/prepare_data.sh`; còn chia sẻ Drive, kiểm tra dung lượng, báo Hòa có được tạo `.tar` không: phải làm tay)
- [ ] **C2** (Ngày 1) Bộ chạy Colab: `setup_env.sh`, `prepare_data.sh`, `requirements-xla.txt`, `SEARLE_runner.ipynb` (một phần: đủ 4 file, `shellcheck` sạch, thư viện giải được cho Python 3.8; chờ chạy notebook trên runtime mới ra R@1 = 25.66, làm cùng H1)
- [x] **C3** (Ngày 1) Gộp `datasets.py`, sửa nuốt lỗi, xử lý batch rỗng, đưa file nhỏ của `outputs/` vào git — 17/10
  - [x] Gộp bản `datasets.py` của repo và bản vá Colab
  - [x] Thay `except: pass` bằng cảnh báo + đếm mẫu lỗi
  - [x] Xử lý batch `None` trong các vòng lặp
  - [x] `outputs/submissions/` vào git
  - [x] `REPORT*.md`, `pip_freeze.txt`, `merge_oti.py`, csv/json của `phi_B32`, log `eval*.log` vào git
- [ ] **C4** (Ngày 2) Module truy xuất chung; `--output-dir`, `--cache-dir`, `--note`; CIRR luôn sinh cả file subset (một phần: code xong, trên dữ liệu giả lập mọi chỉ số và file nộp trùng code cũ; chờ Hòa xác nhận ở H5 trên dữ liệu thật)
- [x] **C5** (Ngày 2) Baseline CLIP `image`, `text`, `sum`; `--exclude-reference` — 17/10
- [x] **C6** (Ngày 3) Cờ tiền xử lý `--targetpad-ratio`, `--interp`, `--tta` (cache phân biệt theo cấu hình) — 17/10
- [x] **C7** (Ngày 3) `check_datasets.py`, `bootstrap_ci.py`, `collect_results.py`; đối chiếu số "Paper" (khớp arXiv v2) — 17/10
- [x] **C8** (Ngày 3) CBIR cổ điển (histogram màu, LBP, HOG, Gabor), `--cbir-features`, `--cbir-text-weight` — 17/10
- [x] **C9** (Ngày 4) Suy giảm, phục hồi, lọc tần số; `--save-quality` (thử trên ảnh thật: phục hồi đúng loại đều tăng PSNR/SSIM) — 18/10
- [ ] **C10** (Ngày 5) Code train: tra token bằng dict, train tập con, `--resume`, ảnh xấu, nhiễu đặc trưng, lịch LR, loại token (một phần: đã viết đủ cờ, `tools/merge_oti.py`, `tools/precompute_degraded_features.py`; kiểm thử so với code cũ chưa đạt: epoch 1 khớp, epoch 2–3 lệch nhẹ; chưa thử `--resume`, `--aug-feature-dir`)
- [x] **C11** (Ngày 5) Bộ tách ràng buộc, phân loại truy vấn, phân tích giữ – sửa; chấm tay 100 câu (parser đúng 97%, phân loại đúng 78%; phiếu chấm ở `tools/grading/`) — 18/10
- [x] **C12** (Ngày 6) Bộ đo thuộc tính; kiểm thử trên ảnh tổng hợp (E2.1 đã chạy: màu 100%, sọc 19/20, mặt nạ Otsu/GrabCut IoU 1,00); `attribute_stats.py` — 18/10
- [x] **C13** (Ngày 7) Xếp hạng lại theo thuộc tính (code xong, thử trên dữ liệu giả lập; họp mốc M3 theo lịch 23/10, theo dõi ở mục Mốc) — 18/10
- [ ] **C14** (Ngày 8) Ước lượng suy giảm; `--restore auto`, `auto_iqa`, `oracle`; `--degrade mixed` (một phần: `oracle`, `--degrade mixed` chạy được; `degradation_estimation.py` đã viết nhưng thiếu file ngưỡng `restore_rules.json` và công cụ hiệu chỉnh, nên `auto` chưa chạy; `auto_iqa` cần thêm `piq` vào `requirements-xla.txt`)
- [x] **C15** (Ngày 8) Kết hợp điểm ảnh + văn bản; ensemble prompt (sửa lỗi mẫu `"$,"` làm mất token `$`) — 18/10
- [ ] **C16** (Ngày 9) Công cụ hình ảnh: ví dụ, bản đồ che khuất, hình suy giảm, PSNR so với điểm truy xuất
- [ ] **C17** (Ngày 10–11) Demo: tab thuộc tính, tab ảnh xấu → phục hồi, so sánh phương pháp, nút "Vì sao?"
- [ ] **C18** (Ngày 11) Chọn cấu hình cuối, gửi lệnh nộp test
- [ ] **C19** (Ngày 12–13) README gốc repo, `CHANGES.md`, báo cáo, slide
- [ ] **C20** (Ngày 14) Video demo, kiểm tra trước khi nộp, nộp bài
- [ ] **C21** (Ngày 9) Code OTI warm-start (`--init`, `--snapshot-steps`, `--token-file`) (một phần: `--token-file`, `--interp-phi`, `--interp-lambda` đã có ở phía đánh giá; `oti_inversion.py` chưa sửa)
- [ ] **C22** (Ngày 9) Công cụ bộ câu GPT: thống kê, làm sạch (P1), `--phrase-match`, `--concept-filter`, sửa lỗi `saved_exp_name` (một phần: `src/gpt_phrases_utils.py` và hai cờ cho `train_phi.py`; còn phía OTI, `phrases_stats.py`, `clean_phrases.py`, lỗi `saved_exp_name`)
- [ ] **C23** (Ngày 11) Notebook sinh câu `colab/gen_phrases.ipynb`, chế độ quy mô nhỏ

## Kế hoạch 2 tuần: Hòa (Colab)

Chi tiết: [nhiem_vu_Hoa.md](nhiem_vu_Hoa.md).

- [ ] **H1** (Ngày 1) Dựng Colab, đóng gói dữ liệu, chạy thử SEARLE gốc trên CIRR val
- [ ] **H2** (Ngày 1) Kiểm tra dữ liệu nhanh CIRR, CIRCO, FashionIQ (E1.2)
- [x] **H3** File subset CIRR (`subset_phi_B32_cirr.json` đã có, hợp lệ) — 16/10
- [ ] **H4** (Ngày 2) Sinh file test của SEARLE gốc; nộp test SEARLE gốc và phi_B32 (B8–B11)
- [ ] **H5** (Ngày 3) Kiểm tra không hồi quy; chạy lại baseline E1.7a–f
- [ ] **H6** (Ngày 4) Baseline CLIP (E1.8) và CBIR cổ điển (E1.9)
- [ ] **H7** (Ngày 5) Kiểm tra dữ liệu bằng công cụ; khoảng tin cậy (E1.4); gom kết quả
- [ ] **H8** (Ngày 6) Phân loại truy vấn (E1.10); phân tích giữ – sửa (E1.11)
- [ ] **H9** (Ngày 7) Thống kê thuộc tính (E2.0); đồng thuận bộ đo (E2.2) → mốc M3
- [ ] **H10** (Ngày 8–9) Quét xếp hạng lại (E2.3); mặt nạ (E2.4); phủ định (E2.5)
- [ ] **H11** (Ngày 10) Kết hợp (E2.6), tổng quát hoá (E2.7), kết hợp điểm (E4.4), ensemble prompt (E4.5)
- [ ] **H12** (Ngày 11) Ví dụ định tính (E1.12), bản đồ che khuất (E1.13), ví dụ có giải thích (E2.8)
- [ ] **H13** (Ngày 12) Nộp test cấu hình cuối (E2.9); khoảng tin cậy bảng chính
- [ ] **H14** (Ngày 13) Tổng hợp bảng, hình → bàn giao Chí
- [ ] **H15** (Ngày 11) OTI warm-start trên CIRCO val (E7.1a–d)
- [ ] **H16** (Ngày 10) Train phi với lịch LR cosine (E8.2a)
- [ ] **H17** (Ngày 7) Liên kết khái niệm CIRCO val (E7.0); OTI không dùng câu GPT (E6.2-Pλ0)
- [ ] **H18** (Ngày 13) OTI với bộ câu GPT mới trên CIRCO val (E6.2)
- [ ] **T1** (Ngày 6, tuỳ chọn) Train phi trên shard s0 (E1.14a = E8.1a)
- [ ] **T2** (Ngày 8, tuỳ chọn) Train phi trên shard s0 + s1 (E1.14b = E8.1b)

## Kế hoạch 2 tuần: Doanh (Colab)

Chi tiết: [nhiem_vu_Doanh.md](nhiem_vu_Doanh.md).

- [ ] **D1** (Ngày 1) Dựng Colab, giải nén ImageNet, kiểm tra file trên Drive
- [ ] **D2** (Ngày 2) Chạy thử train 2 epoch bằng code cũ, đo thời gian
- [ ] **D3** (Ngày 3) Đệm ảnh (E4.1), nội suy (E4.2)
- [ ] **D4** (Ngày 4) Lật ảnh (E4.3); cấu hình tốt nhất trên CIRCO, FIQ
- [ ] **D5** (Ngày 5) Mức sụt giảm với 8 loại suy giảm (E3.1)
- [ ] **D6** (Ngày 5) Phân tích miền tần số (E3.5)
- [ ] **D7** (Ngày 6) Đặc trưng ảnh xấu shard s0; bắt đầu sàng lọc phi bền (E3.8a)
- [ ] **D8** (Ngày 7) Phục hồi cố định (E3.2); sàng lọc tiếp; nhóm left/right (E4.3b)
- [ ] **D9** (Ngày 8) Đánh giá 3 bản sàng lọc, chọn biến thể; đặc trưng ảnh xấu 100k
- [ ] **D10** (Ngày 9) Train phi bền bản đầy đủ (E3.8b); phục hồi tự động (E3.3)
- [ ] **D11** (Ngày 10) PSNR/SSIM so với điểm truy xuất (E3.4); đánh giá phi bền (E3.8c)
- [ ] **D12** (Ngày 11) Độ bền bộ đo thuộc tính (E3.6); hình minh hoạ (E3.7)
- [ ] **D13** (Ngày 12) Chạy bổ sung; nộp test phi bền (nếu được chọn)
- [ ] **D14** (Ngày 13) Tổng hợp bảng, hình → bàn giao Chí
- [ ] **D15** (Ngày 12) OTI warm-start trên CIRR val (E7.2)
- [ ] **D16** (Ngày 12) Thống kê bộ câu (E6.1); sinh bộ câu mới quy mô nhỏ
- [ ] **D17** (Ngày 13) Train phi trên shard s0 với bộ câu mới (E6.4)

## Mốc kiểm tra

- [ ] **M1** (Ngày 2, 18/10) Colab chạy được cho Hòa và Doanh; dữ liệu đã kiểm tra; code cũ ra đúng số; đã nộp test SEARLE gốc và phi_B32
- [ ] **M2** (Ngày 5, 21/10) Quy trình mới ra đúng số cũ; bảng baseline có khoảng tin cậy; code suy giảm, phục hồi, train mới chạy được
- [ ] **M3** (Ngày 7, 23/10) Có E2.0, E2.2 và quyết định giữ hay đổi đóng góp chính
- [ ] **M4** (Ngày 11, 27/10) Đủ kết quả chính của 02, 03, 04; phi bền bản đầy đủ đã train; demo có 2 tab mới
- [ ] **M5** (Ngày 14, 30/10) Báo cáo, slide, video, README hoàn chỉnh; nộp bài

## Việc tiếp theo

1. Chí: sửa lệch của C10 trước tiên (Doanh cần code train từ Ngày 5–6); tạo `restore_rules.json` và công cụ hiệu chỉnh cho C14; làm C21, C16, phần còn lại của C22; cập nhật `CHANGES.md` và bảng giao kèo ở [01](01_nen_tang_danh_gia.md); rồi commit và push. Không push trước khi C10 qua kiểm thử.
2. Chí: làm tay phần còn lại của C1; chạy notebook một lần trên Colab để nghiệm thu C2.
3. Hòa: H1, H2 ngay khi code lên GitHub. Ô "Kiểm tra dữ liệu (E1.2)" của notebook đã dùng được `tools/check_datasets.py`, thay cho đoạn Python chép tay ở H2.
4. Doanh: D1 ngay khi code lên GitHub.
5. Số sơ bộ cho mốc M3 (đo trên file chú thích công khai, chưa phải E2.0 chính thức của Hòa): truy vấn có thuộc tính đo được chiếm FashionIQ 90,7%, CIRR 36,3%, CIRCO 25,5%, vượt ngưỡng 10%.
