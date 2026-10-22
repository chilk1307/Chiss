# Tiến độ dự án

**Cập nhật:** 22/10/2026 (Ngày 6), đối chiếu với `results/` trên Drive và GitHub: Chí đã xong toàn bộ code cho Colab (C3, C5–C16, C21–C24). Doanh xong D1, D2 (21/10). Hòa dừng H5 ở E1.7a (19/10): SEARLE gốc trên CIRR ra R@1 25.59, mốc là 25.66, ngưỡng ±0,05. Từ đó H4–H18 và D3 trở đi chưa chạy. Dữ liệu trên Drive có lỗi: `FashionIQ.tar` 0 byte, `CIRR.tar` (6,0 GB) bị cụt. **Phần chạy Colab trễ 3–4 ngày so với lịch gốc** (theo lịch hôm nay Hòa ở H8, Doanh ở D7); mốc M1 (18/10) chưa đạt.
**Cách dùng:** làm xong việc nào thì đổi `- [ ]` thành `- [x]`, ghi ngày xong ở cuối dòng, rồi cập nhật bảng tóm tắt bên dưới. Việc làm được một phần thì giữ `- [ ]` và ghi "(một phần: …)".

## Tóm tắt

| Phần | Xong | Tổng | Tiến độ | Ghi chú |
|---|---|---|---|---|
| **Giai đoạn 0: tái lập SEARLE** (trước kế hoạch) | 13 | 15 | **87%** | Còn thiếu: điểm test từ server; FashionIQ chỉ có tập con |
| Kế hoạch 2 tuần: Chí (C1–C24) | 17 | 24 | 71% | Xong toàn bộ code cần cho Colab (C3, C5–C16, C21–C24); C2, C4 chờ nghiệm thu trên Colab; C1 cần làm tay; còn C17–C20 (demo, cấu hình cuối, báo cáo, video) |
| Kế hoạch 2 tuần: Hòa (H1–H18, T1–T2) | 1 | 20 | 5% | H3 xong; H1, H5 một phần; H5 dừng ở E1.7a (R@1 25.59, mốc 25.66) |
| Kế hoạch 2 tuần: Doanh (D1–D17) | 2 | 17 | 12% | D1, D2 xong 21/10; D3 là việc kế tiếp |
| **Kế hoạch 2 tuần: cả nhóm** | 20 | 61 | **33%** | Phần chạy trên Colab mới 3/37 việc: H5 đang chặn Hòa (và H4) |
| Mốc kiểm tra M1–M5 | 0 | 5 | 0% | M1 (hạn 18/10) chưa đạt: chưa nộp test, H5 chưa qua |

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

- [ ] **C1** (Ngày 1) Chia sẻ Drive cho Hòa, Doanh; xác nhận vị trí dữ liệu và dung lượng Drive (một phần: Hòa và Doanh đã đọc/ghi được thư mục `SEARLE` trên Colab; còn lại: `datasets/FashionIQ.tar` 0 byte và `datasets/CIRR.tar` bị cụt phải đóng gói lại. Hòa phải dùng thư mục giải nén sẵn trên Drive, Doanh tự tạo `CIRR_D2_val_20261021.zip` (chỉ val). Doanh đọc `outputs` từ một bản sao tên `SEARLE (1)`; `outputs/phi_models/` có 4 thư mục `dryrun_2ep` trùng tên cần dọn)
- [ ] **C2** (Ngày 1) Bộ chạy Colab: `setup_env.sh`, `prepare_data.sh`, `requirements-xla.txt`, `SEARLE_runner.ipynb` (một phần: notebook đã chạy được trên runtime mới (Hòa 19/10 trên T4, Doanh 21/10 trên A100); tiêu chí "R@1 = 25.66" chưa đạt: ra 25.59 trên T4, xem H5)
- [x] **C3** (Ngày 1) Gộp `datasets.py`, sửa nuốt lỗi, xử lý batch rỗng, đưa file nhỏ của `outputs/` vào git — 17/10
  - [x] Gộp bản `datasets.py` của repo và bản vá Colab
  - [x] Thay `except: pass` bằng cảnh báo + đếm mẫu lỗi
  - [x] Xử lý batch `None` trong các vòng lặp
  - [x] `outputs/submissions/` vào git
  - [x] `REPORT*.md`, `pip_freeze.txt`, `merge_oti.py`, csv/json của `phi_B32`, log `eval*.log` vào git
- [ ] **C4** (Ngày 2) Module truy xuất chung; `--output-dir`, `--cache-dir`, `--note`; CIRR luôn sinh cả file subset (một phần: code xong, trên dữ liệu giả lập mọi chỉ số và file nộp trùng code cũ; trên dữ liệu thật mới có E1.7a = 25.59 so với 25.66; chờ làm rõ nguyên nhân ở H5)
- [x] **C5** (Ngày 2) Baseline CLIP `image`, `text`, `sum`; `--exclude-reference` — 17/10
- [x] **C6** (Ngày 3) Cờ tiền xử lý `--targetpad-ratio`, `--interp`, `--tta` (cache phân biệt theo cấu hình) — 17/10
- [x] **C7** (Ngày 3) `check_datasets.py`, `bootstrap_ci.py`, `collect_results.py`; đối chiếu số "Paper" (khớp arXiv v2) — 17/10
- [x] **C8** (Ngày 3) CBIR cổ điển (histogram màu, LBP, HOG, Gabor), `--cbir-features`, `--cbir-text-weight` — 17/10
- [x] **C9** (Ngày 4) Suy giảm, phục hồi, lọc tần số; `--save-quality` (thử trên ảnh thật: phục hồi đúng loại đều tăng PSNR/SSIM) — 18/10
- [x] **C10** (Ngày 5) Code train: tra token bằng dict, train tập con, `--resume`, ảnh xấu, nhiễu đặc trưng, lịch LR, loại token (trên dữ liệu giả lập: trùng từng bit code gốc `70328e1`; `--resume` trùng từng bit chạy liền; mọi cờ mới chạy được; Doanh kiểm lại trên dữ liệu thật ở D2) — 18/10
- [x] **C11** (Ngày 5) Bộ tách ràng buộc, phân loại truy vấn, phân tích giữ – sửa; chấm tay 100 câu (parser đúng 97%, phân loại đúng 78%; phiếu chấm ở `tools/grading/`) — 18/10
- [x] **C12** (Ngày 6) Bộ đo thuộc tính; kiểm thử trên ảnh tổng hợp (E2.1 đã chạy: màu 100%, sọc 19/20, mặt nạ Otsu/GrabCut IoU 1,00); `attribute_stats.py` — 18/10
- [x] **C13** (Ngày 7) Xếp hạng lại theo thuộc tính (code xong, thử trên dữ liệu giả lập; họp mốc M3 theo lịch 23/10, theo dõi ở mục Mốc) — 18/10
- [x] **C14** (Ngày 8) Ước lượng suy giảm; `--restore auto`, `auto_iqa`, `oracle`; `--degrade mixed` (cả 4 chạy được; `tools/calibrate_restore_rules.py` nhận đúng loại suy giảm 86,9%; ngưỡng hiện tại là bản tạm từ 17 ảnh mẫu, phải hiệu chỉnh lại trên 300 ảnh CIRR train trước E3.3) — 18/10
- [x] **C15** (Ngày 8) Kết hợp điểm ảnh + văn bản; ensemble prompt (sửa lỗi mẫu `"$,"` làm mất token `$`) — 18/10
- [x] **C16** (Ngày 9) Công cụ hình ảnh: ví dụ, bản đồ che khuất, hình suy giảm, PSNR so với điểm truy xuất (`visualize_queries.py`, `occlusion_map.py`, `visualize_degradations.py`, `quality_vs_retrieval.py`; thử trên dữ liệu giả lập) — 18/10
- [ ] **C17** (Ngày 10–11) Demo: tab thuộc tính, tab ảnh xấu → phục hồi, so sánh phương pháp, nút "Vì sao?"
- [ ] **C18** (Ngày 11) Chọn cấu hình cuối, gửi lệnh nộp test
- [ ] **C19** (Ngày 12–13) README gốc repo, `CHANGES.md`, báo cáo, slide
- [ ] **C20** (Ngày 14) Video demo, kiểm tra trước khi nộp, nộp bài
- [x] **C21** (Ngày 9) Code OTI warm-start (`--init`, `--snapshot-steps`, `--token-file`) (`--init random` trùng từng bit code gốc; thêm `timing.json`, `--gpt-exp-name none`, nội suy token khi đánh giá) — 18/10
- [x] **C22** (Ngày 9) Công cụ bộ câu GPT: thống kê, làm sạch (P1), `--phrase-match`, `--concept-filter`, sửa lỗi `saved_exp_name` (đọc/ghi file câu dạng luồng, không cần nạp cả 700 MB) — 18/10
- [x] **C23** (Ngày 11) Notebook sinh câu `colab/gen_phrases.ipynb`, chế độ quy mô nhỏ (`tools/generate_phrases_llm.py` chế độ P2 và P3; chạy thử với mô hình nhỏ trên CPU; chạy thật trên Colab ở D16) — 18/10
- [x] **C24** (thêm) Phân tích vector ảnh và vector từ: `tools/vector_analysis.py` `tokens`, `gap`, `drift` (thử trên CIRR, CIRCO, FIQ giả lập; `drift` cho đúng số của lượt `validate.py` cùng cấu hình) — 19/10

## Kế hoạch 2 tuần: Hòa (Colab)

Chi tiết: [nhiem_vu_Hoa.md](nhiem_vu_Hoa.md).

- [ ] **H1** (Ngày 1) Dựng Colab, đóng gói dữ liệu, chạy thử SEARLE gốc trên CIRR val (một phần: Colab chạy được, cache CIRR val đã có trên Drive; đóng gói `.tar` hỏng nên dùng thư mục giải nén sẵn)
- [ ] **H2** (Ngày 1) Kiểm tra dữ liệu nhanh CIRR, CIRCO, FashionIQ (E1.2)
- [x] **H3** File subset CIRR (`subset_phi_B32_cirr.json` đã có, hợp lệ) — 16/10
- [ ] **H4** (Ngày 2) Sinh file test của SEARLE gốc; nộp test SEARLE gốc và phi_B32 (B8–B11) (chưa chạy: Hòa dừng cùng H5; thư mục `outputs/submissions/Hoa_H4_...` đã tạo nhưng chưa có file nộp)
- [ ] **H5** (Ngày 3) Kiểm tra không hồi quy; chạy lại baseline E1.7a–f (một phần: E1.7a chạy 19/10 trên T4 ra R@1 25.59 / R@5 55.49 / R@10 69.17 / R@50 90.58, mốc A100 là 25.66 / 55.47 / 69.24 / 90.58; lệch 0,07 điểm, khoảng 3 truy vấn, vượt ngưỡng 0,05 nên dừng; E1.7b–f chưa chạy)
- [ ] **H6** (Ngày 4) Baseline CLIP (E1.8) và CBIR cổ điển (E1.9)
- [ ] **H7** (Ngày 5) Kiểm tra dữ liệu bằng công cụ; khoảng tin cậy (E1.4); gom kết quả
- [ ] **H8** (Ngày 6) Phân loại truy vấn (E1.10); phân tích giữ – sửa (E1.11); vector ảnh – vector từ (E1.15a–b)
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

- [x] **D1** (Ngày 1) Dựng Colab, giải nén ImageNet, kiểm tra file trên Drive (ImageNet đủ 100.000 ảnh; CIRR val đủ 2.297 ảnh) — 21/10
- [x] **D2** (Ngày 2) Chạy thử train 2 epoch, đo thời gian (A100: epoch 1 R@1 22.24 so với mốc 22.05, epoch 2 22.79 so với 22.53; khoảng 2,4 phút mỗi epoch nên 100 epoch mất khoảng 4 giờ; kết quả ở `results/Doanh_D2_20261021_COMPLETED`) — 21/10
- [ ] **D3** (Ngày 3) Đệm ảnh (E4.1), nội suy (E4.2)
- [ ] **D4** (Ngày 4) Lật ảnh (E4.3); cấu hình tốt nhất trên CIRCO, FIQ
- [ ] **D5** (Ngày 5) Mức sụt giảm với 8 loại suy giảm (E3.1)
- [ ] **D6** (Ngày 5) Phân tích miền tần số (E3.5)
- [ ] **D7** (Ngày 6) Đặc trưng ảnh xấu shard s0; bắt đầu sàng lọc phi bền (E3.8a)
- [ ] **D8** (Ngày 7) Phục hồi cố định (E3.2); sàng lọc tiếp; nhóm left/right (E4.3b)
- [ ] **D9** (Ngày 8) Đánh giá 3 bản sàng lọc, chọn biến thể; đặc trưng ảnh xấu 100k
- [ ] **D10** (Ngày 9) Train phi bền bản đầy đủ (E3.8b); phục hồi tự động (E3.3)
- [ ] **D11** (Ngày 10) PSNR/SSIM so với điểm truy xuất (E3.4); đánh giá phi bền (E3.8c); độ trôi vector (E3.9)
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

1. **Gỡ chặn H5 (việc gấp nhất, Chí quyết định).** Code mới đã trùng từng bit code gốc trên dữ liệu giả lập. Nguyên nhân nhiều khả năng là phần cứng: mốc 25.66 đo trên A100 với torch 1.11, bản này mặc định tính TF32; Hòa chạy trên T4, tính FP32 đầy đủ. Chỉ một trong hai phép thử sau là đủ để kết luận:
   - Hòa chạy lại E1.7a trên **A100** (khoảng 3 phút): ra 25.66 thì C4 đạt;
   - hoặc chạy **code gốc** (`git checkout 70328e1`) trên **T4**: ra 25.59 thì code mới không làm lệch số.
   Sau đó ghi rõ GPU vào bảng mốc của H5 và chạy tiếp E1.7b–f, H4.
2. **Chí: sửa dữ liệu trên Drive (C1).** Đóng gói lại `CIRR.tar` (đủ train, dev, test1) và `FashionIQ.tar`, hoặc sửa `colab/prepare_data.sh` để đọc thẳng các thư mục giải nén sẵn. Dọn các thư mục `dryrun_2ep` trùng tên trong `outputs/phi_models/`. Nhắc Doanh dùng lối tắt `SEARLE` thay cho bản sao `SEARLE (1)`.
3. **Doanh: D3 (E4.1, E4.2), D5, D6 làm được ngay.** Cache CIRR val của Hòa đã có trên Drive, nhưng chỉ dùng lại được với cấu hình mặc định (targetpad 1.25, bicubic). Các cấu hình khác của E4.1, E4.2 phải trích đặc trưng lại.
4. Trước E3.3 (Ngày 9), Chí hiệu chỉnh lại ngưỡng phục hồi tự động trên ảnh CIRR train ở Colab rồi commit `src/restore_rules.json`:
   `$PY tools/calibrate_restore_rules.py --images-dir $DATA/CIRR/train --n 300 --out src/restore_rules.json --report-dir $DRIVE/results/E3.3-calibration`
5. Họp mốc M3 dự kiến 23/10 sẽ thiếu E2.0, E2.2 vì Hòa chưa tới H9. E2.0 chỉ đọc câu mô tả, không cần GPU, nên Hòa chạy được ngay song song với việc gỡ H5. Số sơ bộ của Chí (FashionIQ 90,7%, CIRR 36,3%, CIRCO 25,5%) đã vượt ngưỡng 10%.
