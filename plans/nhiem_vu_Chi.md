# Nhiệm vụ của Chí: code và ý tưởng

**Vai trò:** viết toàn bộ code (`src/`, `tools/`, `colab/`, demo), đưa ra ý tưởng, thiết kế thí nghiệm, chọn cấu hình cuối, viết báo cáo, slide và video. Hòa và Doanh chạy thí nghiệm trên Colab bằng code này.

**Nguyên tắc khi viết code:**
1. Dùng đúng tên tham số trong **giao kèo lệnh chạy** ([01](01_nen_tang_danh_gia.md) mục 1.3). Muốn đổi tên thì sửa bảng đó trước rồi báo nhóm.
2. Mọi tham số mới mặc định giữ hành vi cũ. Sau mỗi thay đổi, CIRR val phải ra đúng số cũ.
3. Code mới đặt trong file mới; mọi file sửa hoặc thêm đều ghi vào `CHANGES.md`.
4. Xong việc nào thì đẩy lên GitHub và nhắn nhóm: "Xong C…, Hòa chạy được H…, Doanh chạy được D…".
5. Ưu tiên việc đang làm người khác phải chờ (cột "Ai đang chờ").

## Lịch theo ngày

| Ngày | Ngày tháng | Việc | Ai đang chờ |
|---|---|---|---|
| 1 | T7 17/10 | C1, C2, C3 | Hòa H1–H2, Doanh D1 (cần notebook trước trưa) |
| 2 | CN 18/10 | C4, C5 | Hòa H5–H6 |
| 3 | T2 19/10 | C6 (làm đầu tiên), C7, C8 | Doanh D3 (chiều), Hòa H6–H7 |
| 4 | T3 20/10 | C9 | Doanh D5–D6 |
| 5 | T4 21/10 | C10, C11 | Doanh D7; Hòa H8, T1 |
| 6 | T5 22/10 | C12, kiểm thử E2.1 | Hòa H9 |
| 7 | T6 23/10 | C13; họp mốc M3 | Hòa H10 |
| 8 | T7 24/10 | C14, C15 | Doanh D10; Hòa H11 |
| 9 | CN 25/10 | C16; C21 (học máy); C22 (bộ câu GPT) | Hòa H12, H15, H18; Doanh D11–D12, D16–D17 |
| 10 | T2 26/10 | C17 demo | — |
| 11 | T3 27/10 | C17 (tiếp), C18 chọn cấu hình cuối; C23 notebook sinh câu (sau mốc M4) | Hòa H13; Doanh D13, D16 |
| 12 | T4 28/10 | C19 README, CHANGES.md, báo cáo | — |
| 13 | T5 29/10 | C19 báo cáo, slide | Nhận bàn giao H14, D14 |
| 14 | T6 30/10 | C20 video, kiểm tra cuối, nộp bài | — |

## Chi tiết từng việc

### C1. Chia sẻ tài nguyên (Ngày 1)
- Chia sẻ thư mục Drive `SEARLE` cho Hòa và Doanh, quyền *Người chỉnh sửa*.
- Nếu repo GitHub đặt riêng tư: thêm Hòa và Doanh làm cộng tác viên.
- Xác nhận vị trí trên Drive của: các dataset (CIRR, CIRCO, FashionIQ), ImageNet test, `outputs/` (GPT_phrases, oti_pseudo_tokens, similar_concepts, phi_models); ghi vào `colab/prepare_data.sh`.
- Kiểm tra dung lượng Drive còn trống: nếu còn đủ khoảng 20 GB, cho phép Hòa đóng gói CIRR và CIRCO thành file `.tar` (H1).

### C2. Bộ chạy Colab (Ngày 1, trước trưa)

| File | Nội dung |
|---|---|
| `colab/setup_env.sh` | Tạo venv Python 3.8 bằng `uv` từ `outputs/pip_freeze.txt` (giống `outputs/scripts/run_eval2.sh`), cài thêm `colab/requirements-xla.txt`; ghi `/content/env.sh` khai báo `DRIVE`, `DATA`, `PY`, `RUNNER` |
| `colab/requirements-xla.txt` | Thư viện xử lý ảnh, gợi ý bản hỗ trợ Python 3.8: `opencv-python-headless`, `scikit-image==0.21.0`, `scipy==1.10.1`; kiểm tra cài được |
| `colab/prepare_data.sh <tập...>` | Giải nén dữ liệu từ Drive ra `/content/data/<tập>` (`cirr`, `circo`, `fiq`, `imagenet`); tạo liên kết `data/{GPT_phrases,oti_pseudo_tokens,similar_concepts,phi_models}` tới `$DRIVE/outputs/...` |
| `colab/SEARLE_runner.ipynb` | Ô 0: `RUNNER`; ô 1: gắn Drive; ô 2: clone hoặc `git pull`; ô 3: cài môi trường; ô 4: chuẩn bị dữ liệu; ô 5: kiểm tra GPU; các ô sau: mẫu ô lệnh `%%bash` + `source /content/env.sh` |

Xong khi: chạy notebook từ đầu trên một runtime mới, `validate.py --eval-type searle --dataset cirr` ra R@1 = 25.66.

### C3. Gộp và sửa `datasets.py` (Ngày 1)
Theo [01](01_nen_tang_danh_gia.md) mục 1.1: gộp bản repo với bản vá Colab; thay `except: pass` bằng cảnh báo + đếm; xử lý batch `None`; đưa các file nhỏ trong `outputs/` vào git.

### C4. Module truy xuất chung (Ngày 2)
Theo [01](01_nen_tang_danh_gia.md) mục 1.3: `src/retrieval_core.py`; `validate.py` và `generate_test_submission.py` dùng chung; `--output-dir` (ghi `metrics.json`, `dump.jsonl`, `log.txt`, một dòng vào `runs_<RUNNER>.csv`), `--cache-dir`, `--note`; CIRR luôn sinh cả file subset.
Xong khi: Hòa xác nhận ở H5 số khớp và file nộp sinh lại trùng file cũ.

### C5. Baseline CLIP (Ngày 2)
`--eval-type image|text|sum`, `--exclude-reference` ([01](01_nen_tang_danh_gia.md) mục 1.8).

### C6. Cờ tiền xử lý (Ngày 3, làm đầu tiên)
`--targetpad-ratio`, `--interp`, `--tta` ([04](04_tien_xu_ly_va_truy_van.md)). Cache phải phân biệt theo cấu hình tiền xử lý.

### C7. Công cụ đánh giá (Ngày 3)
`tools/check_datasets.py`, `tools/bootstrap_ci.py` (có `--baseline`), `tools/collect_results.py`. Đối chiếu số "Paper" với bài báo ([01](01_nen_tang_danh_gia.md) mục 1.6).

### C8. CBIR cổ điển (Ngày 3)
`src/cbir.py`: histogram màu HSV, LBP, HOG, Gabor ([01](01_nen_tang_danh_gia.md) mục 1.9); `--eval-type cbir`, `--cbir-features`, `--cbir-text-weight`; cache đặc trưng cổ điển.

### C9. Suy giảm, phục hồi, lọc tần số (Ngày 4)
`src/degradations.py`, `src/restorations.py`, `src/freq_filters.py`; cờ `--degrade`, `--restore`, `--freq-filter`, `--save-quality` ([03](03_do_ben_va_phuc_hoi_anh.md) mục 3.1–3.2). Chỉ áp cho ảnh tham chiếu, không đụng gallery. Kiểm thử: lưu ảnh trước/sau của mỗi phép lên một thư mục để nhìn bằng mắt.

### C10. Code train (Ngày 5)
- Tra token bằng `dict` thay vì `names_list.index(...)` ([train_phi.py:174-175](../SEARLE-main/src/train_phi.py#L174-L175)); sửa lỗi `train_metrics.csv` chỉ giữ epoch cuối.
- `--train-on-oti-names`, `--resume` (lưu cả optimizer, scaler; checkpoint mỗi 5 epoch).
- `tools/merge_oti.py` (`--shards`, `--out`); `tools/precompute_degraded_features.py`; `--aug-feature-dir`, `--aug-prob`, `--feature-noise` ([03](03_do_ben_va_phuc_hoi_anh.md) mục 3.4).
- Cho luồng học máy ([08](08_hoc_may_cai_tien_phi.md)): `--lr-scheduler {none,cosine}`, `--warmup-epochs`, `--oti-token-type {ema,raw}`.

Xong khi: chạy 2 epoch cấu hình gốc gần khớp 2 dòng đầu của `validation_metrics.csv` cũ; 1 epoch có `--aug-feature-dir` chạy không lỗi; ngắt giữa chừng rồi `--resume` chạy tiếp được.

### C11. Tách ràng buộc và công cụ phân tích (Ngày 5)
`src/attributes/parser.py` ([02](02_kiem_chung_thuoc_tinh.md) mục 2.1); `tools/query_taxonomy.py` (có `--categories`); `tools/keep_edit_analysis.py`. Kiểm thử với khoảng 50 câu tự viết; chấm tay 100 truy vấn ngẫu nhiên để đo độ chính xác của bộ phân loại và parser.

### C12. Bộ đo thuộc tính (Ngày 6)
`src/attributes/meters.py` ([02](02_kiem_chung_thuoc_tinh.md) mục 2.2); `tools/test_meters.py` và tự chạy E2.1 trên máy; `tools/attribute_stats.py` (E2.0, và `--agreement` cho E2.2; nhận thêm `--degrade`, `--restore` để Doanh dùng ở E3.6).

### C13. Xếp hạng lại theo thuộc tính (Ngày 7)
Các cờ `--attr-*` ([02](02_kiem_chung_thuoc_tinh.md) mục 2.3). Chiều Ngày 7: họp mốc M3 với Hòa, quyết định theo E2.0 và E2.2 (ngưỡng ở 02).

### C14. Ước lượng suy giảm và phục hồi tự động (Ngày 8)
`src/degradation_estimation.py`; `--restore auto` (luật; ngưỡng chọn trên ảnh CIRR train), `--restore auto_iqa` (BRISQUE), `--restore oracle` (bảng ánh xạ điền theo kết quả E3.2 của Doanh); `--degrade mixed`.

### C15. Kết hợp điểm và ensemble prompt (Ngày 8)
`--fusion-weight`, `--prompt-set ensemble`, `src/prompts.py` ([04](04_tien_xu_ly_va_truy_van.md) mục 4.4–4.5).

### C16. Công cụ hình ảnh (Ngày 9)
`tools/visualize_queries.py` (có `--compare`, `--explain-attributes`), `tools/occlusion_map.py`, `tools/visualize_degradations.py`, `tools/quality_vs_retrieval.py`.

### C17. Demo (Ngày 10–11)
Theo [05](05_demo_va_bao_cao.md) mục 5.2: tab thuộc tính, tab ảnh xấu → phục hồi, so sánh phương pháp, nút "Vì sao?"; tiếng Việt nếu kịp.

### C18. Chọn cấu hình cuối (Ngày 11)
Dựa trên bảng của Hòa và Doanh: chọn cấu hình cuối của 02 (có thể kèm 04) và quyết định có nộp test phi bền không. Gửi Hòa đúng lệnh cho H13, gửi Doanh lệnh cho D13.

### C19. README, CHANGES.md, báo cáo, slide (Ngày 12–13)
Theo [05](05_demo_va_bao_cao.md) mục 5.4–5.8.

### C20. Video, kiểm tra cuối, nộp bài (Ngày 14)
Theo [05](05_demo_va_bao_cao.md) mục 5.8–5.9.

### C21. Luồng học máy: code OTI warm-start (Ngày 9)
Theo [07](07_hoc_may_oti_warm_start.md) mục 7.1–7.2: khởi tạo token từ phi (`--init`, `--init-phi-exp-name`, `--init-phi-checkpoint`), lưu token theo từng bước (`--snapshot-steps`) trong `oti_inversion.py`; chọn file token khi đánh giá (`--token-file`) trong module chung. Hòa chạy E7.0–E7.1 ngay Ngày 11 (H15), Doanh chạy E7.2 Ngày 12 (D15). Nếu trễ thì dời sang sau mốc M4; luồng xử lý ảnh luôn được ưu tiên trước.

### C22. Bộ câu GPT: công cụ và cờ mới (Ngày 9)
Theo [06](06_sinh_lai_cum_tu_gpt.md): `tools/phrases_stats.py`, `tools/clean_phrases.py` (tạo bộ P1 đã làm sạch); cờ `--phrase-match`, `--concept-filter` cho OTI và train phi; sửa lỗi `saved_exp_name` trong `gpt_phrases_generation.py` (khi chạy tiếp, code đọc `args.saved_exp_name` nhưng không khai báo tham số này). Cần cho Hòa H18 và Doanh D16–D17.

### C23. Bộ câu GPT: notebook sinh câu (Ngày 11, sau mốc M4)
Theo [06](06_sinh_lai_cum_tu_gpt.md): `colab/gen_phrases.ipynb` chạy trên runtime riêng, có chế độ quy mô nhỏ (chỉ khái niệm dùng cho ảnh shard s0, 64 câu mỗi khái niệm). Chạy thử ngắn trước khi giao cho Doanh (D16).

## Quyết định Chí phụ trách

| Khi nào | Quyết định | Dựa trên |
|---|---|---|
| Ngày 7 (M3) | Giữ hay đổi đóng góp chính | E2.0, E2.2 (Hòa) |
| Ngày 8 | Biến thể phi bền để train bản đầy đủ | E3.8a (Doanh) |
| Ngày 11 | Cấu hình cuối để nộp test | Bảng của 02, 03, 04 |
| Hằng ngày | Trả lời lỗi của Hòa, Doanh trong ngày | Báo cáo cuối ngày |

## Mẫu báo cáo cuối ngày

```text
[Chí – Ngày N]
Đã xong: C… (commit …)
Hòa/Doanh chạy được: H…, D…
Đang vướng: …
Ngày mai: C…
```
