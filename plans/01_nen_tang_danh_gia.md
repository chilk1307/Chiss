# 01. Nền tảng đánh giá, baseline và phân tích lỗi

**Tóm tắt:** gom code về **một quy trình duy nhất** dùng chung cho tập val và file nộp test; kiểm tra dữ liệu; mỗi lần chạy tự ghi kết quả ra file; có khoảng tin cậy và điểm test chính thức. Sau đó thêm các baseline (CLIP và xử lý ảnh cổ điển) và phân tích lỗi để biết SEARLE mạnh, yếu ở đâu.

**Thời gian:** Ngày 1–6; ví dụ định tính và bản đồ che khuất ở Ngày 11.
**Người làm:** Chí viết code; Hòa chạy trên Colab. Việc theo ngày: [nhiem_vu_Chi.md](nhiem_vu_Chi.md), [nhiem_vu_Hoa.md](nhiem_vu_Hoa.md).

## 1.1. Gộp code, sửa lỗi nuốt ngoại lệ (Chí: C3)

- Hiện có hai bản `datasets.py`: bản trong repo và bản vá dùng trên Colab (`outputs/scripts/datasets_patched.py` + `patch_fiq.py`). Gộp vào `src/datasets.py`, giữ phần lọc ảnh thiếu và thử mở ảnh nhiều lần.
- Thay `except Exception: pass` ở [datasets.py:133](../SEARLE-main/src/datasets.py#L133) và [datasets.py:256](../SEARLE-main/src/datasets.py#L256) bằng cảnh báo (tên file + lỗi) và đếm số mẫu lỗi; in tổng cuối mỗi lần chạy. Lỗi `ValueError: 'B002NCIATQ' is not in list` trong log FashionIQ cũ nhiều khả năng do mẫu bị bỏ âm thầm.
- `collate_fn` trả `None` khi cả batch rỗng: các vòng lặp phải bỏ qua batch `None`.
- Đưa vào git các file nhỏ đang bị `.gitignore` bỏ qua: `outputs/REPORT*.md`, `outputs/pip_freeze.txt`, `outputs/merge_oti.py`, các file `.csv`/`.json` trong `outputs/phi_models/phi_B32/`, log `outputs/logs/eval*.log`, thư mục `outputs/submissions/`.

## 1.2. Kiểm tra dữ liệu (Chí: C7 viết công cụ; Hòa: H2 kiểm tra nhanh, H7 chạy công cụ)

| Tập | Kiểm tra |
|---|---|
| CIRR | Ảnh khai báo trong `split.rc2.{val,test1}.json` có file trên đĩa |
| CIRCO | Số file trong `unlabeled2017/` so với `image_info_unlabeled2017.json`. Gallery val và test dùng chung bộ ảnh này: nếu thiếu thì cả file nộp test cũng bị ảnh hưởng |
| FashionIQ | Mỗi loại: số bộ ba có đủ ảnh tham chiếu và ảnh đích; số ảnh gallery có file |

- Kết quả ghi vào `results/E1.2-dataset-check/`.
- FashionIQ thiếu khoảng 2.431 ảnh do link chết. Nếu không tìm được bản đầy đủ: cố định danh sách truy vấn hợp lệ vào `results/fiq_val_subset.json`, mọi thí nghiệm dùng đúng danh sách này, mọi bảng ghi "FIQ (tập con)".

## 1.3. Module truy xuất chung và giao kèo lệnh chạy (Chí: C4 và các việc sau)

Logic tạo truy vấn (`a photo of $ that ...`) đang viết lặp ở `validate.py` (dòng 53, 59, 157, 281) và `generate_test_submission.py` (dòng 133, 199). Nếu cải tiến chỉ cài vào một chỗ, file nộp test vẫn chạy logic cũ.

- Tạo `src/retrieval_core.py`; `validate.py` và `generate_test_submission.py` chỉ gọi module này.
- Với cấu hình mặc định, kết quả phải **trùng** bản cũ (Hòa kiểm tra ở H5).
- Với CIRR, `generate_test_submission.py` luôn sinh cả file chính (`metric = recall`) và file subset (`metric = recall_subset`).

### Giao kèo lệnh chạy

Chí cài đúng các tên dưới đây; Hòa và Doanh dùng đúng các tên này trong lệnh. Muốn đổi tên thì sửa bảng trước và báo nhóm.

**`src/validate.py` và `src/generate_test_submission.py`**

| Tham số | Ý nghĩa | Giá trị | Mặc định | Việc |
|---|---|---|---|---|
| `--eval-type` | Phương pháp | có sẵn: `searle`, `searle-xl`, `phi`, `oti`; mới: `image`, `text`, `sum`, `cbir` | bắt buộc | C5, C8 |
| `--output-dir` | Thư mục kết quả của lần chạy | đường dẫn | không ghi | C4 |
| `--cache-dir` | Cache đặc trưng gallery, ảnh tham chiếu, thuộc tính | đường dẫn | không cache | C4 |
| `--note` | Ghi chú cho dòng nhật ký tự động | chuỗi | rỗng | C4 |
| `--clip-model-name` | Backbone CLIP của các baseline `image`, `text`, `sum`, `cbir` (phương pháp khác đọc từ experiment) | tên CLIP | `ViT-B/32` | C5 |
| `--num-workers` | Số tiến trình đọc ảnh | số nguyên | 10 | C4 |
| `--exclude-reference` | Loại ảnh tham chiếu khỏi kết quả (CIRR luôn loại theo giao thức) | cờ | tắt | C5 |
| `--targetpad-ratio` | Tỉ lệ đệm ảnh của targetpad | số thực | 1.25 | C6 |
| `--interp` | Phép nội suy khi đổi kích thước | `nearest`, `bilinear`, `bicubic`, `lanczos` | `bicubic` | C6 |
| `--tta` | Lật ảnh khi truy vấn | `none`, `flip-ref`, `flip-gallery`, `flip-both` | `none` | C6 |
| `--cbir-features` | Đặc trưng cổ điển | `color`, `lbp`, `hog`, `gabor`, `all` | `all` | C8 |
| `--cbir-text-weight` | Trọng số điểm văn bản CLIP cộng vào điểm CBIR | số thực | 0 | C8 |
| `--degrade` | Làm xấu ảnh tham chiếu | `none`, `<loại>:<mức>`, `mixed` (xem [03](03_do_ben_va_phuc_hoi_anh.md)) | `none` | C9 |
| `--restore` | Phục hồi ảnh tham chiếu | `none`, tên phương pháp, `oracle`, `auto`, `auto_iqa` | `none` | C9, C14 |
| `--freq-filter` | Lọc tần số ảnh tham chiếu | `none`, `lowpass:<r>`, `highpass:<r>` (r = tần số cắt, tỉ lệ so với Nyquist) | `none` | C9 |
| `--save-quality` | Ghi PSNR, SSIM của ảnh tham chiếu sau xử lý so với ảnh gốc | cờ | tắt | C9 |
| `--fusion-weight` | α trong `điểm = cos(t, g) + α · cos(v, g)` | số thực | 0 | C15 |
| `--prompt-set` | Bộ mẫu câu | `default`, `ensemble` | `default` | C15 |
| `--attr-rerank` | Bật xếp hạng lại theo thuộc tính | cờ | tắt | C13 |
| `--attr-weight` | Trọng số w | số thực | 0.1 | C13 |
| `--attr-topk` | Số ứng viên K được xếp hạng lại | số nguyên | 100 | C13 |
| `--attr-mask` | Cách tách đối tượng | `none`, `otsu`, `grabcut`, `saliency`, `auto` | `auto` | C13 |
| `--attr-families` | Các họ thuộc tính dùng | `color,brightness,colorfulness,pattern,zoom` | tất cả | C13 |
| `--attr-negation` | Xử lý phủ định | `on`, `off` | `on` | C13 |
| `--submission-name` | Chỉ cho `generate_test_submission.py`: tên file nộp | chuỗi | bắt buộc | có sẵn |

**`src/train_phi.py`** (tham số mới, việc C10)

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--train-on-oti-names` | Chỉ train trên ảnh có token trong `--oti-exp-name` (để train trên 1–2 shard) | tắt |
| `--resume` | Chạy tiếp từ checkpoint cuối (lưu cả optimizer) khi Colab bị ngắt | tắt |
| `--aug-feature-dir` | Thư mục đặc trưng CLIP của ảnh đã làm xấu (từ `tools/precompute_degraded_features.py`) | không |
| `--aug-prob` | Xác suất thay đặc trưng ảnh sạch bằng đặc trưng ảnh xấu | 0 |
| `--feature-noise` | σ của nhiễu Gaussian (tương đối theo độ lớn đặc trưng) cộng vào đầu vào phi | 0 |
| `--lr-scheduler`, `--warmup-epochs` | Lịch learning rate (`none` hoặc `cosine`) và số epoch khởi động, cho [08](08_hoc_may_cai_tien_phi.md) | `none`, 0 |
| `--oti-token-type` | Token đích khi chưng cất: `ema` hoặc `raw`, cho [08](08_hoc_may_cai_tien_phi.md) | `ema` |
| `--features-cache-dir` | Cache đặc trưng CLIP của ảnh train, dùng chung cho mọi lượt train cùng backbone và tập ảnh (bỏ qua bước trích đặc trưng nên chuỗi ngẫu nhiên khác lượt không cache) | không |
| `--phrase-match`, `--concept-filter` | Cách thay khái niệm bằng `$` và cách lọc câu GPT ([06](06_sinh_lai_cum_tu_gpt.md) mục 6.2) | `legacy`, `legacy` |

**Luồng học máy:** tham số của OTI warm-start (`--init`, `--snapshot-steps`, `--token-file`...) nằm ở [07](07_hoc_may_oti_warm_start.md) mục 7.1–7.2.

**Công cụ trong `tools/`**

| Công cụ | Việc làm | Việc |
|---|---|---|
| `check_datasets.py` | Đếm ảnh, truy vấn thiếu của CIRR, CIRCO, FIQ | C7 |
| `bootstrap_ci.py` | Khoảng tin cậy 95%; so sánh cặp với một lần chạy gốc (`--baseline`) | C7 |
| `collect_results.py` | Gom mọi `metrics.json` thành `summary.md` | C7 |
| `merge_oti.py` | Gộp các shard token OTI (`--shards`, `--out`) | C10 |
| `precompute_degraded_features.py` | Đặc trưng CLIP của ảnh ImageNet đã làm xấu | C10 |
| `query_taxonomy.py` | Chỉ số theo nhóm kiểu sửa đổi | C11 |
| `keep_edit_analysis.py` | Phân tích "giữ – sửa" | C11 |
| `test_meters.py` | Kiểm thử bộ đo thuộc tính trên ảnh tổng hợp | C12 |
| `attribute_stats.py` | Thống kê thuộc tính (E2.0); với `--agreement`: độ đồng thuận bộ đo – câu mô tả (E2.2) | C12 |
| `visualize_queries.py` | Lưới ví dụ; `--explain-attributes` cho E2.8 | C16 |
| `occlusion_map.py` | Bản đồ che khuất | C16 |
| `visualize_degradations.py` | Hình ảnh gốc / ảnh xấu / ảnh phục hồi kèm kết quả | C16 |
| `quality_vs_retrieval.py` | Tương quan PSNR/SSIM với điểm truy xuất | C16 |
| `vector_analysis.py` | Vector ảnh và vector từ: `tokens` (E1.15a), `gap` (E1.15b), `drift` (E3.9) | C24 |

**Bộ chạy Colab trong `colab/`** (việc C2): `setup_env.sh`, `prepare_data.sh <tập...>`, `requirements-xla.txt`, `SEARLE_runner.ipynb`.

### Đầu ra của mỗi lần chạy có `--output-dir results/<mã>`

- `metrics.json`: toàn bộ chỉ số, tham số dòng lệnh, commit git, thời gian chạy, loại GPU;
- `dump.jsonl`: mỗi dòng một truy vấn (ảnh tham chiếu, câu mô tả, top-100 kết quả kèm điểm, ground truth; CIRR thêm thứ tự trong group);
- `log.txt`: toàn bộ log;
- một dòng mới trong `results/runs_<RUNNER>.csv` (tên người chạy lấy từ biến môi trường `RUNNER`; mỗi người một file để không ghi đè nhau).

## 1.4. Khoảng tin cậy bootstrap (Chí: C7)

- `tools/bootstrap_ci.py` lấy mẫu lại các truy vấn 1.000 lần từ `dump.jsonl`, báo khoảng tin cậy 95% của mỗi chỉ số; với `--baseline`, báo khoảng tin cậy của hiệu số so với lần chạy gốc.
- Chỉ gọi là "cải thiện" khi khoảng tin cậy 95% của hiệu số không chứa 0.
- Bootstrap chỉ đo nhiễu do chọn truy vấn, không đo nhiễu do seed khi train.

## 1.5. Điểm test chính thức (Hòa: H4, H13)

Đã có đủ file nộp của phi_B32 trong `outputs/submissions/`: CIRR test1 gồm file chính (`metric = recall`) và file subset (`metric = recall_subset`), 4.148 truy vấn; CIRCO test 800 truy vấn.

- Sinh file nộp cho SEARLE gốc để có baseline trên test.
- Nộp lên server CIRR (https://cirr.cecs.anu.edu.au/) và CIRCO (https://circo.micc.unifi.it/); gửi điểm cho Chí ghi vào [nhật ký](nhat_ky_thi_nghiem.md).
- Từ đây, test chỉ dùng cho cấu hình cuối (H13).

## 1.6. Đối chiếu số liệu của paper (Chí: C7)

Hằng số `PAPER_CIRR_VAL`, `PAPER_FIQ_VAL` ([app.py:50-59](../SEARLE-main/demo/app.py#L50-L59)) là cột "Paper" trong demo và trong [00](00_hien_trang_va_baseline.md). Không trộn số val và test trong cùng một bảng.

**Đã đối chiếu (17/10, C7)** với arXiv 2303.15247v2 (bản HTML, không có số trang), SEARLE ViT-B/32. Mọi số trong demo khớp, không phải sửa:

| Tập | Bảng | SEARLE |
|---|---|---|
| CIRR val | Table 9 (cũng là Table 4, 7, 12 cho R@1–R@50) | R@1 25.09, R@5 55.18, R@10 68.79, R@50 90.82, Rs@1/2/3 54.84 / 76.63 / 87.95 |
| FashionIQ val | Table 1 | Shirt 24.44 / 41.61, Dress 18.54 / 39.51, Toptee 25.70 / 46.46, TB 22.89 / 42.53 (R@10 / R@50) |
| CIRCO val | Table 10 | mAP@5/10/25/50 = 6.82 / 7.83 / 9.15 / 9.77 |
| CIRR test | Table 2 | R@1 24.00, R@5 53.42, R@10 66.82, R@50 89.78, Rs@1/2/3 54.89 / 76.60 / 88.19 |
| CIRCO test | Table 3 | mAP@5/10/25/50 = 9.35 / 9.94 / 11.13 / 11.84 |

SEARLE gốc của nhóm trên CIRCO val (B4: mAP@5 6.82, mAP@10 7.78) gần như trùng Table 10, nên gallery CIRCO nhiều khả năng đã đủ ảnh; H2/H7 vẫn kiểm tra lại bằng công cụ. Chưa đối chiếu iSEARLE (arXiv 2405.02951): không cần cho bảng chính.

## 1.7. Chạy lại baseline bằng quy trình mới (Hòa: H5)

| Mã | Mô hình | Tập |
|---|---|---|
| E1.7a-searle-cirr | SEARLE gốc | CIRR val |
| E1.7b-searle-circo | SEARLE gốc | CIRCO val |
| E1.7c-searle-fiq | SEARLE gốc | FIQ val |
| E1.7d-phiB32-cirr | phi_B32 | CIRR val |
| E1.7e-phiB32-circo | phi_B32 | CIRCO val |
| E1.7f-phiB32-fiq | phi_B32 | FIQ val |

CIRR phải khớp bảng ở [00](00_hien_trang_va_baseline.md) (sai lệch ≤ 0.05). Lần chạy này cũng tạo cache đặc trưng gallery mặc định cho cả nhóm.

## 1.8. Baseline dùng CLIP (Chí: C5; Hòa: H6)

| Baseline | Truy vấn | Cho biết |
|---|---|---|
| `image` | đặc trưng ảnh tham chiếu `v` | chỉ "chép" ảnh tham chiếu được bao nhiêu điểm |
| `text` | đặc trưng câu mô tả (không có `$`) | chỉ dùng câu chữ được bao nhiêu điểm |
| `sum` | `normalize(normalize(v) + normalize(t))` | cách ghép đơn giản nhất |

Với `image`, ảnh tham chiếu luôn đứng đầu nếu nằm trong gallery. CIRR đã loại ảnh tham chiếu theo giao thức; với CIRCO và FIQ, chạy bản `--exclude-reference` cho **mọi** phương pháp khi so sánh với các baseline này.

## 1.9. Baseline xử lý ảnh cổ điển: truy xuất ảnh theo nội dung (CBIR) (Chí: C8; Hòa: H6)

Mục đích: cho thấy CLIP hơn các đặc trưng thủ công bao nhiêu, và đưa kiến thức CBIR của môn vào đề tài.

| Đặc trưng | Cách tính | Kiến thức xử lý ảnh |
|---|---|---|
| Histogram màu | HSV, 8 × 4 × 4 bin, chuẩn hoá; so bằng khoảng cách chi-square | Không gian màu, histogram |
| LBP | LBP đồng nhất (P = 8, R = 1) trên ảnh xám, histogram trên lưới 4 × 4 ô | Kết cấu |
| HOG | ảnh xám 128 × 128, ô 16 × 16, 9 hướng | Gradient, đặc trưng hình dạng |
| Gabor | 4 hướng × 3 tần số; trung bình và độ lệch chuẩn năng lượng đáp ứng | Lọc tuyến tính, kết cấu |
| Kết hợp | trung bình điểm tương đồng của các đặc trưng (với cosine, tương đương nối các vector đã chuẩn hoá); histogram dùng `1 − chi²/2` | — |

- `--eval-type cbir` chỉ dùng ảnh tham chiếu (bỏ qua câu mô tả). `--cbir-text-weight` > 0 cộng thêm điểm văn bản CLIP, tạo một baseline "cổ điển + hiện đại" đơn giản.
- CBIR trên CIRCO cần tính đặc trưng cho khoảng 123 nghìn ảnh bằng CPU (1–2 giờ): chỉ chạy nếu còn thời gian.

## 1.10. Phân loại truy vấn theo kiểu sửa đổi (Chí: C11; Hòa: H8)

Bộ luật từ khoá; một truy vấn có thể thuộc nhiều nhóm.

| Nhóm | Từ khoá ví dụ |
|---|---|
| Màu sắc | red, blue, color, colour |
| Độ sáng, độ rực màu | darker, lighter, brighter, colorful, black and white |
| Hoa văn, chất liệu | striped, plaid, floral, pattern, print, solid, plain |
| Số lượng | one, two, three, more, fewer, single, pair, chữ số |
| Vị trí, không gian | left, right, top, bottom, behind, in front, next to |
| Phủ định, bỏ đi | remove, without, no, not |
| Thêm đối tượng | add, with, holding, wearing |
| Nền, bối cảnh | background, outdoors, indoors, beach, grass, snow, street |
| Góc nhìn, phóng to | closer, zoom, angle, view, side, front, back |
| Phong cách | drawing, cartoon, painting, sketch |

- Bảng chỉ số theo nhóm cho mọi phương pháp ở 1.7–1.9.
- Chí chấm tay 100 truy vấn ngẫu nhiên để đo độ chính xác của bộ luật.
- Dùng cho E2.0 và cho phần "Hạn chế" trong báo cáo.

## 1.11. Phân tích "giữ – sửa" (Chí: C11; Hòa: H8)

Với mỗi truy vấn, lấy kết quả top-1 và tính hai độ tương đồng: với ảnh tham chiếu, và với câu mô tả. Vẽ biểu đồ phân tán, phân loại lỗi:
- **chép ảnh tham chiếu:** top-1 rất giống ảnh tham chiếu nhưng sai;
- **bỏ qua ảnh tham chiếu:** top-1 khớp câu chữ nhưng khác hẳn ảnh tham chiếu.

So sánh tỉ lệ hai loại lỗi giữa các baseline và SEARLE.

## 1.12. Ví dụ định tính (Chí: C16; Hòa: H12)

Lưới ảnh: ảnh tham chiếu, câu mô tả, top-5 kết quả, viền xanh cho ảnh đúng. Chọn ngẫu nhiên (ghi seed) 10–20 ví dụ thành công và thất bại mỗi tập, rải đều các nhóm ở 1.10; kèm các từ CLIP gần pseudo-token nhất.

## 1.13. Bản đồ che khuất (Chí: C16; Hòa: H12)

Che lần lượt từng ô của ảnh tham chiếu (ô 32 × 32 px, bước 16 px) bằng màu xám trung tính, tính lại điểm của ảnh đích, vẽ bản đồ mức giảm điểm chồng lên ảnh. Cho biết pseudo-token "nhìn" vào vùng nào. Mỗi ví dụ khoảng 170 lần mã hoá ảnh; chỉ làm cho 10 ví dụ.

## 1.14. Ảnh hưởng của lượng dữ liệu OTI (tuỳ chọn; Chí: C10; Hòa: T1, T2)

Đã có 2 điểm: 6.824 ảnh cho R@1 22.60; 100.000 ảnh cho 25.47. Thêm 2 điểm từ các shard sẵn có:

| Mã | Dữ liệu | Số ảnh | Thời gian train (100 epoch, A100) |
|---|---|---|---|
| E1.14a | `oti_B32_s0` | 33.334 | 2–3 giờ |
| E1.14b | `oti_B32_s0` + `oti_B32_s1` | 66.668 | 4–5 giờ |

Vẽ R@1, R@10 theo số ảnh (thang log). Chưa rõ 6.824 ảnh của lần chạy cũ có trùng shard s0 không; ghi chú khi vẽ.

Đây cũng là E8.1a–b của [08](08_hoc_may_cai_tien_phi.md): chỉ chạy một lần, kết quả dẫn chiếu ở cả hai nơi.

## 1.15. Vector ảnh và vector từ (Chí: C24; Hòa: H8, H15)

SEARLE là một chuỗi vector: ảnh → vector ảnh CLIP → phi → **pseudo-token** (một "từ giả" đặt giữa các vector từ của CLIP) → câu `a photo of $ that ...` → vector truy vấn. Các bảng chỉ số chỉ nhìn đầu cuối của chuỗi; mục này nhìn vào khúc giữa. Công cụ: `tools/vector_analysis.py`.

- **E1.15a, pseudo-token có giống một từ thật không** (`tokens`): so độ lớn của token (phi_B32, SEARLE gốc, token OTI) với độ lớn của 49.406 vector từ CLIP; độ tương đồng với từ thật gần nhất; các token của những ảnh khác nhau có giống nhau quá không (dấu hiệu phi sinh token chung chung); phi và OTI có cho token giống nhau trên cùng một ảnh không. Với `--snap-eval`: đánh giá lại khi thay mỗi token bằng **từ thật gần nhất**, hoặc giữ hướng nhưng **đưa độ lớn về mức của từ thật**. Nếu điểm gần như không đổi thì token hoạt động như một từ; nếu sụt mạnh thì thông tin nằm ở những hướng (hoặc độ lớn) không có từ nào diễn đạt được.
- **E1.15b, khoảng cách giữa phía ảnh và phía chữ** (`gap`): khoảng cách giữa tâm các vector ảnh gallery và tâm của từng kiểu vector truy vấn (ảnh tham chiếu, câu mô tả, `sum`, truy vấn có pseudo-token); thang điểm của từng kiểu (trung bình, độ lệch chuẩn của điểm với gallery, z-score của ảnh đích); hình PCA. Cột "std ảnh / std" giúp đọc trọng số `--fusion-weight` ở E4.4: điểm ảnh–ảnh thường dao động rộng hơn điểm truy vấn–ảnh, nên một α nhỏ có thể đã có tác dụng lớn; E1.15b đo cụ thể tỉ lệ này.

Kết quả dùng cho phần "Phân tích" của báo cáo; đây là phần giải thích, không phải đóng góp chính. Mục cùng loại cho ảnh xấu là E3.9 ([03](03_do_ben_va_phuc_hoi_anh.md)).

## Danh sách thí nghiệm

| Mã | Nội dung | Người chạy | Ngày |
|---|---|---|---|
| E1.2 | Kiểm tra dữ liệu | Hòa | 1, 5 |
| E1.4 | Khoảng tin cậy của các bảng baseline | Hòa | 5 |
| E1.7a–f | Chạy lại baseline bằng quy trình mới | Hòa | 3 |
| E1.8 | Baseline CLIP: `image`, `text`, `sum` | Hòa | 4 |
| E1.9 | Baseline CBIR cổ điển | Hòa | 4 |
| E1.10 | Chỉ số theo nhóm kiểu sửa đổi | Hòa | 6 |
| E1.11 | Phân tích giữ – sửa | Hòa | 6 |
| E1.12 | Ví dụ định tính | Hòa | 11 |
| E1.13 | Bản đồ che khuất | Hòa | 11 |
| E1.14a–b | Lượng dữ liệu OTI (tuỳ chọn) | Hòa | 6–9 |
| E1.15a | Pseudo-token so với từ thật (phi_B32, SEARLE gốc; thêm token OTI sau H15) | Hòa | 6, 11 |
| E1.15b | Khoảng cách phía ảnh – phía chữ, thang điểm từng kiểu truy vấn | Hòa | 6 |

## Tiêu chí hoàn thành

- [ ] Một phiên bản code duy nhất; chạy lại ra đúng số cũ trên CIRR val.
- [ ] Báo cáo kiểm tra dữ liệu cho cả 3 tập.
- [ ] Bảng baseline (SEARLE gốc, phi_B32, `image`, `text`, `sum`, CBIR) cho 3 tập, kèm khoảng tin cậy.
- [ ] Điểm test chính thức của SEARLE gốc và phi_B32.
- [ ] Bảng theo nhóm sửa đổi, biểu đồ giữ – sửa, ít nhất 10 ví dụ định tính và 10 bản đồ che khuất.

## Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Không tìm được ảnh FashionIQ đầy đủ | Dùng tập con cố định, ghi rõ trong báo cáo |
| Server test giới hạn số lần nộp | Chỉ nộp cấu hình cuối; không dùng test để chọn tham số |
| Bộ luật phân loại truy vấn sai | Chấm tay để đo sai số; báo cáo kèm sai số |
| Ví dụ định tính bị chọn lọc có lợi | Chọn ngẫu nhiên trong từng nhóm, ghi seed |

## Lịch sử thay đổi

- 2026-10-16: tạo kế hoạch (gộp nội dung "Sửa nền đánh giá" và "Baseline, phân tích lỗi" của bản cũ).
- 2026-10-17: C2–C8 xong; thêm `--clip-model-name`, `--num-workers` vào giao kèo; ghi kết quả đối chiếu số paper (1.6); ghi cách kết hợp đặc trưng CBIR (1.9).
- 2026-10-18: C9–C15 xong; thêm `--features-cache-dir`, `--phrase-match`, `--concept-filter` vào bảng của `train_phi.py`.
- 2026-10-19: thêm mục 1.15 (vector ảnh và vector từ, E1.15a–b) và công cụ `vector_analysis.py` (C24).
