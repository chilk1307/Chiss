# Thay đổi so với SEARLE gốc

Code gốc: [miccunifi/SEARLE](https://github.com/miccunifi/SEARLE), commit `a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37`.
Mỗi mục ghi: file, việc trong kế hoạch (`plans/`), nội dung thay đổi. Với cấu hình mặc định, mọi thay đổi giữ nguyên kết quả cũ.

## 17/10/2026: Ngày 1–3 của kế hoạch (C2–C8)

### Kiểm chứng không hồi quy

Chạy code cũ (commit `70328e1`) và code mới trên một bộ dữ liệu giả lập có đúng cấu trúc thư mục của CIRR, CIRCO, FashionIQ (CPU, phi_b32):

| Kiểm tra | Kết quả |
|---|---|
| `validate.py` CIRR val, CIRCO val, FashionIQ val (có ảnh thiếu) | Mọi chỉ số trùng khớp |
| `generate_test_submission.py` CIRR (2 file) và CIRCO | File JSON trùng khớp |
| Chạy lại với `--cache-dir` | Cùng số, dùng lại cache |

Trên dữ liệu thật, Hòa kiểm tra lại ở H5 (CIRR R@1 phải là 25.66 với SEARLE gốc, 25.47 với phi_B32).

### Sửa

| File | Việc | Thay đổi |
|---|---|---|
| `src/datasets.py` | C3 | Gộp với bản vá Colab (`outputs/scripts/datasets_patched.py`, `patch_fiq.py`): FashionIQ lọc bộ ba và ảnh gallery thiếu file, in số lượng giữ lại; mở ảnh thử lại khi lỗi đọc tạm thời (Drive), không thử lại khi thiếu file. `except: pass` được thay bằng cảnh báo có tên mẫu và lỗi, đếm theo từng tiến trình; CIRCO cũng được bắt lỗi thay vì dừng giữa chừng. Thêm `image_path()` cho từng dataset và lớp `ImageListDataset`. |
| `src/data_utils.py` | C6 | `targetpad_transform` nhận thêm phép nội suy; thêm `clip_transform`, `build_preprocess`, `preprocess_key`. Mặc định vẫn là targetpad 1.25, bicubic. |
| `src/validate.py` | C4 | Chỉ còn phần dòng lệnh; toàn bộ logic chuyển sang `retrieval_core.py`. Giữ `cirr_compute_val_metrics` (dùng trong `train_phi.py`). In chỉ số đúng định dạng cũ. |
| `src/generate_test_submission.py` | C4 | Chỉ còn phần dòng lệnh, dùng `retrieval_core.py`. CIRR luôn sinh cả file `recall` và `recall_subset`. Với `--output-dir`, chép thêm file nộp vào thư mục kết quả. |
| `src/utils.py`, `src/oti_inversion.py`, `src/image_concepts_association.py` | C3 | Bỏ qua batch `None` (cả batch không đọc được ảnh) thay vì lỗi; `extract_image_features` in số ảnh bị bỏ qua. |
| `demo/app.py` | C7 | Ghi nguồn số "Paper" sau khi đối chiếu với arXiv 2303.15247v2 (Table 9 CIRR val, Table 1 FashionIQ val; số không đổi). |

### Thêm

| File | Việc | Nội dung |
|---|---|---|
| `src/retrieval_core.py` | C4, C5, C6, C8 | Quy trình truy xuất chung cho val và test: nạp mô hình, đặc trưng ảnh (có cache), tạo truy vấn, xếp hạng, chỉ số, file nộp. Phương pháp mới `image`, `text`, `sum`, `cbir`; cờ `--exclude-reference`, `--targetpad-ratio`, `--interp`, `--tta`, `--cbir-features`, `--cbir-text-weight`, `--clip-model-name`, `--num-workers`. Truy vấn có ảnh tham chiếu hoặc ảnh đích không đọc được bị bỏ kèm cảnh báo và được đếm. |
| `src/run_recorder.py` | C4 | `--output-dir`: ghi `metrics.json` (chỉ số, tham số, commit, GPU, thời gian), `dump.jsonl` (mỗi truy vấn một dòng: top-100 kèm điểm, ground truth, chỉ số từng truy vấn), `log.txt`, và thêm một dòng vào `runs_<RUNNER>.csv` ở thư mục cha. `--note` ghi chú. |
| `src/cbir.py` | C8 | Đặc trưng cổ điển: histogram HSV 8×4×4, LBP đồng nhất (P=8, R=1) trên lưới 4×4, HOG (ô 16 px, 9 hướng), năng lượng Gabor (4 hướng × 3 bước sóng). Histogram so bằng chi-square, vector so bằng cosine, nhiều họ thì lấy trung bình. |
| `tools/check_datasets.py` | C7 | Đếm ảnh, truy vấn thiếu file của CIRR, CIRCO, FashionIQ; ghi `report.md`; FashionIQ ghi thêm `fiq_val_subset.json`. |
| `tools/bootstrap_ci.py` | C7 | Khoảng tin cậy 95% từ `dump.jsonl` (1.000 lần lấy mẫu lại, FashionIQ lấy mẫu trong từng loại); `--baseline` cho hiệu số cặp và đánh dấu khác biệt có ý nghĩa. |
| `tools/collect_results.py` | C7 | Gom mọi `metrics.json` thành `summary.md`, một bảng mỗi tập. |
| `colab/setup_env.sh` | C2 | Dựng venv Python 3.8 bằng `uv` từ `outputs/pip_freeze.txt`, cài `requirements-xla.txt`, ghi `/content/env.sh` (`DRIVE`, `DATA`, `PY`, `RUNNER`, `SRC`). |
| `colab/prepare_data.sh` | C1, C2 | Giải nén dataset từ Drive (`CIRR.tar`/`CIRCO.tar`/`FashionIQ.tar`/ImageNet) ra `/content/data`; nếu chưa có `.tar` thì liên kết thư mục Drive kèm cảnh báo; liên kết `data/{GPT_phrases,oti_pseudo_tokens,similar_concepts,phi_models}` tới `$DRIVE/outputs`. |
| `colab/requirements-xla.txt` | C2 | opencv-python-headless 4.10.0.84, scikit-image 0.21.0, scipy 1.10.1, PyWavelets 1.4.1, matplotlib 3.7.5 (ghim numpy 1.23.5, pillow 10.4.0). Đã kiểm tra giải được cùng `pip_freeze.txt` cho Python 3.8 trên Linux, chỉ dùng wheel có sẵn. |
| `colab/SEARLE_runner.ipynb` | C2 | Ô 0 cấu hình (`RUNNER`, `DATASETS`), ô 1 gắn Drive, ô 2 lấy code, ô 3 cài môi trường, ô 4 chuẩn bị dữ liệu, ô 5 kiểm tra GPU; ô kiểm thử (CIRR R@1 = 25.66), ô kiểm tra dữ liệu, mẫu ô lệnh, chạy nền, xem kết quả. |

### Ngoài `SEARLE-main/`

| File | Việc | Thay đổi |
|---|---|---|
| `.gitignore` | C3 | Đưa vào git các file nhỏ của lần tái lập: `outputs/REPORT*.md`, `outputs/pip_freeze.txt`, `outputs/merge_oti.py`, `outputs/logs/eval*.log`, `outputs/phi_models/phi_B32/*.csv, *.json`. |
| `.gitattributes` | C2 | Script `.sh` luôn dùng xuống dòng LF (chạy trên Colab). |

## 18/10/2026: Ngày 4–8 của kế hoạch (C9–C15)

### Kiểm chứng

| Kiểm tra | Kết quả |
|---|---|
| `validate.py` mặc định sau mọi thay đổi | Trùng số cũ trên CIRR, CIRCO, FashionIQ giả lập |
| `train_phi.py` mới so với code gốc (`70328e1`), 3 epoch, CPU | Trọng số phi **trùng từng bit**, loss và chỉ số validation trùng |
| `--resume`: 2 epoch, ngắt, chạy tiếp tới 4 | Trùng từng bit với chạy liền 4 epoch |
| Bộ suy giảm / phục hồi trên ảnh thật (`demo/sample_images`) | Phục hồi đúng loại tăng PSNR, SSIM (ví dụ muối tiêu mức 2: 14 → 34 dB với median) |
| E2.1: bộ đo trên ảnh tổng hợp (`tools/test_meters.py`) | Màu 100% (46 sắc độ); sọc 19/20 (đúng cả hướng); độ sáng, độ rực màu, diện tích đơn điệu đúng; mặt nạ Otsu / GrabCut IoU 1,00 |
| Parser và bộ phân loại, 100 câu thật chấm tay (`tools/grading/`) | Ràng buộc đúng 97%, nhóm sửa đổi đúng 78% |
| Bộ phân loại so với nhãn `semantic_aspects` của CIRCO val | Phủ định P = R = 1,00; góc nhìn P 0,90, R 0,98; số lượng P 0,71, R 0,97 |
| Ước lượng suy giảm (`tools/calibrate_restore_rules.py`, 17 ảnh mẫu) | Nhận đúng loại 86,9%; sai số σ nhiễu trung bình 1,1 |

### Sửa

| File | Việc | Thay đổi |
|---|---|---|
| `src/retrieval_core.py` | C9, C13, C15, C21 | Cờ `--degrade`, `--restore`, `--freq-filter`, `--save-quality` (chỉ ảnh tham chiếu; PSNR/SSIM, suy giảm và phép phục hồi đã dùng của từng ảnh ghi vào `dump.jsonl`, trung bình vào `metrics.json`); `--fusion-weight`, `--prompt-set`; các cờ `--attr-*`; `--token-file`, `--interp-phi`, `--interp-lambda` cho `--eval-type oti`. Cache phân biệt theo cách xử lý ảnh tham chiếu. |
| `src/datasets.py` | C9 | `ImageListDataset` nhận thêm bước xử lý ảnh trước tiền xử lý CLIP. |
| `src/train_phi.py` | C10, C22 | Tra token bằng `dict`; sửa lỗi `train_metrics.csv` chỉ giữ epoch cuối; `--train-on-oti-names`; `--resume` (lưu `checkpoints/resume.pt` mỗi `--save-frequency` epoch: trọng số, optimizer, grad scaler, lịch LR, trạng thái các bộ sinh số ngẫu nhiên); `--aug-feature-dir`, `--aug-prob`; `--feature-noise`; `--lr-scheduler`, `--warmup-epochs`; `--oti-token-type`; `--features-cache-dir`; `--phrase-match`, `--concept-filter`. `comet_ml` thành tuỳ chọn. |
| `src/validate.py` | C10 | `cirr_compute_val_metrics` rút một số ngẫu nhiên như DataLoader của code gốc, để train giữ đúng chuỗi ngẫu nhiên cũ. |
| `colab/requirements-xla.txt` | C14 | Thêm `piq==0.8.0` (BRISQUE). |

### Thêm

| File | Việc | Nội dung |
|---|---|---|
| `src/degradations.py` | C9 | 8 loại suy giảm × 2 mức, `mixed`; ngẫu nhiên theo tên ảnh. |
| `src/restorations.py`, `src/restore_oracle.json` | C9, C14 | 14 phép phục hồi (median, gaussian, bilateral, NLM, Wiener, Richardson–Lucy, unsharp, cân bằng histogram, CLAHE, gamma, Retinex, gray-world, white-patch, deblock); bảng `oracle` tạm thời (điền lại sau E3.2). |
| `src/freq_filters.py` | C9 | Lọc thông thấp / thông cao Gauss trong miền Fourier. |
| `src/reference_processing.py` | C9 | Chuỗi suy giảm → phục hồi → lọc tần số cho ảnh tham chiếu, PSNR/SSIM. |
| `src/degradation_estimation.py`, `src/restore_rules.json` | C14 | Ước lượng σ nhiễu (wavelet), nhiễu xung, độ mờ (Laplacian, năng lượng tần số cao), độ khối JPEG, độ sáng, lệch màu (CIELAB); `--restore auto` (luật, ngưỡng trong `restore_rules.json`) và `auto_iqa` (BRISQUE). Ngưỡng hiện tại hiệu chỉnh tạm trên ảnh mẫu, cần hiệu chỉnh lại trên ảnh CIRR train. |
| `src/prompts.py` | C15 | Bộ mẫu câu; `"a photo of $, {}"` bị bộ tách từ CLIP gộp `$,` thành token khác nên đổi thành `"a photo of $ , {}"`. |
| `src/attributes/parser.py`, `taxonomy.py` | C11 | Tách ràng buộc thuộc tính (màu, độ sáng, độ rực màu, hoa văn, phóng to; phủ định, "đổi A thành B"); 10 nhóm kiểu sửa đổi. |
| `src/attributes/meters.py` | C12 | Mặt nạ (Otsu + hình thái học, GrabCut, spectral residual), màu chủ đạo (k-means CIELAB, CIEDE2000), L*, colourfulness Hasler–Süsstrunk, hoa văn (Canny, tỉ số đỉnh/nền phổ Fourier, LBP, Gabor), diện tích đối tượng; mức thoả mãn từng ràng buộc. |
| `src/attributes/rerank.py` | C13 | Xếp hạng lại top-K; đo thuộc tính một lần mỗi ảnh, cache trong `--cache-dir`. |
| `src/gpt_phrases_utils.py` | C22 | Cách thay khái niệm bằng `$` và lọc khái niệm (`legacy` = hành vi gốc). |
| `tools/test_parser.py`, `tools/grading/` | C11 | 71 câu kiểm thử; phiếu chấm tay 100 câu thật. |
| `tools/query_taxonomy.py`, `tools/keep_edit_analysis.py` | C11 | Chỉ số theo nhóm sửa đổi (kèm khoảng tin cậy, so sánh cặp, phiếu chấm tay); phân tích lỗi giữ – sửa và biểu đồ. |
| `tools/test_meters.py`, `tools/attribute_stats.py` | C12 | E2.1; E2.0, E2.2 (`--agreement`), E3.6 (`--degrade`, `--restore`). |
| `tools/merge_oti.py`, `tools/precompute_degraded_features.py` | C10 | Gộp shard OTI bất kỳ; đặc trưng CLIP của ảnh ImageNet đã làm xấu (lưu theo phần, chạy tiếp được). |
| `tools/calibrate_restore_rules.py` | C14 | Chọn ngưỡng cho `--restore auto`, báo cáo độ chính xác nhận dạng và sai số σ. |

## 18/10/2026 (tiếp): các việc còn lại trước khi train trên Colab (C16, C21, C22, C23)

### Kiểm chứng

| Kiểm tra | Kết quả |
|---|---|
| `oti_inversion.py` mới, `--init random`, so với code gốc (`70328e1`) trên CIRCO giả lập | Token thường và token EMA **trùng từng bit** |
| `--init phi` + `--snapshot-steps 0,6,12` | Snapshot bước cuối trùng token cuối; bước 0: token thường = token EMA; có `timing.json`; snapshot không làm đổi token |
| `--gpt-exp-name none --lambda_gpt 0` | Chạy được, không cần file câu GPT; thiếu `--lambda_gpt 0` thì báo lỗi |
| `validate.py --eval-type oti --token-file ...`, `--interp-phi`, `--interp-lambda` | Chạy được với mọi snapshot |
| Đọc / ghi `concept_to_phrases.pkl` dạng luồng | Đúng với pickle protocol 2–5; file ghi ra nạp được bằng `pickle.load` của Python 3.8 |
| Công cụ hình ảnh (C16) trên các lượt chạy giả lập | Cả 4 công cụ chạy; ví dụ có xếp hạng lại hiển thị lý do ("blue: closest colour dE 5") |
| `generate_phrases_llm.py` với mô hình nhỏ (CPU) | Chế độ `continue` (tiny-gpt2) và `instruct` (SmolLM2-135M-Instruct) chạy hết đường; bộ lọc loại câu không đúng khuôn |

### Sửa

| File | Việc | Thay đổi |
|---|---|---|
| `src/oti_inversion.py` | C21, C22 | `--init {random,phi,searle}`, `--init-phi-exp-name`, `--init-phi-checkpoint`; `--snapshot-steps` (lưu `oti_pseudo_tokens_step<N>.pt`, `ema_oti_pseudo_tokens_step<N>.pt`, chạy tiếp được); `timing.json`; `--gpt-exp-name none` (chỉ cosine loss, không cần file câu); `--phrase-match`, `--concept-filter`; `--num-workers`. Trích đặc trưng ảnh được chuyển lên trước khi khởi tạo token (không đổi chuỗi ngẫu nhiên). |
| `src/gpt_phrases_utils.py` | C22, C23 | Đọc `concept_to_phrases.pkl` dạng luồng (không nạp cả file 700 MB), ghi dạng luồng (protocol 4); tìm khái niệm "dung sai" (dấu nháy, gạch nối, khoảng trắng); khái niệm không còn câu nào bị lọc ra. |
| `src/gpt_phrases_generation.py` | C22 | Sửa lỗi `args.saved_exp_name` chưa khai báo: thêm `--saved-exp-name` (mặc định bằng `--exp-name`). |
| `src/image_concepts_association.py` | — | DataLoader dùng `collate_fn` (bỏ qua ảnh không đọc được thay vì dừng). |

### Thêm

| File | Việc | Nội dung |
|---|---|---|
| `tools/phrases_stats.py` | C22 | E6.1: thống kê bộ câu (trùng lặp, khái niệm bị luật lọc cũ bỏ, khái niệm nằm lẫn trong từ khác, độ dài, distinct-2, khái niệm thực sự được dùng). |
| `tools/clean_phrases.py` | C22 | Tạo bộ P1 đã làm sạch (giữ câu có khái niệm như một từ trọn vẹn, chuẩn hoá cách viết khái niệm, bỏ trùng, ≤ 75 token CLIP). |
| `tools/generate_phrases_llm.py`, `tools/prompts/p3.txt`, `colab/gen_phrases.ipynb` | C23 | Sinh bộ câu P2 (mô hình nền, viết tiếp `a photo of {concept} that`) hoặc P3 (mô hình instruct, câu mô tả chi tiết nhìn thấy được); chế độ quy mô nhỏ theo shard; lưu tiến độ, chạy tiếp được; notebook chạy trên runtime Colab riêng. |
| `tools/run_utils.py` | C16 | Đọc thư mục kết quả, tìm file ảnh, dựng lại mô hình của một lượt chạy từ `metrics.json`. |
| `tools/visualize_queries.py` | C16 | Ví dụ định tính E1.12 (`--compare`), E2.8 (`--explain-attributes`), `--nearest-words`. |
| `tools/occlusion_map.py` | C16 | Bản đồ che khuất E1.13. |
| `tools/visualize_degradations.py` | C16 | Hình E3.7: ảnh gốc / xấu / phục hồi kèm PSNR, SSIM và top-K của từng phiên bản. |
| `tools/quality_vs_retrieval.py` | C16 | E3.4: tương quan Spearman giữa mức tăng PSNR/SSIM và mức tăng điểm truy xuất; phép phục hồi tốt nhất "cho mắt" và "cho truy xuất". |

## 19/10/2026: phân tích vector ảnh và vector từ (C24)

### Kiểm chứng

| Kiểm tra | Kết quả |
|---|---|
| `vector_analysis.py drift` so với `validate.py` cùng cấu hình (phi_B32, CIRR giả lập, `gauss_noise:2`) | Cùng R@1 (4.44): đổi cách xử lý ảnh trong một pipeline cho đúng số của một lượt chạy riêng |
| `tokens --snap-eval`, cột "Pseudo token" | Bằng R@1 của lượt chạy thường cùng phương pháp |
| Cả 3 lệnh trên CIRR, CIRCO, FIQ giả lập; phương pháp phi, OTI (đường dẫn tuyệt đối), `image` | Chạy hết; cấu hình trùng giữa `--configs` và `--runs` chỉ tính một lần; lượt của tập khác bị bỏ qua kèm thông báo |

### Sửa

| File | Việc | Thay đổi |
|---|---|---|
| `src/retrieval_core.py` | C24 | Thêm `RetrievalPipeline.set_reference_processor`: đổi cách xử lý ảnh tham chiếu mà giữ nguyên mô hình đã nạp (đặc trưng và token tham chiếu được tính lại hoặc đọc từ cache). Không dùng với `--eval-type oti` hay `--attr-rerank`. Không đổi kết quả của `validate.py`, `generate_test_submission.py`. |

### Thêm

| File | Việc | Nội dung |
|---|---|---|
| `tools/vector_analysis.py` | C24 | `tokens` (E1.15a): độ lớn, từ thật gần nhất, độ giống nhau giữa các token, độ khớp giữa hai phương pháp (phi, SEARLE, OTI); `--snap-eval` đánh giá lại với token thay bằng từ gần nhất hoặc đưa về độ lớn của từ. `gap` (E1.15b): khoảng cách giữa tâm vector ảnh gallery và tâm từng kiểu vector truy vấn, thang điểm, z-score của ảnh đích, hình PCA. `drift` (E3.9): độ trôi vector ảnh, token, truy vấn của ảnh đã làm xấu / phục hồi / lọc tần số so với ảnh sạch; tương quan Spearman của độ trôi và của PSNR/SSIM với mức thiệt hại (điểm và thứ hạng ảnh đích, chỉ số); bảng "phục hồi có kéo vector về gần ảnh sạch không"; tách theo loại suy giảm với `mixed`. |
