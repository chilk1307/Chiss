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
