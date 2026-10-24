# Hướng dẫn cài đặt các file không có trong repo

Repo này **chỉ chứa mã nguồn**. Các file nặng (checkpoint, token đã invert, danh sách phrase, csv liên kết concept...)
bị loại khỏi git vì vượt giới hạn 100 MB của GitHub. Tài liệu này hướng dẫn lấy lại từng file đó.

> Tất cả đường dẫn bên dưới tính từ thư mục `SEARLE-main/` (chứa `src/`, `data/`, `README.md`).
> Chi tiết các tham số của từng script xem thêm trong [README.md](README.md).

## Mục lục

1. [Tổng quan: file nào thiếu, lấy ở đâu](#1-tổng-quan)
2. [Cài môi trường](#2-cài-môi-trường)
3. [Tải dataset](#3-tải-dataset)
4. [Lấy từng file còn thiếu](#4-lấy-từng-file-còn-thiếu)
5. [Kiểm tra cấu trúc thư mục](#5-kiểm-tra-cấu-trúc-thư-mục)
6. [Chỉ muốn chạy thử nhanh?](#6-chỉ-muốn-chạy-thử-nhanh)
7. [Xử lý lỗi thường gặp](#7-xử-lý-lỗi-thường-gặp)
8. [Chạy demo](#8-chạy-demo)
9. [Chạy trên Google Colab](#9-chạy-trên-google-colab)

---

## 1. Tổng quan

| Đường dẫn (không có trong repo) | Là gì | Dung lượng ước tính | Cách lấy |
|---|---|---|---|
| `data/GPT_phrases/GPTNeo27B/concept_to_phrases.pkl` | Phrase do GPT-Neo sinh cho từng concept | ~700 MB | **Tải sẵn** (mục 4.1) hoặc sinh lại |
| `data/similar_concepts/imagenet/test/*.csv` | Concept gần nhất của từng ảnh ImageNet | ~300 MB | Sinh lại (mục 4.2) |
| `data/oti_pseudo_tokens/imagenet/test/<exp>/*.pt` | Pseudo-token sau OTI | ~200 MB mỗi experiment | Sinh lại (mục 4.3) |
| `data/phi_models/<exp>/checkpoints/phi_*.pt` | Checkpoint mạng phi | ~24 MB mỗi file | Huấn luyện lại (mục 4.4) |
| Trọng số SEARLE đã huấn luyện (`SEARLE_ViT-B32.pt`, `SEARLE_ViT-L14.pt`) | Model phát hành chính thức | — | **Tự động tải** qua `torch.hub` |
| Dataset FashionIQ / CIRR / CIRCO / ImageNet | Dữ liệu đánh giá và huấn luyện | rất lớn | Tải từ trang chính thức (mục 3) |
| `outputs/` (ở **gốc repo**, ngoài `SEARLE-main/`) | Kết quả tái lập của nhóm: checkpoint `phi_B32`, token `oti_B32`, log, file nộp test | vài GB | Google Drive (mục 4.0). Các file nhỏ nằm trong git: `outputs/scripts/`, `outputs/prev_results_oct5/`, `outputs/submissions/`, `outputs/REPORT*.md`, `outputs/pip_freeze.txt`, `outputs/merge_oti.py`, `outputs/logs/eval*.log`, `outputs/phi_models/phi_B32/*.csv, *.json` |
| `results/`, `cache/` (do kế hoạch 2 tuần tạo ra, trên Drive) | Kết quả từng thí nghiệm và cache đặc trưng gallery | tuỳ | Tự sinh khi chạy; không cần tải (mục 4.7) |

Quy tắc chung:

- Chỉ muốn **đánh giá bằng model có sẵn** → chỉ cần mục 2 và 3 (không cần các file ở mục 4).
- Muốn **chạy demo hoặc đánh giá bằng phi_B32 của nhóm** → thêm mục 4.0 (checkpoint `phi_B32`) và mục 8.
- Muốn **dùng ngay các file đã sinh sẵn của nhóm** → tải từ Google Drive (mục 4.0), khỏi chạy lại 4.1 → 4.4.
- Muốn **huấn luyện lại / tái hiện từ đầu** → làm lần lượt 4.1 → 4.2 → 4.3 → 4.4.

---

## 2. Cài môi trường

Khuyến nghị dùng [Anaconda](https://www.anaconda.com/).

```sh
conda create -n searle -y python=3.8
conda activate searle
conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0
pip install comet-ml==3.33.6 transformers==4.24.0 tqdm pandas==1.4.2
pip install git+https://github.com/openai/CLIP.git
pip install setuptools==69.5.1      # comet_ml và CLIP cần pkg_resources; setuptools mới hơn đã bỏ nó
```

Môi trường nhóm đã chạy ổn định (Python 3.8, torch 1.11.0+cu113, torchvision 0.12.0+cu113, transformers 4.24.0, pandas 1.4.2, numpy 1.23.5) được ghi đầy đủ trong `outputs/pip_freeze.txt` (trên Drive, mục 4.0). Cài đúng bản đó nếu gặp lỗi tương thích:

```sh
pip install -r outputs/pip_freeze.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

**Thư viện thêm cho phần xử lý ảnh của kế hoạch** (cần cho `--eval-type cbir` và các phần xử lý ảnh sau này; các phương pháp khác không cần):

```sh
pip install -r colab/requirements-xla.txt
```

File này ghim phiên bản hợp với môi trường Python 3.8 ở trên (opencv-python-headless 4.10.0.84, scikit-image 0.21.0, scipy 1.10.1, PyWavelets 1.4.1, matplotlib 3.7.5).

Clone repo:

```sh
git clone https://github.com/chilk1307/Chiss.git
cd Chiss/SEARLE-main
```

Tạo sẵn các thư mục sẽ chứa file tải về (PowerShell):

```powershell
New-Item -ItemType Directory -Force data\GPT_phrases, data\similar_concepts, data\oti_pseudo_tokens, data\phi_models
```

Linux / macOS:

```sh
mkdir -p data/GPT_phrases data/similar_concepts data/oti_pseudo_tokens data/phi_models
```

---

## 3. Tải dataset

Dataset không nằm trong repo và phải tải từ nguồn chính thức:

| Dataset | Nguồn |
|---|---|
| FashionIQ | https://github.com/XiaoxiaoGuo/fashion-iq |
| CIRR | https://github.com/Cuberick-Orion/CIRR |
| CIRCO | https://github.com/miccunifi/CIRCO |
| ImageNet1K (ILSVRC2012, tập test) | https://image-net.org/index.php |

Cấu trúc thư mục mong đợi của từng dataset được mô tả trong mục *Data Preparation* của [README.md](README.md).
Truyền đường dẫn gốc của dataset qua tham số `--dataset-path`.

Lưu ý riêng cho từng dataset:

- **FashionIQ:** khoảng **2.431 ảnh là link chết** nên không thể tải đủ; `src/datasets.py` của repo đã lọc bỏ các bộ ba thiếu ảnh. Kết quả trên FashionIQ vì vậy chỉ so sánh được **giữa các cấu hình chạy trên cùng tập con**, không so trực tiếp với số trong paper.
- **CIRCO:** cần thêm ảnh COCO 2017 unlabeled (`COCO2017_unlabeled/unlabeled2017/*.jpg`, khoảng 123 nghìn ảnh) và `annotations/image_info_unlabeled2017.json`. Thiếu ảnh sẽ gây `FileNotFoundError` giữa chừng (mục 7).
- **CIRR:** cần cả thư mục `train`, `dev`, `test1` và `cirr/captions`, `cirr/image_splits`. Tập `test1` không có nhãn: chỉ dùng để sinh file nộp (mục 4.6).
- **ImageNet:** chỉ cần tập `test` (100.000 ảnh) cho OTI và train phi. Không cần `train` hay `val`.
- **Tốc độ đọc ảnh:** đọc trực tiếp từ Google Drive rất chậm. Trên Colab hãy giải nén dataset ra ổ cục bộ (`/content/data`) trước khi chạy (mục 9).

---

## 4. Lấy từng file còn thiếu

### 4.0. Tải nhanh từ Google Drive của nhóm (khuyên dùng)

Các file nặng đã sinh sẵn (GPT phrases, similar concepts, OTI pseudo-tokens, checkpoint phi...) được public tại:

**https://drive.google.com/drive/folders/1EZZBD4lZxgVf65RhwHEM8eHAFYPSiZti?usp=sharing**

**Cách 1 — tải bằng trình duyệt:** mở link, chọn thư mục/file cần, bấm *Download*, giải nén rồi đặt đúng đường dẫn như bảng dưới.

**Cách 2 — tải bằng dòng lệnh với `gdown`:**

```sh
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1EZZBD4lZxgVf65RhwHEM8eHAFYPSiZti" -O drive_download
```

Sau khi tải xong, đặt từng thư mục vào `data/` theo bảng:

| Thư mục trên Drive (cấu trúc `outputs/` của nhóm) | Đặt vào |
|---|---|
| `GPT_phrases/` | `data/GPT_phrases/` |
| `similar_concepts/` | `data/similar_concepts/` |
| `oti_pseudo_tokens/` | `data/oti_pseudo_tokens/` |
| `phi_models/` | `data/phi_models/` |

Ví dụ (PowerShell):

```powershell
Copy-Item drive_download\GPT_phrases\*        data\GPT_phrases\        -Recurse -Force
Copy-Item drive_download\similar_concepts\*   data\similar_concepts\   -Recurse -Force
Copy-Item drive_download\oti_pseudo_tokens\*  data\oti_pseudo_tokens\  -Recurse -Force
Copy-Item drive_download\phi_models\*         data\phi_models\         -Recurse -Force
```

Các thư mục còn lại trên Drive (`logs/`, `discarded/`, `prev_results_oct5/`, `scripts/`, `submissions/`, `pip_freeze.txt`, `REPORT*.md`) chỉ để tham khảo; không cần chép vào `data/`.

**Tên experiment của nhóm.** Kết quả tái lập chính có tên `phi_B32` (checkpoint, chữ hoa) và `oti_B32` (token OTI gộp 100.000 ảnh; ba shard thành phần là `oti_B32_s0`, `oti_B32_s1`, `oti_B32_s2`). Các ví dụ lệnh ở mục 4.3 và 4.4 dùng tên thường `oti_b32`, `phi_b32` cho gọn; nếu dùng file của nhóm thì thay bằng tên chữ hoa.

Lưu ý:

- Khi chạy script, truyền `--exp-name` / `--oti-exp-name` / `--phi-checkpoint-name` **đúng tên thư mục** bạn đã đặt vào `data/`, và nhớ phân biệt hoa/thường trên Linux. Ví dụ:
  ```sh
  python src/validate.py --eval-type phi --exp-name phi_B32 --phi-checkpoint-name phi_last.pt --dataset cirr --dataset-path <CIRR>
  ```
- Mỗi thư mục experiment cần có `hyperparameters.json` (validate.py đọc `clip_model_name` và `phi_dropout` từ đó).
- Nếu Drive báo vượt giới hạn tải (quota), bấm *Add shortcut to Drive* hoặc *Make a copy* rồi tải từ Drive của bạn.
- Sau khi đặt xong, kiểm tra lại bằng mục 5.

### 4.1. GPT phrases (`concept_to_phrases.pkl`)

**Cách A — tải bản đã sinh sẵn của tác giả gốc (nhanh nhất, khuyên dùng):**

Linux / macOS:

```sh
cd data/GPT_phrases
wget https://github.com/miccunifi/SEARLE/releases/download/weights/GPTNeo27B.zip
unzip GPTNeo27B.zip
cd ../..
```

Windows (PowerShell):

```powershell
cd data\GPT_phrases
Invoke-WebRequest https://github.com/miccunifi/SEARLE/releases/download/weights/GPTNeo27B.zip -OutFile GPTNeo27B.zip
Expand-Archive GPTNeo27B.zip -DestinationPath .
cd ..\..
```

Sau khi giải nén phải có đúng file:

```
data/GPT_phrases/GPTNeo27B/concept_to_phrases.pkl
```

> Nếu giải nén ra một thư mục lồng thêm (ví dụ `GPTNeo27B/GPTNeo27B/...`) thì di chuyển lên một cấp cho khớp đường dẫn trên.

**Cách B — tự sinh lại (cần GPU, mất nhiều giờ):**

```sh
python src/gpt_phrases_generation.py
```

Kết quả được lưu ở `data/GPT_phrases/GPTNeo27B/`. Có thể thêm `--resume-experiment` để chạy tiếp khi bị gián đoạn.

---

### 4.2. Liên kết concept cho ảnh (`similar_concepts/*.csv`)

Cần dataset ImageNet (mục 3) và file ở mục 4.1.

```sh
python src/image_concepts_association.py --clip-model-name ViT-B/32 --dataset imagenet --dataset-path <đường_dẫn_ImageNet1K> --split test
```

Kết quả: `data/similar_concepts/imagenet/test/ViT-B32_top_250_ensembling_image_texts.csv`.

Dùng `--clip-model-name ViT-L/14` nếu muốn bản SEARLE-XL.

---

### 4.3. Pseudo-token OTI (`oti_pseudo_tokens/*.pt`)

Cần file ở mục 4.1 và 4.2.

```sh
python src/oti_inversion.py --exp-name oti_b32 --clip-model-name ViT-B/32 --dataset imagenet --dataset-path <đường_dẫn_ImageNet1K> --split test
```

Kết quả: `data/oti_pseudo_tokens/imagenet/test/oti_b32/` gồm `oti_pseudo_tokens.pt`,
`ema_oti_pseudo_tokens.pt`, `image_names.pkl`.

Bước này rất nặng (100.000 ảnh × 350 bước tối ưu). Nên chạy trên GPU và dùng `--resume-experiment` khi bị ngắt.
Tên `--exp-name` sẽ được dùng lại ở bước 4.4 (`--oti-exp-name`).

---

### 4.4. Checkpoint mạng phi (`phi_*.pt`)

Cần dataset ImageNet + CIRR (để validation), cùng file ở mục 4.1 và 4.3.

```sh
python src/train_phi.py --exp-name phi_b32 --clip-model-name ViT-B/32 --imagenet-dataset-path <đường_dẫn_ImageNet1K> --cirr-dataset-path <đường_dẫn_CIRR> --oti-exp-name oti_b32 --save-training
```

Kết quả: `data/phi_models/phi_b32/checkpoints/phi_5.pt, phi_10.pt, ..., phi_last.pt`.

Comet.ml là tùy chọn (thêm `--api-key` và `--workspace` nếu có); metric luôn được ghi ra file csv.

Đánh giá checkpoint vừa huấn luyện:

```sh
python src/validate.py --exp-name phi_b32 --eval-type phi --dataset cirr --dataset-path <đường_dẫn_CIRR> --phi-checkpoint-name phi_last.pt
```

---

### 4.5. Model SEARLE chính thức (không cần làm gì)

Khi chạy `--eval-type searle` hoặc `searle-xl`, trọng số được tải tự động từ
`https://github.com/miccunifi/SEARLE/releases/download/weights/` và lưu vào cache của `torch.hub`
(`~/.cache/torch/hub/checkpoints/` hoặc `%USERPROFILE%\.cache\torch\hub\checkpoints\` trên Windows).

Chỉ cần máy có internet ở lần chạy đầu tiên.

---

### 4.6. File nộp test (CIRR, CIRCO)

Các file nộp test của phi_B32 đã nằm sẵn trong repo, không cần sinh lại:

| File | Nội dung | Nộp ở |
|---|---|---|
| `outputs/submissions/phi_B32_cirr.json` | CIRR test1: 4.148 truy vấn, mỗi truy vấn 50 kết quả | [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/) (metric `recall`) |
| `outputs/submissions/subset_phi_B32_cirr.json` | CIRR test1, tập con: 4.148 truy vấn, mỗi truy vấn 3 kết quả | CIRR Evaluation Server (metric `recall_subset`), nộp **cùng** file trên |
| `outputs/submissions/phi_B32_circo.json` | CIRCO test: 800 truy vấn, mỗi truy vấn 50 kết quả | [CIRCO Evaluation Server](https://circo.micc.unifi.it/) |

Muốn sinh file mới (ví dụ cho SEARLE gốc hoặc một checkpoint khác):

```sh
python src/generate_test_submission.py --submission-name <tên> --eval-type searle --dataset cirr  --dataset-path <CIRR>
python src/generate_test_submission.py --submission-name <tên> --eval-type searle --dataset circo --dataset-path <CIRCO>
# với phi của nhóm: --eval-type phi --exp-name phi_B32 --phi-checkpoint-name phi_last.pt
```

File được ghi vào `data/test_submissions/<cirr|circo>/`. Với CIRR, lệnh sinh ra hai file: `<tên>.json` và `subset_<tên>.json`.

Tập test không có nhãn nên **không tự tính được điểm**; điểm do server chấm trả về. Chọn tham số trên tập val, chỉ nộp cấu hình cuối.

---

### 4.7. Thư mục `results/` và `cache/` (tự sinh, không cần tải)

Các công cụ đánh giá của kế hoạch 2 tuần ghi kết quả vào `results/<mã thí nghiệm>/` và cache đặc trưng gallery vào `cache/` (xem `plans/01_nen_tang_danh_gia.md`). Hai thư mục này không đưa vào git và tự được tạo khi chạy với `--output-dir` và `--cache-dir`. Cache chỉ dùng được với **cùng backbone, cùng kiểu tiền xử lý và cùng tập ảnh**; đổi một trong ba thì phải sinh lại.

---

## 5. Kiểm tra cấu trúc thư mục

Sau khi làm xong, cây thư mục `data/` cần có dạng:

```
data/
├── GPT_phrases/
│   └── GPTNeo27B/
│       └── concept_to_phrases.pkl
├── similar_concepts/
│   └── imagenet/test/
│       └── ViT-B32_top_250_ensembling_image_texts.csv
├── oti_pseudo_tokens/
│   └── imagenet/test/oti_b32/
│       ├── oti_pseudo_tokens.pt
│       ├── ema_oti_pseudo_tokens.pt
│       └── image_names.pkl
├── phi_models/
│   └── phi_b32/checkpoints/
│       ├── phi_5.pt
│       ├── ...
│       └── phi_last.pt
├── test_submissions/                   # do generate_test_submission.py tạo ra
│   ├── cirr/<tên>.json, subset_<tên>.json
│   └── circo/<tên>.json
└── oidv7-class-descriptions.csv        # đã có sẵn trong repo
```

Nếu dùng file của nhóm, thay `oti_b32`/`phi_b32` bằng `oti_B32`/`phi_B32`; mỗi thư mục experiment cần có thêm `hyperparameters.json`.

Ở **gốc repo** (ngoài `SEARLE-main/`), các phần sau có trong git: `plans/`, `outputs/scripts/`, `outputs/prev_results_oct5/`, `outputs/submissions/` và các file nhỏ liệt kê ở mục 1. Phần còn lại của `outputs/` (checkpoint, token, phrase, concept) lấy từ Drive.

Kiểm tra nhanh (PowerShell):

```powershell
@(
  "data\GPT_phrases\GPTNeo27B\concept_to_phrases.pkl",
  "data\similar_concepts\imagenet\test\ViT-B32_top_250_ensembling_image_texts.csv",
  "data\oti_pseudo_tokens\imagenet\test\oti_B32\oti_pseudo_tokens.pt",
  "data\oti_pseudo_tokens\imagenet\test\oti_B32\image_names.pkl",
  "data\phi_models\phi_B32\checkpoints\phi_last.pt",
  "data\phi_models\phi_B32\hyperparameters.json",
  "..\outputs\submissions\phi_B32_cirr.json",
  "..\outputs\submissions\subset_phi_B32_cirr.json",
  "..\outputs\submissions\phi_B32_circo.json"
) | ForEach-Object { "{0,-8} {1}" -f (Test-Path $_), $_ }
```

Dòng nào hiện `False` thì file đó còn thiếu; đối chiếu với mục tương ứng ở trên.

---

## 6. Chỉ muốn chạy thử nhanh?

Không cần file nào ở mục 4. Chỉ cần môi trường (mục 2) và một dataset đánh giá (mục 3):

```sh
python src/validate.py --eval-type searle --dataset cirr --dataset-path <đường_dẫn_CIRR>
```

Hoặc dùng ví dụ tối thiểu trong mục *SEARLE Minimal Working Example* của [README.md](README.md) (chỉ cần một ảnh bất kỳ).

---

## 7. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `FileNotFoundError: ... concept_to_phrases.pkl` | Chưa có GPT phrases hoặc giải nén sai thư mục | Làm lại mục 4.1, kiểm tra đúng đường dẫn |
| `FileNotFoundError: ... similar_concepts ... .csv` | Chưa chạy bước liên kết concept | Làm mục 4.2 |
| `FileNotFoundError: ... oti_pseudo_tokens ...` | Chưa chạy OTI hoặc `--oti-exp-name` không khớp `--exp-name` ở bước OTI | Làm mục 4.3, kiểm tra tên experiment (phân biệt hoa/thường) |
| `CUDA out of memory` | Batch size quá lớn | Giảm `--batch-size` |
| Tải `torch.hub` bị lỗi mạng | Không truy cập được GitHub Releases | Tải tay file `.pt` rồi đặt vào cache `torch/hub/checkpoints` |
| Thiếu `clip` / `transformers` | Chưa cài đủ thư viện | Chạy lại các lệnh `pip install` ở mục 2 |
| `ModuleNotFoundError: No module named 'pkg_resources'` khi import `comet_ml` hoặc `clip` | setuptools quá mới (đã bỏ `pkg_resources`) | `pip install setuptools==69.5.1` (demo ghi `setuptools<81`) |
| `FileNotFoundError: ... unlabeled2017/<id>.jpg` khi đánh giá CIRCO | Thiếu ảnh COCO 2017 unlabeled hoặc đường dẫn gốc sai | Kiểm tra mục 3; `--dataset-path` phải trỏ tới thư mục chứa `annotations/` và `COCO2017_unlabeled/` |
| `IndexError: list index out of range` hoặc `ValueError: '<id>' is not in list` khi đánh giá FashionIQ | Dùng `datasets.py` bản gốc, không lọc ảnh thiếu (FashionIQ thiếu khoảng 2.431 ảnh) | Dùng `src/datasets.py` của repo này (đã lọc bộ ba thiếu ảnh); không thay bằng bản gốc |
| `argument --dataset-path: expected one argument` | Biến shell chứa đường dẫn đang rỗng (`$DATA` chưa khai báo) | Khai báo biến hoặc truyền đường dẫn trực tiếp |
| Thiếu RAM khi nạp `concept_to_phrases.pkl` | File 698 MB giải nén thành vài GB trong Python | Dùng máy ≥ 16 GB RAM hoặc Colab; chỉ OTI và train phi cần file này |
| Điểm FashionIQ cao bất thường so với paper | Chạy trên tập con (thiếu ảnh) | Bình thường; chỉ so sánh giữa các cấu hình trên cùng tập con |
| Demo báo không thấy checkpoint | Chưa đặt `phi_*.pt` vào đúng thư mục | Xem mục 8 |
| Không đẩy được file lên GitHub (quá 100 MB) | Thêm nhầm checkpoint, `.pkl`, `.csv` lớn vào git | Các đuôi `.pkl .pt .pth .ckpt .npy .npz .safetensors`, thư mục `outputs/*` (trừ `scripts/`, `prev_results_oct5/`, `submissions/`) và các thư mục dữ liệu lớn của `SEARLE-main/data/` đã nằm trong `.gitignore`; chỉ commit file nhỏ, file nặng đưa lên Drive |

---

## 8. Chạy demo

Demo web (Flask, cổng 5050) nằm ở `demo/app.py`; cần một checkpoint phi để chạy.

**Chỗ demo tìm checkpoint (theo thứ tự):**

1. `<gốc repo>/outputs/phi_models/phi_B32/checkpoints/phi_*.pt` (kết quả của nhóm, từ Drive);
2. `SEARLE-main/data/phi_models/phi_b32/checkpoints/phi_*.pt` (bản nhỏ cũ, nếu có).

Chỉ cần một trong hai. Muốn đúng bản tái lập đầy đủ thì đặt `phi_B32` vào `outputs/phi_models/` ở **gốc repo** (không phải trong `SEARLE-main/data/`). Trang "Đánh giá" của demo còn đọc thêm `outputs/REPORT_results.md`, `outputs/oti_pseudo_tokens/.../oti_B32/image_names.pkl`, `outputs/phi_models/phi_B32/validation_metrics.csv` và `outputs/submissions/*.json`; thiếu file nào thì mục tương ứng hiện "chưa có", demo vẫn chạy.

**Cài thư viện và chạy:**

```sh
pip install -r demo/requirements.txt
python demo/app.py
# Windows: demo\run_demo.bat
```

Mở http://127.0.0.1:5050. Lần đầu CLIP ViT-B/32 được tải về (khoảng 340 MB). Gallery do người dùng tự thêm (tải ảnh lên hoặc nhập đường dẫn thư mục); đặc trưng lưu ở `demo/cache/gallery.pt`.

**Lưu ý:** `demo/requirements.txt` chưa ghim phiên bản và đang chạy trên Python mới hơn môi trường huấn luyện (3.8); nếu cài lẫn vào conda env `searle`, `pip` có thể nâng cấp torch. Nên dùng một môi trường riêng cho demo.

---

## 9. Chạy trên Google Colab

Kế hoạch 2 tuần chạy thí nghiệm trên Colab. Quy ước chung:

- **Drive:** thư mục `SEARLE` của nhóm luôn ở `/content/drive/MyDrive/SEARLE` (thêm lối tắt vào *Drive của tôi* nếu thư mục do người khác chia sẻ).
- **Dữ liệu:** từ 24/10, `$DRIVE/datasets/` có 4 thư mục giải nén đủ ảnh (`CIRR`, `CIRCO`, `FashionIQ`, `ImageNet1K`). CIRR, CIRCO, FashionIQ được liên kết thẳng tới Drive. Đọc ảnh từ Drive rất chậm (lượt đánh giá CIRCO ngày 15/10 phải bỏ dở, `outputs/REPORT.md`), nên mọi lượt đánh giá dùng `--cache-dir`: ảnh gallery chỉ phải đọc một lần cho mỗi cấu hình tiền xử lý. ImageNet (train đọc cả 100.000 ảnh) vẫn được giải nén từ tar ra `/content/data`.
- **Code:** clone `https://github.com/chilk1307/Chiss.git` vào `/content/XLA_SEARLE` ở đầu mỗi phiên, vì ổ cục bộ bị xoá khi runtime ngắt.
- **Môi trường:** Colab có Python mới hơn 3.8. `colab/setup_env.sh` dựng một venv Python 3.8 từ `outputs/pip_freeze.txt` bằng `uv venv --python 3.8`, cài thêm `colab/requirements-xla.txt` và ghi `/content/env.sh` (`DRIVE`, `DATA`, `PY`, `RUNNER`).
- **Bộ chạy:** mở `SEARLE-main/colab/SEARLE_runner.ipynb` trên Colab (*Tệp → Mở sổ tay → GitHub* → `chilk1307/Chiss`), đặt `RUNNER` ở ô 0, chạy ô 0–5. Ô 4 gọi `colab/prepare_data.sh cirr circo fiq imagenet` (chọn tập cần dùng): liên kết `/content/data/<tập>` tới thư mục `$DRIVE/datasets/<tập>` (chỉ khi thiếu thư mục mới thử `<tập>.tar`), giải nén tar ImageNet ra đĩa cục bộ; đồng thời liên kết `data/{GPT_phrases,oti_pseudo_tokens,similar_concepts,phi_models}` tới `$DRIVE/outputs/`.
- **Lưu kết quả lên Drive** ngay khi xong từng lượt chạy (`results/`, `cache/`, checkpoint), vì runtime có thể bị ngắt bất cứ lúc nào. Train phi dài nên có `--resume`.
