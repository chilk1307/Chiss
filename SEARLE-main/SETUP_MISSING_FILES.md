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

Quy tắc chung:

- Chỉ muốn **đánh giá bằng model có sẵn** → chỉ cần mục 2 và 3 (không cần các file ở mục 4).
- Muốn **huấn luyện lại / tái hiện** → làm lần lượt 4.1 → 4.2 → 4.3 → 4.4.

---

## 2. Cài môi trường

Khuyến nghị dùng [Anaconda](https://www.anaconda.com/).

```sh
conda create -n searle -y python=3.8
conda activate searle
conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0
pip install comet-ml==3.33.6 transformers==4.24.0 tqdm pandas==1.4.2
pip install git+https://github.com/openai/CLIP.git
```

Clone repo:

```sh
git clone https://github.com/chilk1307/XLA_SEARLE.git
cd XLA_SEARLE/SEARLE-main
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

---

## 4. Lấy từng file còn thiếu

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
└── oidv7-class-descriptions.csv        # đã có sẵn trong repo
```

Kiểm tra nhanh (PowerShell):

```powershell
@(
  "data\GPT_phrases\GPTNeo27B\concept_to_phrases.pkl",
  "data\similar_concepts\imagenet\test\ViT-B32_top_250_ensembling_image_texts.csv",
  "data\oti_pseudo_tokens\imagenet\test\oti_b32\oti_pseudo_tokens.pt",
  "data\phi_models\phi_b32\checkpoints\phi_last.pt"
) | ForEach-Object { "{0,-8} {1}" -f (Test-Path $_), $_ }
```

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
