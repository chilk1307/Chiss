# Nhiệm vụ train trên Colab (Chí tự chạy)

**Bối cảnh (24/10):** từ nay chỉ còn Chí chạy Colab. Mã việc vẫn giữ tên H… (Hòa), D… (Doanh), T… như cũ để khớp [tien_do.md](tien_do.md); làm xong thì đánh dấu đúng mã đó. File này gom **mọi lượt train phi** (cần A100) và lượt đánh giá ngay sau train, xếp theo thứ tự chạy. Các thí nghiệm chỉ đánh giá (E1.x, E2.x, E3.1–E3.7, E4.x) và lịch từng ngày ở [nhiem_vu_Chi.md](nhiem_vu_Chi.md); chạy ở ô khác trên cùng runtime A100 trong lúc train chạy nền.

**Thời gian ước lượng** lấy từ D2 (21/10, A100): 100.000 ảnh mất khoảng 2,4 phút mỗi epoch, nên 100 epoch mất khoảng 4 giờ. Một shard (khoảng 33.000 ảnh) mất khoảng 1/3 thời gian đó.

## 0. Chuẩn bị (làm một lần)

1. **Dữ liệu trên Drive** (đã giải nén, không dùng tar nữa). Thư mục `SEARLE` của Chí trên Colab nằm ở `/content/drive/MyDrive/SEARLE`:
   ```text
   datasets/CIRR/        cirr/  train/  dev/  test1/
   datasets/CIRCO/       annotations/  COCO2017_unlabeled/
   datasets/FashionIQ/   captions/  image_splits/  images/
   datasets/ImageNet1K/  test/  (+ ILSVRC2012_img_test_v10102019.tar)
   outputs/              oti_pseudo_tokens/  GPT_phrases/  similar_concepts/  phi_models/  submissions/
   ```
   Thư mục `datasets` được chia sẻ ở https://drive.google.com/drive/folders/1LSMCmBN0WXL7o-muTcXmVPfRjsFnWNji.
   `colab/prepare_data.sh` (ô 4) liên kết thẳng CIRR, CIRCO, FashionIQ tới các thư mục này, không mở `CIRR.tar` (bị cụt) và `FashionIQ.tar` (0 byte) nữa. Riêng ImageNet vẫn giải nén `ILSVRC2012_img_test_v10102019.tar` ra đĩa của Colab, vì train đọc cả 100.000 ảnh và một file 13 GB đọc nhanh hơn nhiều so với 100.000 file lẻ trên Drive. Không có tar thì script liên kết tới `ImageNet1K/test`.
2. **Code:** ô 2 lấy code từ GitHub **`chilk1307/Chiss`** (repo mới), nên mọi bản sửa phải được **push lên `main`** trước khi chạy trên Colab.
3. **Notebook:** *Tệp → Mở sổ tay → GitHub* → `chilk1307/Chiss` → `SEARLE-main/colab/SEARLE_runner.ipynb` → *Lưu bản sao vào Drive*. Ở ô 0:
   ```python
   RUNNER = "Chi"
   DATASETS = "cirr circo imagenet"   # thêm fiq vào phiên có đánh giá FashionIQ
   ```
4. **Dọn Drive (nên làm):** xoá 4 thư mục `outputs/phi_models/dryrun_2ep` trùng tên (kết quả D2 đã chép ở `results/Doanh_D2_20261021_COMPLETED`). Drive cho phép nhiều thư mục cùng tên, còn Colab chỉ thấy một trong số đó.

## 1. Quy trình mỗi phiên

1. *Thời gian chạy → Thay đổi loại thời gian chạy* → **A100**.
2. Chạy ô 0 → ô 5. Ô 4 phải in `ImageNet1K: ready (100000 test images)` và `CIRR: ready …`. Giải nén ImageNet mất vài phút mỗi phiên.
3. Mỗi ô lệnh bắt đầu bằng:
   ```bash
   %%bash
   source /content/env.sh        # khai báo DRIVE, DATA, PY, RUNNER
   cd /content/XLA_SEARLE/SEARLE-main
   ```
4. Train chạy nền bằng `nohup … &` để ô kết thúc ngay; theo dõi bằng ô "Chạy nền" thứ hai của notebook:
   ```bash
   tail -n 3 $DRIVE/results/<mã>/train.log
   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
   ```
   A100 chạy được **2 lượt train cùng lúc**; không chạy 3.
5. Checkpoint nằm trong `data/phi_models/<tên>/`, liên kết tới `$DRIVE/outputs/phi_models/`, nên còn sau khi Colab ngắt. **Colab ngắt:** chạy lại ô 0–5, rồi chạy **đúng lệnh cũ, thêm `--resume`**. Không đổi `--exp-name`: chạy lại cùng tên mà quên `--resume` sẽ ghi đè lượt cũ.
6. Giữ tab Colab mở; phiên rảnh quá lâu sẽ bị ngắt.

Biến dùng chung (dán vào đầu mỗi ô train):
```bash
BASE="--clip-model-name ViT-B/32 --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR --save-training"
FCACHE="--features-cache-dir $DRIVE/cache/train_features"
CACHE="--cache-dir $DRIVE/cache_A100"    # cache đặc trưng gallery khi đánh giá (mọi số đo trên A100)
CIRR="--dataset cirr --dataset-path $DATA/CIRR"
CIRCO="--dataset circo --dataset-path $DATA/CIRCO"
FIQ="--dataset fashioniq --dataset-path $DATA/FashionIQ"
```
`$FCACHE` lưu đặc trưng CLIP của ảnh train ở lượt đầu, các lượt sau cùng tập ảnh khỏi trích lại. Lượt nạp từ cache có chuỗi số ngẫu nhiên hơi khác lượt tự trích, nên **các lượt đem so với nhau phải cùng dùng `$FCACHE`**.

## 2. Thứ tự chạy

Hôm nay là Ngày 8 (24/10). M4 hạn 27/10 cần phi bền bản đầy đủ đã train; M5 hạn 30/10.

| Phiên | Ngày | Lượt 1 trên A100 | Lượt 2 song song | Giờ A100 (ước lượng) |
|---|---|---|---|---|
| 1 | 24/10 | Gỡ H5 (E1.7a, 3 phút) → **T1** `phi_scale_s0` | **D7** đặc trưng ảnh xấu shard s0 | khoảng 2 |
| 2 | 25/10 | **D7–D8** sàng lọc `phi_s0_clean`, rồi `phi_s0_noise` | **D8** sàng lọc `phi_s0_degaug` | khoảng 2 |
| 2 (tiếp) | 25/10 | **D9** đánh giá 3 bản sàng lọc, chọn biến thể | **D9** đặc trưng ảnh xấu 100k | khoảng 3 |
| 3 | 26/10 | **D10** `phi_B32_robust` (100 epoch, 100k ảnh) | **H16** `phi_B32_cosine` | khoảng 5–6 |
| 4 | 27/10 | **D11** đánh giá phi bền (E3.8c) và cosine | **T2** `phi_scale_s01` (tuỳ chọn) | khoảng 3 |
| 5 | 28/10 | **D17** train với bộ câu mới (chỉ khi đã có D16) | — | khoảng 3 |
| — | 29–30/10 | Dự phòng: chạy lại lượt hỏng, `--resume` | — | — |

Hết quota GPU thì bỏ theo thứ tự: D17 → T2 → H16. **D10 (phi bền) là lượt quan trọng nhất**, vì là kết quả chính của [03](03_do_ben_va_phuc_hoi_anh.md).

## 3. Lệnh từng lượt

### Phiên 1, bước 0: gỡ chặn H5 (3 phút, đang có sẵn A100)
Mốc 25.66 đo trên A100; Hòa chạy trên T4 ra 25.59. Chạy lại trên A100:
```bash
$PY src/validate.py --eval-type searle $CIRR $CACHE --output-dir $DRIVE/results/E1.7a-searle-cirr
```
- Ra **25.66 (±0,05)**: C4 đạt, H5 hết chặn; ghi "GPU: A100" vào H5 trong [tien_do.md](tien_do.md), rồi chạy tiếp E1.7b–f ([nhiem_vu_Chi.md](nhiem_vu_Chi.md) mục 5).
- Vẫn ra 25.59: chưa train vội; so tiếp bằng code gốc `git checkout 70328e1` trên cùng A100.
- Cache riêng `cache_A100` để không lẫn với cache cũ `$DRIVE/cache` Hòa tạo trên T4.

### T1. Train phi trên shard s0 — E1.14a (= E8.1a)
Khoảng 1,5 giờ. Lượt đầu tạo `$FCACHE` cho shard s0; T1 là mốc so sánh của D17.
```bash
mkdir -p $DRIVE/results/E1.14a-train-s0
nohup $PY src/train_phi.py --exp-name phi_scale_s0 $BASE --oti-exp-name oti_B32_s0 --train-on-oti-names \
  --num-epochs 100 --validation-frequency 5 $FCACHE > $DRIVE/results/E1.14a-train-s0/train.log 2>&1 &
```

### D7. Đặc trưng ảnh xấu shard s0 — chuẩn bị E3.8a
Khoảng 1 giờ, chạy song song với T1; chạy lại cùng lệnh thì tiếp tục từ phần đã xong.
```bash
mkdir -p $DRIVE/results/E3.8a-train
nohup $PY tools/precompute_degraded_features.py --imagenet-path $DATA/ImageNet1K --oti-exp-name oti_B32_s0 \
  --degrade none,gauss_noise:1,gauss_noise:2,gauss_blur:1,gauss_blur:2,jpeg:1,jpeg:2,low_light:1,low_light:2 \
  --out-dir $DRIVE/cache/imagenet_s0 > $DRIVE/results/E3.8a-train/precompute_s0.log 2>&1 &
```

### D7–D8. Sàng lọc 3 biến thể phi bền trên shard s0 — E3.8a
Mỗi bản 50 epoch, 40–60 phút. Cả 3 bản cùng dùng `$FCACHE` (đã có từ T1), nên chỉ khác nhau đúng ở cờ đang thử.
```bash
TRAIN="$BASE --oti-exp-name oti_B32_s0 --train-on-oti-names --num-epochs 50 --validation-frequency 5 $FCACHE"
L=$DRIVE/results/E3.8a-train
nohup $PY src/train_phi.py --exp-name phi_s0_clean  $TRAIN > $L/phi_s0_clean.log 2>&1 &
nohup $PY src/train_phi.py --exp-name phi_s0_degaug $TRAIN --aug-feature-dir $DRIVE/cache/imagenet_s0 --aug-prob 0.5 > $L/phi_s0_degaug.log 2>&1 &
# chạy sau khi một trong hai lượt trên xong:
nohup $PY src/train_phi.py --exp-name phi_s0_noise  $TRAIN --feature-noise 0.1 > $L/phi_s0_noise.log 2>&1 &
```
`phi_s0_noise` (σ = 0,1) cũng là một điểm của E8.5 ([08](08_hoc_may_cai_tien_phi.md)).

### D9. Đánh giá sàng lọc, chọn biến thể; đặc trưng ảnh xấu 100k — E3.8a
Bắt đầu tính đặc trưng 100k trước (2–3 giờ, chạy nền), rồi đánh giá trong lúc chờ:
```bash
nohup $PY tools/precompute_degraded_features.py --imagenet-path $DATA/ImageNet1K --oti-exp-name oti_B32 \
  --degrade none,gauss_noise:1,gauss_noise:2,gauss_blur:1,gauss_blur:2,jpeg:1,jpeg:2,low_light:1,low_light:2 \
  --out-dir $DRIVE/cache/imagenet_full > $DRIVE/results/E3.8a-train/precompute_full.log 2>&1 &

for P in phi_s0_clean phi_s0_degaug phi_s0_noise; do
  for D in none gauss_noise:2 gauss_blur:2 jpeg:2 low_light:2 salt_pepper:2 motion_blur:2 low_res:2 color_cast:2; do
    $PY src/validate.py --eval-type phi --exp-name $P --phi-checkpoint-name phi_last.pt $CACHE $CIRR \
      --degrade $D --output-dir $DRIVE/results/E3.8a-$P-${D/:/}-cirr
  done
done
$PY tools/collect_results.py --results-dir $DRIVE/results --out $DRIVE/results/summary.md
```
4 suy giảm đầu là **đã thấy khi train**, 4 suy giảm sau là **chưa thấy**. Chọn biến thể có R@1 trung bình trên các ảnh xấu cao nhất mà R@1 trên ảnh sạch (`none`) không tụt quá 0,5 điểm so với `phi_s0_clean`. Ghi lựa chọn vào [nhat_ky_thi_nghiem.md](nhat_ky_thi_nghiem.md).

### D10. Train phi bền bản đầy đủ — E3.8b
Cần: `precompute_full.log` báo xong. Khoảng 4–6 giờ. Lệnh dưới giả sử chọn `degaug`; chọn `noise` thì thay hai cờ `--aug-…` bằng `--feature-noise 0.1`.
```bash
mkdir -p $DRIVE/results/E3.8b-train-robust
nohup $PY src/train_phi.py --exp-name phi_B32_robust $BASE --oti-exp-name oti_B32 \
  --num-epochs 100 --validation-frequency 5 \
  --aug-feature-dir $DRIVE/cache/imagenet_full --aug-prob 0.5 > $DRIVE/results/E3.8b-train-robust/train.log 2>&1 &
```
Lượt này **không dùng `$FCACHE`**, để giống `phi_B32` (mốc so sánh, train không dùng cache).

### H16. Train phi với lịch learning rate cosine — E8.2a
Chạy song song với D10, khoảng 4 giờ; so với `phi_B32`, nên cũng không dùng `$FCACHE`.
```bash
mkdir -p $DRIVE/results/E8.2a-train-cosine
nohup $PY src/train_phi.py --exp-name phi_B32_cosine $BASE --oti-exp-name oti_B32 \
  --num-epochs 100 --lr-scheduler cosine --warmup-epochs 5 \
  --validation-frequency 5 > $DRIVE/results/E8.2a-train-cosine/train.log 2>&1 &
```

### D11 + đánh giá H16, T1, T2 — E3.8c, E8.2a, E1.14
Chạy trên A100 như mọi lượt đánh giá khác, để số so được với E1.7d, E3.1.
```bash
ev() {  # <tên phi> <mã thư mục> <cờ thêm…>
  local P=$1 C=$2; shift 2
  $PY src/validate.py --eval-type phi --exp-name $P --phi-checkpoint-name phi_last.pt $CACHE "$@" \
    --output-dir $DRIVE/results/$C
}
# Phi bền: ảnh sạch trên 3 tập, bộ suy giảm trên CIRR và CIRCO (so với E3.1 của phi_B32)
ev phi_B32_robust E3.8c-robust-clean-cirr  $CIRR
ev phi_B32_robust E3.8c-robust-clean-circo $CIRCO
ev phi_B32_robust E3.8c-robust-clean-fiq   $FIQ
for D in gauss_noise:2 gauss_blur:2 jpeg:2 low_light:2 salt_pepper:2 motion_blur:2 low_res:2 color_cast:2; do
  ev phi_B32_robust E3.8c-robust-${D/:/}-cirr  $CIRR  --degrade $D
  ev phi_B32_robust E3.8c-robust-${D/:/}-circo $CIRCO --degrade $D
done
# Cosine (so với E1.7d, E1.7e)
ev phi_B32_cosine E8.2a-cosine-cirr  $CIRR
ev phi_B32_cosine E8.2a-cosine-circo $CIRCO
# Lượng dữ liệu
for P in phi_scale_s0 phi_scale_s01; do
  ev $P E1.14-$P-cirr  $CIRR
  ev $P E1.14-$P-circo $CIRCO
done
R=$DRIVE/results
$PY tools/bootstrap_ci.py --baseline $R/E1.7d-phiB32-cirr --runs $R/E3.8c-robust-clean-cirr $R/E8.2a-cosine-cirr $R/E1.14-*-cirr --out $R/E3.8c-ci-cirr
```
Cần có E1.7d, E1.7e, E3.1 (phi_B32 trên cùng cấu hình) để so; chưa có thì chạy H5 và D5 trong [nhiem_vu_Chi.md](nhiem_vu_Chi.md) trước. Sau khi có phi bền, chạy E3.9 (độ trôi vector, D11 trong [nhiem_vu_Chi.md](nhiem_vu_Chi.md)).

### T2. (Tuỳ chọn) Train phi trên shard s0 + s1 — E1.14b (= E8.1b)
Khoảng 2,5–3 giờ.
```bash
$PY tools/merge_oti.py --shards oti_B32_s0,oti_B32_s1 --out oti_B32_s01
mkdir -p $DRIVE/results/E1.14b-train-s01
nohup $PY src/train_phi.py --exp-name phi_scale_s01 $BASE --oti-exp-name oti_B32_s01 --train-on-oti-names \
  --num-epochs 100 --validation-frequency 5 $FCACHE > $DRIVE/results/E1.14b-train-s01/train.log 2>&1 &
```

### D17. (Tuỳ chọn) Train trên shard s0 với bộ câu mới — E6.4
Cần: bộ P1 `GPTNeo27B_clean` và bộ câu mới sinh ở D16 (`colab/gen_phrases.ipynb`, lệnh ở D16 trong [nhiem_vu_Chi.md](nhiem_vu_Chi.md)). Cấu hình giống T1, chỉ đổi bộ câu; mốc so sánh là T1.
```bash
for P in GPTNeo27B_clean <bộ mới>; do
  mkdir -p $DRIVE/results/E6.4-$P-s0
  nohup $PY src/train_phi.py --exp-name E6.4-$P-s0 $BASE --oti-exp-name oti_B32_s0 --train-on-oti-names \
    --num-epochs 100 --validation-frequency 5 $FCACHE --gpt-exp-name $P > $DRIVE/results/E6.4-$P-s0/train.log 2>&1 &
done
```

### Không thuộc file này: OTI trên tập val
H17 (E7.0, E6.2-Pλ0), H15 (E7.1), D15 (E7.2), H18 (E6.2) tối ưu token cho từng ảnh của tập val, không train mạng. Mỗi lượt vài phút đến dưới 30 phút; lệnh ở [nhiem_vu_Chi.md](nhiem_vu_Chi.md) mục 8.

## 4. Kiểm tra sau mỗi lượt train

1. `tail -n 20 $DRIVE/results/<mã>/train.log` báo đủ số epoch, không có `Traceback`.
2. `data/phi_models/<tên>/checkpoints/phi_last.pt` tồn tại.
3. `data/phi_models/<tên>/validation_metrics.csv`: R@1 CIRR val tăng dần rồi đi ngang. Lượt 100 epoch trên 100k ảnh phải ở quanh 25.5 (`phi_B32` là 25.47); thấp hơn 24 là bất thường, phải xem lại cờ.
4. Đánh dấu việc trong [tien_do.md](tien_do.md), ghi số chính vào [nhat_ky_thi_nghiem.md](nhat_ky_thi_nghiem.md).

## 5. Khi gặp lỗi

| Hiện tượng | Làm gì |
|---|---|
| Ô 4 báo `expected 100000 images … found N` | Giải nén ImageNet bị ngắt: chạy lại ô 4 (script xoá bản dở rồi giải nén lại) |
| Ô 4 báo `… missing` với CIRR, CIRCO hoặc FashionIQ | Kiểm tra lối tắt `MyDrive/SEARLE` và tên thư mục trong `datasets/` (phân biệt hoa thường) |
| `unrecognized arguments` | Code trên Colab cũ: chạy lại ô 2 (lấy từ `chilk1307/Chiss`) |
| `training images have no OTI token` | Thiếu `--train-on-oti-names` ở lượt train trên shard |
| `--aug-prob needs --aug-feature-dir` hoặc thiếu file trong `imagenet_s0`, `imagenet_full` | Lượt tính đặc trưng ảnh xấu chưa xong: xem `precompute_*.log`, chạy lại đúng lệnh để tiếp tục |
| `CUDA out of memory` | Đang chạy quá 2 lượt: dừng bớt (`pkill -f <tên>`), rồi chạy lại lượt bị lỗi với `--resume` |
| Colab ngắt giữa chừng | Ô 0–5, rồi đúng lệnh cũ thêm `--resume` |
| Log không đổi quá 15 phút | `nvidia-smi`: GPU 0% thì có thể đang đọc ảnh từ Drive (ImageNet bị liên kết thay vì giải nén), xem lại output ô 4 |
