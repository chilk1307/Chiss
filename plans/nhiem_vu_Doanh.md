# Nhiệm vụ của Doanh: tiền xử lý, độ bền ảnh và train phi bền trên Colab

**Vai trò:** chạy trên Google Colab các thí nghiệm tiền xử lý ảnh ([04](04_tien_xu_ly_va_truy_van.md)), độ bền với ảnh xấu và phục hồi ảnh ([03](03_do_ben_va_phuc_hoi_anh.md)), train phi bền với ảnh xấu, và một phần luồng học máy. Doanh **không sửa code**: gặp lỗi thì gửi log cho Chí.

**Bàn giao cho Chí (hạn Ngày 13, 29/10):**
1. Bảng tiền xử lý E4.1–E4.3 (kể cả nhóm truy vấn left/right).
2. Bảng mức sụt giảm, phục hồi cố định, phục hồi tự động (E3.1–E3.3).
3. Biểu đồ PSNR/SSIM so với điểm truy xuất (E3.4), đường cong theo tần số cắt (E3.5) và bảng độ trôi vector (E3.9).
4. Kết quả phi bền: sàng lọc và bản đầy đủ (E3.8).
5. Hình minh hoạ (E3.7), độ bền bộ đo thuộc tính (E3.6).
6. Kết quả luồng học máy và bộ câu GPT (E7.2, E6.1, E6.4), nếu đã chạy.

## 0. Cập nhật 18/10: code đã xong, làm sớm được

Toàn bộ code mà các việc D1–D17 cần đã có trên GitHub. Lịch ở mục 3 là lịch gốc; hạn chế duy nhất khi làm sớm là quota GPU. Thứ tự gợi ý:

1. **D1, D2**. D2 giờ chạy bản `train_phi.py` mới trên dữ liệu thật: epoch 1 phải ra R@1 khoảng 22,05 (mục D2); lệch nhiều thì dừng và báo Chí.
2. **D5, D6** (E3.1, E3.5): không phải chờ ai.
3. **D3, D4** (đệm ảnh, nội suy, lật ảnh): chờ **Hòa xong H5** để dùng lại cache gallery; nếu chưa có cache vẫn chạy được, chỉ chậm hơn ở lượt đầu.
4. **D7, D8, D9** (A100): đặc trưng ảnh xấu, sàng lọc phi bền, phục hồi cố định E3.2. Sau D8 gửi Chí bảng E3.2 để điền bảng `oracle`.
5. **D10** (train bản đầy đủ, E3.3): train phi bền chạy được ngay; **E3.3 (phục hồi tự động) chờ Chí hiệu chỉnh và commit `src/restore_rules.json`**, nhớ chạy ô lấy code mới (ô 2) trước khi chạy E3.3.
6. **D16**: lượt 1 chỉ sinh câu cho CIRCO val (vài nghìn khái niệm); **D17** chờ bộ câu của D16 và dùng chung `--features-cache-dir` với T1 của Hòa (xem H-T1).
7. **D15** chờ kết quả E7.1 của Hòa.

Lần chạy notebook đầu tiên chưa ai thử trên Colab thật: nếu một ô lỗi, gửi Chí tên ô và 30 dòng cuối log.

## 1. Chuẩn bị (Ngày 1, làm một lần)

1. **Drive:** mở link thư mục `SEARLE` Chí chia sẻ → chuột phải → *Sắp xếp* → *Thêm lối tắt* → *Drive của tôi*. Trên Colab thư mục nằm ở `/content/drive/MyDrive/SEARLE`.
2. **Notebook:** trên Colab chọn *Tệp → Mở sổ tay → GitHub*, nhập `chilk1307/XLA_SEARLE`, mở `SEARLE-main/colab/SEARLE_runner.ipynb`, *Lưu bản sao vào Drive*. Ở ô 0 đặt `RUNNER = "Doanh"`.
3. Dùng **tài khoản Colab của chính mình**. Phần train cần A100; nếu tài khoản không có A100, báo Chí ngay Ngày 1.

## 2. Quy trình mỗi phiên Colab

1. Chọn GPU: **A100** khi train hoặc tính đặc trưng ImageNet; **T4/L4** khi chỉ đánh giá.
2. Chạy ô 1–5 (gắn Drive, lấy code mới nhất, cài môi trường, giải nén dữ liệu, kiểm tra GPU). Ô 4 thêm `imagenet` vào những ngày có train.
3. Mỗi ô lệnh bắt đầu bằng:
   ```bash
   %%bash
   source /content/env.sh        # khai báo DRIVE, DATA, PY, RUNNER
   cd /content/XLA_SEARLE/SEARLE-main
   ```
4. Train dài thì chạy nền bằng `nohup ... &` (lệnh mẫu ở dưới), theo dõi bằng `tail -n 3 <file log>`.
5. Sau mỗi lượt: kiểm tra `$DRIVE/results/<mã>/metrics.json` và dòng mới trong `$DRIVE/results/runs_Doanh.csv`.
6. Cuối ngày gửi báo cáo theo mẫu ở mục 6.

## 3. Lịch theo ngày

| Ngày | Ngày tháng | Việc | Cần có từ Chí / Hòa | Mã thí nghiệm | GPU |
|---|---|---|---|---|---|
| 1 | T7 17/10 | D1 dựng Colab, giải nén ImageNet, kiểm tra file trên Drive | C2 | — | không cần |
| 2 | CN 18/10 | D2 chạy thử train 2 epoch, đo thời gian | — | — | A100 |
| 3 | T2 19/10 | D3 đệm ảnh, nội suy (buổi chiều) | C6; Hòa xong H5 (cache) | E4.1, E4.2 | T4/L4 |
| 4 | T3 20/10 | D4 lật ảnh; cấu hình tốt nhất trên CIRCO, FIQ | C6 | E4.3 | T4/L4 |
| 5 | T4 21/10 | D5 mức sụt giảm; D6 phân tích tần số | C9 | E3.1, E3.5 | T4/L4 |
| 6 | T5 22/10 | D7 tính đặc trưng ảnh xấu (shard s0); bắt đầu sàng lọc phi bền | C10 | E3.8a | A100 |
| 7 | T6 23/10 | D8 phục hồi cố định; sàng lọc (tiếp); nhóm left/right | C9, C11 | E3.2, E3.8a, E4.3b | A100 |
| 8 | T7 24/10 | D9 đánh giá 3 bản sàng lọc, cùng Chí chọn biến thể; tính đặc trưng ảnh xấu 100k | — | E3.8a | A100 |
| 9 | CN 25/10 | D10 train phi bền bản đầy đủ (nền); phục hồi tự động | C14 | E3.8b, E3.3 | A100 |
| 10 | T2 26/10 | D11 PSNR/SSIM; đánh giá phi bền; độ trôi vector | C16, C24 | E3.4, E3.8c, E3.9 | T4/L4 |
| 11 | T3 27/10 | D12 độ bền bộ đo thuộc tính; hình minh hoạ | C12, C16 | E3.6, E3.7 | T4/L4 |
| 12 | T4 28/10 | D13 chạy bổ sung; nộp test phi bền (nếu Chí quyết định ở C18); D15 OTI warm-start trên CIRR val (học máy, nếu còn GPU); D16 sinh bộ câu GPT quy mô nhỏ | C18; E7.1 của Hòa; C22, C23 | E7.2, E6.1 | T4/L4; A100 cho D16 |
| 13 | T5 29/10 | D14 tổng hợp bảng, hình → bàn giao Chí; D17 train phi trên shard s0 với bộ câu mới | C22 | E6.4 | A100 (cho D17) |
| 14 | T6 30/10 | Dự phòng: chạy lại khi cần; đối chiếu số trong báo cáo | — | — | tuỳ |

## 4. Chi tiết từng nhiệm vụ

Các lệnh dưới đây dùng biến chung:
```bash
PHI="--eval-type phi --exp-name phi_B32 --phi-checkpoint-name phi_last.pt --cache-dir $DRIVE/cache"
CIRR="--dataset cirr --dataset-path $DATA/CIRR"
CIRCO="--dataset circo --dataset-path $DATA/CIRCO"
```

### D1. Dựng Colab, kiểm tra file trên Drive (Ngày 1)
- Làm mục 1, chạy ô 1–5 với ô 4 gồm `cirr circo imagenet`.
- Kiểm tra các file cần cho train có trên Drive:
  ```bash
  ls $DRIVE/outputs/oti_pseudo_tokens/imagenet/test/     # phải có oti_B32, oti_B32_s0, oti_B32_s1, oti_B32_s2
  ls $DRIVE/outputs/GPT_phrases/GPTNeo27B/               # concept_to_phrases.pkl
  ls $DRIVE/outputs/similar_concepts/imagenet/test/      # ViT-B32_top_250_ensembling_image_texts.csv
  ls $DATA/ImageNet1K/test | head -3; ls $DATA/ImageNet1K/test | wc -l   # phải là 100000
  ```

### D2. Chạy thử train 2 epoch (Ngày 2)
Code là bản `train_phi.py` mới (trùng từng bit code gốc trên dữ liệu giả lập, chưa thử trên dữ liệu thật): đây là lần kiểm tra trên dữ liệu thật.
```bash
$PY src/train_phi.py --exp-name dryrun_2ep --clip-model-name ViT-B/32 \
  --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR \
  --oti-exp-name oti_B32 --num-epochs 2 --save-training
```
- Ghi lại thời gian một epoch (dùng để ước lượng GPU cho cả 2 tuần).
- So `$DRIVE/outputs/phi_models/dryrun_2ep/validation_metrics.csv` với 2 dòng đầu của `outputs/phi_models/phi_B32/validation_metrics.csv` (epoch 1: R@1 = 22.05). Lệch nhỏ do khác GPU là bình thường.
- Xong thì xoá thư mục `dryrun_2ep` để tránh nhầm.

### D3. Đệm ảnh và nội suy (Ngày 3, buổi chiều) — E4.1, E4.2
Cần: C6; đợi Hòa báo xong H5 (cache mặc định).
```bash
for RATIO in 1.0 1.25 1.5 2.0; do
  $PY src/validate.py $PHI $CIRR --targetpad-ratio $RATIO --output-dir $DRIVE/results/E4.1-pad$RATIO-cirr
done
$PY src/validate.py $PHI $CIRR --preprocess-type clip --output-dir $DRIVE/results/E4.1-clipcrop-cirr
for I in nearest bilinear bicubic lanczos; do
  $PY src/validate.py $PHI $CIRR --interp $I --output-dir $DRIVE/results/E4.2-interp-$I-cirr
done
```

### D4. Lật ảnh; cấu hình tốt nhất trên CIRCO, FIQ (Ngày 4) — E4.3
```bash
for T in flip-ref flip-gallery flip-both; do
  $PY src/validate.py $PHI $CIRR --tta $T --output-dir $DRIVE/results/E4.3-$T-cirr
done
$PY tools/collect_results.py --results-dir $DRIVE/results --out $DRIVE/results/summary.md
```
- Gửi Chí bảng E4.1–E4.3; Chí chọn cấu hình tốt nhất.
- Chạy cấu hình đó trên CIRCO val và FIQ val, **không chỉnh thêm**: `--output-dir $DRIVE/results/E4.x-best-circo` và `...-best-fiq`.

### D5. Mức sụt giảm (Ngày 5) — E3.1
Cần: C9. Kết quả trên ảnh sạch dùng lại E1.7d, E1.7e của Hòa.
```bash
for D in gauss_noise salt_pepper gauss_blur motion_blur jpeg low_light low_res color_cast; do
  for L in 1 2; do
    $PY src/validate.py $PHI $CIRR  --degrade $D:$L --save-quality --output-dir $DRIVE/results/E3.1-$D$L-cirr
    $PY src/validate.py $PHI $CIRCO --degrade $D:$L --save-quality --output-dir $DRIVE/results/E3.1-$D$L-circo
  done
done
```
32 lượt, mỗi lượt vài phút (chỉ ảnh tham chiếu phải mã hoá lại).

### D6. Phân tích miền tần số (Ngày 5) — E3.5
```bash
for R in 0.05 0.1 0.2 0.3 0.5; do
  $PY src/validate.py $PHI $CIRR --freq-filter lowpass:$R  --output-dir $DRIVE/results/E3.5-lp$R-cirr
  $PY src/validate.py $PHI $CIRR --freq-filter highpass:$R --output-dir $DRIVE/results/E3.5-hp$R-cirr
done
```

### D7. Đặc trưng ảnh xấu (shard s0), bắt đầu sàng lọc phi bền (Ngày 6) — E3.8a
Cần: C10; **A100**.
```bash
# 1) Đặc trưng CLIP của ảnh sạch và 8 phiên bản ảnh xấu (khoảng 1 giờ)
$PY tools/precompute_degraded_features.py --imagenet-path $DATA/ImageNet1K --oti-exp-name oti_B32_s0 \
  --degrade none,gauss_noise:1,gauss_noise:2,gauss_blur:1,gauss_blur:2,jpeg:1,jpeg:2,low_light:1,low_light:2 \
  --out-dir $DRIVE/cache/imagenet_s0
# 2) Bản sàng lọc 1: phi train trên ảnh sạch (mốc so sánh cùng quy mô), 50 epoch
TRAIN="--clip-model-name ViT-B/32 --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR --oti-exp-name oti_B32_s0 --train-on-oti-names --num-epochs 50 --validation-frequency 5 --save-training"
mkdir -p $DRIVE/results/E3.8a-train
nohup $PY src/train_phi.py --exp-name phi_s0_clean $TRAIN > $DRIVE/results/E3.8a-train/phi_s0_clean.log 2>&1 &
```
Mỗi bản sàng lọc 1–1,5 giờ. Colab ngắt: chạy lại đúng lệnh, thêm `--resume`.

### D8. Phục hồi cố định; sàng lọc (tiếp); nhóm left/right (Ngày 7) — E3.2, E3.8a, E4.3b
- **Sàng lọc bản 2 và 3** (có thể chạy song song trên A100):
  ```bash
  nohup $PY src/train_phi.py --exp-name phi_s0_degaug $TRAIN --aug-feature-dir $DRIVE/cache/imagenet_s0 --aug-prob 0.5 > $DRIVE/results/E3.8a-train/phi_s0_degaug.log 2>&1 &
  nohup $PY src/train_phi.py --exp-name phi_s0_noise  $TRAIN --feature-noise 0.1 > $DRIVE/results/E3.8a-train/phi_s0_noise.log 2>&1 &
  ```
  Bản `phi_s0_noise` (σ = 0.1) cũng là một điểm của E8.5 trong [08](08_hoc_may_cai_tien_phi.md); không chạy lại điểm này ở đó.
- **E3.2 phục hồi cố định** (mức 2, CIRR val):

  | Suy giảm | Các phép phục hồi cần thử |
  |---|---|
  | `gauss_noise:2` | `gaussian`, `bilateral`, `nlm` |
  | `salt_pepper:2` | `median` |
  | `gauss_blur:2` | `unsharp`, `wiener`, `richardson_lucy` |
  | `motion_blur:2` | `wiener`, `richardson_lucy` |
  | `jpeg:2` | `deblock`, `bilateral` |
  | `low_light:2` | `histeq`, `clahe`, `gamma`, `retinex` |
  | `low_res:2` | `unsharp` |
  | `color_cast:2` | `gray_world`, `white_patch` |

  Lệnh mẫu cho một ô trong bảng:
  ```bash
  $PY src/validate.py $PHI $CIRR --degrade gauss_noise:2 --restore nlm --save-quality --output-dir $DRIVE/results/E3.2-gauss_noise2-nlm-cirr
  ```
  Gửi Chí bảng kết quả: Chí dùng để điền bảng `oracle` cho C14.
- **E4.3b, nhóm left/right:**
  ```bash
  $PY tools/query_taxonomy.py --dataset cirr --dataset-path $DATA/CIRR --categories spatial \
    --runs $DRIVE/results/E1.7d-phiB32-cirr $DRIVE/results/E4.3-*-cirr --out $DRIVE/results/E4.3b-leftright-cirr
  ```

### D9. Đánh giá 3 bản sàng lọc, chọn biến thể (Ngày 8) — E3.8a
```bash
for P in phi_s0_clean phi_s0_degaug phi_s0_noise; do
  for D in none gauss_noise:2 gauss_blur:2 jpeg:2 low_light:2 salt_pepper:2 motion_blur:2 low_res:2 color_cast:2; do
    $PY src/validate.py --eval-type phi --exp-name $P --phi-checkpoint-name phi_last.pt --cache-dir $DRIVE/cache $CIRR \
      --degrade $D --output-dir $DRIVE/results/E3.8a-$P-${D/:/}-cirr
  done
done
```
- 4 suy giảm đầu (`gauss_noise`, `gauss_blur`, `jpeg`, `low_light`) là **đã thấy khi train**; 4 suy giảm sau là **chưa thấy**.
- Gửi Chí bảng; cùng chọn biến thể cho bản đầy đủ.
- Sau đó tính đặc trưng ảnh xấu cho toàn bộ 100.000 ảnh (chạy nền, 2–3 giờ):
  ```bash
  nohup $PY tools/precompute_degraded_features.py --imagenet-path $DATA/ImageNet1K --oti-exp-name oti_B32 \
    --degrade none,gauss_noise:1,gauss_noise:2,gauss_blur:1,gauss_blur:2,jpeg:1,jpeg:2,low_light:1,low_light:2 \
    --out-dir $DRIVE/cache/imagenet_full > $DRIVE/results/E3.8a-train/precompute_full.log 2>&1 &
  ```

### D10. Train phi bền bản đầy đủ; phục hồi tự động (Ngày 9) — E3.8b, E3.3
- **Train bản đầy đủ** (giả sử chọn `degaug`; nếu chọn biến thể khác thì đổi cờ), 7–9 giờ:
  ```bash
  mkdir -p $DRIVE/results/E3.8b-train-robust
  nohup $PY src/train_phi.py --exp-name phi_B32_robust --clip-model-name ViT-B/32 \
    --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR --oti-exp-name oti_B32 \
    --num-epochs 100 --validation-frequency 5 --save-training \
    --aug-feature-dir $DRIVE/cache/imagenet_full --aug-prob 0.5 > $DRIVE/results/E3.8b-train-robust/train.log 2>&1 &
  ```
  Colab ngắt: chạy lại đúng lệnh, thêm `--resume`.
- **E3.3 phục hồi tự động** (cần C14; `--restore auto` dùng ngưỡng trong `src/restore_rules.json`, hiện là bản tạm: **chỉ chạy E3.3 sau khi Chí báo đã hiệu chỉnh lại ngưỡng trên CIRR train**; `auto_iqa` tải trọng số BRISQUE từ GitHub ở lần dùng đầu nên cần internet; `oracle` đọc bảng `src/restore_oracle.json` mà Chí điền sau E3.2):
  ```bash
  for R in none auto auto_iqa oracle; do
    $PY src/validate.py $PHI $CIRR --degrade mixed --restore $R --save-quality --output-dir $DRIVE/results/E3.3-mixed-$R-cirr
  done
  # Ảnh sạch: phục hồi tự động có làm hỏng ảnh tốt không
  for R in auto auto_iqa; do
    $PY src/validate.py $PHI $CIRR --restore $R --output-dir $DRIVE/results/E3.3-clean-$R-cirr
  done
  ```
  Lặp lại cấu hình tốt nhất trên CIRCO val.

### D11. PSNR/SSIM; đánh giá phi bền; độ trôi vector (Ngày 10) — E3.4, E3.8c, E3.9
```bash
# E3.1 (không phục hồi) là mốc so sánh của từng loại suy giảm, nên phải đưa vào cùng E3.2
$PY tools/quality_vs_retrieval.py --runs $DRIVE/results/E3.1-*-cirr $DRIVE/results/E3.2-* --out $DRIVE/results/E3.4-quality-cirr
# Phi bền (sau khi train xong): ảnh sạch trên 3 tập và bộ suy giảm trên CIRR, CIRCO
ROB="--eval-type phi --exp-name phi_B32_robust --phi-checkpoint-name phi_last.pt --cache-dir $DRIVE/cache"
$PY src/validate.py $ROB $CIRR  --output-dir $DRIVE/results/E3.8c-robust-clean-cirr
$PY src/validate.py $ROB $CIRCO --output-dir $DRIVE/results/E3.8c-robust-clean-circo
$PY src/validate.py $ROB --dataset fashioniq --dataset-path $DATA/FashionIQ --output-dir $DRIVE/results/E3.8c-robust-clean-fiq
for D in gauss_noise:2 gauss_blur:2 jpeg:2 low_light:2 salt_pepper:2 motion_blur:2 low_res:2 color_cast:2; do
  $PY src/validate.py $ROB $CIRR  --degrade $D --output-dir $DRIVE/results/E3.8c-robust-${D/:/}-cirr
  $PY src/validate.py $ROB $CIRCO --degrade $D --output-dir $DRIVE/results/E3.8c-robust-${D/:/}-circo
done
```
So với E3.1 (phi_B32 trên cùng suy giảm).

**E3.9, độ trôi vector** (cần C24; ý nghĩa ở [03](03_do_ben_va_phuc_hoi_anh.md)). Chạy sau E3.3. Công cụ đọc `--degrade`, `--restore`, `--freq-filter` từ chính các thư mục kết quả, lấy đặc trưng ảnh đã xử lý từ cache, rồi so vector ảnh, pseudo-token, vector truy vấn với ảnh sạch. Lượt nào chạy không có `--save-quality` (E3.3 trên ảnh sạch, E3.5, E3.8c) thì cột PSNR/SSIM ghi `n/a`, các cột khác vẫn đủ.
```bash
R=$DRIVE/results
$PY tools/vector_analysis.py drift $CIRR  --cache-dir $DRIVE/cache --method phi:phi_B32 \
  --runs $R/E3.1-*-cirr $R/E3.2-* $R/E3.3-*-cirr $R/E3.5-*-cirr --out $R/E3.9-drift-cirr
$PY tools/vector_analysis.py drift $CIRCO --cache-dir $DRIVE/cache --method phi:phi_B32 \
  --runs $R/E3.1-*-circo --out $R/E3.9-drift-circo
# Phi bền (sau E3.8c): token có ít trôi hơn phi_B32 trên cùng suy giảm không
$PY tools/vector_analysis.py drift $CIRR  --cache-dir $DRIVE/cache --method phi:phi_B32_robust \
  --runs $R/E3.8c-robust-*-cirr --out $R/E3.9-drift-robust-cirr
```
Khoảng 50 cấu hình trên CIRR, mỗi cấu hình chủ yếu tốn thời gian mã hoá lại câu truy vấn. Gửi Chí 3 file `report.md` và hình `drift_vs_metric.png`.

### D12. Độ bền bộ đo thuộc tính; hình minh hoạ (Ngày 11) — E3.6, E3.7
```bash
for D in gauss_noise:2 jpeg:2 low_light:2 color_cast:2; do
  $PY tools/attribute_stats.py --agreement --attr-mask auto --dataset fashioniq --dataset-path $DATA/FashionIQ \
    --degrade $D --cache-dir $DRIVE/cache --out $DRIVE/results/E3.6-agree-${D/:/}-fiq
done
$PY tools/attribute_stats.py --agreement --attr-mask auto --dataset fashioniq --dataset-path $DATA/FashionIQ \
  --degrade color_cast:2 --restore gray_world --cache-dir $DRIVE/cache --out $DRIVE/results/E3.6-agree-colorcast2-grayworld-fiq
$PY tools/visualize_degradations.py --dataset cirr --dataset-path $DATA/CIRR --run $DRIVE/results/E1.7d-phiB32-cirr \
  --degrade gauss_noise:2,low_light:2,jpeg:2,color_cast:2 --restore auto --n 8 --out $DRIVE/results/E3.7-figures
```

### D13. Chạy bổ sung; nộp test phi bền (Ngày 12)
- Chạy các lượt bổ sung Chí yêu cầu.
- Nếu Chí quyết định nộp (C18): sinh file nộp cho `phi_B32_robust` (CIRR và CIRCO) bằng `generate_test_submission.py`, tên `robust_cirr`, `robust_circo`; chép ra `$DRIVE/outputs/submissions/`; báo Hòa để nộp cùng đợt.

### D14. Tổng hợp và bàn giao (Ngày 13)
- `tools/collect_results.py` → `summary.md`; gom hình vào `$DRIVE/results/figures_Doanh/`.
- Gửi Chí danh sách: mã thí nghiệm → một câu kết luận.

### D15. Luồng học máy: OTI warm-start trên CIRR val, nếu còn GPU (Ngày 12) — E7.2
Theo [07](07_hoc_may_oti_warm_start.md): chạy OTI warm-start trên CIRR val với 1–2 cấu hình tốt nhất từ E7.1 của Hòa; Chí gửi lệnh cụ thể sau khi xem E7.1. Trước đó cần liên kết khái niệm cho ảnh của CIRR val (một lần):
```bash
$PY src/image_concepts_association.py --clip-model-name ViT-B/32 --dataset cirr --dataset-path $DATA/CIRR --split val --dataset-mode relative
```
Bỏ qua nếu hết GPU; không ảnh hưởng các phần khác.

### D16. Bộ câu GPT: thống kê và sinh bộ câu mới quy mô nhỏ (Ngày 12) — E6.1
Cần: C22 (công cụ thống kê, làm sạch câu) và C23 (notebook sinh câu `colab/gen_phrases.ipynb`). Ý nghĩa các bộ câu P0–P3 và lệnh chi tiết ở [06](06_sinh_lai_cum_tu_gpt.md). 1–3 giờ A100.
- Tạo bộ P1 nếu trên Drive chưa có `outputs/GPT_phrases/GPTNeo27B_clean` (chỉ cần CPU, khoảng 10–20 phút):
  ```bash
  $PY tools/clean_phrases.py --phrases data/GPT_phrases/GPTNeo27B/concept_to_phrases.pkl --out-exp GPTNeo27B_clean
  ```
- E6.1: thống kê chất lượng cho bộ gốc P0 và bộ đã làm sạch P1 bằng `tools/phrases_stats.py`:
  ```bash
  SC=data/similar_concepts/imagenet/test/ViT-B32_top_250_ensembling_image_texts.csv
  $PY tools/phrases_stats.py --phrases data/GPT_phrases/GPTNeo27B/concept_to_phrases.pkl --similar-concepts $SC --out $DRIVE/results/E6.1-stats-P0
  $PY tools/phrases_stats.py --phrases data/GPT_phrases/GPTNeo27B_clean/concept_to_phrases.pkl --similar-concepts $SC --out $DRIVE/results/E6.1-stats-P1
  ```
- Sinh bộ câu mới (P2 hoặc P3, theo 06) bằng `colab/gen_phrases.ipynb`, 64 câu mỗi khái niệm: lượt 1 `SCOPE = "circo_val"` (vài nghìn khái niệm, cho H18); lượt 2 `SCOPE = "s0"` (gần như mọi khái niệm, cho D17) chỉ khi thời gian đo ở lượt 1 cho thấy còn đủ GPU. Ghi lại thời gian sinh để ước lượng chi phí cho bộ đầy đủ.
- Chạy E6.1 cho bộ mới; báo Hòa tên thư mục bộ câu để chạy H18.
- Nếu trùng giờ GPU với D13, D15: ưu tiên D13 trừ khi Chí báo khác.

### D17. Bộ câu GPT: train phi trên shard s0 với bộ câu mới (Ngày 13) — E6.4
Cần: C22. Cấu hình giống T1 của Hòa (`phi_scale_s0`), chỉ đổi bộ câu; baseline chính là T1, không train lại. Mỗi lượt 2–3 giờ, có thể chạy song song 2 lượt trên A100. Thay `<bộ P1>`, `<bộ mới>` bằng tên thư mục bộ câu trong `data/GPT_phrases/`.
```bash
for P in <bộ P1> <bộ mới>; do
  mkdir -p $DRIVE/results/E6.4-$P-s0
  nohup $PY src/train_phi.py --exp-name E6.4-$P-s0 --clip-model-name ViT-B/32 \
    --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR \
    --oti-exp-name oti_B32_s0 --train-on-oti-names --num-epochs 100 \
    --validation-frequency 5 --save-training --gpt-exp-name $P > $DRIVE/results/E6.4-$P-s0/train.log 2>&1 &
done
```
Đánh giá trên CIRR val và CIRCO val giống E1.14 (đổi `--exp-name`), so với T1 bằng bootstrap. Nếu không xong trong Ngày 13, báo Chí để ghi là kết quả sơ bộ.

## 5. Khi gặp lỗi

| Hiện tượng | Làm gì |
|---|---|
| Báo tham số không hợp lệ (`unrecognized arguments`) | Code mới chưa về: chạy lại ô 2; vẫn lỗi thì báo Chí |
| `FileNotFoundError` về dữ liệu | Chạy lại ô 4 đúng tên tập (thêm `imagenet` khi train) |
| Hết bộ nhớ GPU | Không chạy quá 2 lượt train cùng lúc; khởi động lại thời gian chạy |
| Colab ngắt khi đang train | Chạy lại ô 1–5, rồi chạy đúng lệnh train cũ, thêm `--resume` |
| Train chạy nhưng log không đổi lâu | `nvidia-smi` xem GPU có chạy không; gửi Chí 30 dòng cuối log |
| Số khác kỳ vọng | Không chạy tiếp các bước sau; gửi Chí lệnh, `metrics.json` và 30 dòng cuối log |

## 6. Mẫu báo cáo cuối ngày

```text
[Doanh – Ngày N]
Đã xong: E…, E… (results/…)
Đang train: … (epoch …/…)
Số chính: …
Lỗi/vướng: …
Ngày mai: D…
GPU đã dùng hôm nay: … giờ (loại: …)
```
