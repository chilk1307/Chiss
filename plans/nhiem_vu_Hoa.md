# Nhiệm vụ của Hòa: chạy đánh giá, thí nghiệm và train trên Colab

**Vai trò:** chạy trên Google Colab các lượt đánh giá, nộp test, thí nghiệm của đóng góp chính ([02](02_kiem_chung_thuoc_tinh.md)), các lượt train phân tích lượng dữ liệu và một phần luồng học máy ([07](07_hoc_may_oti_warm_start.md), [08](08_hoc_may_cai_tien_phi.md)). Hòa **không sửa code**: gặp lỗi thì gửi log cho Chí.

**Bàn giao cho Chí (hạn Ngày 13, 29/10):**
1. Bảng baseline 3 tập (CLIP và xử lý ảnh cổ điển), có khoảng tin cậy.
2. Điểm test chính thức (CIRR, CIRCO) của SEARLE gốc, phi_B32 và cấu hình cuối.
3. Kết quả thí nghiệm thuộc tính E2.0–E2.9.
4. Ví dụ định tính, bản đồ che khuất, phân tích vector ảnh – vector từ (E1.15).
5. Kết quả train phân tích lượng dữ liệu, luồng học máy và bộ câu GPT (nếu đã chạy).

## 0. Cập nhật 18/10: code đã xong, làm sớm được

Toàn bộ code mà các việc H1–H18 cần đã có trên GitHub. Lịch ở mục 3 là lịch gốc; hạn chế duy nhất khi làm sớm là quota GPU. Thứ tự gợi ý:

1. **H1, H2, H4** (không cần GPU mạnh).
2. **H5**: làm sớm nhất có thể, vì **Doanh cần cache của H5 để làm D3**; đây cũng là nghiệm thu bộ module mới (C4). Số CIRR lệch quá 0,05 so với bảng kỳ vọng thì dừng và báo Chí.
3. **H6, H7, H8, H9, H17**: code đã đủ. H9 càng sớm càng tốt để Chí họp mốc M3.
4. **H10, H11** chờ Chí chọn trọng số sau khi xem kết quả H9.
5. **T1, H16** (A100, chạy nền nhiều giờ): chỉ chạy khi còn quota; dùng chung A100 với Doanh nên báo nhau trước khi chạy.
6. **H15** làm được ngay sau H17; **H18** chờ Doanh sinh xong bộ câu ở D16.

Lần chạy notebook đầu tiên chưa ai thử trên Colab thật: nếu một ô lỗi, gửi Chí tên ô và 30 dòng cuối log.

## 1. Chuẩn bị (Ngày 1, làm một lần)

1. **Drive:** mở link thư mục `SEARLE` Chí chia sẻ → chuột phải → *Sắp xếp* → *Thêm lối tắt* → chọn *Drive của tôi*. Nhờ vậy trên Colab thư mục luôn nằm ở `/content/drive/MyDrive/SEARLE`, giống máy của Chí.
2. **Notebook:** trên Colab chọn *Tệp → Mở sổ tay → GitHub*, nhập `chilk1307/XLA_SEARLE`, mở `SEARLE-main/colab/SEARLE_runner.ipynb`, rồi *Lưu bản sao vào Drive*. Ở ô 0 đặt `RUNNER = "Hoa"`.
3. Dùng **tài khoản Colab của chính mình** (quota GPU tính theo tài khoản).
4. **Server test:** đọc hướng dẫn nộp bài trên trang CIRR (https://cirr.cecs.anu.edu.au/) và CIRCO (https://circo.micc.unifi.it/); đăng ký nếu trang yêu cầu.

## 2. Quy trình mỗi phiên Colab

1. *Thời gian chạy → Thay đổi loại thời gian chạy:* chọn **T4 hoặc L4** khi đánh giá, **A100** khi train.
2. Chạy lần lượt ô 1 (gắn Drive), ô 2 (lấy code mới nhất), ô 3 (cài môi trường, vài phút), ô 4 (giải nén dữ liệu cần dùng ra `/content/data`), ô 5 (kiểm tra GPU).
3. Mỗi ô lệnh bắt đầu bằng 3 dòng sau, rồi dán lệnh của nhiệm vụ (mục 4):
   ```bash
   %%bash
   source /content/env.sh        # khai báo DRIVE, DATA, PY, RUNNER
   cd /content/XLA_SEARLE/SEARLE-main
   ```
4. Sau mỗi lượt chạy: mở `$DRIVE/results/<mã>/metrics.json` xem đã có số chưa; kiểm tra dòng mới trong `$DRIVE/results/runs_Hoa.csv`.
5. Cuối ngày gửi báo cáo theo mẫu ở mục 6.

## 3. Lịch theo ngày

| Ngày | Ngày tháng | Việc | Cần có từ Chí | Mã thí nghiệm | GPU |
|---|---|---|---|---|---|
| 1 | T7 17/10 | H1 dựng Colab, đóng gói dữ liệu; H2 kiểm tra dữ liệu | C2 | E1.2 | không cần |
| 2 | CN 18/10 | H4 sinh file test của SEARLE gốc; nộp test của SEARLE gốc và phi_B32 (H3 đã xong) | C3 (file nộp phi_B32 trong git) | B8–B11 | T4/L4 |
| 3 | T2 19/10 | H5 kiểm tra không hồi quy, chạy lại baseline | C4 | E1.7a–f | T4/L4 |
| 4 | T3 20/10 | H6 baseline CLIP và CBIR cổ điển | C5, C8 | E1.8, E1.9 | T4/L4 |
| 5 | T4 21/10 | H7 kiểm tra dữ liệu bằng công cụ, khoảng tin cậy, gom kết quả | C7 | E1.2, E1.4 | không cần |
| 6 | T5 22/10 | H8 phân loại truy vấn, giữ – sửa, vector ảnh – vector từ; T1 train nền (tuỳ chọn) | C10, C11, C24 | E1.10, E1.11, E1.15a–b, E1.14a | T4/L4; A100 nếu train |
| 7 | T6 23/10 | H9 thống kê thuộc tính, đồng thuận bộ đo → gửi Chí trước 17h; H17 liên kết khái niệm CIRCO val, OTI không dùng câu GPT | C12 | E2.0, E2.2, E7.0, E6.2-Pλ0 | T4/L4 (cho H17) |
| 8 | T7 24/10 | H10 quét xếp hạng lại; T2 train nền (tuỳ chọn) | C13 | E2.3, E1.14b | T4/L4 (A100 nếu train) |
| 9 | CN 25/10 | H10 (tiếp): mặt nạ, phủ định; đánh giá T1, T2 | C13 | E2.4, E2.5, E1.14 | T4/L4 |
| 10 | T2 26/10 | H11 kết hợp, tổng quát hoá, kết hợp điểm, prompt; H16 train nền (học máy) | C15, C10 | E2.6, E2.7, E4.4, E4.5, E8.2a | T4/L4 + A100 |
| 11 | T3 27/10 | H12 ví dụ định tính, bản đồ che khuất; H15 OTI warm-start trên CIRCO val (học máy) | C16, C21 | E1.12, E1.13, E2.8, E7.1a–d | T4/L4 |
| 12 | T4 28/10 | H13 nộp test cấu hình cuối; khoảng tin cậy bảng chính | C18 | E2.9 | T4/L4 |
| 13 | T5 29/10 | H14 tổng hợp bảng, hình → bàn giao Chí; H18 OTI với bộ câu GPT mới trên CIRCO val | C22; bộ câu của Doanh (D16) | E6.2 | T4/L4 (cho H18) |
| 14 | T6 30/10 | Dự phòng: chạy lại khi cần; đối chiếu số trong báo cáo với `results/` | — | — | tuỳ |

## 4. Chi tiết từng nhiệm vụ

### H1. Dựng Colab và đóng gói dữ liệu (Ngày 1)
- Làm mục 1, rồi chạy ô 1–5 của notebook.
- **Chỉ khi Chí xác nhận Drive còn đủ chỗ (C1):** đóng gói dataset thành một file `.tar` để các phiên sau giải nén nhanh (FashionIQ đã có `FashionIQ.tar`, bỏ qua):
  ```bash
  cd $DRIVE/datasets
  tar -cf CIRR.tar CIRR
  tar -cf CIRCO.tar CIRCO      # khoảng 19 GB ảnh, lâu: để chạy trong lúc làm H2
  ```
- Chạy thử (code trên GitHub đã là bản mới; đây cũng là nghiệm thu bộ chạy Colab):
  ```bash
  $PY src/validate.py --eval-type searle --dataset cirr --dataset-path $DATA/CIRR
  ```
  Kỳ vọng: `cirr_recall_at1 = 25.66` (±0.05). Ghi lại thời gian chạy.

### H2. Kiểm tra dữ liệu nhanh (Ngày 1) — E1.2
Cách nhanh: chạy ô "Kiểm tra dữ liệu (E1.2)" của notebook (gọi `tools/check_datasets.py`, ghi `$DRIVE/results/E1.2-dataset-check/report.md`). Nếu ô đó lỗi, dùng đoạn Python dưới đây (chạy trong một ô **Python**, không phải `%%bash`):
```python
import json, os
DATA = "/content/data"
# CIRR
for s in ["val", "test1"]:
    m = json.load(open(f"{DATA}/CIRR/cirr/image_splits/split.rc2.{s}.json"))
    ok = sum(os.path.exists(os.path.join(f"{DATA}/CIRR", p)) for p in m.values())
    print("CIRR", s, "ảnh khai báo:", len(m), "có file:", ok)
# CIRCO
info = json.load(open(f"{DATA}/CIRCO/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json"))
need = {im["file_name"] for im in info["images"]}
have = set(os.listdir(f"{DATA}/CIRCO/COCO2017_unlabeled/unlabeled2017"))
print("CIRCO ảnh khai báo:", len(need), "có file:", len(need & have), "thiếu:", len(need - have))
# FashionIQ
have = {f[:-4] for f in os.listdir(f"{DATA}/FashionIQ/images")}
for t in ["dress", "shirt", "toptee"]:
    caps = json.load(open(f"{DATA}/FashionIQ/captions/cap.{t}.val.json"))
    ok = sum(c["candidate"] in have and c["target"] in have for c in caps)
    names = json.load(open(f"{DATA}/FashionIQ/image_splits/split.{t}.val.json"))
    print("FIQ", t, "truy vấn:", len(caps), "đủ ảnh:", ok, "| gallery:", len(names), "có file:", sum(n in have for n in names))
```
- Chép kết quả vào `$DRIVE/results/E1.2-dataset-check/report.md`, gửi Chí.
- Nếu CIRCO thiếu ảnh: báo Chí ngay, vì ảnh hưởng cả val lẫn file nộp test.

### H3. File subset CIRR (đã xong)
`subset_phi_B32_cirr.json` đã có trong `outputs/submissions/` của repo (`metric = recall_subset`, 4.148 truy vấn × 3 kết quả). Không cần làm gì.

### H4. Sinh file test của SEARLE gốc, nộp test (Ngày 2) — B8–B11
File nộp của phi_B32 đã đủ 3 file, nằm trong repo sau khi Chí đưa `outputs/submissions/` vào git (C3): `/content/XLA_SEARLE/outputs/submissions/`. Chỉ cần sinh file cho SEARLE gốc:
```bash
$PY src/generate_test_submission.py --submission-name searle_official_cirr  --eval-type searle --dataset cirr  --dataset-path $DATA/CIRR
$PY src/generate_test_submission.py --submission-name searle_official_circo --eval-type searle --dataset circo --dataset-path $DATA/CIRCO
# Chép ra Drive
mkdir -p $DRIVE/outputs/submissions
cp data/test_submissions/cirr/*.json data/test_submissions/circo/*.json $DRIVE/outputs/submissions/
```
- Nộp lên server:
  - CIRR: `phi_B32_cirr.json`, `subset_phi_B32_cirr.json`, `searle_official_cirr.json`, `subset_searle_official_cirr.json`;
  - CIRCO: `phi_B32_circo.json`, `searle_official_circo.json`.
  - Các file của phi_B32 lấy ở `/content/XLA_SEARLE/outputs/submissions/`; file của SEARLE gốc lấy ở `$DRIVE/outputs/submissions/`.
- Gửi điểm nhận được cho Chí để ghi vào nhật ký (B8–B11).

### H5. Kiểm tra không hồi quy, chạy lại baseline (Ngày 3) — E1.7a–f
Cần: C4.
```bash
OPTS="--cache-dir $DRIVE/cache"
PHI="--eval-type phi --exp-name phi_B32 --phi-checkpoint-name phi_last.pt"
$PY src/validate.py --eval-type searle --dataset cirr      --dataset-path $DATA/CIRR      $OPTS --output-dir $DRIVE/results/E1.7a-searle-cirr
$PY src/validate.py --eval-type searle --dataset circo     --dataset-path $DATA/CIRCO     $OPTS --output-dir $DRIVE/results/E1.7b-searle-circo
$PY src/validate.py --eval-type searle --dataset fashioniq --dataset-path $DATA/FashionIQ $OPTS --output-dir $DRIVE/results/E1.7c-searle-fiq
$PY src/validate.py $PHI --dataset cirr      --dataset-path $DATA/CIRR      $OPTS --output-dir $DRIVE/results/E1.7d-phiB32-cirr
$PY src/validate.py $PHI --dataset circo     --dataset-path $DATA/CIRCO     $OPTS --output-dir $DRIVE/results/E1.7e-phiB32-circo
$PY src/validate.py $PHI --dataset fashioniq --dataset-path $DATA/FashionIQ $OPTS --output-dir $DRIVE/results/E1.7f-phiB32-fiq
```
- Kỳ vọng CIRR: SEARLE R@1 = 25.66, phi_B32 R@1 = 25.47 (±0.05). Lệch nhiều hơn → **dừng**, báo Chí.
- Lượt CIRCO đầu tiên lâu (trích đặc trưng khoảng 123 nghìn ảnh gallery); các lượt sau dùng cache.
- **Báo nhóm khi xong**: Doanh cần cache này để bắt đầu D3.
- So file nộp test sinh bằng code mới với file cũ:
  ```bash
  $PY src/generate_test_submission.py --submission-name check_phi_B32_cirr --eval-type phi --exp-name phi_B32 --phi-checkpoint-name phi_last.pt --dataset cirr --dataset-path $DATA/CIRR
  ```
  rồi trong ô Python:
  ```python
  import json
  a = json.load(open("/content/XLA_SEARLE/SEARLE-main/data/test_submissions/cirr/check_phi_B32_cirr.json"))
  b = json.load(open("/content/XLA_SEARLE/outputs/submissions/phi_B32_cirr.json"))
  keys = [k for k in a if k not in ("version", "metric")]
  print(sum(a[k][:10] == b[k][:10] for k in keys), "/", len(keys), "truy vấn có top-10 trùng")
  ```
  Kỳ vọng ≥ 99% (khác loại GPU có thể làm lệch vài truy vấn có điểm sát nhau).

### H6. Baseline CLIP và CBIR cổ điển (Ngày 4) — E1.8, E1.9
Cần: C5, C8.
```bash
OPTS="--cache-dir $DRIVE/cache"
PHI="--eval-type phi --exp-name phi_B32 --phi-checkpoint-name phi_last.pt"
# Baseline CLIP
for T in image text sum; do
  $PY src/validate.py --eval-type $T --dataset cirr      --dataset-path $DATA/CIRR      $OPTS --output-dir $DRIVE/results/E1.8-$T-cirr
  $PY src/validate.py --eval-type $T --dataset circo     --dataset-path $DATA/CIRCO     $OPTS --exclude-reference --output-dir $DRIVE/results/E1.8-$T-circo-ex
  $PY src/validate.py --eval-type $T --dataset fashioniq --dataset-path $DATA/FashionIQ $OPTS --exclude-reference --output-dir $DRIVE/results/E1.8-$T-fiq-ex
done
# SEARLE gốc và phi_B32 cũng chạy với --exclude-reference trên CIRCO, FIQ để so công bằng
$PY src/validate.py --eval-type searle --dataset circo     --dataset-path $DATA/CIRCO     $OPTS --exclude-reference --output-dir $DRIVE/results/E1.8-searle-circo-ex
$PY src/validate.py --eval-type searle --dataset fashioniq --dataset-path $DATA/FashionIQ $OPTS --exclude-reference --output-dir $DRIVE/results/E1.8-searle-fiq-ex
$PY src/validate.py $PHI --dataset circo     --dataset-path $DATA/CIRCO     $OPTS --exclude-reference --output-dir $DRIVE/results/E1.8-phiB32-circo-ex
$PY src/validate.py $PHI --dataset fashioniq --dataset-path $DATA/FashionIQ $OPTS --exclude-reference --output-dir $DRIVE/results/E1.8-phiB32-fiq-ex
# CBIR cổ điển (chạy CPU, vài phút với CIRR và FIQ)
for F in color lbp hog gabor all; do
  $PY src/validate.py --eval-type cbir --cbir-features $F --dataset cirr      --dataset-path $DATA/CIRR      $OPTS --output-dir $DRIVE/results/E1.9-cbir-$F-cirr
  $PY src/validate.py --eval-type cbir --cbir-features $F --dataset fashioniq --dataset-path $DATA/FashionIQ $OPTS --exclude-reference --output-dir $DRIVE/results/E1.9-cbir-$F-fiq-ex
done
# CBIR + văn bản CLIP
$PY src/validate.py --eval-type cbir --cbir-features all --cbir-text-weight 1.0 --dataset cirr --dataset-path $DATA/CIRR $OPTS --output-dir $DRIVE/results/E1.9-cbir-text-cirr
```
- CBIR trên CIRCO: chỉ chạy nếu còn thời gian (tính đặc trưng khoảng 123 nghìn ảnh bằng CPU, 1–2 giờ).
- Kỳ vọng: SEARLE, phi_B32 > `sum` > `text`, `image`; CBIR thấp nhất. Nếu khác thì vẫn ghi lại và báo Chí.

### H7. Kiểm tra dữ liệu bằng công cụ, khoảng tin cậy, gom kết quả (Ngày 5) — E1.2, E1.4
Cần: C7.
```bash
for DS in cirr circo fashioniq; do
  P=$DATA/CIRR; [ $DS = circo ] && P=$DATA/CIRCO; [ $DS = fashioniq ] && P=$DATA/FashionIQ
  $PY tools/check_datasets.py --dataset $DS --dataset-path $P --out $DRIVE/results/E1.2-dataset-check
done
R=$DRIVE/results
$PY tools/bootstrap_ci.py --baseline $R/E1.7d-phiB32-cirr --runs $R/E1.7a-searle-cirr $R/E1.8-*-cirr $R/E1.9-*-cirr --out $R/E1.4-ci-cirr
$PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-circo-ex --runs $R/E1.8-*-circo-ex --out $R/E1.4-ci-circo
$PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-fiq-ex --runs $R/E1.8-*-fiq-ex $R/E1.9-*-fiq-ex --out $R/E1.4-ci-fiq
$PY tools/collect_results.py --results-dir $R --out $R/summary.md
```
Gửi Chí `summary.md` và 3 bảng khoảng tin cậy.

### H8. Phân loại truy vấn, phân tích giữ – sửa, vector ảnh – vector từ (Ngày 6) — E1.10, E1.11, E1.15a–b
Cần: C11, C24.
```bash
R=$DRIVE/results
$PY tools/query_taxonomy.py --dataset cirr --dataset-path $DATA/CIRR --runs $R/E1.7a-searle-cirr $R/E1.7d-phiB32-cirr $R/E1.8-*-cirr --out $R/E1.10-taxonomy-cirr
$PY tools/query_taxonomy.py --dataset circo --dataset-path $DATA/CIRCO --runs $R/E1.8-*-circo-ex --out $R/E1.10-taxonomy-circo
$PY tools/query_taxonomy.py --dataset fashioniq --dataset-path $DATA/FashionIQ --runs $R/E1.8-*-fiq-ex --out $R/E1.10-taxonomy-fiq
$PY tools/keep_edit_analysis.py --dataset cirr --dataset-path $DATA/CIRR --runs $R/E1.7d-phiB32-cirr $R/E1.8-sum-cirr --cache-dir $DRIVE/cache --out $R/E1.11-keepedit-cirr
$PY tools/keep_edit_analysis.py --dataset circo --dataset-path $DATA/CIRCO --runs $R/E1.7e-phiB32-circo $R/E1.8-sum-circo-ex --cache-dir $DRIVE/cache --out $R/E1.11-keepedit-circo
```
**E1.15a, E1.15b: vector ảnh và vector từ** (cần C24; ý nghĩa ở [01](01_nen_tang_danh_gia.md) mục 1.15). Dùng lại cache của H5 nên chỉ vài phút mỗi lệnh; `--snap-eval` chạy thêm 2 lượt đánh giá cho mỗi phương pháp.
```bash
R=$DRIVE/results
$PY tools/vector_analysis.py tokens --dataset cirr  --dataset-path $DATA/CIRR  --cache-dir $DRIVE/cache --sources searle phi:phi_B32 --snap-eval --out $R/E1.15a-tokens-cirr
$PY tools/vector_analysis.py tokens --dataset circo --dataset-path $DATA/CIRCO --cache-dir $DRIVE/cache --sources searle phi:phi_B32 --snap-eval --out $R/E1.15a-tokens-circo
$PY tools/vector_analysis.py gap    --dataset cirr  --dataset-path $DATA/CIRR  --cache-dir $DRIVE/cache --sources phi:phi_B32 --out $R/E1.15b-gap-cirr
$PY tools/vector_analysis.py gap    --dataset circo --dataset-path $DATA/CIRCO --cache-dir $DRIVE/cache --sources phi:phi_B32 --out $R/E1.15b-gap-circo
```
Gửi Chí 4 file `report.md` cùng hình `tokens.png`, `gap.png`.

### T1. (Tuỳ chọn) Train phi trên 1 shard, chạy nền từ Ngày 6 — E1.14a
Cần: C10 (đã xong); runtime **A100**; ô 4 thêm `imagenet`. Khoảng 2–3 giờ. Đây cũng là E8.1a của [08](08_hoc_may_cai_tien_phi.md): chỉ chạy một lần, kết quả dẫn chiếu ở cả hai nơi. `--features-cache-dir` lưu đặc trưng CLIP của ảnh train để các lượt train sau (của Doanh và Hòa, cùng tập ảnh) khỏi trích lại.
```bash
mkdir -p $DRIVE/results/E1.14a-train-s0
nohup $PY src/train_phi.py --exp-name phi_scale_s0 --clip-model-name ViT-B/32 \
  --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR \
  --oti-exp-name oti_B32_s0 --train-on-oti-names --num-epochs 100 \
  --validation-frequency 5 --save-training --features-cache-dir $DRIVE/cache/train_features \
  > $DRIVE/results/E1.14a-train-s0/train.log 2>&1 &
```
Lượt nạp đặc trưng từ cache bỏ qua bước trích đặc trưng nên chuỗi số ngẫu nhiên hơi khác lượt tự trích (lượt đầu tiên, tạo ra cache, thì giống lượt không dùng cache). Khác biệt này chỉ ở mức nhiễu huấn luyện, nhưng để so sánh công bằng thì mọi lượt train đem so với T1 (D17, H16...) nên cùng dùng `--features-cache-dir $DRIVE/cache/train_features`.
- Theo dõi: `tail -n 3 $DRIVE/results/E1.14a-train-s0/train.log`.
- Colab ngắt: chạy lại ô 1–5, rồi chạy đúng lệnh trên, thêm `--resume`.

### H9. Thống kê thuộc tính, đồng thuận bộ đo (Ngày 7) — E2.0, E2.2
Cần: C12 (đã xong). **Gửi kết quả cho Chí trước 17h** để họp mốc M3. E2.0 chỉ đọc câu mô tả nên không cần GPU. Số sơ bộ Chí đo trên file chú thích công khai (truy vấn có ít nhất một ràng buộc): FIQ 90,7%, CIRR 36,3%, CIRCO 25,5%; số của Hòa trên tập con có đủ ảnh sẽ hơi khác. E2.2 báo, với mỗi họ thuộc tính, tỉ lệ đồng thuận giữa bộ đo và câu mô tả, kèm mức ngẫu nhiên để so.
```bash
R=$DRIVE/results
$PY tools/attribute_stats.py --dataset cirr      --dataset-path $DATA/CIRR      --out $R/E2.0-stats-cirr
$PY tools/attribute_stats.py --dataset fashioniq --dataset-path $DATA/FashionIQ --out $R/E2.0-stats-fiq
$PY tools/attribute_stats.py --dataset circo     --dataset-path $DATA/CIRCO     --out $R/E2.0-stats-circo
$PY tools/attribute_stats.py --agreement --attr-mask auto --dataset cirr      --dataset-path $DATA/CIRR      --cache-dir $DRIVE/cache --out $R/E2.2-agree-cirr
$PY tools/attribute_stats.py --agreement --attr-mask auto --dataset fashioniq --dataset-path $DATA/FashionIQ --cache-dir $DRIVE/cache --out $R/E2.2-agree-fiq
$PY tools/attribute_stats.py --agreement --attr-mask auto --dataset circo     --dataset-path $DATA/CIRCO     --cache-dir $DRIVE/cache --out $R/E2.2-agree-circo
```

### H10. Quét xếp hạng lại theo thuộc tính (Ngày 8–9) — E2.3, E2.4, E2.5
Cần: C13.
```bash
PHI="--eval-type phi --exp-name phi_B32 --phi-checkpoint-name phi_last.pt --cache-dir $DRIVE/cache"
# E2.3: từng họ thuộc tính × trọng số, trên CIRR val và FIQ val (40 lượt, mỗi lượt vài phút)
for FAM in color brightness colorfulness pattern zoom; do
  for W in 0.05 0.1 0.2 0.3; do
    $PY src/validate.py $PHI --dataset cirr --dataset-path $DATA/CIRR \
      --attr-rerank --attr-families $FAM --attr-weight $W --output-dir $DRIVE/results/E2.3-$FAM-w$W-cirr
    $PY src/validate.py $PHI --dataset fashioniq --dataset-path $DATA/FashionIQ --exclude-reference \
      --attr-rerank --attr-families $FAM --attr-weight $W --output-dir $DRIVE/results/E2.3-$FAM-w$W-fiq
  done
done
$PY tools/collect_results.py --results-dir $DRIVE/results --out $DRIVE/results/summary.md
```
- Gửi Chí `summary.md`; Chí chọn trọng số tốt nhất cho mỗi họ.
- **Ngày 9**, với cấu hình Chí gửi:
  - E2.3 (tiếp): `--attr-topk 50` và `--attr-topk 100`;
  - E2.4: `--attr-mask none`, `otsu`, `grabcut`, `saliency`;
  - E2.5: `--attr-negation on` và `off`.
  Đặt tên thư mục theo mẫu `E2.4-mask-otsu-cirr`, `E2.5-neg-off-fiq`.
- **Ngày 9**, đánh giá T1, T2 (nếu đã train):
  ```bash
  for P in phi_scale_s0 phi_scale_s01; do
    $PY src/validate.py --eval-type phi --exp-name $P --phi-checkpoint-name phi_last.pt --dataset cirr --dataset-path $DATA/CIRR --cache-dir $DRIVE/cache --output-dir $DRIVE/results/E1.14-$P-cirr
  done
  ```

### T2. (Tuỳ chọn) Train phi trên 2 shard, chạy nền từ Ngày 8 — E1.14b
Cần: C10; **A100**; khoảng 4–5 giờ. Đây cũng là E8.1b của [08](08_hoc_may_cai_tien_phi.md).
```bash
$PY tools/merge_oti.py --shards oti_B32_s0,oti_B32_s1 --out oti_B32_s01
mkdir -p $DRIVE/results/E1.14b-train-s01
nohup $PY src/train_phi.py --exp-name phi_scale_s01 --clip-model-name ViT-B/32 \
  --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR \
  --oti-exp-name oti_B32_s01 --train-on-oti-names --num-epochs 100 \
  --validation-frequency 5 --save-training > $DRIVE/results/E1.14b-train-s01/train.log 2>&1 &
```

### H11. Kết hợp và tổng quát hoá (Ngày 10) — E2.6, E2.7, E4.4, E4.5
Cần: C15.
- **E2.6:** chạy cấu hình kết hợp mọi họ thuộc tính (Chí gửi lệnh) trên CIRR val và FIQ val.
- **E4.4:** kết hợp điểm ảnh + văn bản trên CIRR val:
  ```bash
  for A in 0.05 0.1 0.2 0.3 0.5; do
    $PY src/validate.py $PHI --dataset cirr --dataset-path $DATA/CIRR --fusion-weight $A --output-dir $DRIVE/results/E4.4-fusion-a$A-cirr
  done
  ```
- **E4.5:** `$PY src/validate.py $PHI --dataset cirr --dataset-path $DATA/CIRR --prompt-set ensemble --output-dir $DRIVE/results/E4.5-ensemble-cirr`
- **E2.7:** cấu hình cuối (Chí gửi) trên CIRCO val và FIQ val, **không chỉnh lại tham số**.

### H12. Ví dụ định tính, bản đồ che khuất (Ngày 11) — E1.12, E1.13, E2.8
Cần: C16. Thay `<cuối>` bằng mã thư mục cấu hình cuối mà Chí gửi.
```bash
R=$DRIVE/results
$PY tools/visualize_queries.py --run $R/E1.7d-phiB32-cirr --compare $R/<cuối> --dataset cirr --dataset-path $DATA/CIRR --n 20 --seed 0 --out $R/E1.12-examples-cirr
$PY tools/occlusion_map.py --run $R/E1.7d-phiB32-cirr --dataset cirr --dataset-path $DATA/CIRR --n 10 --seed 0 --out $R/E1.13-occlusion-cirr
$PY tools/visualize_queries.py --explain-attributes --run $R/<cuối> --dataset fashioniq --dataset-path $DATA/FashionIQ --n 20 --seed 0 --out $R/E2.8-explained-fiq
```

### H13. Nộp test cấu hình cuối (Ngày 12) — E2.9
- Chạy `generate_test_submission.py` với đúng các cờ của cấu hình cuối (Chí gửi ở C18) cho CIRR và CIRCO, đặt tên `final_cirr`, `final_circo`; chép ra `$DRIVE/outputs/submissions/`; nộp server; gửi điểm cho Chí.
- Khoảng tin cậy cho bảng chính: `tools/bootstrap_ci.py --baseline <phi_B32 cùng tập> --runs <cấu hình cuối>` trên 3 tập val.

### H14. Tổng hợp và bàn giao (Ngày 13)
- `tools/collect_results.py` → bản `summary.md` mới nhất.
- Gom hình vào `$DRIVE/results/figures_Hoa/`.
- Gửi Chí danh sách: mã thí nghiệm → một câu kết luận.

### H15. Luồng học máy: OTI warm-start trên CIRCO val (Ngày 11) — E7.1a–d
Cần: C21; E7.0 đã chạy ở H17 (Ngày 7), không chạy lại. Ý nghĩa từng cấu hình ở [07](07_hoc_may_oti_warm_start.md). CIRCO val chỉ có 220 truy vấn, mỗi lượt OTI vài phút. Lượt E7.1a (khởi tạo ngẫu nhiên, bộ câu gốc) cũng là E6.2-P0 của [06](06_sinh_lai_cum_tu_gpt.md); H18 sẽ so với kết quả này.
```bash
# E7.1a–d: mỗi lượt lưu token ở nhiều bước trong cùng một lần chạy
OTI="--clip-model-name ViT-B/32 --dataset circo --dataset-path $DATA/CIRCO --split val --snapshot-steps 0,10,25,50,100,200,350"
WARM="--init-phi-exp-name phi_B32 --init-phi-checkpoint phi_last.pt"
$PY src/oti_inversion.py --exp-name E7.1a-random-lr0.02-circo  $OTI --init random --learning-rate 0.02
$PY src/oti_inversion.py --exp-name E7.1b-phi-lr0.02-circo     $OTI --init phi $WARM --learning-rate 0.02
$PY src/oti_inversion.py --exp-name E7.1c-phi-lr0.005-circo    $OTI --init phi $WARM --learning-rate 0.005
$PY src/oti_inversion.py --exp-name E7.1d-searle-lr0.02-circo  $OTI --init searle --learning-rate 0.02
$PY src/oti_inversion.py --exp-name E7.1d-searle-lr0.005-circo $OTI --init searle --learning-rate 0.005
```
Token lưu trong `data/oti_pseudo_tokens/circo/val/<tên>/` (liên kết tới Drive), mỗi bước một cặp file `oti_pseudo_tokens_step<N>.pt` và `ema_oti_pseudo_tokens_step<N>.pt`; thời gian từng batch ở `timing.json`. Sau đó đánh giá từng bước đã lưu, **cả token thường lẫn token EMA** (ví dụ cho E7.1b):
```bash
E=E7.1b-phi-lr0.02-circo
for S in 0 10 25 50 100 200 350; do
  for T in oti_pseudo_tokens ema_oti_pseudo_tokens; do
    $PY src/validate.py --eval-type oti --exp-name $E --token-file ${T}_step$S.pt --dataset circo --dataset-path $DATA/CIRCO \
      --cache-dir $DRIVE/cache --output-dir $DRIVE/results/E7.1b-step$S-$T-circo
  done
done
```
Lặp cho E7.1a, c, d (đổi `E` và tên thư mục kết quả). Gửi Chí bảng kết quả; Chí chọn 1–2 cấu hình cho E7.2 (Doanh, D15). E7.1e (chỉ cosine loss) chạy nếu Chí yêu cầu.

Sau đó chạy E1.15a lần nữa, thêm token OTI (token EMA cuối của mỗi lượt): cho biết token OTI khác token phi bao nhiêu (độ lớn, hướng), và khởi tạo từ phi có giữ token OTI gần phi không.
```bash
$PY tools/vector_analysis.py tokens --dataset circo --dataset-path $DATA/CIRCO --cache-dir $DRIVE/cache \
  --sources phi:phi_B32 oti:E7.1a-random-lr0.02-circo oti:E7.1b-phi-lr0.02-circo oti:E6.2-Plambda0-circo \
  --snap-eval --out $DRIVE/results/E1.15a-tokens-oti-circo
```

### H16. Luồng học máy: train phi với lịch learning rate (chạy nền từ Ngày 10) — E8.2a
Cần: C10; **A100**; khoảng 6–7 giờ. Chi tiết ở [08](08_hoc_may_cai_tien_phi.md).
```bash
mkdir -p $DRIVE/results/E8.2a-train-cosine
nohup $PY src/train_phi.py --exp-name phi_B32_cosine --clip-model-name ViT-B/32 \
  --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR \
  --oti-exp-name oti_B32 --num-epochs 100 --lr-scheduler cosine --warmup-epochs 5 \
  --validation-frequency 5 --save-training > $DRIVE/results/E8.2a-train-cosine/train.log 2>&1 &
```
Ngày 11–12: đánh giá `phi_B32_cosine` trên CIRR val và CIRCO val (giống lệnh E1.7d, E1.7e, đổi `--exp-name`).

### H17. Bộ câu GPT: chuẩn bị CIRCO val và OTI không dùng câu GPT (Ngày 7) — E7.0, E6.2-Pλ0
Không cần code mới. Ý nghĩa ở [06](06_sinh_lai_cum_tu_gpt.md): lượt "không dùng câu GPT" cho biết câu GPT đóng góp bao nhiêu. Dưới 15 phút.
```bash
# E7.0: liên kết khái niệm cho ảnh của CIRCO val (dùng chung cho 06 và 07)
$PY src/image_concepts_association.py --clip-model-name ViT-B/32 --dataset circo --dataset-path $DATA/CIRCO --split val --dataset-mode relative
# E6.2-Pλ0: OTI 350 bước trên CIRCO val, tắt GPT loss (thư mục dùng chữ "Plambda0" thay cho "Pλ0")
$PY src/oti_inversion.py --exp-name E6.2-Plambda0-circo --clip-model-name ViT-B/32 --dataset circo --dataset-path $DATA/CIRCO --split val --lambda_gpt 0
$PY src/validate.py --eval-type oti --exp-name E6.2-Plambda0-circo --dataset circo --dataset-path $DATA/CIRCO --cache-dir $DRIVE/cache --output-dir $DRIVE/results/E6.2-Plambda0-circo
```

### H18. Bộ câu GPT: OTI với các bộ câu mới trên CIRCO val (Ngày 13) — E6.2
Cần: C22 và bộ câu mới Doanh sinh ở D16. Dưới 30 phút. Mỗi bộ câu là một thư mục trong `data/GPT_phrases/` (tên thư mục theo [06](06_sinh_lai_cum_tu_gpt.md)); dùng cùng seed với E7.1a để so sánh công bằng:
```bash
for P in <bộ P1> <bộ mới của D16>; do
  $PY src/oti_inversion.py --exp-name E6.2-$P-circo --clip-model-name ViT-B/32 --dataset circo --dataset-path $DATA/CIRCO --split val --gpt-exp-name $P
  $PY src/validate.py --eval-type oti --exp-name E6.2-$P-circo --dataset circo --dataset-path $DATA/CIRCO --cache-dir $DRIVE/cache --output-dir $DRIVE/results/E6.2-$P-circo
done
```
So với E7.1a (= E6.2-P0) và E6.2-Pλ0 (H17), kèm khoảng tin cậy bootstrap.

## 5. Khi gặp lỗi

| Hiện tượng | Làm gì |
|---|---|
| Báo tham số không hợp lệ (`unrecognized arguments`) | Code mới chưa về: chạy lại ô 2; vẫn lỗi thì báo Chí |
| `FileNotFoundError` về dữ liệu | Chạy lại ô 4 đúng tên tập; kiểm tra lối tắt Drive |
| Hết bộ nhớ GPU (`CUDA out of memory`) | Khởi động lại thời gian chạy; không chạy 2 lượt đánh giá cùng lúc |
| Colab ngắt khi đang train | Chạy lại ô 1–5, rồi chạy đúng lệnh train cũ, thêm `--resume` |
| Số khác kỳ vọng | Không chạy tiếp các bước sau; gửi Chí lệnh, `metrics.json` và 30 dòng cuối `log.txt` |

Khi báo lỗi luôn gửi kèm: mã thí nghiệm, lệnh đã chạy, 30 dòng cuối của log.

## 6. Mẫu báo cáo cuối ngày

```text
[Hòa – Ngày N]
Đã xong: E…, E… (results/…)
Số chính: …
Lỗi/vướng: …
Ngày mai: H…
GPU đã dùng hôm nay: … giờ (loại: …)
```
