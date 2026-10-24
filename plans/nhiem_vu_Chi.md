# Nhiệm vụ (Chí làm một mình)

**Từ 24/10:** Chí làm toàn bộ dự án: code, chạy Colab, báo cáo. File này gộp ba file nhiệm vụ cũ (Chí, Hòa, Doanh). **Mã việc giữ nguyên** để khớp [tien_do.md](tien_do.md):

| Mã | Nội dung | Trước 24/10 |
|---|---|---|
| **C** | Code, ý tưởng, quyết định, demo, báo cáo | Chí |
| **H**, **T** | Đánh giá, nộp test, thí nghiệm thuộc tính, train phân tích lượng dữ liệu | Hòa |
| **D** | Tiền xử lý, độ bền và phục hồi ảnh, train phi bền | Doanh |

Lệnh của **mọi lượt train phi** (T1, T2, D7–D10, H16, D17) nằm ở [nhiem_vu_train_Colab.md](nhiem_vu_train_Colab.md); file này chỉ dẫn chiếu tới đó.

**Nguyên tắc khi sửa code** (giữ từ trước):
1. Dùng đúng tên tham số trong giao kèo lệnh chạy ([01](01_nen_tang_danh_gia.md) mục 1.3).
2. Mọi tham số mới mặc định giữ hành vi cũ; sau mỗi thay đổi, CIRR val phải ra đúng số cũ.
3. Code mới đặt trong file mới; mọi file sửa hoặc thêm đều ghi vào `CHANGES.md`.
4. Sửa xong thì đẩy lên GitHub **`chilk1307/Chiss`** (repo mới; notebook clone từ đây) trước khi chạy trên Colab.

## 1. Tình trạng (24/10, Ngày 8)

| Nhóm việc | Xong | Còn lại |
|---|---|---|
| Code (C) | C3, C5–C16, C21–C24 | C1, C2, C4 chờ nghiệm thu; hiệu chỉnh `restore_rules.json`, điền `restore_oracle.json`; C17–C20 |
| Đánh giá, thuộc tính (H) | H3 | H1, H2, H4–H18; H5 dừng ở E1.7a (T4 ra 25.59, mốc 25.66) |
| Độ bền, tiền xử lý (D) | D1, D2 | D3–D17 |
| Mốc | — | M1, M2, M3 quá hạn; M4 hạn 27/10; M5 hạn 30/10 |

**Dữ liệu:** `datasets/` trên Drive đã có 4 thư mục giải nén đủ ảnh (`CIRR`, `CIRCO`, `FashionIQ`, `ImageNet1K`). `colab/prepare_data.sh` dùng thẳng các thư mục này; riêng ImageNet vẫn giải nén tar còn tốt ra đĩa Colab, vì train đọc cả 100.000 ảnh.

## 2. Lịch còn lại (Ngày 8–14)

Mỗi ngày mở **một runtime A100**: train chạy nền (`nohup`), đánh giá chạy ở ô khác trong lúc chờ. Như vậy mọi số đều đo trên cùng một loại GPU.

| Ngày | Ngày tháng | Train nền (A100) | Đánh giá, thí nghiệm | Code, quyết định |
|---|---|---|---|---|
| 8 | T7 24/10 | T1; D7 đặc trưng ảnh xấu s0 | **H5** (gỡ chặn bằng E1.7a trên A100, rồi E1.7b–f); H2/H7 kiểm tra dữ liệu; H4 sinh file test SEARLE gốc | Đẩy code lên `Chiss`; C1 dọn Drive |
| 9 | CN 25/10 | D7–D8 sàng lọc 3 bản phi bền | H6 baseline; **H9** E2.0, E2.2 (mốc M3 muộn); D5 E3.1; D6 E3.5 | Họp M3 (tự quyết); hiệu chỉnh `restore_rules.json` |
| 10 | T2 26/10 | D9 đặc trưng 100k → **D10** phi bền; H16 cosine | D9 đánh giá sàng lọc; H7 khoảng tin cậy; **H10** E2.3; D3, D4 (E4.1–E4.3); D8 E3.2 | Chọn biến thể phi bền; điền `restore_oracle.json` |
| 11 | T3 27/10 | D10, H16 (tiếp) | H10 (tiếp) E2.4, E2.5; **H11**; D10 E3.3; H8 | **C17** demo; mốc M4 |
| 12 | T4 28/10 | T2 (tuỳ chọn) | D11 (E3.8c, E3.4, E3.9); D12; H12; H13 nộp test cấu hình cuối; D13 | **C18** chọn cấu hình cuối |
| 13 | T5 29/10 | D17 (tuỳ chọn) | Học máy nếu còn thời gian: H17, H15, D15, D16, H18 | **C19** README, báo cáo, slide |
| 14 | T6 30/10 | — | Chạy lại lượt hỏng | **C20** video, kiểm tra cuối, nộp bài |

**Thiếu thời gian thì cắt theo thứ tự:** học máy (H15, H17, H18, D15–D17, T2, H16) → D6, D12, E1.15 → T1. **Không cắt:** H5, H4, H6–H7, H9–H11, D3–D5, D8, D10–D11, C17–C20.

## 3. Colab: chuẩn bị và quy trình mỗi phiên

**Một lần:**
1. Thư mục `SEARLE` của Chí nằm sẵn ở `/content/drive/MyDrive/SEARLE` (`datasets/`, `outputs/`, `cache/`, `results/`).
2. *Tệp → Mở sổ tay → GitHub* → `chilk1307/Chiss` → `SEARLE-main/colab/SEARLE_runner.ipynb` → *Lưu bản sao vào Drive*. Ô 0: `RUNNER = "Chi"`, `DATASETS = "cirr circo fiq imagenet"` (bỏ `imagenet` vào ngày không train).
3. Trang nộp test: CIRR https://cirr.cecs.anu.edu.au/, CIRCO https://circo.micc.unifi.it/ (đăng ký nếu trang yêu cầu).

**Mỗi phiên:** chọn A100 → chạy ô 0–5 → mỗi ô lệnh bắt đầu bằng:
```bash
%%bash
source /content/env.sh        # khai báo DRIVE, DATA, PY, RUNNER
cd /content/XLA_SEARLE/SEARLE-main
CACHE="--cache-dir $DRIVE/cache_A100"
PHI="--eval-type phi --exp-name phi_B32 --phi-checkpoint-name phi_last.pt $CACHE"
CIRR="--dataset cirr --dataset-path $DATA/CIRR"
CIRCO="--dataset circo --dataset-path $DATA/CIRCO"
FIQ="--dataset fashioniq --dataset-path $DATA/FashionIQ"
R=$DRIVE/results
```
- **Cache:** mọi lượt đánh giá dùng `$DRIVE/cache_A100`. Cache cũ `$DRIVE/cache` được tạo trên T4, cho số lệch một chút (xem H5), nên không trộn hai loại trong cùng một bảng.
- Sau mỗi lượt: xem `$R/<mã>/metrics.json` và dòng mới trong `$R/runs_Chi.csv`; ghi số chính vào [nhat_ky_thi_nghiem.md](nhat_ky_thi_nghiem.md), đánh dấu việc trong [tien_do.md](tien_do.md).
- Gom kết quả bất kỳ lúc nào: `$PY tools/collect_results.py --results-dir $R --out $R/summary.md`.

## 4. Việc code và quyết định (C)

| Việc | Trạng thái |
|---|---|
| C3, C5–C16, C21–C24 | Xong 17–19/10; chi tiết từng file ở [CHANGES.md](../SEARLE-main/CHANGES.md) |
| C1 | Còn: xoá 4 thư mục `outputs/phi_models/dryrun_2ep` trùng tên trên Drive. Không cần đóng gói lại tar nữa |
| C2 | Code xong; nghiệm thu khi E1.7a ra 25.66 trên runtime mới (H5) |
| C4 | Code xong; nghiệm thu ở H5 (số khớp, file nộp sinh lại trùng file cũ) |
| C14 (tiếp) | Hiệu chỉnh ngưỡng `--restore auto` trên 300 ảnh CIRR train **trước E3.3**; điền `oracle` sau E3.2 |
| C17–C20 | Chưa |

### C14 (tiếp). Hiệu chỉnh phục hồi tự động (Ngày 9–10)
```bash
$PY tools/calibrate_restore_rules.py --images-dir $DATA/CIRR/train --n 300 --out src/restore_rules.json --report-dir $R/E3.3-calibration
```
Chép `src/restore_rules.json` từ Colab về máy, commit, push, rồi chạy lại ô 2 trước E3.3. Sau E3.2 (D8): điền `src/restore_oracle.json` bằng phép phục hồi tốt nhất của từng loại suy giảm, commit, push.

### Quyết định

| Khi nào | Quyết định | Dựa trên |
|---|---|---|
| Ngày 9 (M3, muộn) | Giữ hay đổi đóng góp chính (ngưỡng ở [02](02_kiem_chung_thuoc_tinh.md)) | E2.0, E2.2 (H9). Số sơ bộ trên file chú thích: FIQ 90,7%, CIRR 36,3%, CIRCO 25,5%, đều vượt ngưỡng 10% |
| Ngày 10 | Trọng số tốt nhất mỗi họ thuộc tính | E2.3 (H10) |
| Ngày 10 | Biến thể phi bền cho bản đầy đủ | E3.8a (D9) |
| Ngày 10 | Cấu hình tiền xử lý tốt nhất | E4.1–E4.3 (D3, D4) |
| Ngày 12 | Cấu hình cuối để nộp test; có nộp phi bền không | Bảng của 02, 03, 04 |

### C17. Demo (Ngày 11)
Theo [05](05_demo_va_bao_cao.md) mục 5.2: tab thuộc tính, tab ảnh xấu → phục hồi, so sánh phương pháp, nút "Vì sao?"; tiếng Việt nếu kịp.

### C18. Chọn cấu hình cuối (Ngày 12)
Chọn cấu hình cuối của 02 (có thể kèm 04) và quyết định có nộp test phi bền không; rồi chạy H13, D13.

### C19. README, CHANGES.md, báo cáo, slide (Ngày 13)
Theo [05](05_demo_va_bao_cao.md) mục 5.4–5.8. Rà lại phiếu chấm tay `tools/grading/parser_grading_seed1.csv` (do trợ lý AI chấm: parser đúng 97%, bộ phân loại đúng 78%) trước khi đưa số vào báo cáo.

### C20. Video, kiểm tra cuối, nộp bài (Ngày 14)
Theo [05](05_demo_va_bao_cao.md) mục 5.8–5.9.

## 5. Nền tảng: dữ liệu, không hồi quy, nộp test (H1–H7)

### H1, H2, H7. Dữ liệu (Ngày 8) — E1.2
Ô 4 phải báo `ready` cho cả 4 tập. Kiểm tra đủ ảnh bằng công cụ (cũng là ô "Kiểm tra dữ liệu (E1.2)" của notebook):
```bash
for DS in cirr circo fashioniq; do
  P=$DATA/CIRR; [ $DS = circo ] && P=$DATA/CIRCO; [ $DS = fashioniq ] && P=$DATA/FashionIQ
  $PY tools/check_datasets.py --dataset $DS --dataset-path $P --out $R/E1.2-dataset-check
done
```
FashionIQ trước đây thiếu khoảng 2.431 ảnh; với thư mục mới, báo cáo phải không còn thiếu ảnh nào. Nếu CIRCO thiếu ảnh thì ảnh hưởng cả val lẫn file nộp test: sửa dữ liệu trước khi chạy tiếp.

### H5. Không hồi quy, chạy lại baseline (Ngày 8) — E1.7a–f
Bước gỡ chặn (E1.7a trên A100, cách đọc kết quả) ở [nhiem_vu_train_Colab.md](nhiem_vu_train_Colab.md) mục 3. Khi E1.7a ra 25.66 (±0,05):
```bash
$PY src/validate.py --eval-type searle $CIRCO $CACHE --output-dir $R/E1.7b-searle-circo
$PY src/validate.py --eval-type searle $FIQ   $CACHE --output-dir $R/E1.7c-searle-fiq
$PY src/validate.py $PHI $CIRR  --output-dir $R/E1.7d-phiB32-cirr
$PY src/validate.py $PHI $CIRCO --output-dir $R/E1.7e-phiB32-circo
$PY src/validate.py $PHI $FIQ   --output-dir $R/E1.7f-phiB32-fiq
```
- Kỳ vọng CIRR phi_B32 R@1 = 25.47 (±0,05).
- **Lượt CIRCO đầu (E1.7b) rất lâu:** đọc khoảng 123 nghìn ảnh gallery thẳng từ Drive. Ngày 15/10 lượt này chậm tới mức phải bỏ dở (`outputs/REPORT.md`). Chạy nó **sớm nhất, chạy nền** (`nohup … > $R/E1.7b.log 2>&1 &`) để đặc trưng gallery vào `cache_A100`; mọi lượt CIRCO sau cùng cấu hình tiền xử lý (E1.7e, E1.8, E3.1, …) chỉ đọc 220 ảnh tham chiếu. Nếu sau 1 giờ log vẫn chưa qua 10%, dừng lại và đóng gói `datasets/CIRCO/COCO2017_unlabeled` thành tar để giải nén ra đĩa Colab.
- FIQ giờ chạy trên **tập đầy đủ**: số sẽ khác lượt cũ trên tập con; lượt này hoàn thành mục "FashionIQ val đầy đủ" của Giai đoạn 0.
- So file nộp test sinh bằng code mới với file cũ (nghiệm thu C4):
  ```bash
  $PY src/generate_test_submission.py --submission-name check_phi_B32_cirr --eval-type phi --exp-name phi_B32 --phi-checkpoint-name phi_last.pt $CIRR
  ```
  rồi trong ô Python:
  ```python
  import json
  a = json.load(open("/content/XLA_SEARLE/SEARLE-main/data/test_submissions/cirr/check_phi_B32_cirr.json"))
  b = json.load(open("/content/XLA_SEARLE/outputs/submissions/phi_B32_cirr.json"))
  keys = [k for k in a if k not in ("version", "metric")]
  print(sum(a[k][:10] == b[k][:10] for k in keys), "/", len(keys), "truy vấn có top-10 trùng")
  ```
  Kỳ vọng ≥ 99%.

### H3. File subset CIRR (đã xong 16/10)
`subset_phi_B32_cirr.json` đã có trong `outputs/submissions/`.

### H4. Sinh file test SEARLE gốc, nộp test (Ngày 8) — B8–B11
```bash
$PY src/generate_test_submission.py --submission-name searle_official_cirr  --eval-type searle $CIRR
$PY src/generate_test_submission.py --submission-name searle_official_circo --eval-type searle $CIRCO
mkdir -p $DRIVE/outputs/submissions
cp data/test_submissions/cirr/*.json data/test_submissions/circo/*.json $DRIVE/outputs/submissions/
```
Nộp lên server:
- CIRR: `phi_B32_cirr.json`, `subset_phi_B32_cirr.json` (ở `/content/XLA_SEARLE/outputs/submissions/`), `searle_official_cirr.json`, `subset_searle_official_cirr.json` (ở `$DRIVE/outputs/submissions/`);
- CIRCO: `phi_B32_circo.json`, `searle_official_circo.json`.

Ghi điểm vào nhật ký (B8–B11) và mục "Điểm test từ server" của Giai đoạn 0.

### H6. Baseline CLIP và CBIR cổ điển (Ngày 9) — E1.8, E1.9
```bash
for T in image text sum; do
  $PY src/validate.py --eval-type $T $CIRR  $CACHE --output-dir $R/E1.8-$T-cirr
  $PY src/validate.py --eval-type $T $CIRCO $CACHE --exclude-reference --output-dir $R/E1.8-$T-circo-ex
  $PY src/validate.py --eval-type $T $FIQ   $CACHE --exclude-reference --output-dir $R/E1.8-$T-fiq-ex
done
# SEARLE gốc và phi_B32 cũng chạy với --exclude-reference trên CIRCO, FIQ để so công bằng
$PY src/validate.py --eval-type searle $CIRCO $CACHE --exclude-reference --output-dir $R/E1.8-searle-circo-ex
$PY src/validate.py --eval-type searle $FIQ   $CACHE --exclude-reference --output-dir $R/E1.8-searle-fiq-ex
$PY src/validate.py $PHI $CIRCO --exclude-reference --output-dir $R/E1.8-phiB32-circo-ex
$PY src/validate.py $PHI $FIQ   --exclude-reference --output-dir $R/E1.8-phiB32-fiq-ex
# CBIR cổ điển (CPU, vài phút với CIRR và FIQ)
for F in color lbp hog gabor all; do
  $PY src/validate.py --eval-type cbir --cbir-features $F $CIRR $CACHE --output-dir $R/E1.9-cbir-$F-cirr
  $PY src/validate.py --eval-type cbir --cbir-features $F $FIQ  $CACHE --exclude-reference --output-dir $R/E1.9-cbir-$F-fiq-ex
done
$PY src/validate.py --eval-type cbir --cbir-features all --cbir-text-weight 1.0 $CIRR $CACHE --output-dir $R/E1.9-cbir-text-cirr
```
CBIR trên CIRCO chỉ chạy nếu còn thời gian (1–2 giờ CPU). Kỳ vọng: SEARLE, phi_B32 > `sum` > `text`, `image`; CBIR thấp nhất.

### H7. Khoảng tin cậy, gom kết quả (Ngày 10) — E1.4
```bash
$PY tools/bootstrap_ci.py --baseline $R/E1.7d-phiB32-cirr --runs $R/E1.7a-searle-cirr $R/E1.8-*-cirr $R/E1.9-*-cirr --out $R/E1.4-ci-cirr
$PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-circo-ex --runs $R/E1.8-*-circo-ex --out $R/E1.4-ci-circo
$PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-fiq-ex --runs $R/E1.8-*-fiq-ex $R/E1.9-*-fiq-ex --out $R/E1.4-ci-fiq
$PY tools/collect_results.py --results-dir $R --out $R/summary.md
```

## 6. Đóng góp chính: kiểm chứng thuộc tính (H8–H14)

### H8. Phân loại truy vấn, giữ – sửa, vector ảnh – vector từ (Ngày 11) — E1.10, E1.11, E1.15a–b
```bash
$PY tools/query_taxonomy.py $CIRR  --runs $R/E1.7a-searle-cirr $R/E1.7d-phiB32-cirr $R/E1.8-*-cirr --out $R/E1.10-taxonomy-cirr
$PY tools/query_taxonomy.py $CIRCO --runs $R/E1.8-*-circo-ex --out $R/E1.10-taxonomy-circo
$PY tools/query_taxonomy.py $FIQ   --runs $R/E1.8-*-fiq-ex --out $R/E1.10-taxonomy-fiq
$PY tools/keep_edit_analysis.py $CIRR  --runs $R/E1.7d-phiB32-cirr $R/E1.8-sum-cirr $CACHE --out $R/E1.11-keepedit-cirr
$PY tools/keep_edit_analysis.py $CIRCO --runs $R/E1.7e-phiB32-circo $R/E1.8-sum-circo-ex $CACHE --out $R/E1.11-keepedit-circo
# E1.15 (ý nghĩa ở 01 mục 1.15): vài phút mỗi lệnh nhờ cache
$PY tools/vector_analysis.py tokens $CIRR  $CACHE --sources searle phi:phi_B32 --snap-eval --out $R/E1.15a-tokens-cirr
$PY tools/vector_analysis.py tokens $CIRCO $CACHE --sources searle phi:phi_B32 --snap-eval --out $R/E1.15a-tokens-circo
$PY tools/vector_analysis.py gap    $CIRR  $CACHE --sources phi:phi_B32 --out $R/E1.15b-gap-cirr
$PY tools/vector_analysis.py gap    $CIRCO $CACHE --sources phi:phi_B32 --out $R/E1.15b-gap-circo
```

### H9. Thống kê thuộc tính, đồng thuận bộ đo (Ngày 9) — E2.0, E2.2 → mốc M3
E2.0 chỉ đọc câu mô tả, không cần GPU.
```bash
$PY tools/attribute_stats.py $CIRR  --out $R/E2.0-stats-cirr
$PY tools/attribute_stats.py $FIQ   --out $R/E2.0-stats-fiq
$PY tools/attribute_stats.py $CIRCO --out $R/E2.0-stats-circo
$PY tools/attribute_stats.py --agreement --attr-mask auto $CIRR  $CACHE --out $R/E2.2-agree-cirr
$PY tools/attribute_stats.py --agreement --attr-mask auto $FIQ   $CACHE --out $R/E2.2-agree-fiq
$PY tools/attribute_stats.py --agreement --attr-mask auto $CIRCO $CACHE --out $R/E2.2-agree-circo
```
Đủ số thì quyết định M3 (mục 4) và ghi vào nhật ký.

### H10. Quét xếp hạng lại theo thuộc tính (Ngày 10–11) — E2.3, E2.4, E2.5
```bash
# E2.3: từng họ thuộc tính × trọng số trên CIRR val và FIQ val (40 lượt, mỗi lượt vài phút)
for FAM in color brightness colorfulness pattern zoom; do
  for W in 0.05 0.1 0.2 0.3; do
    $PY src/validate.py $PHI $CIRR --attr-rerank --attr-families $FAM --attr-weight $W --output-dir $R/E2.3-$FAM-w$W-cirr
    $PY src/validate.py $PHI $FIQ --exclude-reference --attr-rerank --attr-families $FAM --attr-weight $W --output-dir $R/E2.3-$FAM-w$W-fiq
  done
done
$PY tools/collect_results.py --results-dir $R --out $R/summary.md
```
Chọn trọng số tốt nhất mỗi họ, rồi với cấu hình đó:
- E2.3 (tiếp): `--attr-topk 50` và `--attr-topk 100`;
- E2.4: `--attr-mask none`, `otsu`, `grabcut`, `saliency`;
- E2.5: `--attr-negation on` và `off`.

Đặt tên thư mục theo mẫu `E2.4-mask-otsu-cirr`, `E2.5-neg-off-fiq`.

### H11. Kết hợp, tổng quát hoá, kết hợp điểm, prompt (Ngày 11) — E2.6, E2.7, E4.4, E4.5
- **E2.6:** kết hợp mọi họ thuộc tính với trọng số đã chọn (`--attr-families` liệt kê các họ, cách nhau bằng dấu phẩy) trên CIRR val và FIQ val.
- **E4.4:**
  ```bash
  for A in 0.05 0.1 0.2 0.3 0.5; do
    $PY src/validate.py $PHI $CIRR --fusion-weight $A --output-dir $R/E4.4-fusion-a$A-cirr
  done
  ```
- **E4.5:** `$PY src/validate.py $PHI $CIRR --prompt-set ensemble --output-dir $R/E4.5-ensemble-cirr`
- **E2.7:** cấu hình cuối trên CIRCO val và FIQ val, **không chỉnh lại tham số**.

### H12. Ví dụ định tính, bản đồ che khuất (Ngày 12) — E1.12, E1.13, E2.8
Thay `<cuối>` bằng thư mục kết quả của cấu hình cuối.
```bash
$PY tools/visualize_queries.py --run $R/E1.7d-phiB32-cirr --compare $R/<cuối> $CIRR --n 20 --seed 0 --out $R/E1.12-examples-cirr
$PY tools/occlusion_map.py --run $R/E1.7d-phiB32-cirr $CIRR --n 10 --seed 0 --out $R/E1.13-occlusion-cirr
$PY tools/visualize_queries.py --explain-attributes --run $R/<cuối> $FIQ --n 20 --seed 0 --out $R/E2.8-explained-fiq
```

### H13. Nộp test cấu hình cuối (Ngày 12) — E2.9
- `generate_test_submission.py` với đúng cờ của cấu hình cuối (C18) cho CIRR và CIRCO, tên `final_cirr`, `final_circo`; chép ra `$DRIVE/outputs/submissions/`; nộp server.
- Khoảng tin cậy bảng chính: `tools/bootstrap_ci.py --baseline <phi_B32 cùng tập> --runs <cấu hình cuối>` trên 3 tập val.

### H14, D14. Tổng hợp (Ngày 13)
`tools/collect_results.py` → `summary.md` mới nhất; gom hình vào `$R/figures/`; viết mỗi mã thí nghiệm một câu kết luận, đưa vào báo cáo (C19).

## 7. Đóng góp thứ hai: độ bền, phục hồi ảnh, tiền xử lý (D3–D13)

### D3. Đệm ảnh, nội suy (Ngày 10) — E4.1, E4.2
```bash
for RATIO in 1.0 1.25 1.5 2.0; do
  $PY src/validate.py $PHI $CIRR --targetpad-ratio $RATIO --output-dir $R/E4.1-pad$RATIO-cirr
done
$PY src/validate.py $PHI $CIRR --preprocess-type clip --output-dir $R/E4.1-clipcrop-cirr
for I in nearest bilinear bicubic lanczos; do
  $PY src/validate.py $PHI $CIRR --interp $I --output-dir $R/E4.2-interp-$I-cirr
done
```

### D4. Lật ảnh; cấu hình tốt nhất trên CIRCO, FIQ (Ngày 10) — E4.3
```bash
for T in flip-ref flip-gallery flip-both; do
  $PY src/validate.py $PHI $CIRR --tta $T --output-dir $R/E4.3-$T-cirr
done
```
Chọn cấu hình tốt nhất của E4.1–E4.3, chạy trên CIRCO val và FIQ val **không chỉnh thêm**: `--output-dir $R/E4.x-best-circo`, `...-best-fiq`.

### D5. Mức sụt giảm (Ngày 9) — E3.1
Ảnh sạch dùng lại E1.7d, E1.7e. 32 lượt, mỗi lượt vài phút.
```bash
for D in gauss_noise salt_pepper gauss_blur motion_blur jpeg low_light low_res color_cast; do
  for L in 1 2; do
    $PY src/validate.py $PHI $CIRR  --degrade $D:$L --save-quality --output-dir $R/E3.1-$D$L-cirr
    $PY src/validate.py $PHI $CIRCO --degrade $D:$L --save-quality --output-dir $R/E3.1-$D$L-circo
  done
done
```

### D6. Phân tích miền tần số (Ngày 9) — E3.5
```bash
for RT in 0.05 0.1 0.2 0.3 0.5; do
  $PY src/validate.py $PHI $CIRR --freq-filter lowpass:$RT  --output-dir $R/E3.5-lp$RT-cirr
  $PY src/validate.py $PHI $CIRR --freq-filter highpass:$RT --output-dir $R/E3.5-hp$RT-cirr
done
```

### D7, D8, D9, D10. Train phi bền (Ngày 8–11) — E3.8a, E3.8b
Lệnh ở [nhiem_vu_train_Colab.md](nhiem_vu_train_Colab.md): đặc trưng ảnh xấu (s0, rồi 100k), sàng lọc 3 biến thể trên shard s0, đánh giá sàng lọc, train bản đầy đủ `phi_B32_robust`.

### D8. Phục hồi cố định; nhóm left/right (Ngày 10) — E3.2, E4.3b
**E3.2** (mức 2, CIRR val), mỗi ô trong bảng là một lượt:

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

```bash
$PY src/validate.py $PHI $CIRR --degrade gauss_noise:2 --restore nlm --save-quality --output-dir $R/E3.2-gauss_noise2-nlm-cirr
```
Xong thì điền `src/restore_oracle.json` (mục 4, C14).

**E4.3b**, nhóm left/right:
```bash
$PY tools/query_taxonomy.py $CIRR --categories spatial --runs $R/E1.7d-phiB32-cirr $R/E4.3-*-cirr --out $R/E4.3b-leftright-cirr
```

### D10. Phục hồi tự động (Ngày 11) — E3.3
Cần `src/restore_rules.json` đã hiệu chỉnh (C14) và `restore_oracle.json` đã điền; chạy lại ô 2 trước. `auto_iqa` tải trọng số BRISQUE ở lần dùng đầu (cần internet).
```bash
for RS in none auto auto_iqa oracle; do
  $PY src/validate.py $PHI $CIRR --degrade mixed --restore $RS --save-quality --output-dir $R/E3.3-mixed-$RS-cirr
done
# Ảnh sạch: phục hồi tự động có làm hỏng ảnh tốt không
for RS in auto auto_iqa; do
  $PY src/validate.py $PHI $CIRR --restore $RS --output-dir $R/E3.3-clean-$RS-cirr
done
```
Lặp lại cấu hình tốt nhất trên CIRCO val.

### D11. PSNR/SSIM, đánh giá phi bền, độ trôi vector (Ngày 12) — E3.4, E3.8c, E3.9
Đánh giá `phi_B32_robust` (E3.8c) ở [nhiem_vu_train_Colab.md](nhiem_vu_train_Colab.md) mục D11.
```bash
# E3.4: E3.1 (không phục hồi) là mốc của từng loại suy giảm, nên đưa vào cùng E3.2
$PY tools/quality_vs_retrieval.py --runs $R/E3.1-*-cirr $R/E3.2-* --out $R/E3.4-quality-cirr
# E3.9 (ý nghĩa ở 03): chạy sau E3.3; lượt không có --save-quality thì cột PSNR/SSIM ghi n/a
$PY tools/vector_analysis.py drift $CIRR  $CACHE --method phi:phi_B32 \
  --runs $R/E3.1-*-cirr $R/E3.2-* $R/E3.3-*-cirr $R/E3.5-*-cirr --out $R/E3.9-drift-cirr
$PY tools/vector_analysis.py drift $CIRCO $CACHE --method phi:phi_B32 --runs $R/E3.1-*-circo --out $R/E3.9-drift-circo
$PY tools/vector_analysis.py drift $CIRR  $CACHE --method phi:phi_B32_robust --runs $R/E3.8c-robust-*-cirr --out $R/E3.9-drift-robust-cirr
```

### D12. Độ bền bộ đo thuộc tính; hình minh hoạ (Ngày 12) — E3.6, E3.7
```bash
for D in gauss_noise:2 jpeg:2 low_light:2 color_cast:2; do
  $PY tools/attribute_stats.py --agreement --attr-mask auto $FIQ --degrade $D $CACHE --out $R/E3.6-agree-${D/:/}-fiq
done
$PY tools/attribute_stats.py --agreement --attr-mask auto $FIQ --degrade color_cast:2 --restore gray_world $CACHE --out $R/E3.6-agree-colorcast2-grayworld-fiq
$PY tools/visualize_degradations.py $CIRR --run $R/E1.7d-phiB32-cirr \
  --degrade gauss_noise:2,low_light:2,jpeg:2,color_cast:2 --restore auto --n 8 --out $R/E3.7-figures
```

### D13. Nộp test phi bền (Ngày 12, nếu C18 chọn nộp)
`generate_test_submission.py` cho `phi_B32_robust` (CIRR, CIRCO), tên `robust_cirr`, `robust_circo`; chép ra `$DRIVE/outputs/submissions/`; nộp cùng đợt với H13.

## 8. Học máy (tuỳ chọn, Ngày 13) — 06, 07, 08

Train phi của luồng này (T1, T2, H16, D17) ở [nhiem_vu_train_Colab.md](nhiem_vu_train_Colab.md). Các lượt dưới đây tối ưu token OTI trên tập val, mỗi lượt vài phút đến dưới 30 phút.

### H17. Chuẩn bị CIRCO val, OTI không dùng câu GPT — E7.0, E6.2-Pλ0
```bash
$PY src/image_concepts_association.py --clip-model-name ViT-B/32 $CIRCO --split val --dataset-mode relative
$PY src/oti_inversion.py --exp-name E6.2-Plambda0-circo --clip-model-name ViT-B/32 $CIRCO --split val --lambda_gpt 0
$PY src/validate.py --eval-type oti --exp-name E6.2-Plambda0-circo $CIRCO $CACHE --output-dir $R/E6.2-Plambda0-circo
```

### H15. OTI warm-start trên CIRCO val — E7.1a–d
Cần H17. E7.1a cũng là E6.2-P0 ([06](06_sinh_lai_cum_tu_gpt.md)).
```bash
OTI="--clip-model-name ViT-B/32 $CIRCO --split val --snapshot-steps 0,10,25,50,100,200,350"
WARM="--init-phi-exp-name phi_B32 --init-phi-checkpoint phi_last.pt"
$PY src/oti_inversion.py --exp-name E7.1a-random-lr0.02-circo  $OTI --init random --learning-rate 0.02
$PY src/oti_inversion.py --exp-name E7.1b-phi-lr0.02-circo     $OTI --init phi $WARM --learning-rate 0.02
$PY src/oti_inversion.py --exp-name E7.1c-phi-lr0.005-circo    $OTI --init phi $WARM --learning-rate 0.005
$PY src/oti_inversion.py --exp-name E7.1d-searle-lr0.02-circo  $OTI --init searle --learning-rate 0.02
$PY src/oti_inversion.py --exp-name E7.1d-searle-lr0.005-circo $OTI --init searle --learning-rate 0.005
# Đánh giá từng bước đã lưu, cả token thường lẫn EMA (lặp cho từng lượt, đổi E)
E=E7.1b-phi-lr0.02-circo
for S in 0 10 25 50 100 200 350; do
  for TK in oti_pseudo_tokens ema_oti_pseudo_tokens; do
    $PY src/validate.py --eval-type oti --exp-name $E --token-file ${TK}_step$S.pt $CIRCO $CACHE --output-dir $R/${E%%-*}-step$S-$TK-circo
  done
done
```
Sau đó, E1.15a có thêm token OTI:
```bash
$PY tools/vector_analysis.py tokens $CIRCO $CACHE \
  --sources phi:phi_B32 oti:E7.1a-random-lr0.02-circo oti:E7.1b-phi-lr0.02-circo oti:E6.2-Plambda0-circo \
  --snap-eval --out $R/E1.15a-tokens-oti-circo
```

### D15. OTI warm-start trên CIRR val — E7.2
Liên kết khái niệm cho CIRR val (một lần), rồi chạy 1–2 cấu hình tốt nhất của E7.1 với `$CIRR` thay `$CIRCO`:
```bash
$PY src/image_concepts_association.py --clip-model-name ViT-B/32 $CIRR --split val --dataset-mode relative
```

### D16. Bộ câu GPT: thống kê, sinh bộ câu mới quy mô nhỏ — E6.1
```bash
# P1 (CPU, 10–20 phút) nếu Drive chưa có outputs/GPT_phrases/GPTNeo27B_clean
$PY tools/clean_phrases.py --phrases data/GPT_phrases/GPTNeo27B/concept_to_phrases.pkl --out-exp GPTNeo27B_clean
SC=data/similar_concepts/imagenet/test/ViT-B32_top_250_ensembling_image_texts.csv
$PY tools/phrases_stats.py --phrases data/GPT_phrases/GPTNeo27B/concept_to_phrases.pkl --similar-concepts $SC --out $R/E6.1-stats-P0
$PY tools/phrases_stats.py --phrases data/GPT_phrases/GPTNeo27B_clean/concept_to_phrases.pkl --similar-concepts $SC --out $R/E6.1-stats-P1
```
Sinh bộ câu mới bằng `colab/gen_phrases.ipynb` (runtime riêng): lượt 1 `SCOPE = "circo_val"` (cho H18); lượt 2 `SCOPE = "s0"` (cho D17) chỉ khi còn GPU.

### H18. OTI với bộ câu mới trên CIRCO val — E6.2
```bash
for P in GPTNeo27B_clean <bộ mới của D16>; do
  $PY src/oti_inversion.py --exp-name E6.2-$P-circo --clip-model-name ViT-B/32 $CIRCO --split val --gpt-exp-name $P
  $PY src/validate.py --eval-type oti --exp-name E6.2-$P-circo $CIRCO $CACHE --output-dir $R/E6.2-$P-circo
done
```
So với E7.1a (= E6.2-P0) và E6.2-Pλ0, kèm khoảng tin cậy bootstrap.

## 9. Khi gặp lỗi

| Hiện tượng | Làm gì |
|---|---|
| `unrecognized arguments` | Code trên Colab cũ: chạy lại ô 2 (lấy từ `chilk1307/Chiss`) |
| `FileNotFoundError` về dữ liệu | Chạy lại ô 4 đúng tên tập; kiểm tra `MyDrive/SEARLE/datasets/<tập>` |
| `CUDA out of memory` | Đang chạy 2 lượt train cùng lúc với đánh giá: chờ một lượt train xong, hoặc dừng bớt rồi `--resume` |
| Colab ngắt | Ô 0–5; train thì chạy lại đúng lệnh cũ, thêm `--resume`; đánh giá thì chạy lại (cache còn trên Drive) |
| Số khác kỳ vọng | Dừng các bước phụ thuộc; xem `log.txt` của lượt đó, so cờ với lượt mốc |

## 10. Mẫu nhật ký cuối ngày

Ghi vào [nhat_ky_thi_nghiem.md](nhat_ky_thi_nghiem.md):
```text
[Ngày N – dd/10]
Đã xong: C…, H…, D… (results/…)
Đang train: … (epoch …/…)
Số chính: …
Quyết định: …
Vướng: …
Ngày mai: …
GPU đã dùng: … giờ A100
```
