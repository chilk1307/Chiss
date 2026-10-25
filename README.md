# Cải tiến SEARLE bằng xử lý ảnh cổ điển

Đồ án môn Xử lý ảnh. Nhóm tái lập **SEARLE** (Baldrati et al., ICCV 2023), một phương pháp truy xuất ảnh kết hợp
zero-shot dựa trên CLIP. Đầu vào là một ảnh tham chiếu và một câu mô tả thay đổi; đầu ra là các ảnh đích trong gallery.
Nhóm thêm hai lớp xử lý ảnh cổ điển lên trên SEARLE:

1. **Kiểm chứng thuộc tính (đóng góp chính).** Rút các ràng buộc (màu, độ sáng, độ sặc sỡ, hoa văn, độ phóng to) từ câu
   mô tả bằng luật. Đo chúng trên vùng đối tượng của từng ảnh ứng viên: k-means trong CIELAB với khoảng cách CIEDE2000,
   L*, chỉ số Hasler–Süsstrunk, FFT, LBP, Gabor, và mặt nạ Otsu, GrabCut hoặc spectral residual. Sau đó xếp lại top-K
   của CLIP theo mức ảnh đáp ứng câu mô tả.
2. **Độ bền với ảnh xấu.** Làm xấu ảnh tham chiếu theo 8 loại suy giảm, rồi phục hồi bằng lọc trung vị, bilateral, NLM,
   Wiener, Richardson–Lucy, unsharp, CLAHE, gamma, Retinex, gray-world, white-patch hoặc khử khối. Có ba cách chọn phép
   phục hồi: cố định theo loại, tự động theo luật, hoặc theo BRISQUE. Ngoài ra train lại mạng φ với đặc trưng của ảnh
   xấu (φ bền).

## Kết quả chính (val, CLIP ViT-B/32)

| | CIRR R@1 | CIRR R@5 | CIRR Rsubset@1 | CIRCO mAP@10* | FashionIQ R@10* |
|---|---|---|---|---|---|
| SEARLE (φ công bố), chạy lại | 25.66 | 55.47 | 55.08 | 9.89 | 23.01 |
| φ_B32, tái lập | 25.47 | 55.23 | 55.20 | 9.27 | 22.74 |
| φ_B32 + kiểm chứng thuộc tính | **25.81** | **55.70** † | **55.56** † | 9.39 | 22.94 |
| φ bền, ảnh sạch | 25.45 | 55.75 † | 54.68 | – | – |

\* bỏ ảnh tham chiếu khỏi gallery. † khoảng tin cậy bootstrap 95% của hiệu số so với φ_B32 không chứa 0.

Kết quả trên ảnh xấu (CIRR R@1, trung bình 8 loại suy giảm ở mức 2):

| | R@1 |
|---|---|
| φ_B32, không phục hồi | 24.07 |
| φ bền | 24.59 |
| φ_B32 + phục hồi đúng loại | 24.75 |
| Ảnh sạch (tham chiếu) | 25.47 |

Mọi con số đều lấy từ `MyDrive/SEARLE/results/<mã thí nghiệm>/metrics.json`. Báo cáo đầy đủ (Claude Docs) có thêm
phân tích lỗi, tiền xử lý, lọc tần số, các khoảng tin cậy và hạn chế.

## Cấu trúc

```
SEARLE-main/
  src/                     mã SEARLE và phần thêm của nhóm
    retrieval_core.py      pipeline chung cho validate.py và generate_test_submission.py
    attributes/            bộ phân tích câu (parser.py), bộ đo (meters.py), xếp lại (rerank.py), phân loại câu (taxonomy.py)
    degradations.py        8 loại suy giảm × 2 mức
    restorations.py        các phép phục hồi; restore_rules.json (luật tự động), restore_oracle.json (phép tốt nhất theo loại)
    freq_filters.py        lọc thông thấp / thông cao Gauss trong miền tần số
    cbir.py                CBIR cổ điển (HSV, LBP, HOG, Gabor)
  tools/                   phân tích: bootstrap_ci, query_taxonomy, keep_edit_analysis, vector_analysis, occlusion_map, ...
  colab/                   SEARLE_runner.ipynb, setup_env.sh, prepare_data.sh, queue/ (các hàng đợi đã chạy)
  demo/                    demo web Flask (python demo/app.py → http://127.0.0.1:5050)
plans/                     kế hoạch và mã thí nghiệm (E1.x tái lập và phân tích, E2.x thuộc tính, E3.x độ bền, E4.x tiền xử lý)
```

## Chạy lại

- **Colab:** mở `SEARLE-main/colab/SEARLE_runner.ipynb`, chọn GPU A100 và chạy các ô 0–5 (Drive, code, môi trường, dữ liệu).
  Sau đó khởi động các hàng đợi trong `colab/queue/` (xem `colab/queue/README.md`). Toàn bộ thí nghiệm mất khoảng một ngày
  A100 khi chạy song song; riêng mỗi lần train φ 100 epoch mất khoảng 6 giờ.
- **Một lượt đánh giá:**
  ```bash
  python src/validate.py --eval-type phi --exp-name phi_B32 --phi-checkpoint-name phi_last.pt \
      --dataset cirr --dataset-path $DATA/CIRR \
      --attr-rerank --attr-families color,brightness,zoom --attr-weight 1.0 --attr-topk 50 --attr-mask grabcut --attr-negation off
  ```
  Thêm `--degrade motion_blur:2 --restore auto` để thử ảnh xấu và phục hồi.
- **File nộp server test:** `src/generate_test_submission.py` với cùng cờ và `--submission-name final_cirr`.
  Các file đã sinh nằm ở `MyDrive/SEARLE/outputs/submissions/`.
- **Demo:** `python SEARLE-main/demo/app.py`. Biến `SEARLE_PHI_DIR` chọn mô hình φ, `SEARLE_DEMO_PORT` đổi cổng.

## Tham khảo

Baldrati, Agnolucci, Bertini, Del Bimbo. *Zero-Shot Composed Image Retrieval with Textual Inversion*, ICCV 2023.
Mã gốc: https://github.com/miccunifi/SEARLE (giấy phép trong `SEARLE-main/LICENSE`).
