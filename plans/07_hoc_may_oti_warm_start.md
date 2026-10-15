# 07. Học máy: OTI khởi tạo từ phi (warm-start)

**Mục tiêu:** giảm mạnh số bước tối ưu của OTI bằng cách khởi tạo pseudo-token từ dự đoán của phi thay vì ngẫu nhiên; đồng thời kiểm tra xem tinh chỉnh vài bước lúc truy vấn có tốt hơn dùng phi thuần hay không.

**Vai trò:** đóng góp **học máy**, chạy song song với các đóng góp xử lý ảnh ([02](02_kiem_chung_thuoc_tinh.md), [03](03_do_ben_va_phuc_hoi_anh.md)).
**Ước lượng:** 5–7 ngày công, 5–10 giờ GPU.
**Phụ thuộc:** [01](01_nen_tang_danh_gia.md) (module truy xuất chung ở mục 1.3, khoảng tin cậy bootstrap ở mục 1.4).
**Liên quan:** OTI dùng câu GPT cho GPT loss; các bộ câu mới (P1–P3) và thí nghiệm so sánh ở [06](06_sinh_lai_cum_tu_gpt.md). E7.0 dưới đây dùng chung cho 06.

## Phân công trong kế hoạch 2 tuần

| Mã | Ngày | Người | Việc |
|---|---|---|---|
| H17 | 7 | Hòa | E7.0 cho CIRCO val (dùng chung với [06](06_sinh_lai_cum_tu_gpt.md)) |
| C21 | 9 | Chí | Code mục 7.1–7.2: `--init`, `--init-phi-exp-name`, `--init-phi-checkpoint`, `--snapshot-steps`, `--token-file` |
| H15 | 11 | Hòa | E7.1a–d trên CIRCO val (E7.0 đã có từ H17). Thư mục kết quả: `E7.1a-random-lr0.02-circo`, `E7.1b-phi-lr0.02-circo`, `E7.1c-phi-lr0.005-circo`, `E7.1d-searle-lr0.02-circo`, `E7.1d-searle-lr0.005-circo`. E7.1e chỉ chạy khi Chí yêu cầu |
| D15 | 12 (nếu còn GPU) | Doanh | E7.0 cho CIRR val + E7.2 |
| — | Sau ngày 12 | Người còn GPU | E7.3–E7.5; E7.6 (cần [08](08_hoc_may_cai_tien_phi.md) mục 8.0–8.1) |
| — | Sau E7.1e | Chí | Mục 7.8 (chế độ chính xác cao trong demo), chỉ khi H2 đúng |

E7.1a-random-lr0.02-circo cũng là E6.2-P0-circo của [06](06_sinh_lai_cum_tu_gpt.md).

## Bối cảnh

- **OTI** tối ưu riêng một pseudo-token cho từng ảnh: 350 bước AdamW, khởi tạo ngẫu nhiên `N(0, 0.02²)` ([oti_inversion.py:144-146](../SEARLE-main/src/oti_inversion.py#L144-L146)). Rất chậm: khoảng 61 giây cho một batch 32 ảnh (đo khi chạy 3 tiến trình song song trên A100).
- **phi** (SEARLE) dự đoán token chỉ bằng một lần chạy MLP, rất nhanh, nhưng chỉ là xấp xỉ của OTI (phi được train để chưng cất từ token OTI).
- Paper có phiên bản **SEARLE-OTI** (chạy OTI trực tiếp trên ảnh truy vấn). Theo paper, OTI không phải lúc nào cũng tốt hơn phi trên mọi tập. Vì vậy đây là một **câu hỏi nghiên cứu**, chưa chắc có lợi; kết quả âm tính vẫn đưa vào báo cáo được.

## Giả thuyết

- **H1 (tốc độ):** khởi tạo bằng phi, OTI đạt chất lượng bằng OTI khởi tạo ngẫu nhiên 350 bước chỉ với ≤ 50 bước.
- **H2 (chất lượng):** token phi + tinh chỉnh vài bước cho kết quả truy xuất tốt hơn token phi thuần.
- **H3 (nội suy):** `token = (1 − λ) · token_phi + λ · token_OTI` với λ ở giữa tốt hơn cả hai đầu mút.

## Cài đặt

### 7.1. Sửa `src/oti_inversion.py`

| Tham số mới | Ý nghĩa |
|---|---|
| `--init {random,phi,searle}` | Cách khởi tạo token. `random` = hành vi cũ |
| `--init-phi-exp-name`, `--init-phi-checkpoint` | phi tự train dùng khi `--init phi` (ví dụ `phi_B32`, `phi_last.pt`) |
| `--snapshot-steps 0,10,25,50,100,200,350` | Lưu token tại các bước này trong cùng một lần chạy |

Thay đổi trong code:
1. Chuyển đoạn trích đặc trưng ảnh ([oti_inversion.py:148-151](../SEARLE-main/src/oti_inversion.py#L148-L151)) lên trước đoạn khởi tạo token.
2. Khi `--init` khác `random`: `oti_pseudo_tokens = phi(batch_im_features.float()).detach().clone()`. Với `searle`, nạp phi gốc qua `torch.hub` giống [validate.py:462-463](../SEARLE-main/src/validate.py#L462-L463).
3. Trong vòng lặp tối ưu ([oti_inversion.py:182](../SEARLE-main/src/oti_inversion.py#L182)), khi bước hiện tại nằm trong `--snapshot-steps` thì lưu bản sao của token thường và token EMA. Cuối cùng ghi `oti_pseudo_tokens_step{N}.pt` và `ema_oti_pseudo_tokens_step{N}.pt`.
4. Ghi thời gian chạy từng batch vào `timing.json`.

Nhờ snapshot, **một lần chạy 350 bước cho ra cả đường cong**, không phải chạy riêng từng số bước.

### 7.2. Đánh giá token OTI qua module truy xuất chung

`--eval-type oti` của `validate.py` cũ luôn đọc `ema_oti_pseudo_tokens.pt` ([validate.py:421](../SEARLE-main/src/validate.py#L421)). Trong module truy xuất chung ([01](01_nen_tang_danh_gia.md) mục 1.3), thêm:
- `--token-file <tên file>`: chọn snapshot cần đánh giá.
- `--interp-phi <exp>:<checkpoint>` và `--interp-lambda <λ>`: nội suy với token phi (cho H3).

### 7.3. Cạm bẫy: EMA khi số bước ít

EMA với hệ số 0.99 có "trí nhớ" khoảng 100 bước và được khởi tạo bằng chính token ban đầu ([oti_inversion.py:175](../SEARLE-main/src/oti_inversion.py#L175)). Khi chỉ chạy 10–50 bước:
- Khởi tạo ngẫu nhiên: token EMA vẫn gần như ngẫu nhiên, làm baseline bị thiệt oan.
- Khởi tạo bằng phi: token EMA gần như vẫn là token phi.

Vì vậy luôn đánh giá **cả token thường lẫn token EMA** cho mỗi snapshot và báo cáo cả hai.

### 7.4. Điều kiện cần trước khi chạy OTI trên tập val (E7.0)

`oti_inversion.py` đọc file liên kết concept của đúng tập và split ([oti_inversion.py:81-82](../SEARLE-main/src/oti_inversion.py#L81-L82)). Cần chạy trước:

```sh
python src/image_concepts_association.py --clip-model-name ViT-B/32 --dataset circo --dataset-path <CIRCO> --split val --dataset-mode relative
python src/image_concepts_association.py --clip-model-name ViT-B/32 --dataset cirr --dataset-path <CIRR> --split val --dataset-mode relative
```

OTI dùng câu GPT (`concept_to_phrases.pkl`, 698 MB) cho phần regularization; Colab phải có file này ở `data/GPT_phrases/GPTNeo27B/`.

## Thí nghiệm

Chạy CIRCO val trước (chỉ 220 truy vấn, mỗi lần chạy vài phút), sau đó mới đến CIRR val. Mã đầy đủ thêm hậu tố tập theo quy ước README, ví dụ `E7.1b-phi-lr0.02-circo`.

| Mã | Tập | Khởi tạo | lr | λ_gpt | Ghi chú |
|---|---|---|---|---|---|
| E7.1a | CIRCO val | random | 0.02 | 0.5 | = SEARLE-OTI của paper |
| E7.1b | CIRCO val | phi (phi_B32) | 0.02 | 0.5 | |
| E7.1c | CIRCO val | phi (phi_B32) | 0.005 | 0.5 | lr nhỏ để token không bị đẩy xa điểm khởi tạo ngay từ đầu |
| E7.1d | CIRCO val | searle | 0.02 và 0.005 | 0.5 | |
| E7.1e | CIRCO val | cấu hình tốt nhất | | 0 | Chỉ dùng cosine loss: không cần câu GPT (~700 MB), cần cho demo ([05](05_demo_va_bao_cao.md)) |
| E7.2 | CIRR val | 2–3 cấu hình tốt nhất từ E7.1 | | | |
| E7.3 | FIQ val (tập con) | cấu hình tốt nhất | | | Ghi rõ "tập con" vì thiếu ảnh |
| E7.4 | CIRR val, CIRCO val | nội suy λ = 0, 0.25, 0.5, 0.75, 1 | | | H3 |
| E7.5 | — | đo độ trễ mỗi truy vấn | | | batch 1 và batch 32; GPU và CPU |

Mọi lần chạy E7.1–E7.3 đều lưu snapshot ở các bước 0, 10, 25, 50, 100, 200, 350.

Ước lượng thời gian một lần chạy: `số ảnh / 32 × (số bước / 350) × ≤ 61 giây`. Ví dụ CIRCO val 350 bước: 7 batch, dưới 7 phút.

### Thí nghiệm mở rộng (nếu còn thời gian)

**E7.6: tạo dữ liệu train phi nhanh hơn.** Chạy OTI warm-start 50 bước trên một shard ImageNet (33.334 ảnh; ước tính 2–3 giờ thay vì khoảng 15 giờ), train phi trên token mới, rồi so với phi train trên token OTI 350 bước của cùng shard (cần [08](08_hoc_may_cai_tien_phi.md) mục 8.0 và 8.1).

Lưu ý: phi_B32 đã được train trên chính các ảnh này, nên đây là một dạng tự chưng cất (self-distillation). Cần ghi rõ trong báo cáo.

### Liên kết với phần xử lý ảnh

- **E7.7 (tuỳ chọn):** chạy cấu hình warm-start tốt nhất trên ảnh tham chiếu bị suy giảm (bộ suy giảm của [03](03_do_ben_va_phuc_hoi_anh.md)). Câu hỏi: tinh chỉnh vài bước theo chính ảnh truy vấn có giúp token bền hơn phi thuần khi ảnh xấu không?

## 7.8. Đưa vào demo: chế độ "chính xác cao" (chỉ làm nếu H2 đúng)

- Tinh chỉnh token bằng OTI warm-start k bước ngay lúc truy vấn; hiển thị thời gian xử lý.
- OTI đầy đủ cần câu GPT (~700 MB) và danh sách concept của từng ảnh, quá nặng cho demo. Dùng biến thể chỉ có cosine loss (λ_gpt = 0) nếu E7.1e cho thấy vẫn có lợi.
- Viết hàm riêng `refine_token(image_features, init_token, steps, lr)` trong demo, dùng `get_templates()` của [utils.py:120](../SEARLE-main/src/utils.py#L120).
- Trên CPU sẽ chậm: chỉ bật khi có GPU, giới hạn số bước. Giao diện demo ở [05](05_demo_va_bao_cao.md).

## Kết quả cần có

1. **Đường cong chính:** trục x = số bước (thang log), trục y = CIRR R@1 và CIRCO mAP@10; mỗi đường là một kiểu khởi tạo (random, phi, searle); nét liền = token thường, nét đứt = token EMA; thêm đường ngang = phi thuần.
2. **Bảng độ trễ:** phi thuần, warm-start 10/25/50 bước, OTI 350 bước (ms/truy vấn).
3. **Bảng nội suy (H3):** chỉ số theo λ.
4. **Kết luận** cho H1, H2, H3, kèm khoảng tin cậy bootstrap.

## Tiêu chí hoàn thành

- [ ] `oti_inversion.py` có `--init`, `--snapshot-steps`; chạy với `--init random` (cùng seed) ra cùng token như trước, sai lệch không đáng kể.
- [ ] Có đường cong cho CIRCO val và CIRR val.
- [ ] Trả lời H1–H3 bằng số liệu có khoảng tin cậy.
- [ ] Nếu H2 đúng: chuyển cấu hình tốt nhất sang demo (mục 7.8).

## Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| OTI không tốt hơn phi (H2 sai) | Vẫn báo cáo; phân tích nguyên nhân (OTI bám chi tiết thị giác, khó kết hợp với văn bản?); thử nội suy H3 |
| lr 0.02 đẩy token ra xa điểm khởi tạo ngay những bước đầu | Đã có cấu hình lr 0.005; có thể thêm warmup cho lr |
| Thời gian GPU trên Colab hạn chế | Dùng snapshot (một lần chạy cho cả đường cong); chạy CIRCO trước |
| Tranh GPU với các thí nghiệm xử lý ảnh | Xử lý ảnh ưu tiên trước mốc M3; luồng học máy chạy vào thời gian GPU rảnh |

## Lịch sử thay đổi

- 2026-10-16: tạo kế hoạch (bản cũ, số 03).
- 2026-10-16: chuyển thành file 07 theo yêu cầu giữ phần học máy; cập nhật liên kết sang cấu trúc kế hoạch 2 tuần; đổi mã thí nghiệm E3 → E7; thêm mục vị trí trong lịch và E7.7.
- 2026-10-16: gộp kế hoạch: liên kết sang 06 (sinh lại cụm từ GPT); E7.6 không còn để sau BTL; thêm mục 7.8 (chế độ chính xác cao của demo, chuyển từ kế hoạch demo cũ); bỏ liên kết tới bản lưu trữ đã xoá.
