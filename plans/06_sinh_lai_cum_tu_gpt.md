# 06. Học máy: sinh lại cụm từ GPT (`concept_to_phrases.pkl`)

**Mục tiêu:** thay bộ câu do GPT-Neo 2.7B sinh ra (bản tác giả phát hành) bằng bộ câu sạch hơn và gần với câu sửa đổi của bài toán CIR hơn, rồi đo xem bộ câu mới có giúp OTI và phi tốt hơn không.

**Vai trò:** đóng góp **học máy**, thuộc kế hoạch chính, chạy song song với phần xử lý ảnh.
**Ước lượng:** 3–4 ngày công; 3–8 giờ GPU A100 để sinh câu (đo thử trước, mục 6.3), cộng thời gian đánh giá ở mục 6.4.
**Phụ thuộc:** [01](01_nen_tang_danh_gia.md) (module truy xuất chung, bootstrap); E7.0 của [07](07_hoc_may_oti_warm_start.md) (liên kết concept cho tập val, dùng chung); hạ tầng train ở [08](08_hoc_may_cai_tien_phi.md) mục 8.0.

## File này là gì, dùng để làm gì

- **Là gì:** từ điển `{khái niệm: [256 câu]}`. Với mỗi khái niệm trong 20.931 khái niệm của Open Images V7 ([oidv7-class-descriptions.csv](../SEARLE-main/data/oidv7-class-descriptions.csv)), GPT-Neo 2.7B viết tiếp câu `a photo of {khái niệm} that` ([gpt_phrases_generation.py:66-72](../SEARLE-main/src/gpt_phrases_generation.py#L66-L72)). Tổng khoảng 5,4 triệu câu, file nặng 698 MB.
- **Dùng để làm gì:** tạo **GPT loss** (điều chuẩn) khi chạy OTI và khi train phi. Lấy một câu của khái niệm gần với ảnh, thay khái niệm bằng `$`, rồi buộc câu chứa pseudo-token có đặc trưng CLIP giống câu gốc ([oti_inversion.py:189-213](../SEARLE-main/src/oti_inversion.py#L189-L213), [train_phi.py:179-191](../SEARLE-main/src/train_phi.py#L179-L191)). Nhờ vậy `$` hoạt động như một danh từ thật và ghép được với câu mô tả thay đổi.
- **Không dùng** khi đánh giá, sinh file nộp test hay chạy demo. Muốn thấy tác dụng của bộ câu mới phải chạy OTI hoặc train phi (mục 6.4 sắp xếp từ rẻ đến đắt).

## Vấn đề đã thấy khi đọc code

| # | Vấn đề | Vị trí | Ảnh hưởng |
|---|---|---|---|
| 1 | Mô hình sinh câu cũ (GPT-Neo 2.7B, 2021), temperature 0.5; `max_length=35` tính **cả phần prompt**, nên phần viết tiếp chỉ khoảng 25 token | [gpt_phrases_generation.py:69-72](../SEARLE-main/src/gpt_phrases_generation.py#L69-L72), [:103](../SEARLE-main/src/gpt_phrases_generation.py#L103) | Câu ngắn, dễ lặp, ít chi tiết thị giác |
| 2 | Xử lý hậu kỳ chỉ thay xuống dòng và dấu chấm thừa; không lọc trùng, không lọc câu lạc đề | [gpt_phrases_generation.py:18-21](../SEARLE-main/src/gpt_phrases_generation.py#L18-L21) | Lãng phí lượt lấy mẫu, nhiễu cho loss |
| 3 | Quy tắc lọc khái niệm quá chặt: chỉ cần **một** câu không chứa đúng chuỗi khái niệm là bỏ **cả** khái niệm | [oti_inversion.py:86-94](../SEARLE-main/src/oti_inversion.py#L86-L94), [train_phi.py:117-125](../SEARLE-main/src/train_phi.py#L117-L125) | Mất khái niệm (thường do tokenizer đổi ký tự đặc biệt khi giải mã) |
| 4 | Không nhất quán khi thay `$`: OTI chỉ thay **lần xuất hiện đầu** (`replace(w, " $ ", 1)`), train_phi thay **mọi lần** (`replace(w, " $ ")`), cả hai đều không xét ranh giới từ | [oti_inversion.py:195-196](../SEARLE-main/src/oti_inversion.py#L195-L196), [train_phi.py:181-182](../SEARLE-main/src/train_phi.py#L181-L182) | Ở train_phi, khái niệm nằm lẫn trong từ khác bị thay sai (ví dụ "ant" trong "giant" thành "gi $"); câu có khái niệm lặp lại thì có nhiều `$` |
| 5 | `--resume-experiment` dùng `args.saved_exp_name` nhưng argparse không khai báo tham số này | [gpt_phrases_generation.py:41-45](../SEARLE-main/src/gpt_phrases_generation.py#L41-L45) | Chạy tiếp sẽ lỗi `AttributeError` |
| 6 | Sinh tuần tự từng khái niệm, không gộp batch giữa các khái niệm | [gpt_phrases_generation.py:58-77](../SEARLE-main/src/gpt_phrases_generation.py#L58-L77) | Rất chậm nếu sinh lại bằng code cũ |
| 7 | File danh sách khái niệm trong repo đã bị sửa so với bản gốc (lần kiểm tra `git status` trước đây báo `M data/oidv7-class-descriptions.csv`) | `data/oidv7-class-descriptions.csv` | Phải xác định đúng danh sách khái niệm trước khi sinh lại, nếu không bộ câu mới lệch với file `similar_concepts` |

**Đã đo (E6.1 trên P0, 18/10):** vấn đề 2 nhỏ (câu trùng 0,06%; câu dài quá 75 token CLIP 0%); vấn đề 3 gần như không có (luật lọc cũ chỉ bỏ 2/20.931 khái niệm, đều do mất dấu cách trước dấu nháy, đã cứu được bằng `clean_phrases.py`); vấn đề 4: khái niệm nằm lẫn trong từ khác 0%, nhưng **12,1% câu chứa khái niệm nhiều lần**, khi đó `train_phi.py` gốc thay mọi lần xuất hiện bằng `$` mà chỉ `$` đầu được thay bằng pseudo-token (các `$` sau thành nhiễu) — `--phrase-match word` sửa đúng chỗ này. Nội dung câu thường không mang thông tin thị giác ("a photo of Abdomen that is not a photo"), độ đa dạng distinct-2 trung bình 0,365: lý do chính để thử P3. Làm sạch P1 chỉ bỏ 3.489/5.358.336 câu, nên P1 gần như trùng P0.

## Các bộ câu sẽ so sánh

| Mã | Bộ câu | Cách tạo | Câu hỏi trả lời được |
|---|---|---|---|
| P0 | Bản gốc `GPTNeo27B` | Có sẵn | Mốc so sánh |
| P1 | P0 đã làm sạch | Mục 6.2, không cần GPU | Làm sạch thôi đã đủ chưa? |
| P2 | Sinh lại bằng LLM mới, **giữ nguyên prompt** `a photo of {khái niệm} that` | Mục 6.3 | Chỉ đổi mô hình ngôn ngữ thì được bao nhiêu? |
| P3 | Sinh bằng LLM instruct, câu viết tiếp **giống câu sửa đổi CIR** (thuộc tính thị giác: màu, chất liệu, số lượng, bối cảnh, góc nhìn) | Mục 6.3 | Câu "thị giác", cùng miền với truy vấn có tốt hơn không? |
| Pλ0 | Không dùng câu (`--lambda-gpt 0`, OTI `--lambda_gpt 0`) | Không cần file | Câu GPT đóng góp bao nhiêu? (trùng E8.3d của [08](08_hoc_may_cai_tien_phi.md), dùng chung kết quả) |

Lý do giữ tiền tố `a photo of {khái niệm} that` cho P2 và P3: truy vấn lúc đánh giá có dạng `a photo of $ that {câu sửa đổi}`, nên câu điều chuẩn có cùng khuôn sẽ khớp với cách dùng `$` khi truy vấn.

## 6.1. Thống kê bộ câu hiện tại (CPU)

Script mới `tools/phrases_stats.py`, đầu vào là một file `concept_to_phrases.pkl`, đầu ra là `stats.json` và vài câu mẫu.

- Số khái niệm, số câu mỗi khái niệm, tỉ lệ câu trùng trong cùng khái niệm.
- Số khái niệm bị bỏ theo quy tắc lọc hiện tại (vấn đề 3), kèm ví dụ.
- Tỉ lệ câu mà khái niệm xuất hiện nhiều lần, hoặc nằm lẫn trong từ khác (vấn đề 4).
- Phân bố độ dài (số từ), tỉ lệ câu lặp từ, tỉ lệ ký tự ngoài ASCII.
- **Khái niệm thực sự được dùng:** hợp của top-15 khái niệm mỗi ảnh (OTI) và top-150 (phi) trong file `similar_concepts` của ImageNet test. Nếu tập này nhỏ hơn nhiều so với 20.931 thì chỉ cần sinh lại cho các khái niệm này.

Lưu ý bộ nhớ: nạp nguyên file 698 MB vào Python cần vài GB RAM. Trên Colab thì đủ; trên máy cá nhân ít RAM thì đọc tuần tự bằng `pickletools.genops` (không unpickle, an toàn với file tải về và chỉ giữ một khái niệm trong bộ nhớ mỗi lúc).

## 6.2. Làm sạch và sửa cách dùng câu (P1, không cần GPU)

**Làm sạch dữ liệu** (script `tools/clean_phrases.py`, ghi ra `data/GPT_phrases/GPTNeo27B_clean/concept_to_phrases.pkl`):
1. Bỏ câu trùng (so sau khi chuyển chữ thường, bỏ dấu câu).
2. Giữ câu khi khái niệm xuất hiện **như một từ trọn vẹn** (biểu thức chính quy có ranh giới từ, không phân biệt hoa thường); bỏ từng câu không đạt thay vì bỏ cả khái niệm.
3. Chuẩn hoá khái niệm bị tokenizer làm đổi ký tự để cứu các khái niệm đang bị lọc.
4. Cắt câu dài hơn 77 token CLIP (CLIP vốn cắt, nhưng cắt sớm giúp thống kê đúng).

**Sửa cách dùng trong code** (giá trị mặc định giữ hành vi cũ):
- `--phrase-match {legacy,word}` cho cả `oti_inversion.py` và `train_phi.py`: `word` = thay **lần xuất hiện đầu** theo ranh giới từ, giống nhau ở cả hai file.
- `--concept-filter {legacy,per-phrase}`: `per-phrase` = lọc từng câu, không bỏ cả khái niệm.
- Sửa lỗi `saved_exp_name` (vấn đề 5).

Kiểm tra không hồi quy: với `legacy`, 2 epoch đầu của train phi ra đúng số cũ (như [08](08_hoc_may_cai_tien_phi.md) mục 8.0).

## 6.3. Sinh lại bằng LLM mới (P2, P3)

**Môi trường riêng:** môi trường dự án là Python 3.8 + torch 1.11, quá cũ cho các thư viện sinh văn bản tốc độ cao (vLLM). Bước sinh câu là bước độc lập, nên chạy trong **một runtime Colab riêng**; đầu ra là file pickle cùng định dạng (ghi bằng `protocol=4` để Python 3.8 đọc được), code OTI và phi dùng lại nguyên qua `--gpt-exp-name`.

**Mô hình:** chọn một LLM mở, chạy được trên một GPU Colab.
- P2: mô hình nền (base) cỡ 1–3B tham số để viết tiếp prompt `a photo of {khái niệm} that`, giống cách GPT-Neo được dùng.
- P3: mô hình instruct cỡ 3–8B tham số.
- Ghi tên mô hình, phiên bản, tham số sinh vào `hyperparameters.json`. **Kiểm tra giấy phép** của mô hình trước khi dùng và ghi vào báo cáo.

**Prompt cho P3** (cố định, lưu vào repo `tools/prompts/p3.txt`), ý chính:
> Viết N câu tiếng Anh ngắn, mỗi câu bắt đầu bằng "a photo of {khái niệm} that", phần sau mô tả một đặc điểm **nhìn thấy được** (màu, hình dạng, chất liệu, số lượng, vị trí, bối cảnh, ánh sáng, góc chụp). Mỗi câu phải chứa đúng từ "{khái niệm}". Trả về danh sách JSON.

**Hậu kỳ chung cho P2, P3:** cùng bộ lọc của mục 6.2 (ranh giới từ, bỏ trùng, ≤ 77 token CLIP), bỏ câu mở đầu kiểu "Here are…", lọc câu không phải tiếng Anh. Số câu mỗi khái niệm: mặc định 256 như P0; nếu tốn quá thì 64 và ghi rõ.

**Quy mô nhỏ trước (D16):** E6.1 cho thấy top-150 khái niệm của phi đã phủ 99% trong 20.931 khái niệm (đo trên 6.824 ảnh), nên "khái niệm của shard s0" gần như là toàn bộ. Vì vậy chạy theo hai lượt, 64 câu mỗi khái niệm:
1. `SCOPE = "circo_val"`: top-15 khái niệm của ảnh CIRCO val (cần E7.0 của H17), vài nghìn khái niệm, đủ cho H18 (E6.2);
2. `SCOPE = "s0"` (gần như mọi khái niệm) cho D17 (E6.4), chỉ khi lượt 1 đo được thời gian nằm trong ngân sách GPU.

**Script:** `colab/gen_phrases.ipynb` (runtime riêng), lưu tiến độ mỗi 500 khái niệm lên Drive để chạy tiếp khi Colab ngắt.

## 6.4. Đánh giá (từ rẻ đến đắt)

| Mức | Mã | Cách đo | Chi phí | Ghi chú |
|---|---|---|---|---|
| Nội tại | E6.1 | `tools/phrases_stats.py` cho P0–P3 + độ đa dạng (distinct-2) | CPU, vài phút mỗi bộ | Có bảng so sánh chất lượng bộ câu |
| Rẻ | E6.2 | **SEARLE-OTI trên CIRCO val** với từng bộ câu (P0, P1, P2, P3, Pλ0); cùng seed, 350 bước | Dưới 7 phút mỗi lần | OTI dùng câu trực tiếp nên đây là cách nhanh nhất thấy tác dụng. Cần E7.0 |
| Rẻ | E6.3 | Như E6.2 trên CIRR val, 2 bộ tốt nhất + P0 | Khoảng 1 giờ mỗi lần (ước tính) | Xác nhận trên tập lớn hơn |
| Trung bình | E6.4 | **Train phi trên shard s0** với từng bộ câu, cấu hình giống T1 (`phi_scale_s0`), dùng token OTI có sẵn | 2–3 giờ mỗi lần | Chỉ GPT loss của phi thay đổi; token đích vẫn sinh từ P0. Baseline = T1 (= E8.1a) |
| Đắt | E6.5 | Chạy lại OTI trên shard s0 với bộ câu tốt nhất, rồi train phi, so với E8.1a (phi train trên token cũ của s0) | Khoảng 15 giờ A100 cho OTI một shard (lần trước, 3 shard chạy song song) + 2–3 giờ train | Chỉ làm khi E6.2–E6.4 cho thấy lợi rõ. Có thể dùng OTI warm-start (E7.6 ở [07](07_hoc_may_oti_warm_start.md)) để giảm chi phí |

Mọi so sánh dùng khoảng tin cậy bootstrap ([01](01_nen_tang_danh_gia.md) mục 1.4). Mã đầy đủ thêm hậu tố bộ câu và tập, ví dụ `E6.2-P3-circo`.

## Bảng phân công (người, ngày)

File nhiệm vụ [nhiem_vu_Chi.md](nhiem_vu_Chi.md) (từ 24/10 gộp cả việc của Hòa và Doanh) trỏ tới bảng này. Lịch xếp vào chỗ còn trống của luồng xử lý ảnh; mã việc nối tiếp các mã đã dùng trong file nhiệm vụ.

| Mã | Ngày | Người | Việc | Cần code mới? | GPU |
|---|---|---|---|---|---|
| H17 | 7 (T6 23/10) | Hòa | E7.0 cho CIRCO val (liên kết concept, script có sẵn); **E6.2-Pλ0-circo**: OTI 350 bước trên CIRCO val với `--lambda_gpt 0` | Không (code cũ) | Dưới 15 phút |
| C22 | 9 (CN 25/10) | Chí | `tools/phrases_stats.py`, `tools/clean_phrases.py` (tạo P1); cờ `--phrase-match`, `--concept-filter` cho OTI và train phi; sửa lỗi `saved_exp_name` | — | — |
| C23 | 11 (T3 27/10, sau M4) | Chí | `colab/gen_phrases.ipynb` (runtime riêng), chế độ quy mô nhỏ | — | Chạy thử ngắn |
| D16 | 12 (T4 28/10) | Doanh | E6.1 cho P0 và P1; **sinh P2 hoặc P3 quy mô nhỏ**: chỉ các khái niệm dùng cho ảnh shard s0, 64 câu mỗi khái niệm; E6.1 cho bộ mới | Có (C22, C23) | 1–3 giờ A100 |
| H18 | 13 (T5 29/10) | Hòa | **E6.2** trên CIRCO val với P1 và bộ mới của D16 (cùng seed với E7.1a) | Có (C22) | Dưới 30 phút |
| D17 | 13 (T5 29/10) | Doanh | **E6.4**: train phi trên shard s0 với P1 và bộ mới, cấu hình giống T1 (`phi_scale_s0`); so với T1 | Có (C22) | 2–3 giờ mỗi lần |
| — | Sau ngày 14 hoặc khi còn GPU | Tuỳ | Sinh bộ câu đầy đủ; E6.3 (CIRR val); E6.5 | | |

Dùng chung, không chạy hai lần:
- **E6.2-P0-circo = E7.1a-random-lr0.02-circo** (H15, ngày 11). H18 so với kết quả của H15.
- **E7.0** chạy ở H17 (ngày 7); H15 dùng lại, bỏ qua bước E7.0.
- **Baseline của E6.4 = T1** (`phi_scale_s0`, = E1.14a = E8.1a), không train lại.

Vì D16 chạy ở quy mô nhỏ (khái niệm của shard s0, 64 câu), kết quả E6.2 và E6.4 là bằng chứng sơ bộ; ghi rõ quy mô khi báo cáo.

## Kết quả cần có

1. Bảng chất lượng nội tại của P0–P3 (E6.1).
2. Bảng CIRCO val và CIRR val theo bộ câu, gồm cả Pλ0 (E6.2–E6.3).
3. Bảng phi sàng lọc theo bộ câu (E6.4).
4. Kết luận: làm sạch (P1) có đủ không, đổi mô hình (P2) và đổi kiểu câu (P3) mỗi thứ đóng góp bao nhiêu, kèm khoảng tin cậy.

## Tiêu chí hoàn thành

- [ ] Có `stats.json` cho P0 và kết luận về vấn đề 2–4.
- [ ] Có P1, và ít nhất một trong P2 hoặc P3.
- [ ] Có E6.2 trên CIRCO val cho mọi bộ câu đã tạo, kèm Pλ0.
- [ ] Các cờ mới mặc định giữ hành vi cũ, đã qua kiểm tra không hồi quy.

## Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| LLM không giữ đúng từ khái niệm, viết lan man | Bộ lọc hậu kỳ; đo tỉ lệ câu bị loại; giảm temperature |
| Sinh câu tốn GPU hơn dự kiến | Đo thử 200 khái niệm trước; chỉ sinh cho khái niệm thực sự được dùng; giảm còn 64 câu mỗi khái niệm |
| Bộ câu mới không làm tốt hơn (chênh lệch nằm trong nhiễu) | Vẫn báo cáo như một ablation về vai trò của câu điều chuẩn; Pλ0 cho biết giới hạn trên của ảnh hưởng |
| Danh sách khái niệm bị lệch (vấn đề 7) | So file CSV hiện tại với bản gốc của repo SEARLE trước khi sinh |
| Giấy phép mô hình sinh câu | Chọn mô hình giấy phép mở, ghi rõ trong báo cáo |

## Lịch sử thay đổi

- 2026-10-16: tạo file. Chuyển "Sinh lại cụm từ GPT" từ danh sách để sau BTL vào kế hoạch chính theo yêu cầu; gộp phần giải thích về `concept_to_phrases.pkl` từ file 06 cũ (`06_de_xuat_sau_btl.md`, đã xoá).
- 2026-10-16: thêm bảng phân công H17, C22, C23, D16, H18, D17 theo chỗ trống trong lịch; E6.4 dùng T1 (`phi_scale_s0`) làm baseline cùng quy mô; sinh câu quy mô nhỏ trước.
- 2026-10-18: C22, C23 xong; ghi kết quả E6.1 trên P0 và P1; quy mô nhỏ đổi thành hai lượt (CIRCO val trước, shard s0 sau) vì top-150 khái niệm của phi phủ gần hết danh sách.
