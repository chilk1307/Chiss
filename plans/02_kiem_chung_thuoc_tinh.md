# 02. Đóng góp chính: CLIP đề xuất, xử lý ảnh kiểm chứng

**Tóm tắt:** SEARLE (dựa trên CLIP) tìm ra top-100 ảnh ứng viên. Sau đó dùng **kỹ thuật xử lý ảnh cổ điển** đo trực tiếp trên ảnh các thuộc tính mà câu mô tả yêu cầu (màu, độ sáng, độ rực màu, hoa văn, mức phóng to), rồi xếp hạng lại. Không cần train, chạy trên CPU, giải thích được vì sao một ảnh được đẩy lên.

**Thời gian:** code Ngày 5–7; thí nghiệm Ngày 6–12.
**Người làm:** Chí viết code (C11, C12, C13); Hòa chạy thí nghiệm (H9–H13); Doanh chạy E3.6 dùng bộ đo của file này.

**Đóng góp dự kiến:**
1. Bộ đo thuộc tính thị giác bằng xử lý ảnh cổ điển, được kiểm chứng trên ảnh tổng hợp và trên cặp ảnh thật của benchmark.
2. Bước xếp hạng lại không cần train, giải thích được ("ảnh này được đẩy lên vì tối hơn 23%"), cắm được sau SEARLE hoặc bất kỳ phương pháp ZS-CIR nào.
3. Phân tích theo từng loại thuộc tính: khi nào xử lý ảnh cổ điển bổ sung được cho CLIP, khi nào không.

## Vì sao làm hướng này

- Bộ mã hoá văn bản của CLIP xử lý kém câu so sánh, phủ định và việc gắn thuộc tính cho đúng đối tượng; hiện tượng "túi từ" này đã được chỉ ra trong bài ARO (Yuksekgonul et al., ICLR 2023). SEARLE dồn mọi thứ vào câu `a photo of $ that <mô tả>` nên thừa hưởng các điểm yếu đó.
- Nhiều câu sửa đổi trong CIR là thuộc tính đo được trực tiếp trên ảnh: "is darker", "is red", "has stripes", "is more colorful", "black and white", "closer view". Tỉ lệ cụ thể sẽ đo ở E2.0.
- Xử lý ảnh cổ điển đo các thuộc tính này một cách tường minh, giải thích được và không cần dữ liệu train.

## Giả thuyết

- **H1:** với truy vấn có thuộc tính đo được, xếp hạng lại cải thiện R@K (CIRR, FIQ) và mAP@K (CIRCO) so với SEARLE; truy vấn không có thuộc tính giữ nguyên kết quả.
- **H2:** thuộc tính tương đối ("tối hơn", "rực hơn") được lợi nhiều hơn thuộc tính tuyệt đối ("màu đỏ"), vì CLIP yếu hơn ở phép so sánh.
- **H3:** tách đối tượng khỏi nền trước khi đo làm bộ đo chính xác hơn, nhất là với ảnh cảnh (CIRR, CIRCO).
- **H4:** xử lý phủ định tường minh ("no stripes", "not red") cải thiện rõ trên nhóm truy vấn phủ định.

## Sơ đồ

```text
ảnh tham chiếu + câu sửa đổi
   │
   ├──► SEARLE (phi_B32) ──► top-K ứng viên, điểm CLIP
   │
   └──► bộ tách ràng buộc (từ khoá) ──► các ràng buộc {loại, giá trị, phủ định}
                                              │
   ảnh tham chiếu, ảnh ứng viên ──► bộ đo thuộc tính (xử lý ảnh cổ điển) ──► mức thoả mãn c ∈ [0, 1]
                                              │
                 điểm mới = điểm CLIP (chuẩn hoá) + w · trung bình(c) ──► xếp hạng lại top-K
```

## 2.1. Bộ tách ràng buộc từ câu mô tả (Chí: C11, Ngày 5)

File mới `src/attributes/parser.py`, dùng từ điển và luật (không cần LLM).

| Loại | Từ khoá ví dụ | Ràng buộc với ảnh ứng viên |
|---|---|---|
| Màu tuyệt đối | red, blue, green, yellow, black, white, gray/grey, pink, purple, orange, brown, beige, navy | màu chủ đạo gần với màu đó |
| Độ sáng tương đối | darker, lighter, brighter, dimmer | L* thấp hơn / cao hơn ảnh tham chiếu |
| Độ rực màu | more colorful, colorful, less colorful, black and white, monochrome, grayscale | chỉ số colorfulness, độ bão hoà cao hơn / thấp hơn |
| Hoa văn | striped/stripes, plaid/checkered, polka dot, floral, patterned/print, solid/plain; thêm hướng: vertical, horizontal | năng lượng cạnh, tính tuần hoàn, hướng sọc |
| Mức phóng to | closer, close-up, zoom in, zoomed out, further away | diện tích đối tượng lớn hơn / nhỏ hơn |

- **Phủ định:** "not", "no", "without", "less" đảo chiều ràng buộc trong phạm vi vài từ phía sau.
- **FashionIQ:** mỗi truy vấn có 2 câu mô tả, gộp ràng buộc của cả hai.
- **Kiểm thử:** bộ khoảng 50 câu tự viết kèm ràng buộc mong đợi; Chí chấm tay thêm 100 câu ngẫu nhiên của CIRR và FIQ để đo độ chính xác.

## 2.2. Bộ đo thuộc tính (Chí: C12, Ngày 6)

File mới `src/attributes/meters.py`, dùng OpenCV và scikit-image; ảnh thu nhỏ về cạnh dài 256 px cho nhanh.

| Bộ đo | Cách tính | Kiến thức xử lý ảnh |
|---|---|---|
| Mặt nạ đối tượng | FashionIQ (nền trắng): ngưỡng Otsu trên độ bão hoà và độ sáng, phép mở/đóng hình thái, giữ thành phần liên thông lớn nhất. CIRR, CIRCO (ảnh cảnh): GrabCut khởi tạo bằng hình chữ nhật ở giữa ảnh, hoặc bản đồ nổi bật spectral residual (Hou & Zhang, CVPR 2007) | Phân ngưỡng, hình thái học, phân vùng, miền tần số |
| Màu chủ đạo | k-means (k = 3–5) trên điểm ảnh của đối tượng trong không gian CIELAB; khoảng cách CIEDE2000 tới màu chuẩn của từ chỉ màu | Không gian màu, phân cụm |
| Độ sáng | trung vị L* của vùng đối tượng | Không gian CIELAB, histogram |
| Độ rực màu | chỉ số colorfulness của Hasler & Süsstrunk (2003); độ bão hoà trung bình trong HSV | Xử lý ảnh màu |
| Hoa văn | mật độ cạnh Canny trong vùng đối tượng; tỉ lệ năng lượng của đỉnh mạnh nhất (ngoài thành phần một chiều) trong phổ biên độ Fourier, vì sọc tạo đỉnh rõ; entropy histogram LBP | Phát hiện cạnh, biến đổi Fourier, kết cấu |
| Hướng sọc | góc của đỉnh phổ Fourier; hoặc hướng có năng lượng lớn nhất trong bộ lọc Gabor 4 hướng | Miền tần số, lọc Gabor |
| Mức phóng to | tỉ lệ diện tích mặt nạ trên diện tích ảnh | Phân vùng |

**Mức thoả mãn c ∈ [0, 1]:**
- Thuộc tính tuyệt đối (màu): `c = Σ_k p_k · exp(−ΔE_k / τ)`, với `p_k` là tỉ lệ điểm ảnh của cụm k.
- Thuộc tính tương đối (ví dụ "tối hơn"): `c = sigmoid((L_tham_chiếu − L_ứng_viên − δ) / τ)`.
- Phủ định: `c' = 1 − c`.

**Cache:** thuộc tính của mỗi ảnh chỉ tính một lần. Chỉ cần tính cho ảnh tham chiếu và các ảnh nằm trong top-K, không cần toàn bộ gallery (CIRCO có khoảng 123 nghìn ảnh).

**Kiểm thử trên ảnh tổng hợp (E2.1, Chí chạy trên máy):** ảnh đơn sắc từng màu; sọc hình sin nhiều tần số và hướng; một ảnh được làm tối/sáng, tăng/giảm bão hoà theo từng mức; hình tròn nhiều kích thước. Bộ đo phải xếp đúng thứ tự trên các ảnh này.

## 2.3. Xếp hạng lại (Chí: C13, Ngày 7)

- Lấy top-K của SEARLE (mặc định K = 100, để có thể thay đổi cả R@50) cùng điểm CLIP.
- Chuẩn hoá điểm CLIP trong top-K (z-score) để cùng thang với c, rồi `điểm mới = điểm CLIP + w · trung bình(c)`; chọn w trên CIRR val.
- Truy vấn không có ràng buộc nào: giữ nguyên kết quả.
- Cài vào module chung qua các tham số `--attr-*` ([01](01_nen_tang_danh_gia.md) mục 1.3).
- Kết hợp được với kết hợp điểm và ensemble prompt ở [04](04_tien_xu_ly_va_truy_van.md).

## Thí nghiệm

| Mã | Nội dung | Mục đích | Người chạy | Ngày |
|---|---|---|---|---|
| E2.0 | Tỉ lệ truy vấn có từng loại thuộc tính trong CIRR val, FIQ val, CIRCO val | Quyết định đi tiếp hay không | Hòa | 7 |
| E2.1 | Kiểm thử bộ đo trên ảnh tổng hợp | Bộ đo đo đúng thứ cần đo | Chí | 6 |
| E2.2 | Độ đồng thuận bộ đo – câu mô tả trên cặp (tham chiếu, đích) thật. Ví dụ: với truy vấn "darker", bao nhiêu phần trăm cặp có L*(đích) < L*(tham chiếu)? | Bộ đo khớp cách người viết mô tả | Hòa | 7 |
| E2.3 | Xếp hạng lại với từng họ thuộc tính riêng; quét w = 0.05, 0.1, 0.2, 0.3 và K = 50, 100 | H1, H2 | Hòa | 8–9 |
| E2.4 | Mặt nạ: không tách / Otsu / GrabCut / spectral residual | H3 | Hòa | 9 |
| E2.5 | Có và không xử lý phủ định | H4 | Hòa | 9 |
| E2.6 | Kết hợp mọi họ thuộc tính; kết hợp thêm với [04](04_tien_xu_ly_va_truy_van.md) | Cấu hình cuối | Hòa | 10 |
| E2.7 | Cấu hình cuối trên CIRCO val và FIQ val, **không chỉnh lại tham số** | Tổng quát hoá | Hòa | 10 |
| E2.8 | 10–20 ví dụ kèm lời giải thích | Báo cáo, demo | Hòa | 11 |
| E2.9 | Nộp test CIRR và CIRCO cho cấu hình cuối | Số chính thức | Hòa | 12 |

**Điểm quyết định ở mốc M3 (Ngày 7), sau E2.0 và E2.2** (ngưỡng gợi ý, nhóm có thể chỉnh): nếu truy vấn có thuộc tính chiếm dưới 10% ở cả CIRR và FIQ, **hoặc** độ đồng thuận ở E2.2 dưới 65% với hầu hết các loại, thì [03](03_do_ben_va_phuc_hoi_anh.md) thành đóng góp chính, file này giữ lại như một phân tích (xem README mục 6).

## Kết quả cần có

1. Bảng tỉ lệ thuộc tính (E2.0).
2. Bảng kiểm chứng bộ đo (E2.1, E2.2).
3. Bảng kết quả chính: SEARLE và SEARLE + xếp hạng lại, trên nhóm có thuộc tính và trên toàn bộ truy vấn, cho cả 3 tập, kèm khoảng tin cậy.
4. Bảng ablation (E2.3–E2.5).
5. Hình ví dụ kèm lời giải thích; sơ đồ phương pháp.

## Tiêu chí hoàn thành

- [ ] Parser và bộ đo có kiểm thử (E2.1, độ chính xác của parser).
- [ ] Trả lời H1–H4 bằng số liệu có khoảng tin cậy.
- [ ] Cấu hình cuối có số trên 3 tập val và trên test.
- [ ] Tắt `--attr-rerank` thì kết quả trùng baseline.

## Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Từ chỉ màu nói về một vật nhỏ trong ảnh cảnh ("make the dog brown"), bộ đo trên toàn ảnh đo sai | Báo cáo riêng FIQ (một đối tượng, nền trắng) và CIRR; dùng mặt nạ nổi bật cho ảnh cảnh; nêu rõ giới hạn |
| Parser bỏ sót cách diễn đạt | Mở rộng từ điển dựa trên E2.0 và kết quả chấm tay |
| Cải thiện chỉ trên nhóm nhỏ, tổng gần như không đổi | Báo cáo riêng từng nhóm; đó vẫn là một phát hiện có giá trị |
| Thang điểm CLIP và mức thoả mãn khác nhau | Chuẩn hoá trong top-K; chọn w trên val |
| FIQ thiếu ảnh | Dùng tập con cố định ([01](01_nen_tang_danh_gia.md) mục 1.2) |
| Đã có công trình tương tự | Tra cứu kỹ ([05](05_demo_va_bao_cao.md) mục 5.4) trước khi viết là "mới"; nếu có thì so sánh và nêu rõ điểm khác |

## Lịch sử thay đổi

- 2026-10-16: tạo kế hoạch.
