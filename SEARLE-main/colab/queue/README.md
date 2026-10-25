# Hàng đợi chạy nền trên Colab

Các script này chạy toàn bộ thí nghiệm của đề tài trên một runtime Colab A100, nối tiếp nhau và chạy nền (`nohup`),
nên không phụ thuộc vào ô đang mở. Chúng được đặt ở `MyDrive/SEARLE/queue/` khi chạy.

- `common.sh`: biến dùng chung (`$R` = `MyDrive/SEARLE/results`, cờ `$PHI`, `$CIRR`, `$CIRCO`, `$FIQ`) và hàm `run <thư mục> <lệnh>`.
  `run` bỏ qua lượt đã có `.done`, ghi `stdout.log`, đánh dấu `.done` khi thành công, nên chạy lại script sau khi Colab ngắt không lặp việc cũ.
- `boot.sh`: dựng lại phiên sau khi Colab ngắt (clone/pull code, môi trường, dữ liệu, bản CIRCO trên ổ cục bộ), rồi khởi động mọi hàng đợi.
- Huấn luyện: `t1.sh` (T1, sàng lọc E3.8a), `t2.sh` (D10, φ bền `phi_B32_robust`), `t3.sh` (H16, `phi_B32_cosine`).
- Đánh giá: `q1.sh` (tái lập, baseline, E3.1, E3.5, E4.1–E4.3), `q2.sh` (E3.2, E2.3, E4.4–E4.5, E3.6–E3.7, phân tích E1.10–E1.15, E3.4),
  `q3.sh` (đánh giá sàng lọc E3.8a, tự chọn biến thể), `q4.sh` (E3.8c, E8.2a, E3.9), `q5.sh` (E2.3 mở rộng, E2.6, E2.4, E2.5, E2.7, khoảng tin cậy),
  `q6.sh` (E3.3), `q7.sh` (trọng số lớn), `q8.sh` (file nộp test H13/D13, hình E1.12/E1.13/E2.8, khoảng tin cậy φ bền).

Khởi động một hàng đợi:

```bash
source /content/env.sh
nohup bash $DRIVE/queue/q8.sh >> $DRIVE/queue/q8.log 2>&1 &
```

Lưu ý: kiểm tra một hàng đợi đang chạy bằng `pgrep -f "queue/q8[.]sh"` (dấu `[.]` để lệnh không tự khớp với chính nó).
