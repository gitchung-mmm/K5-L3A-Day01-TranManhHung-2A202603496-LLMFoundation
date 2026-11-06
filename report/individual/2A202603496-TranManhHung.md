# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603496
- **Họ và tên:** Trần Mạnh Hùng
- **SEED** 9460

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc_1.jpg`](hand_calc_1.jpg) — ảnh tính tay có các bước | ☑ |
| [`hand_calc_2.jpg`](hand_calc_2.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

*Bảng tổng hợp so sánh các phương pháp thu được từ bài thực hành:*

| Phương pháp | F1 (test toàn bộ) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 0.925 | 1.000 | 0.09 ms | $0.00 |
| **2. BoW + MLP (deep learning)** | 0.977 | 1.000 | 0.27 ms | $0.00 |
| **3a. Encoder (MiniLM) + kNN** | 0.941 | 0.952 | 68.50 ms | $0.00 |
| **3b. Encoder (MiniLM) + LogReg** | 0.971 | 1.000 | 45.20 ms | $0.00 |
| **3c. Encoder (MiniLM) + MLP head** | 0.968 | 1.000 | 46.10 ms | $0.00 |
| **4a. LLM nhỏ zero-shot** | — | 0.950 | 1250.00 ms | $0.05 |
| **4b. LLM lớn zero-shot** | — | 0.975 | 2400.00 ms | $0.50 |

*(Ghi chú: Các hàng LLM chỉ đo trên tập SUB 40 tin để tiết kiệm chi phí; các mô hình còn lại đo trên toàn bộ tập Test 776 tin).*

## 3. Dự đoán và thực tế

3 dự đoán có độ lệch đáng chú ý nhất giữa suy đoán ban đầu và thực nghiệm:

1. **Dự đoán `ml_lr` (Learning rate = 50 ở Phần 1):**
   - *Dự đoán ban đầu:* Tôi dự đoán `lr=50` sẽ khiến mô hình nhảy vọt qua lại và hàm Loss dao động răng cưa (zigzag) dữ dội suốt 300 epochs và không thể giảm.
   - *Thực tế:* Sau khi vọt lên đỉnh nhọn ở 2-3 epoch đầu (~2.7), Loss bất ngờ tụt thẳng đứng xuống sát 0 (`0.005`).
   - *Lý do lệch:* Do `lr` quá lớn làm trọng số $w$ bị thổi phồng cực đại, khiến hàm Sigmoid bão hòa hoàn toàn (ép xác suất $p$ về sát 1 hoặc 0). Mô hình trở nên "tự tin thái quá" khiến hàm Loss trên giấy tờ tụt về sát 0 nhưng thực chất các gradient đã bị triệt tiêu về 0 và tê liệt việc học.

2. **Dự đoán `dl_curve` (So sánh F1 giữa MLP và Logistic Regression ở Phần 2):**
   - *Dự đoán ban đầu:* Tôi nghĩ MLP sẽ chỉ ngang bằng hoặc nhỉnh hơn Logistic Regression một chút xíu do cùng dùng chung input Bag-of-Words.
   - *Thực tế:* MLP đạt F1 test tới 0.977 (cao hơn rõ rệt so với 0.925 của LogReg ở Phần 1).
   - *Lý do lệch:* Nhờ có 64 nơ-ron lớp ẩn và hàm kích hoạt ReLU phi tuyến, MLP đã tự học cách bắt cặp các tổ hợp từ khóa đi chung (ví dụ cụm từ khẩn cấp kết hợp tiền tệ) tốt hơn nhiều so với việc chỉ cộng điểm từng từ độc lập của LogReg.

3. **Dự đoán `tradeoff` (Vị trí trên đường Pareto ở Phần 5):**
   - *Dự đoán ban đầu:* Dự đoán cả BoW+LogReg và Encoder+LogReg sẽ cùng nằm trên đường Pareto.
   - *Thực tế:* Trên tập SUB 40 tin, chỉ có duy nhất `1. BoW + LogReg` nằm trên đường Pareto.
   - *Lý do lệch:* Do tập SUB chỉ có 40 tin, BoW+LogReg đã đạt điểm tối đa $F_1 = 1.0$ (tỷ lệ lỗi $1-F_1 = 0\%$) lại có tốc độ siêu nhanh (~0.09 ms). Một khi đã đạt lỗi 0% ở độ trễ cực thấp, nó lấn át (dominate) toàn bộ các mô hình khác trên biểu đồ tập SUB.

## 4. Phân tích 50 case khó

- **Mô hình tụt điểm nhiều nhất:** Các mô hình thuộc họ Bag-of-Words (cả LogReg và MLP) tụt dốc thảm hại nhất, từ mức $F_1 \approx 0.93 - 0.98$ trên tập test thường rơi xuống chỉ còn khoảng $0.45 - 0.55$ trên 50 case khó.
- **Nhóm case khó nhất với BoW:** Nhóm *"Tưởng spam nhưng không phải"* và nhóm *"Né từ khoá"*.
- **2 case cụ thể và cơ chế khiến model sai:**
  1. *Case 1 (Tưởng spam nhưng là ham):* `"Mum says the prize for the school raffle is a cake lol"` (Nhãn thật: HAM).
     - *Cơ chế sai của BoW:* BoW bắt trúng từ `"prize"`. Từ này có trọng số dương rất lớn trong từ điển, làm điểm số $z$ vọt lên cao khiến mô hình chặn nhầm thành SPAM (báo động giả). Mô hình Embedding và LLM hiểu được ngữ cảnh đời thường ("school raffle", "cake", "mum") nên phân loại đúng.
  2. *Case 2 (Spam ngụy trang né từ khoá):* `"Exclusive: get a brand new phone at no cost, just reply with your address"` (Nhãn thật: SPAM).
     - *Cơ chế sai của BoW:* Người gửi dùng cụm từ `"at no cost"` thay vì dùng từ khóa `"free"`. BoW không tìm thấy từ khóa quen thuộc trong từ điển 2.000 từ nên cho rằng tin này vô hại (HAM) và bỏ lọt. Mô hình Embedding nhận biết `"at no cost"` đồng nghĩa với `"free"` nên bắt chính xác là SPAM.

## 5. Thử thách cá nhân

- **Đề nhận được:** **Đề #5 — Embedding:** Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.
- **4 tin nhắn tự thiết kế:**
  1. *Spam 1:* `"Someone left an unclaimed package for you, verify details at tinyurl.com/pkg92 to receive it"` (Nhãn thật: 1 - SPAM).
  2. *Spam 2:* `"Security alert: Unauthorized access detected on your card. Click link below immediately to prevent suspension"` (Nhãn thật: 1 - SPAM).
  3. *Ham 1:* `"Are you free this evening to help me practice for tomorrow's presentation?"` (Nhãn thật: 0 - HAM).
  4. *Ham 2:* `"Our team managed to win the championship game last night, celebrate with us later!"` (Nhãn thật: 0 - HAM).
- **Kết quả thực nghiệm:**
  - **BoW + LogReg:** Đúng 0/4 tin (bỏ lọt cả 2 tin spam do không có từ khóa lộ liễu; chặn nhầm cả 2 tin ham vì thấy có chữ `free` và `win`).
  - **Embedding + LogReg:** Đúng 4/4 tin tuyệt đối (100%).
- **Giải thích:** BoW chỉ kiểm tra sự có mặt của từ ngữ rời rạc nên dễ bị đánh lừa bởi từ đồng nghĩa hoặc ngữ cảnh thông thường. Embedding mã hóa câu thành vector 384 chiều thể hiện đúng ý định giao tiếp tổng thể, phân biệt được "free time" (rảnh rỗi) khác với "free prize" (lừa đảo).

## 6. Phương án cho sản phẩm thật

- **Bài toán:** Lọc 10 triệu SMS/ngày (~116 tin/giây), yêu cầu độ trễ $p95 < 50\text{ ms/tin}$.
- **Lựa chọn phương án:** Tôi đề xuất giải pháp kiến trúc hỗn hợp **2 tầng (Cascading Router)**:
  - **Tầng 1 (Chính - 95% lưu lượng):** Dùng mô hình **3b. Encoder (MiniLM) + LogReg**. 
    - *Số liệu chứng minh:* Độ trễ $p95 \approx 45.2\text{ ms}$ (đạt chuẩn SLA $< 50\text{ ms}$), throughput cao, chi phí phần cứng rẻ ($0 USD tiền gọi API), $F_1$ rất cao (~0.97) và có khả năng chống chọi các case ngụy trang.
    - Nếu xác suất dự đoán $p > 0.85$ (chắc chắn spam) hoặc $p < 0.15$ (chắc chắn ham), chốt nhãn ngay tại tầng 1.
  - **Tầng 2 (Vùng lưỡng lự - 5% lưu lượng còn lại):** Với các tin nhắn nằm trong vùng phân vân ($0.15 \le p \le 0.85$), chuyển tiếp sang **LLM nhỏ (gpt-6-luna / Haiku)** với prompt có thẻ phòng thủ `<sms>`.
- **Đánh đổi chấp nhận:**
  - Loại bỏ hoàn toàn việc dùng LLM cho 100% lưu lượng vì độ trễ quá lớn ($> 1.000\text{ ms}$) và chi phí cực kỳ đắt đỏ (~15.000 USD/tháng cho model nhỏ, ~150.000 USD/tháng cho model lớn).
  - Chấp nhận đầu tư cụm máy chủ có GPU vừa phải để chạy mô hình Encoder MiniLM nhằm đổi lấy độ bền vững cao trước các kỹ thuật spam biến tướng ngoài đời thực.

## 7. AI log và điều đã học

### Bảng ghi nhận AI log:

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| :--- | :--- | :--- | :--- |
| **Phần 1** | Hỏi về bản chất của Gradient Descent và cách cập nhật trọng số | Giải thích cơ chế dùng sai số $(p - y)$ nhân tần suất từ để sửa $w$, giải thích $w$ là núm vặn điểm số | Tự kiểm tra lại 4 bước trong `train_logreg`, quan sát loss giảm từ 0.69 về 0.3 |
| **Phần 2** | Hỏi về ý nghĩa của 64 nơ-ron lớp ẩn và tác dụng của hàm `nn.ReLU()` | Giải thích nơ-ron giúp gộp các từ lại để hiểu ngữ cảnh, ReLU tạo tính phi tuyến | Tính nhẩm 128.129 tham số, kiểm tra hiện tượng Overfitting tại epoch 12 trên biểu đồ val loss |
| **Phần 3** | Hỏi về cơ chế Cosine Similarity và tại sao Embedding né được từ khóa | Giải thích ánh xạ câu thành vector 384 chiều, từ đồng nghĩa nằm gần nhau trên bản đồ nghĩa | Chạy Thí nghiệm 3.B, kiểm chứng BoW đoán sai tin "free for lunch" còn Embedding đoán đúng |
| **Phần 4 & 5** | Hỏi về thiết kế Router lai giữa Embedding và LLM, phân tích đường Pareto | Hướng dẫn tính chi phí theo triệu token, phân tích đánh đổi SLA độ trễ < 50ms | So sánh biểu đồ Pareto thực tế, hoàn thành 4 tin nhắn tự viết cho Thử thách Đề #5 |

### Điều tôi hiểu ra hôm nay:
Trước buổi lab, tôi từng nghĩ Deep Learning hay LLM luôn là lựa chọn tốt nhất cho mọi bài toán AI. Hôm nay tôi đã hiểu ra rằng mỗi thế hệ AI đều có sự đánh đổi rõ rệt giữa **Chất lượng - Độ trễ - Chi phí**: một mô hình ML/Embedding nhỏ gọn, chạy nhanh và miễn phí hoàn toàn có thể đánh bại LLM khổng lồ trong các ứng dụng thực tế đòi hỏi phản hồi mili-giây. Ngoài ra, tôi học được bài học đắt giá rằng **Accuracy trên tập test rất dễ đánh lừa**, và việc kiểm thử qua các case đối kháng (Stress Test) là bước bắt buộc để xây dựng một hệ thống AI thực sự bền vững.