# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là hiện tượng "nghịch đảo chỉ số" (inverted evidence) ở NB4: cấu hình `attn_only` (r=283) đạt loss huấn luyện thấp hơn cấu hình `correct` (0.0531 < 0.0549), nhưng khi đánh giá trên tác vụ mục tiêu thì lại thua sút rõ rệt (0.8200 so với 0.9000). Điều này cho thấy loss huấn luyện chỉ là một chỉ số thay thế (proxy metric) và hoàn toàn có thể gây ngộ nhận nếu không đo đạc trực tiếp trên bài toán thực tế.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở khâu xác thực loss mask và chat template ở NB1. Ban đầu tôi nghĩ bước huấn luyện trên GPU sẽ tốn thời gian nhất, nhưng thực tế việc xử lý hiện tượng token merging qua ranh giới thẻ `<think>\n\n</think>\n\n` của kiến trúc Qwen3.5 mới là thử thách kỹ thuật tinh tế nhất, đòi hỏi phải dùng offset mapping thay vì diff token danh sách đơn thuần.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab này, tôi từng tin rằng: "Cứ fine-tune là mô hình sẽ tự động thông minh hơn và vượt trội hoàn toàn so với mô hình gốc". Giờ đây tôi nhận ra rằng nếu base model được prompt tử tế (prompt engineering chỉn chu), nó đã có thể giải quyết được 76.5% bài toán mà không tốn chi phí huấn luyện. Fine-tuning chỉ thực sự có giá trị khi nó vượt qua được chính mốc prompt tối ưu đó một cách thuyết phục mà không làm rơi rụng năng lực tổng quát.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant để hỗ trợ phân tích ma trận tham số các tầng chiếu của Qwen3.5 (nhận diện thêm 5 module linear-attention của Gated DeltaNet) và tự động hóa các hàm chuẩn hóa chuỗi tiếng Việt. Chỗ AI assistant dễ sai nhất là xu hướng mặc định đề xuất huấn luyện với precision bf16 (vốn không được hỗ trợ phần cứng trên chip Nvidia Turing T4 của Colab) và thường quên mask phần prompt của người dùng nếu không được nhắc nhở tường minh.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm là **đóng băng một tập kiểm thử đánh giá độc lập (eval set) và xây dựng một prompt tối ưu làm mốc chuẩn (baseline-b)**. Tôi sẽ đo đạc năng lực của mô hình gốc với prompt tối ưu trước. Nếu prompt tối ưu đã đáp ứng được yêu cầu của khách hàng về độ chính xác và chi phí, tôi sẽ khuyên khách hàng chưa cần vội fine-tune. Nếu bắt buộc fine-tune, mốc đóng băng này sẽ là bằng chứng khách quan duy nhất để chứng minh giá trị của dự án.
