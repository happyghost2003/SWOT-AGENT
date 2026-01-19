# 🔎 SWOT AGENT - Công cụ phân tích SWOT cho F&B

Ứng dụng phân tích SWOT tự động sử dụng AI (Google Gemini) cho ngành F&B tại Việt Nam.

## ✨ Tính năng

- 📊 **Phân tích SWOT tự động** - Nhập tên quán hoặc upload file CSV
- ⚔️ **So sánh đối thủ** - So sánh SWOT giữa quán của bạn và đối thủ
- 📈 **So sánh nhiều quán** - So sánh và xếp hạng nhiều quán cùng lúc
- 🔍 **Phân tích chi nhánh** - Phân tích SWOT cho từng chi nhánh cụ thể
- 📥 **Xuất Excel** - Xuất kết quả phân tích để dùng với Power BI
- 📊 **Biểu đồ trực quan** - Hiển thị biểu đồ SWOT đẹp mắt

## 🛠️ Cài đặt thư viện

```bash
pip install streamlit pandas google-generativeai plotly openpyxl
```

Hoặc cài đầy đủ:

```bash
pip install -r requirements.txt
```

## 🚀 Khởi động ứng dụng

```bash
python3 -m streamlit run app.py
```

Hoặc:

```bash
streamlit run app.py
```

Sau đó mở trình duyệt và truy cập: **http://localhost:8501**

## 📁 Cấu trúc project

```
SWOT AGENT/
├── app.py              # File chính của ứng dụng
├── main.py             # File phụ (nếu có)
├── data/               # Thư mục chứa file CSV mẫu
├── requirements.txt    # Danh sách thư viện cần cài
└── README.md           # File này
```

## 📝 Hướng dẫn sử dụng

### 1. Phân tích SWOT đơn giản
- Nhập tên quán vào ô "Tên quán"
- Bấm nút "Phân tích SWOT"

### 2. So sánh với đối thủ
- Nhập tên quán của bạn
- Upload nhiều file CSV (của quán mình + đối thủ)
- AI sẽ tự động nhận diện và so sánh

### 3. Upload file CSV
- Mỗi file CSV là dữ liệu của 1 quán
- Đặt tên file rõ ràng (VD: `phuc_long.csv`, `starbucks.csv`)
- AI sẽ đọc toàn bộ dữ liệu từ file

## ⚙️ Cấu hình API Key

1. Copy file `.env.example` thành `.env`:
```bash
cp .env.example .env
```

2. Mở file `.env` và thay API key của bạn:
```
GOOGLE_API_KEY=your-api-key-here
```

3. Lấy API key tại: https://makersuite.google.com/app/apikey

> ⚠️ **Lưu ý:** File `.env` đã được thêm vào `.gitignore` nên sẽ KHÔNG bị push lên GitHub.

## 📦 Requirements

- Python 3.8+
- streamlit
- pandas
- google-generativeai
- plotly
- openpyxl (để xuất Excel)
- python-dotenv

## 🌐 Deploy lên Streamlit Cloud

1. Push code lên GitHub (file `.env` sẽ không bị push)
2. Truy cập https://share.streamlit.io/ và đăng nhập bằng GitHub
3. Chọn repo và branch
4. Vào **Settings > Secrets** và thêm:
```toml
GOOGLE_API_KEY = "your-api-key-here"
```
5. Bấm **Deploy**

## Tác giả

Phòng AI - SWOT Agent

