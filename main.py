"""
SWOT AGENT - Phân Tích Quán Cafe/Nhà Hàng
Sử dụng Google Gemini LLM
"""

import os
import glob
import pandas as pd
import google.generativeai as genai

# ============================================
# CẤU HÌNH API
# ============================================
GOOGLE_API_KEY = "AIzaSyDw_uBs_QUItg2KqiQF9cMu6pHW--pvJR8"
genai.configure(api_key=GOOGLE_API_KEY)

# Khởi tạo model
model = genai.GenerativeModel('models/gemini-flash-latest')


# ============================================
# ĐỌC VÀ XỬ LÝ CSV
# ============================================
def load_all_csv(data_folder="data"):
    """Đọc tất cả file CSV trong thư mục data"""
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    if not csv_files:
        return None, "Không tìm thấy file CSV nào trong thư mục data/"
    
    all_data = []
    file_info = []
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            file_info.append({
                "file": os.path.basename(file_path),
                "rows": len(df),
                "columns": list(df.columns)
            })
            print(f"✓ Đã đọc: {os.path.basename(file_path)} ({len(df)} dòng)")
        except Exception as e:
            print(f"✗ Lỗi đọc {file_path}: {e}")
    
    return all_data, file_info


def summarize_csv_data(dataframes, file_info):
    """Tóm tắt dữ liệu từ CSV để gửi cho AI"""
    if not dataframes:
        return ""
    
    summary = "📊 DỮ LIỆU TỪ CSV:\n"
    
    for i, (df, info) in enumerate(zip(dataframes, file_info)):
        summary += f"\n--- File: {info['file']} ---\n"
        summary += f"Số dòng: {info['rows']}\n"
        summary += f"Các cột: {', '.join(info['columns'])}\n"
        
        # Thống kê cơ bản cho các cột số
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                summary += f"- {col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.0f}\n"
        
        # Sample data
        summary += f"Mẫu dữ liệu:\n{df.head(5).to_string()}\n"
    
    return summary


# ============================================
# PHÂN TÍCH SWOT VỚI GEMINI
# ============================================
def call_gemini(prompt):
    """Gọi Gemini API"""
    response = model.generate_content(prompt)
    return response.text


def analyze_swot_by_name(shop_name):
    """Phân tích SWOT chỉ dựa trên tên quán"""
    prompt = f"""
Bạn là chuyên gia phân tích kinh doanh F&B tại Việt Nam.

🏪 QUÁN CẦN PHÂN TÍCH: {shop_name}

YÊU CẦU:
1. Hãy tìm hiểu và phân tích quán này (dựa trên kiến thức của bạn về thị trường F&B Việt Nam)
2. Thực hiện phân tích SWOT chi tiết:

📗 STRENGTHS (Điểm mạnh):
- ...

📕 WEAKNESSES (Điểm yếu):
- ...

📘 OPPORTUNITIES (Cơ hội):
- ...

📙 THREATS (Thách thức):
- ...

💡 ĐỀ XUẤT CHIẾN LƯỢC:
- Đưa ra 3 đề xuất cụ thể để cải thiện kinh doanh

Hãy phân tích chi tiết, thực tế và phù hợp với thị trường Việt Nam.
"""
    return call_gemini(prompt)


def analyze_swot_with_csv(shop_name, csv_summary):
    """Phân tích SWOT kết hợp CSV data và tên quán"""
    prompt = f"""
Bạn là chuyên gia phân tích kinh doanh F&B tại Việt Nam.

🏪 QUÁN CẦN PHÂN TÍCH: {shop_name}

{csv_summary}

YÊU CẦU:
Dựa trên dữ liệu CSV ở trên VÀ kiến thức của bạn về quán này, hãy phân tích SWOT:

📗 STRENGTHS (Điểm mạnh):
- Phân tích dựa trên data thực tế

📕 WEAKNESSES (Điểm yếu):
- Chỉ ra vấn đề từ data

📘 OPPORTUNITIES (Cơ hội):
- Cơ hội phát triển

📙 THREATS (Thách thức):
- Rủi ro tiềm ẩn

💡 ĐỀ XUẤT CHIẾN LƯỢC:
- 3 đề xuất cụ thể dựa trên data

📈 INSIGHTS TỪ DATA:
- Những điểm đáng chú ý từ dữ liệu

Phân tích thật chi tiết và actionable!
"""
    return call_gemini(prompt)


def analyze_csv_only(csv_summary):
    """Phân tích SWOT chỉ từ CSV data"""
    prompt = f"""
Bạn là chuyên gia phân tích kinh doanh F&B tại Việt Nam.

{csv_summary}

YÊU CẦU:
Dựa trên dữ liệu CSV ở trên, hãy phân tích SWOT cho quán/nhà hàng này:

📗 STRENGTHS (Điểm mạnh):
- Điểm mạnh từ menu, giá cả, sản phẩm

📕 WEAKNESSES (Điểm yếu):
- Vấn đề có thể nhận ra từ data

📘 OPPORTUNITIES (Cơ hội):
- Cơ hội cải thiện

📙 THREATS (Thách thức):
- Rủi ro và thách thức

💡 ĐỀ XUẤT CHIẾN LƯỢC:
- 3 đề xuất cụ thể để tăng doanh thu

Phân tích chi tiết và đưa ra insights hữu ích!
"""
    return call_gemini(prompt)


# ============================================
# MAIN MENU
# ============================================
def print_menu():
    print("\n" + "="*50)
    print("🔍 SWOT AGENT - Phân Tích Quán Cafe/Nhà Hàng")
    print("="*50)
    print("1. Nhập tên quán")
    print("2. Phân tích từ file CSV trong thư mục data/")
    print("3. Kết hợp: Tên quán + CSV data")
    print("4. Thoát")
    print("="*50)


def main():
    while True:
        print_menu()
        choice = input("Chọn chế độ (1-4): ").strip()
        
        if choice == "1":
            # Chế độ 1: Chỉ nhập tên quán
            shop_name = input("\n🏪 Nhập tên quán: ").strip()
            if shop_name:
                print("\n⏳ Đang phân tích...\n")
                try:
                    result = analyze_swot_by_name(shop_name)
                    print(result)
                except Exception as e:
                    print(f"❌ Lỗi: {e}")
            else:
                print("❌ Vui lòng nhập tên quán!")
                
        elif choice == "2":
            # Chế độ 2: Chỉ từ CSV
            print("\n📁 Đang đọc các file CSV...")
            dataframes, file_info = load_all_csv()
            
            if dataframes:
                csv_summary = summarize_csv_data(dataframes, file_info)
                print("\n⏳ Đang phân tích...\n")
                try:
                    result = analyze_csv_only(csv_summary)
                    print(result)
                except Exception as e:
                    print(f"❌ Lỗi: {e}")
            else:
                print(f"❌ {file_info}")
                
        elif choice == "3":
            # Chế độ 3: Kết hợp
            shop_name = input("\n🏪 Nhập tên quán: ").strip()
            
            print("\n📁 Đang đọc các file CSV...")
            dataframes, file_info = load_all_csv()
            
            if dataframes and shop_name:
                csv_summary = summarize_csv_data(dataframes, file_info)
                print("\n⏳ Đang phân tích kết hợp...\n")
                try:
                    result = analyze_swot_with_csv(shop_name, csv_summary)
                    print(result)
                except Exception as e:
                    print(f"❌ Lỗi: {e}")
            elif not shop_name:
                print("❌ Vui lòng nhập tên quán!")
            else:
                print(f"❌ {file_info}")
                
        elif choice == "4":
            print("\n👋 Tạm biệt!")
            break
        else:
            print("❌ Vui lòng chọn 1-4")


if __name__ == "__main__":
    main()
