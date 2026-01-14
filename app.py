"""
SWOT AGENT - Web Interface
Sử dụng Streamlit + Google Gemini LLM
"""

import os
import glob
import pandas as pd
import google.generativeai as genai
import streamlit as st
import json
import re
from datetime import datetime
from io import BytesIO

# ============================================
# CẤU HÌNH API
# ============================================
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('models/gemini-flash-latest')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Đặc vụ SWOT của Phòng AI - Phân Tích Quán",
    page_icon="🔎",
    layout="wide"
)

# ============================================
# CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .swot-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .strength-box { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; }
    .weakness-box { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; }
    .opportunity-box { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; }
    .threat-box { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNCTIONS
# ============================================
def load_all_csv(data_folder="data"):
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
        except Exception as e:
            st.error(f"Lỗi đọc {file_path}: {e}")
    return all_data, file_info


def summarize_csv_data(dataframes, file_info):
    if not dataframes:
        return ""
    summary = "📊 DỮ LIỆU TỪ CSV:\n"
    for i, (df, info) in enumerate(zip(dataframes, file_info)):
        summary += f"\n--- File: {info['file']} ---\n"
        summary += f"Số dòng: {info['rows']}\n"
        summary += f"Các cột: {', '.join(info['columns'])}\n"
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                summary += f"- {col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.0f}\n"
        summary += f"Mẫu dữ liệu:\n{df.head(5).to_string()}\n"
    return summary


def call_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text


def analyze_swot_with_scores(shop_name, csv_summary=""):
    """Phân tích SWOT và trả về điểm số cho biểu đồ"""
    context = f"\n{csv_summary}" if csv_summary else ""
    
    prompt = f"""
Bạn là chuyên gia phân tích kinh doanh và là một Data Analyst trong lĩnh vực F&B tại Việt Nam.

🏪 QUÁN CẦN PHÂN TÍCH: {shop_name}
{context}

YÊU CẦU:
1. Phân tích SWOT chi tiết
2. Cho điểm từ 1-10 cho mỗi yếu tố SWOT (dựa trên độ mạnh/yếu)
3. Trả về kết quả theo format sau:

QUAN_TRONG: Trả về một block JSON ở cuối với format:
```json
{{
    "shop_name": "{shop_name}",
    "scores": {{
        "strengths": <điểm 1-10>,
        "weaknesses": <điểm 1-10>,
        "opportunities": <điểm 1-10>,
        "threats": <điểm 1-10>
    }},
    "summary": {{
        "strengths": ["điểm mạnh 1", "điểm mạnh 2", "điểm mạnh 3"],
        "weaknesses": ["điểm yếu 1", "điểm yếu 2", "điểm yếu 3"],
        "opportunities": ["cơ hội 1", "cơ hội 2", "cơ hội 3"],
        "threats": ["thách thức 1", "thách thức 2", "thách thức 3"]
    }}
}}
```

Bây giờ hãy phân tích chi tiết:

📗 STRENGTHS (Điểm mạnh):
- ...

📕 WEAKNESSES (Điểm yếu):
- ...

📘 OPPORTUNITIES (Cơ hội):
- ...

📙 THREATS (Thách thức):
- ...

💡 ĐỀ XUẤT CHIẾN LƯỢC:
- 3 đề xuất cụ thể

Cuối cùng, đưa ra block JSON như yêu cầu.
"""
    return call_gemini(prompt)


def analyze_competitor_comparison(my_shop, competitor_shop):
    """So sánh SWOT giữa 2 quán"""
    prompt = f"""
Bạn là chuyên gia phân tích kinh doanh và là một Data Analyst trong lĩnh vực F&B tại Việt Nam.

⚔️ SO SÁNH ĐỐI THỦ CẠNH TRANH:
- 🏪 QUÁN CỦA BẠN: {my_shop}
- 🎯 ĐỐI THỦ: {competitor_shop}

YÊU CẦU:
1. Phân tích SWOT cho CẢ HAI quán
2. So sánh và đối chiếu điểm mạnh/yếu
3. Đề xuất chiến lược cạnh tranh

QUAN_TRONG: Trả về một block JSON ở cuối với format:
```json
{{
    "my_shop": {{
        "name": "{my_shop}",
        "scores": {{
            "strengths": <điểm 1-10>,
            "weaknesses": <điểm 1-10>,
            "opportunities": <điểm 1-10>,
            "threats": <điểm 1-10>
        }},
        "summary": {{
            "strengths": ["điểm mạnh 1", "điểm mạnh 2", "điểm mạnh 3"],
            "weaknesses": ["điểm yếu 1", "điểm yếu 2", "điểm yếu 3"],
            "opportunities": ["cơ hội 1", "cơ hội 2", "cơ hội 3"],
            "threats": ["thách thức 1", "thách thức 2", "thách thức 3"]
        }}
    }},
    "competitor": {{
        "name": "{competitor_shop}",
        "scores": {{
            "strengths": <điểm 1-10>,
            "weaknesses": <điểm 1-10>,
            "opportunities": <điểm 1-10>,
            "threats": <điểm 1-10>
        }},
        "summary": {{
            "strengths": ["điểm mạnh 1", "điểm mạnh 2", "điểm mạnh 3"],
            "weaknesses": ["điểm yếu 1", "điểm yếu 2", "điểm yếu 3"],
            "opportunities": ["cơ hội 1", "cơ hội 2", "cơ hội 3"],
            "threats": ["thách thức 1", "thách thức 2", "thách thức 3"]
        }}
    }},
    "competitive_advantages": ["lợi thế 1", "lợi thế 2", "lợi thế 3"],
    "areas_to_improve": ["cần cải thiện 1", "cần cải thiện 2", "cần cải thiện 3"],
    "strategies": ["chiến lược 1", "chiến lược 2", "chiến lược 3"]
}}
```

Bây giờ hãy phân tích chi tiết:

## 🏪 PHÂN TÍCH {my_shop}:
📗 STRENGTHS: ...
📕 WEAKNESSES: ...
📘 OPPORTUNITIES: ...
📙 THREATS: ...

## 🎯 PHÂN TÍCH {competitor_shop}:
📗 STRENGTHS: ...
📕 WEAKNESSES: ...
📘 OPPORTUNITIES: ...
📙 THREATS: ...

## ⚔️ SO SÁNH & KẾT LUẬN:
- Lợi thế cạnh tranh của bạn
- Điểm cần cải thiện
- Đề xuất chiến lược

Cuối cùng, đưa ra block JSON như yêu cầu.
"""
    return call_gemini(prompt)


def analyze_specific_branch(brand_name, branch_location, csv_summary=""):
    """Phân tích SWOT cho một chi nhánh cụ thể (không phải toàn chuỗi)"""
    context = f"\n{csv_summary}" if csv_summary else ""
    
    prompt = f"""
Bạn là chuyên gia phân tích kinh doanh và là một Data Analyst trong lĩnh vực F&B tại Việt Nam.

🔍 TÌM KIẾM CHUYÊN SÂU - PHÂN TÍCH CHI NHÁNH CỤ THỂ:
- 🏪 THƯƠNG HIỆU: {brand_name}
- 📍 CHI NHÁNH: {branch_location}
{context}

⚠️ LƯU Ý QUAN TRỌNG:
- Đây là phân tích cho MỘT CHI NHÁNH CỤ THỂ, KHÔNG PHẢI cả chuỗi
- Tập trung vào đặc điểm riêng của chi nhánh này tại vị trí "{branch_location}"
- Phân tích dựa trên:
  + Vị trí địa lý cụ thể (khu vực, đặc điểm dân cư, giao thông)
  + Đối thủ cạnh tranh tại khu vực đó
  + Đặc điểm khách hàng mục tiêu tại địa điểm
  + Thuận lợi/khó khăn riêng của vị trí này

YÊU CẦU:
1. Phân tích SWOT chi tiết CHO CHI NHÁNH NÀY (không phải toàn chuỗi)
2. Cho điểm từ 1-10 cho mỗi yếu tố SWOT
3. Đề xuất chiến lược phù hợp với vị trí cụ thể

QUAN_TRONG: Trả về một block JSON ở cuối với format:
```json
{{
    "brand_name": "{brand_name}",
    "branch_location": "{branch_location}",
    "analysis_type": "specific_branch",
    "scores": {{
        "strengths": <điểm 1-10>,
        "weaknesses": <điểm 1-10>,
        "opportunities": <điểm 1-10>,
        "threats": <điểm 1-10>
    }},
    "location_analysis": {{
        "area_characteristics": "Đặc điểm khu vực",
        "target_customers": "Khách hàng mục tiêu tại đây",
        "nearby_competitors": ["đối thủ 1", "đối thủ 2", "đối thủ 3"],
        "traffic_level": "Mức độ giao thông"
    }},
    "summary": {{
        "strengths": ["điểm mạnh chi nhánh 1", "điểm mạnh chi nhánh 2", "điểm mạnh chi nhánh 3"],
        "weaknesses": ["điểm yếu chi nhánh 1", "điểm yếu chi nhánh 2", "điểm yếu chi nhánh 3"],
        "opportunities": ["cơ hội địa phương 1", "cơ hội địa phương 2", "cơ hội địa phương 3"],
        "threats": ["thách thức địa phương 1", "thách thức địa phương 2", "thách thức địa phương 3"]
    }},
    "local_strategies": ["chiến lược địa phương 1", "chiến lược địa phương 2", "chiến lược địa phương 3"]
}}
```

Bây giờ hãy phân tích chi tiết CHI NHÁNH "{brand_name} - {branch_location}":

📍 PHÂN TÍCH VỊ TRÍ:
- Đặc điểm khu vực...
- Khách hàng mục tiêu...
- Đối thủ gần đó...

📗 STRENGTHS (Điểm mạnh của chi nhánh này):
- ...

📕 WEAKNESSES (Điểm yếu của chi nhánh này):
- ...

📘 OPPORTUNITIES (Cơ hội tại địa điểm này):
- ...

📙 THREATS (Thách thức tại địa điểm này):
- ...

💡 ĐỀ XUẤT CHIẾN LƯỢC CHO CHI NHÁNH:
- 3 đề xuất cụ thể phù hợp với vị trí

Cuối cùng, đưa ra block JSON như yêu cầu.
"""
    return call_gemini(prompt)


def extract_branch_json(response_text):
    """Trích xuất JSON từ phân tích chi nhánh"""
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    except:
        pass
    
    return {
        "brand_name": "Unknown",
        "branch_location": "Unknown",
        "analysis_type": "specific_branch",
        "scores": {"strengths": 7, "weaknesses": 5, "opportunities": 6, "threats": 4},
        "location_analysis": {
            "area_characteristics": "Chưa xác định",
            "target_customers": "Chưa xác định",
            "nearby_competitors": ["Đối thủ 1", "Đối thủ 2"],
            "traffic_level": "Trung bình"
        },
        "summary": {
            "strengths": ["Thương hiệu mạnh", "Vị trí tốt", "Menu đa dạng"],
            "weaknesses": ["Giá cao", "Không gian hạn chế", "Thời gian chờ"],
            "opportunities": ["Mở rộng thị trường", "Delivery", "Marketing số"],
            "threats": ["Cạnh tranh", "Chi phí tăng", "Xu hướng thay đổi"]
        },
        "local_strategies": ["Tập trung khách hàng địa phương", "Khuyến mãi theo khu vực", "Hợp tác địa phương"]
    }


def display_branch_charts(branch_data, brand_name, branch_location):
    """Hiển thị biểu đồ cho phân tích chi nhánh"""
    scores = branch_data.get("scores", {})
    summary = branch_data.get("summary", {})
    location_analysis = branch_data.get("location_analysis", {})
    
    full_name = f"{brand_name} - {branch_location}"
    
    # Phân tích vị trí
    st.subheader("📍 Phân tích vị trí chi nhánh")
    loc_col1, loc_col2 = st.columns(2)
    with loc_col1:
        st.info(f"**🏙️ Đặc điểm khu vực:** {location_analysis.get('area_characteristics', 'N/A')}")
        st.info(f"**🚗 Mức độ giao thông:** {location_analysis.get('traffic_level', 'N/A')}")
    with loc_col2:
        st.info(f"**👥 Khách hàng mục tiêu:** {location_analysis.get('target_customers', 'N/A')}")
        competitors = location_analysis.get('nearby_competitors', [])
        if competitors:
            st.warning(f"**🎯 Đối thủ gần đây:** {', '.join(competitors[:3])}")
    
    # Biểu đồ SWOT
    st.markdown("---")
    st.subheader("📊 Biểu đồ SWOT chi nhánh")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_data = pd.DataFrame({
            'Yếu tố': ['💪 Strengths', '⚠️ Weaknesses', '🚀 Opportunities', '⚡ Threats'],
            'Điểm': [
                scores.get('strengths', 7),
                scores.get('weaknesses', 5),
                scores.get('opportunities', 6),
                scores.get('threats', 4)
            ]
        })
        st.bar_chart(chart_data.set_index('Yếu tố'))
    
    with col2:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("💪 Strengths", f"{scores.get('strengths', 7)}/10", "Chi nhánh")
            st.metric("🚀 Opportunities", f"{scores.get('opportunities', 6)}/10", "Địa phương")
        with m2:
            st.metric("⚠️ Weaknesses", f"{scores.get('weaknesses', 5)}/10", "Chi nhánh")
            st.metric("⚡ Threats", f"{scores.get('threats', 4)}/10", "Địa phương")
    
    # SWOT Grid
    st.subheader("🎯 Ma trận SWOT chi nhánh")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("""
        <div class="swot-box strength-box">
            <h4>💪 STRENGTHS (Chi nhánh)</h4>
        </div>
        """, unsafe_allow_html=True)
        for item in summary.get('strengths', [])[:3]:
            st.markdown(f"✅ {item}")
        
        st.markdown("""
        <div class="swot-box opportunity-box">
            <h4>🚀 OPPORTUNITIES (Địa phương)</h4>
        </div>
        """, unsafe_allow_html=True)
        for item in summary.get('opportunities', [])[:3]:
            st.markdown(f"🎯 {item}")
    
    with c2:
        st.markdown("""
        <div class="swot-box weakness-box">
            <h4>⚠️ WEAKNESSES (Chi nhánh)</h4>
        </div>
        """, unsafe_allow_html=True)
        for item in summary.get('weaknesses', [])[:3]:
            st.markdown(f"⚠️ {item}")
        
        st.markdown("""
        <div class="swot-box threat-box">
            <h4>⚡ THREATS (Địa phương)</h4>
        </div>
        """, unsafe_allow_html=True)
        for item in summary.get('threats', [])[:3]:
            st.markdown(f"🔥 {item}")
    
    # Chiến lược địa phương
    st.markdown("---")
    st.subheader("💡 Chiến lược cho chi nhánh")
    local_strategies = branch_data.get('local_strategies', [])
    for idx, strat in enumerate(local_strategies, 1):
        st.success(f"**{idx}.** {strat}")
    
    # Export
    st.markdown("---")
    st.subheader("📥 Xuất kết quả")
    
    excel_buffer = BytesIO()
    
    # Sheet 1: Điểm số
    scores_df = pd.DataFrame({
        "Brand": [brand_name] * 4,
        "Branch_Location": [branch_location] * 4,
        "Category": ["Strengths", "Weaknesses", "Opportunities", "Threats"],
        "Category_VN": ["Điểm mạnh", "Điểm yếu", "Cơ hội", "Thách thức"],
        "Score": [
            scores.get('strengths', 7),
            scores.get('weaknesses', 5),
            scores.get('opportunities', 6),
            scores.get('threats', 4)
        ],
        "Type": ["Internal", "Internal", "External", "External"],
        "Analysis_Type": ["Specific_Branch"] * 4,
        "Analyzed_Date": [datetime.now().strftime("%Y-%m-%d")] * 4
    })
    
    # Sheet 2: Phân tích vị trí
    location_df = pd.DataFrame({
        "Field": ["Brand", "Branch_Location", "Area_Characteristics", "Target_Customers", "Traffic_Level", "Analyzed_Date"],
        "Value": [
            brand_name,
            branch_location,
            location_analysis.get('area_characteristics', ''),
            location_analysis.get('target_customers', ''),
            location_analysis.get('traffic_level', ''),
            datetime.now().strftime("%Y-%m-%d")
        ]
    })
    
    # Sheet 3: Chi tiết SWOT
    details_list = []
    for category, items in summary.items():
        category_vn = {
            'strengths': 'Điểm mạnh',
            'weaknesses': 'Điểm yếu', 
            'opportunities': 'Cơ hội',
            'threats': 'Thách thức'
        }.get(category, category)
        
        for idx, item in enumerate(items[:5], 1):
            details_list.append({
                "Brand": brand_name,
                "Branch_Location": branch_location,
                "Category": category.capitalize(),
                "Category_VN": category_vn,
                "Order": idx,
                "Detail": item
            })
    details_df = pd.DataFrame(details_list)
    
    # Sheet 4: Đối thủ gần đây
    competitors = location_analysis.get('nearby_competitors', [])
    competitors_df = pd.DataFrame({
        "Brand": [brand_name] * len(competitors),
        "Branch_Location": [branch_location] * len(competitors),
        "Nearby_Competitor": competitors,
        "Order": list(range(1, len(competitors) + 1))
    }) if competitors else pd.DataFrame()
    
    # Sheet 5: Chiến lược địa phương
    strategies_df = pd.DataFrame({
        "Brand": [brand_name] * len(local_strategies),
        "Branch_Location": [branch_location] * len(local_strategies),
        "Strategy": local_strategies,
        "Order": list(range(1, len(local_strategies) + 1))
    }) if local_strategies else pd.DataFrame()
    
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        scores_df.to_excel(writer, sheet_name='SWOT_Scores', index=False)
        location_df.to_excel(writer, sheet_name='Location_Analysis', index=False)
        details_df.to_excel(writer, sheet_name='SWOT_Details', index=False)
        if not competitors_df.empty:
            competitors_df.to_excel(writer, sheet_name='Nearby_Competitors', index=False)
        if not strategies_df.empty:
            strategies_df.to_excel(writer, sheet_name='Local_Strategies', index=False)
    
    excel_buffer.seek(0)
    
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.download_button(
            label="📊 Tải Excel (Power BI)",
            data=excel_buffer,
            file_name=f"swot_branch_{brand_name}_{branch_location.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with exp_col2:
        json_str = json.dumps(branch_data, ensure_ascii=False, indent=2)
        st.download_button(
            label="📋 Tải JSON",
            data=json_str,
            file_name=f"swot_branch_{brand_name}_{branch_location.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )



def extract_comparison_json(response_text):
    """Trích xuất JSON so sánh từ response"""
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    except:
        pass
    
    # Default fallback
    return {
        "my_shop": {
            "name": "Quán của bạn",
            "scores": {"strengths": 7, "weaknesses": 5, "opportunities": 6, "threats": 4},
            "summary": {
                "strengths": ["Thương hiệu", "Vị trí", "Menu"],
                "weaknesses": ["Giá", "Không gian", "Phục vụ"],
                "opportunities": ["Mở rộng", "Online", "Marketing"],
                "threats": ["Cạnh tranh", "Chi phí", "Xu hướng"]
            }
        },
        "competitor": {
            "name": "Đối thủ",
            "scores": {"strengths": 6, "weaknesses": 6, "opportunities": 5, "threats": 5},
            "summary": {
                "strengths": ["Giá rẻ", "Đông khách", "Nổi tiếng"],
                "weaknesses": ["Chất lượng", "Dịch vụ", "Sáng tạo"],
                "opportunities": ["Franchise", "App", "Event"],
                "threats": ["Bão hòa", "Nhân sự", "Nguyên liệu"]
            }
        },
        "competitive_advantages": ["Chất lượng cao hơn", "Dịch vụ tốt hơn"],
        "areas_to_improve": ["Giá cả cạnh tranh", "Marketing mạnh hơn"],
        "strategies": ["Tập trung chất lượng", "Khuyến mãi thông minh", "Xây dựng cộng đồng"]
    }


def extract_json_from_response(response_text):
    """Trích xuất JSON từ response"""
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    except:
        pass
    
    return {
        "shop_name": "Unknown",
        "scores": {"strengths": 7, "weaknesses": 5, "opportunities": 6, "threats": 4},
        "summary": {
            "strengths": ["Thương hiệu mạnh", "Vị trí tốt", "Menu đa dạng"],
            "weaknesses": ["Giá cao", "Không gian hạn chế", "Thời gian chờ"],
            "opportunities": ["Mở rộng thị trường", "Delivery", "Marketing số"],
            "threats": ["Cạnh tranh", "Chi phí tăng", "Xu hướng thay đổi"]
        }
    }


def clean_result_text(result_text):
    """Loại bỏ JSON block và các header không cần thiết khỏi kết quả hiển thị"""
    cleaned = result_text
    # Xóa JSON block
    cleaned = re.sub(r'```json\s*.*?\s*```', '', cleaned, flags=re.DOTALL)
    # Xóa các header liên quan JSON
    cleaned = re.sub(r'QUAN_TRONG:.*?(?=📗|$)', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'##?\s*KHỐI.*?\n', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'##?\s*KẾT QUẢ JSON.*?\n', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\*\*KHỐI.*?\*\*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Cuối cùng.*?JSON.*?\n', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def display_swot_charts(swot_data, shop_name):
    """Hiển thị biểu đồ SWOT"""
    scores = swot_data.get("scores", {})
    summary = swot_data.get("summary", {})
    
    # Row 1: Biểu đồ điểm số
    st.subheader("📊 Biểu đồ phân tích SWOT")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_data = pd.DataFrame({
            'Yếu tố': ['💪 Strengths', '⚠️ Weaknesses', '🚀 Opportunities', '⚡ Threats'],
            'Điểm': [
                scores.get('strengths', 7),
                scores.get('weaknesses', 5),
                scores.get('opportunities', 6),
                scores.get('threats', 4)
            ]
        })
        st.bar_chart(chart_data.set_index('Yếu tố'))
    
    with col2:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("💪 Strengths", f"{scores.get('strengths', 7)}/10", "Điểm mạnh")
            st.metric("🚀 Opportunities", f"{scores.get('opportunities', 6)}/10", "Cơ hội")
        with m2:
            st.metric("⚠️ Weaknesses", f"{scores.get('weaknesses', 5)}/10", "Điểm yếu")
            st.metric("⚡ Threats", f"{scores.get('threats', 4)}/10", "Thách thức")
    
    # Row 2: SWOT Grid
    st.subheader("🎯 Ma trận SWOT")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("""
        <div class="swot-box strength-box">
            <h4>💪 STRENGTHS</h4>
        </div>
        """, unsafe_allow_html=True)
        for item in summary.get('strengths', [])[:3]:
            st.markdown(f"✅ {item}")
        
        st.markdown("""
        <div class="swot-box opportunity-box">
            <h4>🚀 OPPORTUNITIES</h4>
        </div>
        """, unsafe_allow_html=True)
        for item in summary.get('opportunities', [])[:3]:
            st.markdown(f"🎯 {item}")
    
    with c2:
        st.markdown("""
        <div class="swot-box weakness-box">
            <h4>⚠️ WEAKNESSES</h4>
        </div>
        """, unsafe_allow_html=True)
        for item in summary.get('weaknesses', [])[:3]:
            st.markdown(f"⚠️ {item}")
        
        st.markdown("""
        <div class="swot-box threat-box">
            <h4>⚡ THREATS</h4>
        </div>
        """, unsafe_allow_html=True)
        for item in summary.get('threats', [])[:3]:
            st.markdown(f"🔥 {item}")
    
    # Row 3: Export buttons
    st.markdown("---")
    st.subheader("📥 Xuất kết quả")
    
    # Prepare export data
    swot_data["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    swot_data["shop_name"] = shop_name
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        # ===== EXCEL EXPORT (Best for Power BI) =====
        excel_buffer = BytesIO()
        
        # Sheet 1: Điểm số SWOT (dạng bảng cho biểu đồ)
        scores_df = pd.DataFrame({
            "Shop_Name": [shop_name] * 4,
            "Category": ["Strengths", "Weaknesses", "Opportunities", "Threats"],
            "Category_VN": ["Điểm mạnh", "Điểm yếu", "Cơ hội", "Thách thức"],
            "Score": [
                scores.get('strengths', 7),
                scores.get('weaknesses', 5),
                scores.get('opportunities', 6),
                scores.get('threats', 4)
            ],
            "Type": ["Internal", "Internal", "External", "External"],
            "Impact": ["Positive", "Negative", "Positive", "Negative"],
            "Analyzed_Date": [datetime.now().strftime("%Y-%m-%d")] * 4,
            "Analyzed_Time": [datetime.now().strftime("%H:%M:%S")] * 4
        })
        
        # Sheet 2: Chi tiết SWOT (dạng danh sách cho filter)
        details_list = []
        for category, items in summary.items():
            category_vn = {
                'strengths': 'Điểm mạnh',
                'weaknesses': 'Điểm yếu', 
                'opportunities': 'Cơ hội',
                'threats': 'Thách thức'
            }.get(category, category)
            
            for idx, item in enumerate(items[:5], 1):  # Lấy tối đa 5 items
                details_list.append({
                    "Shop_Name": shop_name,
                    "Category": category.capitalize(),
                    "Category_VN": category_vn,
                    "Order": idx,
                    "Detail": item,
                    "Score": scores.get(category, 5),
                    "Analyzed_Date": datetime.now().strftime("%Y-%m-%d")
                })
        
        details_df = pd.DataFrame(details_list)
        
        # Sheet 3: Metadata
        metadata_df = pd.DataFrame({
            "Field": ["Shop Name", "Analysis Date", "Analysis Time", "Strengths Score", "Weaknesses Score", "Opportunities Score", "Threats Score", "Overall Score", "Data Source"],
            "Value": [
                shop_name,
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M:%S"),
                scores.get('strengths', 7),
                scores.get('weaknesses', 5),
                scores.get('opportunities', 6),
                scores.get('threats', 4),
                round((scores.get('strengths', 7) + scores.get('opportunities', 6) - scores.get('weaknesses', 5) - scores.get('threats', 4) + 20) / 4, 1),
                "AI Analysis"
            ]
        })
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            scores_df.to_excel(writer, sheet_name='SWOT_Scores', index=False)
            details_df.to_excel(writer, sheet_name='SWOT_Details', index=False)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        excel_buffer.seek(0)
        
        st.download_button(
            label="� Tải Excel (Power BI)",
            data=excel_buffer,
            file_name=f"swot_{shop_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with export_col2:
        # CSV export (structured for Power BI)
        csv_data = pd.DataFrame({
            "Shop_Name": [shop_name] * 4,
            "Category": ["Strengths", "Weaknesses", "Opportunities", "Threats"],
            "Category_VN": ["Điểm mạnh", "Điểm yếu", "Cơ hội", "Thách thức"],
            "Score": [
                scores.get('strengths', 7),
                scores.get('weaknesses', 5), 
                scores.get('opportunities', 6),
                scores.get('threats', 4)
            ],
            "Detail_1": [
                summary.get('strengths', [''])[0] if summary.get('strengths') else '',
                summary.get('weaknesses', [''])[0] if summary.get('weaknesses') else '',
                summary.get('opportunities', [''])[0] if summary.get('opportunities') else '',
                summary.get('threats', [''])[0] if summary.get('threats') else ''
            ],
            "Detail_2": [
                summary.get('strengths', ['', ''])[1] if len(summary.get('strengths', [])) > 1 else '',
                summary.get('weaknesses', ['', ''])[1] if len(summary.get('weaknesses', [])) > 1 else '',
                summary.get('opportunities', ['', ''])[1] if len(summary.get('opportunities', [])) > 1 else '',
                summary.get('threats', ['', ''])[1] if len(summary.get('threats', [])) > 1 else ''
            ],
            "Detail_3": [
                summary.get('strengths', ['', '', ''])[2] if len(summary.get('strengths', [])) > 2 else '',
                summary.get('weaknesses', ['', '', ''])[2] if len(summary.get('weaknesses', [])) > 2 else '',
                summary.get('opportunities', ['', '', ''])[2] if len(summary.get('opportunities', [])) > 2 else '',
                summary.get('threats', ['', '', ''])[2] if len(summary.get('threats', [])) > 2 else ''
            ],
            "Type": ["Internal", "Internal", "External", "External"],
            "Impact": ["Positive", "Negative", "Positive", "Negative"],
            "Analyzed_Date": [datetime.now().strftime("%Y-%m-%d")] * 4
        })
        st.download_button(
            label="� Tải CSV",
            data=csv_data.to_csv(index=False).encode('utf-8-sig'),  # UTF-8 BOM for Excel compatibility
            file_name=f"swot_{shop_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with export_col3:
        json_str = json.dumps(swot_data, ensure_ascii=False, indent=2)
        st.download_button(
            label="📋 Tải JSON",
            data=json_str,
            file_name=f"swot_{shop_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


# ============================================
# MAIN UI
# ============================================
st.markdown('<h1 class="main-header">🔎 Đặc Vụ SWOT của Phòng AI 🕵🏻‍♀️ </h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888;">Phân Tích Quán </p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Nhập tên quán", "📁 Phân tích CSV", "🔗 Kết hợp", "⚔️ So sánh đối thủ", "🔍 Tìm kiếm chuyên sâu"])

with tab1:
    st.subheader("Nhập tên quán")
    shop_name = st.text_input("🏪 Tên quán:", placeholder="Ví dụ: Highlands Coffee, The Coffee House...")
    
    if st.button("🚀 Phân tích SWOT", key="btn1"):
        if shop_name:
            with st.spinner("⏳ Đang phân tích..."):
                try:
                    result = analyze_swot_with_scores(shop_name)
                    swot_data = extract_json_from_response(result)
                    
                    # Hiển thị biểu đồ
                    display_swot_charts(swot_data, shop_name)
                    
                    # Hiển thị phân tích chi tiết (không có JSON)
                    st.markdown("---")
                    st.subheader("📋 Phân tích chi tiết")
                    clean_text = clean_result_text(result)
                    st.markdown(clean_text)
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
        else:
            st.warning("Vui lòng nhập tên quán!")

with tab2:
    st.subheader("Phân tích từ file CSV")
    st.info("📁 Đặt file CSV vào thư mục `data/` để phân tích")
    
    uploaded_file = st.file_uploader("Hoặc upload file CSV:", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(10))
        
        if st.button("🚀 Phân tích SWOT từ file", key="btn2"):
            with st.spinner("⏳ Đang phân tích..."):
                try:
                    summary = f"📊 DỮ LIỆU TỪ CSV:\n"
                    summary += f"Số dòng: {len(df)}\n"
                    summary += f"Các cột: {', '.join(df.columns)}\n"
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            summary += f"- {col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.0f}\n"
                    summary += f"Mẫu dữ liệu:\n{df.head(5).to_string()}\n"
                    
                    result = analyze_swot_with_scores("Quán từ CSV", summary)
                    swot_data = extract_json_from_response(result)
                    
                    display_swot_charts(swot_data, "CSV_Analysis")
                    st.markdown("---")
                    st.subheader("📋 Phân tích chi tiết")
                    clean_text = clean_result_text(result)
                    st.markdown(clean_text)
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
    else:
        if st.button("🔄 Đọc từ thư mục data/", key="btn_folder"):
            dataframes, file_info = load_all_csv()
            if dataframes:
                for info in file_info:
                    st.success(f"✓ {info['file']} ({info['rows']} dòng)")
                
                with st.spinner("⏳ Đang phân tích..."):
                    csv_summary = summarize_csv_data(dataframes, file_info)
                    result = analyze_swot_with_scores("Quán từ CSV", csv_summary)
                    swot_data = extract_json_from_response(result)
                    
                    display_swot_charts(swot_data, "CSV_Analysis")
                    st.markdown("---")
                    st.subheader("📋 Phân tích chi tiết")
                    clean_text = clean_result_text(result)
                    st.markdown(clean_text)
            else:
                st.warning(file_info)

with tab3:
    st.subheader("Kết hợp: Tên quán + CSV")
    shop_name_3 = st.text_input("🏪 Tên quán:", key="shop3", placeholder="Ví dụ: Starbucks...")
    uploaded_file_3 = st.file_uploader("📁 Upload CSV:", type=['csv'], key="csv3")
    
    if st.button("🚀 Phân tích kết hợp", key="btn3"):
        if shop_name_3 and uploaded_file_3:
            df = pd.read_csv(uploaded_file_3)
            with st.spinner("⏳ Đang phân tích kết hợp..."):
                try:
                    summary = f"📊 DỮ LIỆU TỪ CSV:\n"
                    summary += f"Số dòng: {len(df)}\n"
                    summary += f"Các cột: {', '.join(df.columns)}\n"
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            summary += f"- {col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.0f}\n"
                    summary += f"Mẫu dữ liệu:\n{df.head(5).to_string()}\n"
                    
                    result = analyze_swot_with_scores(shop_name_3, summary)
                    swot_data = extract_json_from_response(result)
                    
                    display_swot_charts(swot_data, shop_name_3)
                    st.markdown("---")
                    st.subheader("📋 Phân tích chi tiết")
                    clean_text = clean_result_text(result)
                    st.markdown(clean_text)
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
        else:
            st.warning("Vui lòng nhập tên quán và upload file CSV!")

with tab4:
    st.subheader("⚔️ So sánh với đối thủ cạnh tranh")
    st.info("Nhập tên quán của bạn và đối thủ để AI phân tích so sánh SWOT")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        my_shop_name = st.text_input("🏪 Quán của bạn:", placeholder="Ví dụ: Highlands Coffee...", key="my_shop")
    with col_input2:
        competitor_name = st.text_input("🎯 Đối thủ:", placeholder="Ví dụ: The Coffee House...", key="competitor")
    
    if st.button("⚔️ Phân tích so sánh", key="btn_compare"):
        if my_shop_name and competitor_name:
            with st.spinner("⏳ Đang phân tích so sánh..."):
                try:
                    result = analyze_competitor_comparison(my_shop_name, competitor_name)
                    comparison_data = extract_comparison_json(result)
                    
                    # ===== BIỂU ĐỒ SO SÁNH =====
                    st.markdown("---")
                    st.subheader("📊 Biểu đồ so sánh SWOT")
                    
                    my_scores = comparison_data.get("my_shop", {}).get("scores", {})
                    comp_scores = comparison_data.get("competitor", {}).get("scores", {})
                    
                    # Bar chart so sánh
                    comparison_df = pd.DataFrame({
                        "Yếu tố": ["Strengths", "Weaknesses", "Opportunities", "Threats"],
                        my_shop_name: [
                            my_scores.get("strengths", 7),
                            my_scores.get("weaknesses", 5),
                            my_scores.get("opportunities", 6),
                            my_scores.get("threats", 4)
                        ],
                        competitor_name: [
                            comp_scores.get("strengths", 6),
                            comp_scores.get("weaknesses", 6),
                            comp_scores.get("opportunities", 5),
                            comp_scores.get("threats", 5)
                        ]
                    })
                    
                    st.bar_chart(comparison_df.set_index("Yếu tố"))
                    
                    # Metrics so sánh
                    st.subheader("📈 Điểm số chi tiết")
                    met1, met2 = st.columns(2)
                    with met1:
                        st.markdown(f"### 🏪 {my_shop_name}")
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("💪 Strengths", f"{my_scores.get('strengths', 7)}/10")
                            st.metric("🚀 Opportunities", f"{my_scores.get('opportunities', 6)}/10")
                        with m2:
                            st.metric("⚠️ Weaknesses", f"{my_scores.get('weaknesses', 5)}/10")
                            st.metric("⚡ Threats", f"{my_scores.get('threats', 4)}/10")
                    
                    with met2:
                        st.markdown(f"### 🎯 {competitor_name}")
                        m3, m4 = st.columns(2)
                        with m3:
                            st.metric("💪 Strengths", f"{comp_scores.get('strengths', 6)}/10")
                            st.metric("🚀 Opportunities", f"{comp_scores.get('opportunities', 5)}/10")
                        with m4:
                            st.metric("⚠️ Weaknesses", f"{comp_scores.get('weaknesses', 6)}/10")
                            st.metric("⚡ Threats", f"{comp_scores.get('threats', 5)}/10")
                    
                    # Lợi thế & Chiến lược
                    st.markdown("---")
                    st.subheader("🎯 Kết luận và Chiến lược")
                    
                    adv_col, imp_col = st.columns(2)
                    with adv_col:
                        st.markdown("#### ✅ Lợi thế của bạn")
                        for adv in comparison_data.get("competitive_advantages", []):
                            st.markdown(f"- {adv}")
                    
                    with imp_col:
                        st.markdown("#### ⚠️ Cần cải thiện")
                        for imp in comparison_data.get("areas_to_improve", []):
                            st.markdown(f"- {imp}")
                    
                    st.markdown("#### 💡 Đề xuất chiến lược")
                    for idx, strat in enumerate(comparison_data.get("strategies", []), 1):
                        st.markdown(f"{idx}. {strat}")
                    
                    # ===== EXPORT EXCEL =====
                    st.markdown("---")
                    st.subheader("📥 Xuất kết quả so sánh")
                    
                    excel_buffer = BytesIO()
                    
                    # Sheet 1: Điểm so sánh
                    scores_compare_df = pd.DataFrame({
                        "Shop": [my_shop_name, competitor_name],
                        "Type": ["Quán của bạn", "Đối thủ"],
                        "Strengths": [my_scores.get("strengths", 7), comp_scores.get("strengths", 6)],
                        "Weaknesses": [my_scores.get("weaknesses", 5), comp_scores.get("weaknesses", 6)],
                        "Opportunities": [my_scores.get("opportunities", 6), comp_scores.get("opportunities", 5)],
                        "Threats": [my_scores.get("threats", 4), comp_scores.get("threats", 5)],
                        "Analyzed_Date": [datetime.now().strftime("%Y-%m-%d")] * 2
                    })
                    
                    # Sheet 2: Chi tiết quán của bạn
                    my_summary = comparison_data.get("my_shop", {}).get("summary", {})
                    my_details = []
                    for cat, items in my_summary.items():
                        for idx, item in enumerate(items[:3], 1):
                            my_details.append({
                                "Shop": my_shop_name,
                                "Category": cat.capitalize(),
                                "Order": idx,
                                "Detail": item
                            })
                    my_details_df = pd.DataFrame(my_details)
                    
                    # Sheet 3: Chi tiết đối thủ
                    comp_summary = comparison_data.get("competitor", {}).get("summary", {})
                    comp_details = []
                    for cat, items in comp_summary.items():
                        for idx, item in enumerate(items[:3], 1):
                            comp_details.append({
                                "Shop": competitor_name,
                                "Category": cat.capitalize(),
                                "Order": idx,
                                "Detail": item
                            })
                    comp_details_df = pd.DataFrame(comp_details)
                    
                    # Sheet 4: Chiến lược
                    strategy_df = pd.DataFrame({
                        "Type": ["Lợi thế"] * len(comparison_data.get("competitive_advantages", [])) + 
                                ["Cần cải thiện"] * len(comparison_data.get("areas_to_improve", [])) +
                                ["Chiến lược"] * len(comparison_data.get("strategies", [])),
                        "Content": comparison_data.get("competitive_advantages", []) + 
                                   comparison_data.get("areas_to_improve", []) +
                                   comparison_data.get("strategies", [])
                    })
                    
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        scores_compare_df.to_excel(writer, sheet_name='Comparison_Scores', index=False)
                        my_details_df.to_excel(writer, sheet_name='My_Shop_Details', index=False)
                        comp_details_df.to_excel(writer, sheet_name='Competitor_Details', index=False)
                        strategy_df.to_excel(writer, sheet_name='Strategies', index=False)
                    
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Tải Excel So Sánh (Power BI)",
                        data=excel_buffer,
                        file_name=f"swot_comparison_{my_shop_name}_vs_{competitor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # Phân tích chi tiết
                    st.markdown("---")
                    st.subheader("📋 Phân tích chi tiết")
                    clean_text = clean_result_text(result)
                    st.markdown(clean_text)
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
        else:
            st.warning("Vui lòng nhập tên cả 2 quán!")

with tab5:
    st.subheader("🔍 Tìm kiếm chuyên sâu - Phân tích chi nhánh cụ thể")
    st.info("""
    **Khác biệt với phân tích thông thường:**
    - Phân tích thông thường: Đánh giá TOÀN BỘ chuỗi (VD: "Phúc Long" = phân tích cả thương hiệu)
    - Tìm kiếm chuyên sâu: Chỉ phân tích MỘT CHI NHÁNH cụ thể (VD: "Phúc Long Lê Văn Khương")
    
    👉 Phù hợp khi bạn muốn đánh giá điểm mạnh/yếu của một địa điểm cửa hàng cụ thể!
    """)
    
    col_brand, col_branch = st.columns(2)
    with col_brand:
        brand_name = st.text_input(
            "🏪 Tên thương hiệu:", 
            placeholder="VD: Phúc Long, Highlands, The Coffee House...",
            key="deep_brand"
        )
    with col_branch:
        branch_location = st.text_input(
            "📍 Địa chỉ chi nhánh:",
            placeholder="VD: Lê Văn Khương, Quang Trung Q12, Vincom Thủ Đức...",
            key="deep_branch"
        )
    
    # Ví dụ gợi ý
    st.markdown("**💡 Ví dụ cách nhập:**")
    example_col1, example_col2, example_col3 = st.columns(3)
    with example_col1:
        st.caption("🏪 Phúc Long + 📍 Lê Văn Khương")
    with example_col2:
        st.caption("🏪 Highlands + 📍 Vincom Thủ Đức")
    with example_col3:
        st.caption("🏪 Starbucks + 📍 Nguyễn Huệ Q1")
    
    # Optional: Upload CSV để phân tích thêm
    with st.expander("📁 Upload dữ liệu bổ sung (tùy chọn)"):
        branch_csv = st.file_uploader("Upload CSV dữ liệu chi nhánh:", type=['csv'], key="branch_csv")
    
    if st.button("🔍 Phân tích chi nhánh", key="btn_deep_search"):
        if brand_name and branch_location:
            with st.spinner(f"⏳ Đang phân tích chi nhánh {brand_name} - {branch_location}..."):
                try:
                    csv_summary = ""
                    if 'branch_csv' in dir() and branch_csv is not None:
                        df = pd.read_csv(branch_csv)
                        csv_summary = f"📊 DỮ LIỆU BỔ SUNG:\n"
                        csv_summary += f"Số dòng: {len(df)}\n"
                        csv_summary += f"Các cột: {', '.join(df.columns)}\n"
                        csv_summary += f"Mẫu dữ liệu:\n{df.head(5).to_string()}\n"
                    
                    result = analyze_specific_branch(brand_name, branch_location, csv_summary)
                    branch_data = extract_branch_json(result)
                    
                    # Hiển thị biểu đồ và thông tin
                    display_branch_charts(branch_data, brand_name, branch_location)
                    
                    # Phân tích chi tiết
                    st.markdown("---")
                    st.subheader("📋 Phân tích chi tiết")
                    clean_text = clean_result_text(result)
                    st.markdown(clean_text)
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
        else:
            st.warning("Vui lòng nhập cả tên thương hiệu và địa chỉ chi nhánh!")

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #666;">SWOT Agent v1.0 | Made with AI BROTHERHOOD </p>', unsafe_allow_html=True)
