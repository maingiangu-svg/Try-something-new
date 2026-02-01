import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os

#-------logic AI--------
class SmartCoffeeBrain:
    def __init__(self):
        self.file_name = 'sales_history.csv'
        self._init_data()
        self._train_models()

    def _init_data(self):
        # Menu Data
        self.menu_data = {
            'Espresso':   [0, 9, 1],
            'Bac Xiu':    [9, 2, 0],
            'Tra Dao':    [7, 1, 0],
            'Capuchino':  [5, 4, 1],
            'Americano':  [0, 8, 0],
            'Latte Nong': [6, 2, 1],
            'Tra Sua':    [10, 0, 0]
        }
        self.menu_df = pd.DataFrame(self.menu_data).T
        self.menu_df.columns = ['Sweetness', 'Bitterness', 'Temp_Type']

        # Sales History (Load or Create)
        if os.path.exists(self.file_name):
            self.sales_history = pd.read_csv(self.file_name)
        else:
            self.sales_history = pd.DataFrame({
                'Day_Index': [1, 2, 3, 4, 5],
                'Cups_Sold': [20, 22, 25, 30, 35]
            })
            self.sales_history.to_csv(self.file_name, index=False)

    def _train_models(self):
        # Recommendation Model
        self.recommender = NearestNeighbors(n_neighbors=1)
        self.recommender.fit(self.menu_df)
        
        # Forecasting Model (Polynomial)
        self.forecaster = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        self.forecaster.fit(self.sales_history[['Day_Index']], self.sales_history['Cups_Sold'])

    def suggest(self, temp, pref):
        target_temp = 0 if temp > 25 else 1
        vec = [10, 0, target_temp] if pref == 'Ngọt' else [0, 10, target_temp]
        idx = self.recommender.kneighbors([vec], return_distance=False)[0][0]
        return self.menu_df.index[idx]

    def predict(self, day):
        return int(self.forecaster.predict([[day]])[0])

    def add_data(self, day, cups):
        new_row = pd.DataFrame({'Day_Index': [day], 'Cups_Sold': [cups]})
        self.sales_history = pd.concat([self.sales_history, new_row], ignore_index=True)
        self.sales_history.to_csv(self.file_name, index=False)
        self._train_models() 

# Giao diện web
# Khởi tạo AI
if 'brain' not in st.session_state:
    st.session_state.brain = SmartCoffeeBrain()

st.set_page_config(page_title="Smart Cafe AI", page_icon="☕")

st.title("☕ Hệ Thống Quản Lý Cafe AI")
st.markdown("---")

# Chia cột giao diện
col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Bảng Điều Khiển")
    mode = st.radio("Chọn chức năng:", ["Tư vấn khách hàng", "Dự báo doanh thu", "Nhập dữ liệu bán"])

with col2:
    if mode == "Tư vấn khách hàng":
        st.subheader("🤖 AI Gợi Ý Đồ Uống")
        
        temp = st.slider("Nhiệt độ ngoài trời (°C)", 10, 45, 30)
        pref = st.selectbox("Khách thích khẩu vị nào?", ["Ngọt", "Đắng/Cafe mạnh"])
        
        if st.button("Phân tích ngay"):
            result = st.session_state.brain.suggest(temp, pref)
            st.success(f"🎯 AI đề xuất món: **{result}**")
            
            if temp > 25:
                st.info("💡 Lý do: Trời nóng nên AI chọn đồ uống lạnh.")
            else:
                st.info("💡 Lý do: Trời lạnh nên AI chọn đồ uống nóng.")

    elif mode == "Dự báo doanh thu":
        st.subheader("📈 Dự Báo Tương Lai")
        
        next_day = st.number_input("Dự báo cho ngày thứ:", min_value=1, value=len(st.session_state.brain.sales_history)+1)
        
        if st.button("Chạy mô hình dự báo"):
            pred_val = st.session_state.brain.predict(next_day)
            st.metric(label=f"Doanh số dự kiến ngày {next_day}", value=f"{pred_val} ly")
            
            st.write("### Biểu đồ xu hướng bán hàng")
            chart_data = st.session_state.brain.sales_history.set_index('Day_Index')
            st.line_chart(chart_data)

    elif mode == "Nhập dữ liệu bán":
        st.subheader("📝 Cập Nhật Dữ Liệu Thực Tế")
        
        d_day = st.number_input("Ngày thứ:", min_value=1)
        d_cups = st.number_input("Số ly bán được:", min_value=0)
        
        if st.button("Lưu vào cơ sở dữ liệu"):
            st.session_state.brain.add_data(d_day, d_cups)
            st.toast("Đã lưu thành công! AI đã thông minh hơn.", icon="✅")
            st.dataframe(st.session_state.brain.sales_history.tail(5))

# Footer
st.markdown("---")
st.caption("Developed by AI CMC Student | Powered by Python & Streamlit")