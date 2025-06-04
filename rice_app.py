import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ✅ 한글 폰트 설정 (Windows용)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows: 'Malgun Gothic', Mac: 'AppleGothic', Linux/Streamlit Cloud: 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지

# 📊 데이터 불러오기
df = pd.read_csv("rice_data.csv")

# 🎯 사용자 입력
st.title("쌀 생산량 예측 웹앱")
st.markdown("4~10월 기후 정보와 pH 농도를 기반으로 **생산량(톤)** 을 예측합니다.")

temperature = st.number_input("예상 평균 기온 (°C)", min_value=15.0, max_value=35.0, step=0.1, value=21.0)
rainfall = st.number_input("예상 총 강수량 (mm)", min_value=0.0, max_value=2000.0, step=10.0, value=1000.0)

# ✅ 학습: ph별로 모델 만들기
target = '생산량'
feature_cols = ['기온', '강수량']
ph_values = sorted(df['ph'].unique())

fig, ax = plt.subplots()
for ph in ph_values:
    df_ph = df[df['ph'] == ph]
    X = df_ph[feature_cols]
    y = df_ph[target]
    
    model = LinearRegression()
    model.fit(X, y)

    pred_y = model.predict([[temperature, rainfall]])
    ax.scatter(temperature, pred_y, label=f"ph {ph}: {pred_y[0]:.1f}톤")

# 🔍 그래프 설정
ax.set_xlabel("기온 (°C)")
ax.set_ylabel("예측 생산량 (톤)")
ax.set_title("pH 농도별 쌀 생산량 예측")
ax.set_xlim(20.3, 22.5)
ax.legend()

# 📈 출력
st.pyplot(fig)
