import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False
if not any(f.name == 'NanumGothic' for f in fm.fontManager.ttflist):
    font_url = "https://github.com/naver/nanumfont/blob/master/ttf/NanumGothic.ttf?raw=true"
    font_path = os.path.join(os.getcwd(), "NanumGothic.ttf")
    urllib.request.urlretrieve(font_url, font_path)
    fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'NanumGothic'

# 데이터 로드
df = pd.read_csv("rice_data.csv")

# 회귀 모델 학습
X = df[['rainfall', 'temperature', 'ph']]
y = df['yield']
model = LinearRegression()
model.fit(X, y)

# 예측 결과
y_pred = model.predict(X)

# 성능 지표
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

# Streamlit UI
st.title("🌾 회귀 기반 쌀 생산량 예측 앱")

st.markdown("### 📊 회귀 모델 정확도 지표")
st.write(f"- 결정계수 (R²): **{r2:.3f}**")
st.write(f"- 평균제곱오차 (MSE): **{mse:.3f}**")
st.write(f"- 평균제곱근오차 (RMSE): **{rmse:.3f}**")

# 사용자 입력
rainfall = st.slider("강수량 (mm)", 500, 2000, 1000, 50)
temperature = st.slider("기온 (°C)", 20.0, 23.0, 21.5, 0.1)
selected_ph = st.slider("pH 농도", 5.0, 7.0, 5.8, 0.1)

# 예측
input_data = pd.DataFrame([[rainfall, temperature, selected_ph]],
                          columns=['rainfall', 'temperature', 'ph'])
prediction = model.predict(input_data)[0]
st.success(f"예측 쌀 생산량: {prediction:.2f} 톤")

# pH 변화 시각화
ph_range = np.linspace(5.0, 7.0, 100)
predicted_yields = []

for ph in ph_range:
    input_row = pd.DataFrame([[rainfall, temperature, ph]],
                             columns=['rainfall', 'temperature', 'ph'])
    yield_pred = model.predict(input_row)[0]
    predicted_yields.append(yield_pred)

fig, ax = plt.subplots()
ax.plot(ph_range, predicted_yields, '-', color='green')
ax.set_xlabel("pH 농도")
ax.set_ylabel("예측 쌀 생산량 (톤)")
ax.set_title("pH 농도에 따른 예측 생산량")
st.pyplot(fig)
