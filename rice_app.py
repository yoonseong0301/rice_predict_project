import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우에서 한글 폰트 설정

# 사용자 정의 함수 (완만한 ^ 그래프)
def custom_yield(rainfall, temperature, ph):
    base = (rainfall * 0.01) + (temperature * 2)

    # 완만하게 증가/감소하는 구조로 변경 (기울기 ±5)
    if ph <= 5.8:
        return base + (ph - 5.0) * 5   # 양의 기울기
    else:
        return base + (5.8 - ph) * 5   # 음의 기울기

# Streamlit UI
st.title("🌾 쌀 생산량 예측기")
st.write("강수량과 기온을 선택하면 pH 농도별 예상 생산량을 확인할 수 있어요.")

# 슬라이더 입력
rain_input = st.slider("강수량 (mm)", 500, 2000, step=10, value=1200)
temp_input = st.slider("기온 (℃)", 15.0, 30.0, step=0.1, value=22.0)

# pH 범위
ph_range = np.linspace(5.0, 7.0, 40)

# 예측
predicted_yields = [custom_yield(rain_input, temp_input, ph) for ph in ph_range]

# 그래프 출력
fig, ax = plt.subplots()
ax.plot(ph_range, predicted_yields, marker='o', color='darkblue')
ax.set_xlabel("pH 농도")
ax.set_ylabel("예측 쌀 생산량 (톤)")
ax.set_title("pH 농도에 따른 예측 생산량")
ax.grid(True)
st.pyplot(fig)

# 최고 생산량 출력
max_yield = max(predicted_yields)
max_ph = ph_range[np.argmax(predicted_yields)]
st.success(f"🌟 최고 생산량은 {max_yield:,.0f} 톤 (pH = {max_ph:.2f}) 기준")
