import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="연평균 기온 선형회귀 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 연평균 기온 선형회귀 분석")


# ============================================================
# CSV 자동 불러오기
# ============================================================

CSV_FILE = "temperature.csv"

try:
    df = pd.read_csv(
        CSV_FILE,
        encoding="utf-8-sig"
    )
except UnicodeDecodeError:
    df = pd.read_csv(
        CSV_FILE,
        encoding="cp949"
    )


# ============================================================
# 컬럼 자동 인식
# ============================================================

df.columns = df.columns.astype(str).str.strip()

# 연도 컬럼 찾기
year_candidates = [
    c for c in df.columns
    if "연도" in c or "year" in c.lower()
]

# 기온 컬럼 찾기
temp_candidates = [
    c for c in df.columns
    if "기온" in c or "temperature" in c.lower()
    or "temp" in c.lower()
]

if year_candidates:
    YEAR_COL = year_candidates[0]
else:
    YEAR_COL = df.columns[0]

if temp_candidates:
    TEMP_COL = temp_candidates[0]
else:
    # 연도 컬럼을 제외한 첫 번째 컬럼
    TEMP_COL = [
        c for c in df.columns
        if c != YEAR_COL
    ][0]


# ============================================================
# 데이터 전처리
# ============================================================

data = df[[YEAR_COL, TEMP_COL]].copy()

data[YEAR_COL] = pd.to_numeric(
    data[YEAR_COL],
    errors="coerce"
)

data[TEMP_COL] = pd.to_numeric(
    data[TEMP_COL],
    errors="coerce"
)

data = data.dropna()

data[YEAR_COL] = data[YEAR_COL].astype(int)

data = data.sort_values(YEAR_COL)

# 1906~2025
data = data[
    (data[YEAR_COL] >= 1906) &
    (data[YEAR_COL] <= 2025)
].copy()


# ============================================================
# 데이터 분리
# ============================================================

train_50 = data[
    (data[YEAR_COL] >= 1956) &
    (data[YEAR_COL] <= 2005)
]

train_100 = data[
    (data[YEAR_COL] >= 1906) &
    (data[YEAR_COL] <= 2005)
]

test = data[
    (data[YEAR_COL] >= 2006) &
    (data[YEAR_COL] <= 2025)
]


# ============================================================
# 모델
# ============================================================

model_50 = LinearRegression()
model_100 = LinearRegression()

model_50.fit(
    train_50[[YEAR_COL]],
    train_50[TEMP_COL]
)

model_100.fit(
    train_100[[YEAR_COL]],
    train_100[TEMP_COL]
)


# ============================================================
# 테스트 예측
# ============================================================

X_test = test[[YEAR_COL]]
y_test = test[TEMP_COL]

pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)


# ============================================================
# 평가
# ============================================================

mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)

mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)


# ============================================================
# 화면
# ============================================================

st.header("분석 방법")

st.write("• 50년 학습: 1956~2005년")
st.write("• 100년 학습: 1906~2005년")
st.write("• 공통 테스트: 2006~2025년")
st.write("• 평가 지표: MAE, MSE, R²")


# ============================================================
# 회귀선
# ============================================================

st.header("📈 회귀선 비교")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1956~2005년 학습")

    st.write(
        f"기울기: **{model_50.coef_[0]:.6f} °C/년**"
    )

    st.write(
        f"10년당 변화: **{model_50.coef_[0] * 10:.4f} °C**"
    )

    st.write(
        f"회귀식: "
        f"기온 = {model_50.coef_[0]:.6f} × 연도 "
        f"{model_50.intercept_:+.3f}"
    )


with col2:
    st.subheader("1906~2005년 학습")

    st.write(
        f"기울기: **{model_100.coef_[0]:.6f} °C/년**"
    )

    st.write(
        f"10년당 변화: **{model_100.coef_[0] * 10:.4f} °C**"
    )

    st.write(
        f"회귀식: "
        f"기온 = {model_100.coef_[0]:.6f} × 연도 "
        f"{model_100.intercept_:+.3f}"
    )


# ============================================================
# 성능 비교
# ============================================================

st.header("🎯 2006~2025년 테스트 성능")

results = pd.DataFrame({
    "모델": [
        "1956~2005 (50년)",
        "1906~2005 (100년)"
    ],
    "기울기": [
        model_50.coef_[0],
        model_100.coef_[0]
    ],
    "MAE": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    results.style.format({
        "기울기": "{:.6f}",
        "MAE": "{:.4f}",
        "MSE": "{:.4f}",
        "R²": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 실제값 vs 예측값
# ============================================================

st.header("📊 2006~2025년 실제값 vs 예측값")

prediction = pd.DataFrame({
    "연도": test[YEAR_COL].values,
    "실제 기온": y_test.values,
    "50년 회귀 예측": pred_50,
    "100년 회귀 예측": pred_100
})

st.dataframe(
    prediction.style.format({
        "실제 기온": "{:.2f}",
        "50년 회귀 예측": "{:.2f}",
        "100년 회귀 예측": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 그래프
# ============================================================

st.header("📉 2006~2025년 예측 그래프")

chart = prediction.set_index("연도")

st.line_chart(
    chart,
    use_container_width=True
)


# ============================================================
# 전체 기간
# ============================================================

st.header("🌡️ 1906~2025년 연평균 기온")

full_chart = data.set_index(YEAR_COL)[[TEMP_COL]]

st.line_chart(
    full_chart,
    use_container_width=True
)


# ============================================================
# 회귀선 전체 비교
# ============================================================

st.header("📈 전체 기간 회귀선 비교")

years = np.arange(1906, 2026)

regression = pd.DataFrame({
    "연도": years,
    "50년 회귀선": model_50.predict(
        years.reshape(-1, 1)
    ),
    "100년 회귀선": model_100.predict(
        years.reshape(-1, 1)
    )
})

regression = regression.set_index("연도")

st.line_chart(
    regression,
    use_container_width=True
)


# ============================================================
# 결과 해석
# ============================================================

st.header("📝 결과 요약")

if mae_50 < mae_100:
    mae_best = "50년 모델"
else:
    mae_best = "100년 모델"

if mse_50 < mse_100:
    mse_best = "50년 모델"
else:
    mse_best = "100년 모델"

if r2_50 > r2_100:
    r2_best = "50년 모델"
else:
    r2_best = "100년 모델"

st.write(
    f"""
    **회귀선 기울기**
    
    - 50년 학습: {model_50.coef_[0]:.6f} °C/년
    - 100년 학습: {model_100.coef_[0]:.6f} °C/년
    
    **예측 성능**
    
    - MAE 우수: **{mae_best}**
    - MSE 우수: **{mse_best}**
    - R² 우수: **{r2_best}**
    """
)
