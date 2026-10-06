

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="연평균 기온 선형회귀 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 연평균 기온 선형회귀 분석")

st.markdown("""
### 분석 방법

- **50년 학습:** 1956~2005년
- **100년 학습:** 1906~2005년
- **공통 테스트:** 2006~2025년
- **평가 지표:** MAE, MSE, R²
- 두 학습기간의 **회귀선 기울기와 예측 성능 비교**
""")


# ============================================================
# 1. 데이터 불러오기
# ============================================================

FILE_NAME = "annual_temperature.csv"

try:
    df = pd.read_csv(FILE_NAME)

except FileNotFoundError:
    st.error(
        f"데이터 파일 `{FILE_NAME}`을 찾을 수 없습니다.\n\n"
        "main.py와 같은 폴더에 CSV 파일을 넣어주세요."
    )
    st.stop()


# ============================================================
# 2. 컬럼명 설정
# ============================================================

# 실제 CSV의 컬럼명이 다르면 이 두 줄만 수정하세요.
YEAR_COL = "연도"
TEMP_COL = "연평균기온"


# ============================================================
# 3. 컬럼 존재 여부 확인
# ============================================================

if YEAR_COL not in df.columns:
    st.error(
        f"연도 컬럼 `{YEAR_COL}`을 찾을 수 없습니다.\n\n"
        f"현재 CSV 컬럼: {list(df.columns)}"
    )
    st.stop()


if TEMP_COL not in df.columns:
    st.error(
        f"기온 컬럼 `{TEMP_COL}`을 찾을 수 없습니다.\n\n"
        f"현재 CSV 컬럼: {list(df.columns)}"
    )
    st.stop()


# ============================================================
# 4. 데이터 전처리
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

# 결측값 제거
data = data.dropna(
    subset=[YEAR_COL, TEMP_COL]
)

# 연도순 정렬
data = data.sort_values(
    YEAR_COL
).reset_index(drop=True)


# ============================================================
# 5. 1906~2025년 데이터만 사용
# ============================================================

data = data[
    (data[YEAR_COL] >= 1906) &
    (data[YEAR_COL] <= 2025)
].copy()


# ============================================================
# 6. 데이터 확인
# ============================================================

if len(data) == 0:
    st.error(
        "1906~2025년 사이의 데이터가 없습니다."
    )
    st.stop()


st.subheader("📊 데이터 확인")

col1, col2, col3 = st.columns(3)

col1.metric(
    "데이터 시작 연도",
    f"{int(data[YEAR_COL].min())}년"
)

col2.metric(
    "데이터 마지막 연도",
    f"{int(data[YEAR_COL].max())}년"
)

col3.metric(
    "전체 데이터 수",
    f"{len(data)}개"
)


with st.expander("원본 데이터 보기"):
    st.dataframe(
        data,
        use_container_width=True
    )


# ============================================================
# 7. 학습 / 테스트 데이터 분리
# ============================================================

# ------------------------------------------------------------
# 50년 학습
# 1956~2005
# ------------------------------------------------------------

train_50 = data[
    (data[YEAR_COL] >= 1956) &
    (data[YEAR_COL] <= 2005)
].copy()


# ------------------------------------------------------------
# 100년 학습
# 1906~2005
# ------------------------------------------------------------

train_100 = data[
    (data[YEAR_COL] >= 1906) &
    (data[YEAR_COL] <= 2005)
].copy()


# ------------------------------------------------------------
# 공통 테스트
# 2006~2025
# ------------------------------------------------------------

test = data[
    (data[YEAR_COL] >= 2006) &
    (data[YEAR_COL] <= 2025)
].copy()


# ============================================================
# 8. 데이터 개수 확인
# ============================================================

st.subheader("🗂️ 학습 / 테스트 데이터 분할")

split_table = pd.DataFrame({
    "구분": [
        "50년 학습",
        "100년 학습",
        "공통 테스트"
    ],

    "기간": [
        "1956~2005",
        "1906~2005",
        "2006~2025"
    ],

    "데이터 수": [
        len(train_50),
        len(train_100),
        len(test)
    ]
})

st.dataframe(
    split_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 9. 데이터가 충분한지 확인
# ============================================================

if len(train_50) < 2:
    st.error("1956~2005년 학습 데이터가 부족합니다.")
    st.stop()

if len(train_100) < 2:
    st.error("1906~2005년 학습 데이터가 부족합니다.")
    st.stop()

if len(test) < 1:
    st.error("2006~2025년 테스트 데이터가 없습니다.")
    st.stop()


# ============================================================
# 10. X / y 설정
# ============================================================

X_train_50 = train_50[[YEAR_COL]]
y_train_50 = train_50[TEMP_COL]

X_train_100 = train_100[[YEAR_COL]]
y_train_100 = train_100[TEMP_COL]

X_test = test[[YEAR_COL]]
y_test = test[TEMP_COL]


# ============================================================
# 11. 선형회귀 모델 생성
# ============================================================

model_50 = LinearRegression()
model_100 = LinearRegression()


# ============================================================
# 12. 모델 학습
# ============================================================

model_50.fit(
    X_train_50,
    y_train_50
)

model_100.fit(
    X_train_100,
    y_train_100
)


# ============================================================
# 13. 회귀계수
# ============================================================

slope_50 = float(
    model_50.coef_[0]
)

intercept_50 = float(
    model_50.intercept_
)

slope_100 = float(
    model_100.coef_[0]
)

intercept_100 = float(
    model_100.intercept_
)


# ============================================================
# 14. 테스트 데이터 예측
# ============================================================

pred_50 = model_50.predict(
    X_test
)

pred_100 = model_100.predict(
    X_test
)


# ============================================================
# 15. 성능 평가
# ============================================================

# ------------------------------------------------------------
# 50년 모델
# ------------------------------------------------------------

mae_50 = mean_absolute_error(
    y_test,
    pred_50
)

mse_50 = mean_squared_error(
    y_test,
    pred_50
)

r2_50 = r2_score(
    y_test,
    pred_50
)


# ------------------------------------------------------------
# 100년 모델
# ------------------------------------------------------------

mae_100 = mean_absolute_error(
    y_test,
    pred_100
)

mse_100 = mean_squared_error(
    y_test,
    pred_100
)

r2_100 = r2_score(
    y_test,
    pred_100
)


# ============================================================
# 16. 회귀식 출력
# ============================================================

st.subheader("📈 회귀선")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 최근 50년 학습")

    st.latex(
        rf"""
        \hat{{T}} =
        {slope_50:.6f} \times Year
        {intercept_50:+.3f}
        """
    )

    st.write(
        f"기울기: **{slope_50:.6f} °C/년**"
    )

    st.write(
        f"10년당 변화: **{slope_50 * 10:.4f} °C/10년**"
    )


with col2:

    st.markdown("### 최근 100년 학습")

    st.latex(
        rf"""
        \hat{{T}} =
        {slope_100:.6f} \times Year
        {intercept_100:+.3f}
        """
    )

    st.write(
        f"기울기: **{slope_100:.6f} °C/년**"
    )

    st.write(
        f"10년당 변화: **{slope_100 * 10:.4f} °C/10년**"
    )


# ============================================================
# 17. 기울기 비교
# ============================================================

slope_difference = (
    slope_50 - slope_100
)

st.subheader("📐 회귀선 기울기 비교")

col1, col2, col3 = st.columns(3)

col1.metric(
    "50년 모델",
    f"{slope_50:.6f} °C/년"
)

col2.metric(
    "100년 모델",
    f"{slope_100:.6f} °C/년"
)

col3.metric(
    "기울기 차이",
    f"{slope_difference:.6f} °C/년"
)


if slope_difference > 0:

    st.info(
        "50년 학습 회귀선의 기울기가 "
        "100년 학습 회귀선보다 큽니다."
    )

elif slope_difference < 0:

    st.info(
        "100년 학습 회귀선의 기울기가 "
        "50년 학습 회귀선보다 큽니다."
    )

else:

    st.info(
        "두 회귀선의 기울기가 동일합니다."
    )


# ============================================================
# 18. 성능 비교 결과
# ============================================================

st.subheader(
    "🎯 2006~2025년 테스트 데이터 예측 성능"
)

results = pd.DataFrame({

    "모델": [
        "1956~2005 (50년)",
        "1906~2005 (100년)"
    ],

    "학습기간": [
        "1956~2005",
        "1906~2005"
    ],

    "기울기 (°C/년)": [
        slope_50,
        slope_100
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
        "기울기 (°C/년)": "{:.6f}",
        "MAE": "{:.6f}",
        "MSE": "{:.6f}",
        "R²": "{:.6f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 19. 주요 성능 지표를 크게 표시
# ============================================================

st.markdown("### 성능 지표")

st.markdown(
    "※ MAE와 MSE는 **낮을수록 좋고**, R²는 **높을수록 좋습니다.**"
)


col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("#### MAE")

    st.metric(
        "50년 모델",
        f"{mae_50:.4f}"
    )

    st.metric(
        "100년 모델",
        f"{mae_100:.4f}"
    )


with col2:

    st.markdown("#### MSE")

    st.metric(
        "50년 모델",
        f"{mse_50:.4f}"
    )

    st.metric(
        "100년 모델",
        f"{mse_100:.4f}"
    )


with col3:

    st.markdown("#### R²")

    st.metric(
        "50년 모델",
        f"{r2_50:.4f}"
    )

    st.metric(
        "100년 모델",
        f"{r2_100:.4f}"
    )


# ============================================================
# 20. 어느 모델이 더 좋은지 자동 판단
# ============================================================

st.subheader("🏆 모델 성능 비교")


# MAE
if mae_50 < mae_100:

    mae_result = "50년 모델"

elif mae_100 < mae_50:

    mae_result = "100년 모델"

else:

    mae_result = "동일"


# MSE
if mse_50 < mse_100:

    mse_result = "50년 모델"

elif mse_100 < mse_50:

    mse_result = "100년 모델"

else:

    mse_result = "동일"


# R2
if r2_50 > r2_100:

    r2_result = "50년 모델"

elif r2_100 > r2_50:

    r2_result = "100년 모델"

else:

    r2_result = "동일"


comparison = pd.DataFrame({

    "평가지표": [
        "MAE",
        "MSE",
        "R²"
    ],

    "더 좋은 모델": [
        mae_result,
        mse_result,
        r2_result
    ],

    "판단 기준": [
        "낮을수록 좋음",
        "낮을수록 좋음",
        "높을수록 좋음"
    ]
})


st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 21. 테스트 기간 실제값 vs 예측값
# ============================================================

st.subheader(
    "📊 2006~2025년 실제값 vs 예측값"
)

prediction = test.copy()

prediction["50년 모델 예측"] = pred_50

prediction["100년 모델 예측"] = pred_100

prediction["50년 모델 오차"] = (
    prediction[TEMP_COL]
    - prediction["50년 모델 예측"]
)

prediction["100년 모델 오차"] = (
    prediction[TEMP_COL]
    - prediction["100년 모델 예측"]
)


st.dataframe(
    prediction.style.format({
        TEMP_COL: "{:.2f}",
        "50년 모델 예측": "{:.2f}",
        "100년 모델 예측": "{:.2f}",
        "50년 모델 오차": "{:.2f}",
        "100년 모델 오차": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 22. 테스트 기간 그래프
# matplotlib 없이 Streamlit line chart 사용
# ============================================================

st.subheader(
    "📉 2006~2025년 예측 그래프"
)

chart_data = prediction[
    [
        YEAR_COL,
        TEMP_COL,
        "50년 모델 예측",
        "100년 모델 예측"
    ]
].copy()

chart_data = chart_data.set_index(
    YEAR_COL
)

chart_data.columns = [
    "실제 연평균기온",
    "50년 모델",
    "100년 모델"
]

st.line_chart(
    chart_data,
    use_container_width=True
)


# ============================================================
# 23. 전체 기간 실제 데이터
# ============================================================

st.subheader(
    "🌡️ 1906~2025년 연평균 기온"
)

full_chart = data[
    [
        YEAR_COL,
        TEMP_COL
    ]
].copy()

full_chart = full_chart.set_index(
    YEAR_COL
)

full_chart.columns = [
    "연평균기온"
]

st.line_chart(
    full_chart,
    use_container_width=True
)


# ============================================================
# 24. 회귀선 비교 데이터 생성
# ============================================================

years = np.arange(
    1906,
    2026
)

regression_data = pd.DataFrame({
    "연도": years
})

regression_data["실제 연평균기온"] = (
    data.set_index(YEAR_COL)[TEMP_COL]
    .reindex(years)
)

regression_data["50년 회귀선"] = (
    model_50.predict(
        years.reshape(-1, 1)
    )
)

regression_data["100년 회귀선"] = (
    model_100.predict(
        years.reshape(-1, 1)
    )
)

regression_data = regression_data.set_index(
    "연도"
)


# ============================================================
# 25. 전체 기간 회귀선 비교
# ============================================================

st.subheader(
    "📈 1906~2025년 실제값과 회귀선 비교"
)

st.line_chart(
    regression_data,
    use_container_width=True
)


# ============================================================
# 26. 결과 해석
# ============================================================

st.subheader("📝 결과 해석")

if mae_50 < mae_100:

    mae_text = (
        "50년 학습 모델이 MAE에서 더 우수합니다."
    )

else:

    mae_text = (
        "100년 학습 모델이 MAE에서 더 우수합니다."
    )


if mse_50 < mse_100:

    mse_text = (
        "50년 학습 모델이 MSE에서 더 우수합니다."
    )

else:

    mse_text = (
        "100년 학습 모델이 MSE에서 더 우수합니다."
    )


if r2_50 > r2_100:

    r2_text = (
        "50년 학습 모델이 R²에서 더 우수합니다."
    )

else:

    r2_text = (
        "100년 학습 모델이 R²에서 더 우수합니다."
    )


st.write(
    f"""
    **① 기울기**

    - 50년 모델: **{slope_50:.6f} °C/년**
    - 100년 모델: **{slope_100:.6f} °C/년**
    - 기울기 차이: **{slope_difference:.6f} °C/년**
    
    **② 테스트 성능**

    - {mae_text}
    - {mse_text}
    - {r2_text}
    
    **③ 테스트 기간**

    두 모델 모두 동일하게 **2006~2025년**을 테스트 데이터로
    사용했기 때문에 두 모델의 성능을 직접 비교할 수 있습니다.
    """
)


# ============================================================
# 27. CSV 다운로드
# ============================================================

st.subheader("💾 결과 다운로드")

csv_result = prediction.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="2006~2025년 예측 결과 CSV 다운로드",
    data=csv_result,
    file_name="temperature_prediction_2006_2025.csv",
    mime="text/csv"
)


csv_model = results.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="모델 성능 비교 결과 CSV 다운로드",
    data=csv_model,
    file_name="temperature_model_comparison.csv",
    mime="text/csv"
)
