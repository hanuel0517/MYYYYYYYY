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
# 1. CSV 파일 업로드
# ============================================================

st.subheader("📂 데이터 파일")

uploaded_file = st.file_uploader(
    "연평균 기온 CSV 파일을 업로드하세요.",
    type=["csv"]
)

if uploaded_file is None:
    st.info("⬆️ 위 버튼을 눌러 CSV 파일을 업로드하세요.")
    st.stop()


# ============================================================
# 2. CSV 읽기
# ============================================================

try:
    df = pd.read_csv(
        uploaded_file,
        encoding="utf-8-sig"
    )

except UnicodeDecodeError:

    try:
        df = pd.read_csv(
            uploaded_file,
            encoding="cp949"
        )

    except Exception as e:
        st.error(f"CSV 파일을 읽을 수 없습니다: {e}")
        st.stop()

except Exception as e:

    st.error(f"CSV 파일을 읽을 수 없습니다: {e}")
    st.stop()


# ============================================================
# 3. 컬럼명 정리
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# 4. 데이터 미리보기
# ============================================================

st.success(
    f"파일 업로드 완료: {uploaded_file.name}"
)

st.write(
    f"데이터 크기: **{df.shape[0]}행 × {df.shape[1]}열**"
)

with st.expander("원본 데이터 미리보기"):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# 5. 연도 컬럼 / 기온 컬럼 선택
# ============================================================

st.subheader("🔧 분석에 사용할 컬럼 선택")

columns = list(df.columns)

if len(columns) < 2:

    st.error(
        "분석하려면 최소 2개의 컬럼이 필요합니다."
    )

    st.stop()


# ------------------------------------------------------------
# 연도 컬럼 자동 추천
# ------------------------------------------------------------

year_candidates = [
    col for col in columns
    if any(
        word in col.lower()
        for word in ["연도", "year", "년도"]
    )
]

if len(year_candidates) > 0:
    default_year_index = columns.index(
        year_candidates[0]
    )
else:
    default_year_index = 0


year_col = st.selectbox(
    "① 연도 컬럼",
    columns,
    index=default_year_index
)


# ------------------------------------------------------------
# 기온 컬럼 자동 추천
# ------------------------------------------------------------

temp_candidates = [
    col for col in columns
    if any(
        word in col.lower()
        for word in [
            "기온",
            "온도",
            "temperature",
            "temp"
        ]
    )
]

if len(temp_candidates) > 0:

    default_temp_index = columns.index(
        temp_candidates[0]
    )

else:

    # 연도 컬럼을 제외한 첫 번째 컬럼
    other_columns = [
        col for col in columns
        if col != year_col
    ]

    if len(other_columns) > 0:
        default_temp_index = columns.index(
            other_columns[0]
        )
    else:
        default_temp_index = 0


temp_col = st.selectbox(
    "② 연평균 기온 컬럼",
    columns,
    index=default_temp_index
)


if year_col == temp_col:

    st.error(
        "연도 컬럼과 기온 컬럼은 서로 달라야 합니다."
    )

    st.stop()


# ============================================================
# 6. 데이터 전처리
# ============================================================

data = df[
    [year_col, temp_col]
].copy()


# ------------------------------------------------------------
# 연도 숫자 변환
# ------------------------------------------------------------

data[year_col] = pd.to_numeric(
    data[year_col],
    errors="coerce"
)


# ------------------------------------------------------------
# 기온 숫자 변환
# ------------------------------------------------------------

data[temp_col] = pd.to_numeric(
    data[temp_col],
    errors="coerce"
)


# ------------------------------------------------------------
# 결측값 제거
# ------------------------------------------------------------

before_count = len(data)

data = data.dropna(
    subset=[
        year_col,
        temp_col
    ]
)

after_count = len(data)


# ------------------------------------------------------------
# 연도 정수화
# ------------------------------------------------------------

data[year_col] = data[year_col].astype(int)


# ------------------------------------------------------------
# 연도순 정렬
# ------------------------------------------------------------

data = data.sort_values(
    year_col
).reset_index(drop=True)


# ============================================================
# 7. 1906~2025년 데이터 선택
# ============================================================

data = data[
    (data[year_col] >= 1906) &
    (data[year_col] <= 2025)
].copy()


# ============================================================
# 8. 데이터 확인
# ============================================================

if len(data) == 0:

    st.error(
        "1906~2025년 사이의 데이터가 없습니다."
    )

    st.stop()


st.subheader("📊 분석 데이터")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "시작 연도",
        f"{data[year_col].min()}년"
    )

with col2:

    st.metric(
        "마지막 연도",
        f"{data[year_col].max()}년"
    )

with col3:

    st.metric(
        "분석 데이터 수",
        f"{len(data)}개"
    )


# ============================================================
# 9. 학습 / 테스트 데이터 분리
# ============================================================

# ------------------------------------------------------------
# 최근 50년 학습
# 1956~2005
# ------------------------------------------------------------

train_50 = data[
    (data[year_col] >= 1956) &
    (data[year_col] <= 2005)
].copy()


# ------------------------------------------------------------
# 최근 100년 학습
# 1906~2005
# ------------------------------------------------------------

train_100 = data[
    (data[year_col] >= 1906) &
    (data[year_col] <= 2005)
].copy()


# ------------------------------------------------------------
# 공통 테스트
# 2006~2025
# ------------------------------------------------------------

test = data[
    (data[year_col] >= 2006) &
    (data[year_col] <= 2025)
].copy()


# ============================================================
# 10. 데이터 개수 확인
# ============================================================

st.subheader("🗂️ 학습 / 테스트 데이터 분할")

split_df = pd.DataFrame({

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
    split_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 11. 데이터 충분성 확인
# ============================================================

if len(train_50) < 2:

    st.error(
        "1956~2005년 데이터가 부족합니다."
    )

    st.stop()


if len(train_100) < 2:

    st.error(
        "1906~2005년 데이터가 부족합니다."
    )

    st.stop()


if len(test) < 1:

    st.error(
        "2006~2025년 테스트 데이터가 없습니다."
    )

    st.stop()


# ============================================================
# 12. X / y 설정
# ============================================================

X_train_50 = train_50[
    [year_col]
]

y_train_50 = train_50[
    temp_col
]


X_train_100 = train_100[
    [year_col]
]

y_train_100 = train_100[
    temp_col
]


X_test = test[
    [year_col]
]

y_test = test[
    temp_col
]


# ============================================================
# 13. 선형회귀 모델
# ============================================================

model_50 = LinearRegression()

model_100 = LinearRegression()


# ============================================================
# 14. 모델 학습
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
# 15. 회귀선 기울기 / 절편
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
# 16. 테스트 데이터 예측
# ============================================================

pred_50 = model_50.predict(
    X_test
)

pred_100 = model_100.predict(
    X_test
)


# ============================================================
# 17. 평가 지표
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
# 18. 회귀선
# ============================================================

st.subheader("📈 회귀선")

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        "### 50년 학습 모델"
    )

    st.latex(
        rf"""
        \hat{{T}}
        =
        {slope_50:.6f}
        \times Year
        {intercept_50:+.3f}
        """
    )

    st.write(
        f"**기울기:** {slope_50:.6f} °C/년"
    )

    st.write(
        f"**10년당 변화:** "
        f"{slope_50 * 10:.4f} °C/10년"
    )


with col2:

    st.markdown(
        "### 100년 학습 모델"
    )

    st.latex(
        rf"""
        \hat{{T}}
        =
        {slope_100:.6f}
        \times Year
        {intercept_100:+.3f}
        """
    )

    st.write(
        f"**기울기:** {slope_100:.6f} °C/년"
    )

    st.write(
        f"**10년당 변화:** "
        f"{slope_100 * 10:.4f} °C/10년"
    )


# ============================================================
# 19. 기울기 비교
# ============================================================

st.subheader("📐 회귀선 기울기 비교")

slope_difference = (
    slope_50 - slope_100
)

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
        "두 회귀선의 기울기가 같습니다."
    )


# ============================================================
# 20. 모델 성능 비교
# ============================================================

st.subheader(
    "🎯 2006~2025년 테스트 성능 비교"
)

results = pd.DataFrame({

    "학습 모델": [
        "1956~2005 (50년)",
        "1906~2005 (100년)"
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
# 21. 평가 지표를 크게 표시
# ============================================================

st.markdown(
    "### 평가 지표"
)

st.caption(
    "MAE와 MSE는 낮을수록 좋고, R²는 높을수록 좋습니다."
)

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("#### MAE")

    st.metric(
        "50년 학습",
        f"{mae_50:.4f}"
    )

    st.metric(
        "100년 학습",
        f"{mae_100:.4f}"
    )


with col2:

    st.markdown("#### MSE")

    st.metric(
        "50년 학습",
        f"{mse_50:.4f}"
    )

    st.metric(
        "100년 학습",
        f"{mse_100:.4f}"
    )


with col3:

    st.markdown("#### R²")

    st.metric(
        "50년 학습",
        f"{r2_50:.4f}"
    )

    st.metric(
        "100년 학습",
        f"{r2_100:.4f}"
    )


# ============================================================
# 22. 어느 모델이 더 좋은지 비교
# ============================================================

st.subheader("🏆 모델 성능 비교 결과")


if mae_50 < mae_100:

    mae_result = "50년 학습 모델"

elif mae_100 < mae_50:

    mae_result = "100년 학습 모델"

else:

    mae_result = "동일"


if mse_50 < mse_100:

    mse_result = "50년 학습 모델"

elif mse_100 < mse_50:

    mse_result = "100년 학습 모델"

else:

    mse_result = "동일"


if r2_50 > r2_100:

    r2_result = "50년 학습 모델"

elif r2_100 > r2_50:

    r2_result = "100년 학습 모델"

else:

    r2_result = "동일"


comparison = pd.DataFrame({

    "평가 지표": [
        "MAE",
        "MSE",
        "R²"
    ],

    "더 좋은 모델": [
        mae_result,
        mse_result,
        r2_result
    ],

    "기준": [
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
# 23. 2006~2025 실제값 vs 예측값
# ============================================================

st.subheader(
    "📊 2006~2025년 실제값과 예측값"
)


prediction = test.copy()

prediction["실제값"] = prediction[
    temp_col
]

prediction["50년 모델 예측"] = (
    pred_50
)

prediction["100년 모델 예측"] = (
    pred_100
)

prediction["50년 모델 오차"] = (
    prediction[temp_col]
    - prediction["50년 모델 예측"]
)

prediction["100년 모델 오차"] = (
    prediction[temp_col]
    - prediction["100년 모델 예측"]
)


# 보기 좋은 컬럼 순서
prediction_display = prediction[
    [
        year_col,
        "실제값",
        "50년 모델 예측",
        "100년 모델 예측",
        "50년 모델 오차",
        "100년 모델 오차"
    ]
].copy()


st.dataframe(
    prediction_display.style.format({
        "실제값": "{:.2f}",
        "50년 모델 예측": "{:.2f}",
        "100년 모델 예측": "{:.2f}",
        "50년 모델 오차": "{:.2f}",
        "100년 모델 오차": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 24. 2006~2025년 그래프
# ============================================================

st.subheader(
    "📉 2006~2025년 실제값과 예측값 그래프"
)


chart_data = prediction_display[
    [
        year_col,
        "실제값",
        "50년 모델 예측",
        "100년 모델 예측"
    ]
].copy()


chart_data = chart_data.set_index(
    year_col
)


st.line_chart(
    chart_data,
    use_container_width=True
)


# ============================================================
# 25. 1906~2025 전체 데이터 그래프
# ============================================================

st.subheader(
    "🌡️ 1906~2025년 연평균 기온"
)


full_chart = data[
    [
        year_col,
        temp_col
    ]
].copy()


full_chart = full_chart.set_index(
    year_col
)


full_chart.columns = [
    "연평균 기온"
]


st.line_chart(
    full_chart,
    use_container_width=True
)


# ============================================================
# 26. 전체 기간 회귀선 비교
# ============================================================

years = np.arange(
    1906,
    2026
)


regression_data = pd.DataFrame({
    "연도": years
})


# 실제값
actual_series = (
    data
    .set_index(year_col)[temp_col]
)


regression_data[
    "실제 연평균 기온"
] = actual_series.reindex(
    years
).values


# 50년 회귀선
regression_data[
    "50년 회귀선"
] = model_50.predict(
    years.reshape(-1, 1)
)


# 100년 회귀선
regression_data[
    "100년 회귀선"
] = model_100.predict(
    years.reshape(-1, 1)
)


regression_data = (
    regression_data
    .set_index("연도")
)


st.subheader(
    "📈 1906~2025년 실제값과 회귀선"
)


st.line_chart(
    regression_data,
    use_container_width=True
)


# ============================================================
# 27. 자동 해석
# ============================================================

st.subheader("📝 분석 결과 해석")


st.markdown(
    f"""
### 회귀선 기울기

- **1956~2005년 학습:** {slope_50:.6f} °C/년
- **1906~2005년 학습:** {slope_100:.6f} °C/년
- **차이:** {slope_difference:.6f} °C/년

따라서 50년 학습 회귀선은 10년 기준
**{slope_50 * 10:.4f} °C** 변화하는 추세를,
100년 학습 회귀선은 10년 기준
**{slope_100 * 10:.4f} °C** 변화하는 추세를 나타냅니다.
"""
)


st.markdown(
    f"""
### 2006~2025년 예측 성능

| 지표 | 50년 학습 | 100년 학습 |
|---|---:|---:|
| MAE | {mae_50:.4f} | {mae_100:.4f} |
| MSE | {mse_50:.4f} | {mse_100:.4f} |
| R² | {r2_50:.4f} | {r2_100:.4f} |

- MAE가 더 낮은 모델: **{mae_result}**
- MSE가 더 낮은 모델: **{mse_result}**
- R²가 더 높은 모델: **{r2_result}**
"""
)


# ============================================================
# 28. 결과 다운로드
# ============================================================

st.subheader("💾 결과 다운로드")


# ------------------------------------------------------------
# 예측 결과 CSV
# ------------------------------------------------------------

prediction_csv = prediction_display.to_csv(
    index=False,
    encoding="utf-8-sig"
)


st.download_button(
    label="📥 2006~2025 예측 결과 다운로드",
    data=prediction_csv,
    file_name="temperature_prediction_2006_2025.csv",
    mime="text/csv"
)


# ------------------------------------------------------------
# 모델 비교 결과 CSV
# ------------------------------------------------------------

results_csv = results.to_csv(
    index=False,
    encoding="utf-8-sig"
)


st.download_button(
    label="📥 모델 성능 비교 결과 다운로드",
    data=results_csv,
    file_name="temperature_model_comparison.csv",
    mime="text/csv"
)


# ============================================================
# 끝
# ============================================================

st.success(
    "분석이 완료되었습니다."
)
