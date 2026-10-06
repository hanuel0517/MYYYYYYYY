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

st.write(
    "1956~2005년과 1906~2005년의 연평균 기온을 "
    "각각 학습하여 2006~2025년을 예측하고 비교합니다."
)

# ==========================================================
# 연평균 기온 데이터
# ==========================================================
#
# 아래 부분에 '연도, 연평균기온' 데이터를 넣으면 됩니다.
#
# 예:
# 1956, 11.7
# 1957, 12.1
#
# 실제 데이터는 기상청 기온분석의 연평균 평균기온 자료를 사용하세요.
# ==========================================================

raw_data = """
연도,기온
"""

# 데이터 읽기
try:
    from io import StringIO

    df = pd.read_csv(
        StringIO(raw_data.strip())
    )

    df["연도"] = pd.to_numeric(
        df["연도"],
        errors="coerce"
    )

    df["기온"] = pd.to_numeric(
        df["기온"],
        errors="coerce"
    )

    df = df.dropna()

except:
    df = pd.DataFrame(
        columns=["연도", "기온"]
    )

# ==========================================================
# 데이터가 없을 경우 안내
# ==========================================================

if len(df) == 0:

    st.warning(
        "아직 연평균 기온 데이터가 입력되지 않았습니다."
    )

    st.info(
        "기상청 기온분석에서 서울의 연도별 평균기온 자료를 "
        "복사하여 raw_data 부분에 넣어주세요."
    )

    st.code(
        '''raw_data = """
연도,기온
1956,11.7
1957,12.1
1958,11.5
...
2025,14.0
"""'
    )

    st.stop()

# ==========================================================
# 데이터 정렬
# ==========================================================

df = df.sort_values("연도").reset_index(drop=True)

# ==========================================================
# 데이터 확인
# ==========================================================

st.subheader("📊 연평균 기온 데이터")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# 전체 기온 그래프
# ==========================================================

st.subheader("📈 연평균 기온 변화")

st.line_chart(
    df.set_index("연도")["기온"]
)

# ==========================================================
# 학습 / 테스트 데이터 분리
# ==========================================================

# 최근 50년
train_50 = df[
    (df["연도"] >= 1956) &
    (df["연도"] <= 2005)
].copy()

# 최근 100년
train_100 = df[
    (df["연도"] >= 1906) &
    (df["연도"] <= 2005)
].copy()

# 공통 테스트 데이터
test = df[
    (df["연도"] >= 2006) &
    (df["연도"] <= 2025)
].copy()

# ==========================================================
# 데이터 개수
# ==========================================================

st.subheader("📌 학습 및 테스트 데이터")

split_result = pd.DataFrame({
    "구분": [
        "최근 50년 학습",
        "최근 100년 학습",
        "공통 테스트"
    ],

    "기간": [
        "1956~2005",
        "1906~2005",
        "2006~2025"
    ],

    "데이터 개수": [
        len(train_50),
        len(train_100),
        len(test)
    ]
})

st.dataframe(
    split_result,
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# 데이터가 부족한 경우
# ==========================================================

if len(train_50) < 2:
    st.error(
        "1956~2005년 데이터가 충분하지 않습니다."
    )
    st.stop()

if len(train_100) < 2:
    st.error(
        "1906~2005년 데이터가 충분하지 않습니다."
    )
    st.stop()

if len(test) < 2:
    st.error(
        "2006~2025년 테스트 데이터가 충분하지 않습니다."
    )
    st.stop()

# ==========================================================
# 선형회귀 모델
# ==========================================================

model_50 = LinearRegression()
model_100 = LinearRegression()

model_50.fit(
    train_50[["연도"]],
    train_50["기온"]
)

model_100.fit(
    train_100[["연도"]],
    train_100["기온"]
)

# ==========================================================
# 테스트 데이터 예측
# ==========================================================

prediction_50 = model_50.predict(
    test[["연도"]]
)

prediction_100 = model_100.predict(
    test[["연도"]]
)

actual = test["기온"]

# ==========================================================
# 기울기
# ==========================================================

slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]

intercept_50 = model_50.intercept_
intercept_100 = model_100.intercept_

# ==========================================================
# 평가 지표
# ==========================================================

mae_50 = mean_absolute_error(
    actual,
    prediction_50
)

mse_50 = mean_squared_error(
    actual,
    prediction_50
)

r2_50 = r2_score(
    actual,
    prediction_50
)

mae_100 = mean_absolute_error(
    actual,
    prediction_100
)

mse_100 = mean_squared_error(
    actual,
    prediction_100
)

r2_100 = r2_score(
    actual,
    prediction_100
)

# ==========================================================
# 결과
# ==========================================================

st.subheader("📐 회귀선 결과")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 최근 50년 모델")

    st.write(
        f"회귀식: "
        f"**기온 = {slope_50:.5f} × 연도 "
        f"+ {intercept_50:.3f}**"
    )

    st.metric(
        "기울기",
        f"{slope_50:.5f} ℃/년"
    )

    st.metric(
        "MAE",
        f"{mae_50:.4f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse_50:.4f}"
    )

    st.metric(
        "R²",
        f"{r2_50:.4f}"
    )

with col2:

    st.markdown("### 최근 100년 모델")

    st.write(
        f"회귀식: "
        f"**기온 = {slope_100:.5f} × 연도 "
        f"+ {intercept_100:.3f}**"
    )

    st.metric(
        "기울기",
        f"{slope_100:.5f} ℃/년"
    )

    st.metric(
        "MAE",
        f"{mae_100:.4f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse_100:.4f}"
    )

    st.metric(
        "R²",
        f"{r2_100:.4f}"
    )

# ==========================================================
# 비교표
# ==========================================================

st.subheader("🔎 최근 50년 vs 최근 100년")

result = pd.DataFrame({

    "모델": [
        "최근 50년",
        "최근 100년"
    ],

    "학습기간": [
        "1956~2005",
        "1906~2005"
    ],

    "기울기(℃/년)": [
        slope_50,
        slope_100
    ],

    "MAE(℃)": [
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
    result.round(4),
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# 테스트 데이터 예측
# ==========================================================

comparison = pd.DataFrame({

    "연도": test["연도"].values,

    "실제 기온": actual.values,

    "50년 모델 예측": prediction_50,

    "100년 모델 예측": prediction_100
})

st.subheader("🎯 2006~2025년 예측 결과")

st.dataframe(
    comparison.round(3),
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# 실제값과 예측값 그래프
# ==========================================================

st.subheader("📉 테스트 데이터 예측 비교")

chart = comparison.set_index("연도")

st.line_chart(
    chart[
        [
            "실제 기온",
            "50년 모델 예측",
            "100년 모델 예측"
        ]
    ]
)

# ==========================================================
# 회귀선 전체 비교
# ==========================================================

st.subheader("📈 실제 기온과 두 회귀선")

graph_df = df.copy()

graph_df["50년 회귀선"] = (
    model_50.predict(
        graph_df[["연도"]]
    )
)

graph_df["100년 회귀선"] = (
    model_100.predict(
        graph_df[["연도"]]
    )
)

graph_df = graph_df.set_index("연도")

st.line_chart(
    graph_df[
        [
            "기온",
            "50년 회귀선",
            "100년 회귀선"
        ]
    ]
)

# ==========================================================
# 자동 해석
# ==========================================================

st.subheader("📝 결과 해석")

if mae_50 < mae_100:
    mae_result = "최근 50년 모델의 MAE가 더 낮아 예측 오차가 작았다."
else:
    mae_result = "최근 100년 모델의 MAE가 더 낮아 예측 오차가 작았다."

if mse_50 < mse_100:
    mse_result = "최근 50년 모델의 MSE가 더 낮았다."
else:
    mse_result = "최근 100년 모델의 MSE가 더 낮았다."

if r2_50 > r2_100:
    r2_result = "최근 50년 모델의 R²가 더 높아 테스트 기간의 변동을 더 잘 설명했다."
else:
    r2_result = "최근 100년 모델의 R²가 더 높아 테스트 기간의 변동을 더 잘 설명했다."

st.write(
    f"""
**기울기 비교:**  
최근 50년 모델의 기울기는 **{slope_50:.5f} ℃/년**,
최근 100년 모델의 기울기는 **{slope_100:.5f} ℃/년**이다.

따라서 학습 기간에 따라 장기적인 기온 변화 추세의
기울기가 어떻게 달라지는지 비교할 수 있다.

**예측 성능 비교:**  
{mae_result}

{mse_result}

{r2_result}

MAE와 MSE는 **작을수록** 예측 성능이 좋고,
R²는 **1에 가까울수록** 실제 변화량을 잘 설명한다.
"""
)
