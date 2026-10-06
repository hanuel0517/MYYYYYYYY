import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="연평균 기온 선형회귀 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 연평균 기온 선형회귀 분석")

st.write(
    "서울 관측소의 연평균 기온 자료를 이용하여 "
    "최근 50년과 최근 100년의 선형회귀 모델을 비교합니다."
)

# --------------------------------------------------
# 데이터 가져오기
# --------------------------------------------------
# Data Clock Korea에서 서울 관측소의 연평균 기온 데이터를
# 자동으로 가져옵니다.
#
# 해당 자료는 기상청 서울 관측소의 일별 평균기온을
# 연 단위로 평균한 자료입니다.
# --------------------------------------------------

URL = "https://www.dataclockkorea.com/climate/seoul/"

@st.cache_data
def load_data():

    try:
        tables = pd.read_html(URL)

        # 웹페이지에서 연도/기온 데이터가 들어있는 표 찾기
        for table in tables:

            columns = [str(c).lower() for c in table.columns]

            if len(table.columns) >= 2:

                # 첫 두 열을 연도와 기온으로 사용
                temp = table.iloc[:, :2].copy()

                temp.columns = ["year", "temp"]

                temp["year"] = pd.to_numeric(
                    temp["year"],
                    errors="coerce"
                )

                temp["temp"] = (
                    temp["temp"]
                    .astype(str)
                    .str.replace("℃", "", regex=False)
                    .str.replace("°C", "", regex=False)
                )

                temp["temp"] = pd.to_numeric(
                    temp["temp"],
                    errors="coerce"
                )

                temp = temp.dropna()

                temp = temp[
                    (temp["year"] >= 1908) &
                    (temp["year"] <= 2025)
                ]

                if len(temp) >= 50:
                    return temp.sort_values("year").reset_index(drop=True)

        return None

    except Exception:
        return None


df = load_data()

# --------------------------------------------------
# 데이터 불러오기 실패 처리
# --------------------------------------------------

if df is None or len(df) == 0:

    st.error(
        "기온 데이터를 자동으로 불러오지 못했습니다."
    )

    st.info(
        "인터넷 연결 또는 데이터 제공 사이트의 형식이 "
        "변경되었을 수 있습니다."
    )

    st.stop()

# --------------------------------------------------
# 데이터 확인
# --------------------------------------------------

st.subheader("📊 사용 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "자료 시작 연도",
        int(df["year"].min())
    )

with col2:
    st.metric(
        "자료 마지막 연도",
        int(df["year"].max())
    )

with col3:
    st.metric(
        "전체 데이터 수",
        len(df)
    )

# --------------------------------------------------
# 데이터 그래프
# --------------------------------------------------

st.line_chart(
    df.set_index("year")["temp"]
)

# --------------------------------------------------
# 학습 / 테스트 데이터 분리
# --------------------------------------------------

# 최근 50년
# 1956~2005 학습
train_50 = df[
    (df["year"] >= 1956) &
    (df["year"] <= 2005)
].copy()

# 최근 100년
# 1908~2005 학습
#
# 실제 서울 관측 자료는 1908년부터 시작하므로
# 1906~1907년은 데이터가 없어 자동으로 제외됨.
train_100 = df[
    (df["year"] >= 1906) &
    (df["year"] <= 2005)
].copy()

# 공통 테스트 데이터
test = df[
    (df["year"] >= 2006) &
    (df["year"] <= 2025)
].copy()

# --------------------------------------------------
# 데이터 개수 확인
# --------------------------------------------------

st.subheader("📌 데이터 분할")

split_table = pd.DataFrame({
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

    "실제 사용 데이터": [
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

# --------------------------------------------------
# 선형회귀 모델
# --------------------------------------------------

X50 = train_50[["year"]]
y50 = train_50["temp"]

X100 = train_100[["year"]]
y100 = train_100["temp"]

Xtest = test[["year"]]
ytest = test["temp"]

model50 = LinearRegression()
model100 = LinearRegression()

model50.fit(X50, y50)
model100.fit(X100, y100)

# --------------------------------------------------
# 테스트 데이터 예측
# --------------------------------------------------

pred50 = model50.predict(Xtest)
pred100 = model100.predict(Xtest)

# --------------------------------------------------
# 회귀식
# --------------------------------------------------

slope50 = model50.coef_[0]
intercept50 = model50.intercept_

slope100 = model100.coef_[0]
intercept100 = model100.intercept_

# --------------------------------------------------
# 평가 지표
# --------------------------------------------------

mae50 = mean_absolute_error(
    ytest,
    pred50
)

mse50 = mean_squared_error(
    ytest,
    pred50
)

r2_50 = r2_score(
    ytest,
    pred50
)

mae100 = mean_absolute_error(
    ytest,
    pred100
)

mse100 = mean_squared_error(
    ytest,
    pred100
)

r2_100 = r2_score(
    ytest,
    pred100
)

# --------------------------------------------------
# 결과 출력
# --------------------------------------------------

st.subheader("📈 선형회귀 결과")

col1, col2 = st.columns(2)

# 50년 모델
with col1:

    st.markdown("### 최근 50년 모델")

    st.write(
        f"**학습 기간:** 1956~2005년"
    )

    st.write(
        f"**회귀식:** "
        f"기온 = {slope50:.5f} × 연도 "
        f"+ {intercept50:.3f}"
    )

    st.metric(
        "기울기",
        f"{slope50:.5f} ℃/년"
    )

    st.metric(
        "MAE",
        f"{mae50:.3f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse50:.3f}"
    )

    st.metric(
        "R²",
        f"{r2_50:.3f}"
    )


# 100년 모델
with col2:

    st.markdown("### 최근 100년 모델")

    st.write(
        f"**학습 기간:** 1906~2005년"
    )

    st.write(
        f"**실제 사용:** 1908~2005년"
    )

    st.write(
        f"**회귀식:** "
        f"기온 = {slope100:.5f} × 연도 "
        f"+ {intercept100:.3f}"
    )

    st.metric(
        "기울기",
        f"{slope100:.5f} ℃/년"
    )

    st.metric(
        "MAE",
        f"{mae100:.3f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse100:.3f}"
    )

    st.metric(
        "R²",
        f"{r2_100:.3f}"
    )

# --------------------------------------------------
# 모델 비교표
# --------------------------------------------------

st.subheader("🔎 50년 vs 100년 모델 비교")

result = pd.DataFrame({

    "모델": [
        "최근 50년",
        "최근 100년"
    ],

    "학습 기간": [
        "1956~2005",
        "1908~2005"
    ],

    "기울기 (℃/년)": [
        slope50,
        slope100
    ],

    "MAE (℃)": [
        mae50,
        mae100
    ],

    "MSE": [
        mse50,
        mse100
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

# --------------------------------------------------
# 테스트 기간 예측 결과
# --------------------------------------------------

st.subheader("🔮 2006~2025년 테스트 결과")

comparison = pd.DataFrame({

    "연도": test["year"].values,

    "실제 연평균 기온 (℃)": ytest.values,

    "50년 모델 예측 (℃)": pred50,

    "100년 모델 예측 (℃)": pred100

})

comparison["50년 모델 오차 (℃)"] = (
    comparison["실제 연평균 기온 (℃)"]
    - comparison["50년 모델 예측 (℃)"]
)

comparison["100년 모델 오차 (℃)"] = (
    comparison["실제 연평균 기온 (℃)"]
    - comparison["100년 모델 예측 (℃)"]
)

st.dataframe(
    comparison.round(3),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 회귀선 그래프용 데이터
# --------------------------------------------------

graph = df.copy()

graph["50년 회귀선"] = np.nan
graph["100년 회귀선"] = np.nan

mask50 = graph["year"] >= 1956
mask100 = graph["year"] >= 1908

graph.loc[mask50, "50년 회귀선"] = (
    model50.predict(
        graph.loc[mask50, ["year"]]
    )
)

graph.loc[mask100, "100년 회귀선"] = (
    model100.predict(
        graph.loc[mask100, ["year"]]
    )
)

# --------------------------------------------------
# 실제 기온 + 회귀선
# --------------------------------------------------

st.subheader("📉 실제 기온과 회귀선 비교")

chart_data = graph.set_index("year")[
    [
        "temp",
        "50년 회귀선",
        "100년 회귀선"
    ]
]

chart_data.columns = [
    "실제 연평균 기온",
    "1956~2005 회귀선",
    "1908~2005 회귀선"
]

st.line_chart(chart_data)

# --------------------------------------------------
# 테스트 기간만 비교
# --------------------------------------------------

st.subheader("🎯 테스트 기간 예측 비교")

test_graph = comparison.set_index("연도")[
    [
        "실제 연평균 기온 (℃)",
        "50년 모델 예측 (℃)",
        "100년 모델 예측 (℃)"
    ]
]

st.line_chart(test_graph)

# --------------------------------------------------
# 자동 해석
# --------------------------------------------------

st.subheader("📝 결과 해석")

if mae50 < mae100:
    better_mae = "최근 50년 모델"
else:
    better_mae = "최근 100년 모델"

if mse50 < mse100:
    better_mse = "최근 50년 모델"
else:
    better_mse = "최근 100년 모델"

if r2_50 > r2_100:
    better_r2 = "최근 50년 모델"
else:
    better_r2 = "최근 100년 모델"

st.write(
    f"""
### 회귀선 기울기

최근 50년을 학습한 모델의 기울기는
**{slope50:.5f} ℃/년**이고,
최근 100년을 학습한 모델의 기울기는
**{slope100:.5f} ℃/년**이다.

따라서 두 기간을 사용했을 때
장기적인 기온 상승 추세를 나타내는 회귀선의
기울기가 서로 다르게 나타난다.

### 예측 성능

2006~2025년을 공통 테스트 데이터로 사용한 결과,

- **MAE가 더 낮은 모델:** {better_mae}
- **MSE가 더 낮은 모델:** {better_mse}
- **R²가 더 높은 모델:** {better_r2}

MAE와 MSE는 값이 작을수록 예측 오차가 작고,
R²는 1에 가까울수록 테스트 데이터의 변동을
잘 설명하는 모델이다.
"""
)

# --------------------------------------------------
# 데이터 출처
# --------------------------------------------------

st.caption(
    "자료: 기상청 서울 관측소 연평균 기온 자료를 기반으로 한 "
    "Data Clock Korea의 연도별 집계 자료"
)
