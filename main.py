import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. 연도별 연평균 기온 데이터
# ============================================================
# 서울 연평균 기온 데이터
# year = 연도
# temp = 연평균 기온(℃)

data = {
    "year": [
        # 1906~2025 연도를 넣는 부분
    ],
    "temp": [
        # 해당 연도의 연평균 기온을 넣는 부분
    ]
}

df = pd.DataFrame(data)

# 데이터가 제대로 들어왔는지 확인
print(df.head())
print(df.tail())

# ============================================================
# 2. 데이터 분할
# ============================================================

# 최근 50년 학습 데이터
train_50 = df[
    (df["year"] >= 1956) &
    (df["year"] <= 2005)
]

# 최근 100년 학습 데이터
train_100 = df[
    (df["year"] >= 1906) &
    (df["year"] <= 2005)
]

# 공통 테스트 데이터
test = df[
    (df["year"] >= 2006) &
    (df["year"] <= 2025)
]

# ============================================================
# 3. 선형회귀 모델
# ============================================================

model_50 = LinearRegression()
model_100 = LinearRegression()

model_50.fit(
    train_50[["year"]],
    train_50["temp"]
)

model_100.fit(
    train_100[["year"]],
    train_100["temp"]
)

# ============================================================
# 4. 테스트 데이터 예측
# ============================================================

prediction_50 = model_50.predict(test[["year"]])
prediction_100 = model_100.predict(test[["year"]])

actual = test["temp"]

# ============================================================
# 5. 평가 지표 계산
# ============================================================

# 50년 모델
mae_50 = mean_absolute_error(actual, prediction_50)
mse_50 = mean_squared_error(actual, prediction_50)
r2_50 = r2_score(actual, prediction_50)

# 100년 모델
mae_100 = mean_absolute_error(actual, prediction_100)
mse_100 = mean_squared_error(actual, prediction_100)
r2_100 = r2_score(actual, prediction_100)

# ============================================================
# 6. 회귀선 기울기
# ============================================================

slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]

intercept_50 = model_50.intercept_
intercept_100 = model_100.intercept_

# ============================================================
# 7. 결과 출력
# ============================================================

print("\n==============================")
print("최근 50년 학습 모델")
print("==============================")

print(
    f"회귀식: y = {slope_50:.4f}x + {intercept_50:.2f}"
)

print(f"기울기: {slope_50:.4f} ℃/년")
print(f"MAE: {mae_50:.4f} ℃")
print(f"MSE: {mse_50:.4f}")
print(f"R²: {r2_50:.4f}")


print("\n==============================")
print("최근 100년 학습 모델")
print("==============================")

print(
    f"회귀식: y = {slope_100:.4f}x + {intercept_100:.2f}"
)

print(f"기울기: {slope_100:.4f} ℃/년")
print(f"MAE: {mae_100:.4f} ℃")
print(f"MSE: {mse_100:.4f}")
print(f"R²: {r2_100:.4f}")

# ============================================================
# 8. 두 모델 비교
# ============================================================

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

print("\n==============================")
print("50년 vs 100년 모델 비교")
print("==============================")

print(result.round(4).to_string(index=False))

# ============================================================
# 9. 2006~2025 실제값과 예측값 비교
# ============================================================

comparison = pd.DataFrame({
    "연도": test["year"].values,
    "실제 기온": actual.values,
    "50년 모델 예측": prediction_50,
    "100년 모델 예측": prediction_100
})

print("\n==============================")
print("테스트 데이터 예측 결과")
print("==============================")

print(comparison.round(2).to_string(index=False))

# ============================================================
# 10. 회귀선 그래프
# ============================================================

plt.figure(figsize=(12, 6))

# 실제 연평균 기온
plt.scatter(
    df["year"],
    df["temp"],
    label="실제 연평균 기온",
    alpha=0.6
)

# 50년 회귀선
years_50 = np.arange(1956, 2026).reshape(-1, 1)
plt.plot(
    years_50,
    model_50.predict(years_50),
    label="1956~2005 회귀선",
    linewidth=2
)

# 100년 회귀선
years_100 = np.arange(1906, 2026).reshape(-1, 1)
plt.plot(
    years_100,
    model_100.predict(years_100),
    label="1906~2005 회귀선",
    linewidth=2
)

# 학습/테스트 경계
plt.axvline(
    2005,
    linestyle="--",
    linewidth=1
)

plt.xlabel("연도")
plt.ylabel("연평균 기온 (℃)")
plt.title("연평균 기온 선형회귀 분석")
plt.legend()
plt.grid(alpha=0.3)

plt.show()
