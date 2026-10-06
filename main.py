# ============================================================
# 연평균 기온 선형회귀 분석
#
# 학습 데이터
#   1) 최근 50년: 1956~2005
#   2) 최근 100년: 1906~2005
#
# 공통 테스트 데이터
#   2006~2025
#
# 평가 지표
#   MAE, MSE, R²
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ------------------------------------------------------------
# 1. 데이터 불러오기
# ------------------------------------------------------------

# 파일명은 본인의 데이터 파일명으로 변경하세요.
# 예: "annual_temperature.csv"
df = pd.read_csv("annual_temperature.csv")


# ------------------------------------------------------------
# 2. 컬럼명 확인
# ------------------------------------------------------------

print("데이터 컬럼:")
print(df.columns.tolist())

print("\n데이터 앞부분:")
print(df.head())


# ============================================================
# 중요:
# 아래 두 변수의 컬럼명을 실제 데이터에 맞게 수정하세요.
# ============================================================

YEAR_COL = "연도"
TEMP_COL = "연평균기온"


# ------------------------------------------------------------
# 3. 필요한 데이터만 선택
# ------------------------------------------------------------

data = df[[YEAR_COL, TEMP_COL]].copy()

# 숫자로 변환
data[YEAR_COL] = pd.to_numeric(data[YEAR_COL], errors="coerce")
data[TEMP_COL] = pd.to_numeric(data[TEMP_COL], errors="coerce")

# 결측값 제거
data = data.dropna(subset=[YEAR_COL, TEMP_COL])

# 연도순 정렬
data = data.sort_values(YEAR_COL).reset_index(drop=True)


# ------------------------------------------------------------
# 4. 분석 기간 제한
# ------------------------------------------------------------

data = data[
    (data[YEAR_COL] >= 1906) &
    (data[YEAR_COL] <= 2025)
].copy()

print("\n분석 데이터 기간:")
print(data[YEAR_COL].min(), "~", data[YEAR_COL].max())

print("총 데이터 개수:", len(data))


# ------------------------------------------------------------
# 5. 학습 / 테스트 데이터 분리
# ------------------------------------------------------------

# 최근 50년 학습
train_50 = data[
    (data[YEAR_COL] >= 1956) &
    (data[YEAR_COL] <= 2005)
].copy()

# 최근 100년 학습
train_100 = data[
    (data[YEAR_COL] >= 1906) &
    (data[YEAR_COL] <= 2005)
].copy()

# 공통 테스트 데이터
test = data[
    (data[YEAR_COL] >= 2006) &
    (data[YEAR_COL] <= 2025)
].copy()


print("\n==============================")
print("데이터 분할")
print("==============================")

print(f"50년 학습 데이터: {len(train_50)}개")
print(f"100년 학습 데이터: {len(train_100)}개")
print(f"공통 테스트 데이터: {len(test)}개")


# ------------------------------------------------------------
# 6. X, y 설정
# ------------------------------------------------------------

X_train_50 = train_50[[YEAR_COL]]
y_train_50 = train_50[TEMP_COL]

X_train_100 = train_100[[YEAR_COL]]
y_train_100 = train_100[TEMP_COL]

X_test = test[[YEAR_COL]]
y_test = test[TEMP_COL]


# ------------------------------------------------------------
# 7. 선형회귀 모델 생성 및 학습
# ------------------------------------------------------------

model_50 = LinearRegression()
model_100 = LinearRegression()

model_50.fit(X_train_50, y_train_50)
model_100.fit(X_train_100, y_train_100)


# ------------------------------------------------------------
# 8. 회귀계수 확인
# ------------------------------------------------------------

slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_


print("\n==============================")
print("회귀선")
print("==============================")

print(
    f"50년 모델 (1956~2005): "
    f"기온 = {slope_50:.6f} × 연도 + {intercept_50:.3f}"
)

print(
    f"100년 모델 (1906~2005): "
    f"기온 = {slope_100:.6f} × 연도 + {intercept_100:.3f}"
)


# ------------------------------------------------------------
# 9. 테스트 데이터 예측
# ------------------------------------------------------------

y_pred_50 = model_50.predict(X_test)
y_pred_100 = model_100.predict(X_test)


# ------------------------------------------------------------
# 10. 평가 지표 계산
# ------------------------------------------------------------

# 50년 모델
mae_50 = mean_absolute_error(y_test, y_pred_50)
mse_50 = mean_squared_error(y_test, y_pred_50)
r2_50 = r2_score(y_test, y_pred_50)

# 100년 모델
mae_100 = mean_absolute_error(y_test, y_pred_100)
mse_100 = mean_squared_error(y_test, y_pred_100)
r2_100 = r2_score(y_test, y_pred_100)


# ------------------------------------------------------------
# 11. 결과 비교표
# ------------------------------------------------------------

results = pd.DataFrame({
    "학습기간": [
        "1956~2005 (50년)",
        "1906~2005 (100년)"
    ],
    
    "기울기": [
        slope_50,
        slope_100
    ],
    
    "절편": [
        intercept_50,
        intercept_100
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
print("테스트 데이터 예측 성능 비교")
print("(공통 테스트: 2006~2025)")
print("==============================")

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ------------------------------------------------------------
# 12. 어떤 모델이 더 좋은지 자동 비교
# ------------------------------------------------------------

print("\n==============================")
print("모델 성능 비교")
print("==============================")


if mae_50 < mae_100:
    print("MAE: 50년 학습 모델이 더 좋음")
elif mae_50 > mae_100:
    print("MAE: 100년 학습 모델이 더 좋음")
else:
    print("MAE: 두 모델의 성능이 동일함")


if mse_50 < mse_100:
    print("MSE: 50년 학습 모델이 더 좋음")
elif mse_50 > mse_100:
    print("MSE: 100년 학습 모델이 더 좋음")
else:
    print("MSE: 두 모델의 성능이 동일함")


if r2_50 > r2_100:
    print("R²: 50년 학습 모델이 더 좋음")
elif r2_50 < r2_100:
    print("R²: 100년 학습 모델이 더 좋음")
else:
    print("R²: 두 모델의 성능이 동일함")


# ------------------------------------------------------------
# 13. 기울기 비교
# ------------------------------------------------------------

print("\n==============================")
print("회귀선 기울기 비교")
print("==============================")

print(f"50년 기울기  : {slope_50:.6f} °C/년")
print(f"100년 기울기 : {slope_100:.6f} °C/년")

slope_diff = slope_50 - slope_100

print(f"기울기 차이  : {slope_diff:.6f} °C/년")

# 10년당 변화량
print("\n10년당 예상 기온 변화:")
print(f"50년 모델  : {slope_50 * 10:.4f} °C/10년")
print(f"100년 모델 : {slope_100 * 10:.4f} °C/10년")


# ------------------------------------------------------------
# 14. 테스트 기간의 실제값 vs 예측값 저장
# ------------------------------------------------------------

prediction_result = test.copy()

prediction_result["예측기온_50년"] = y_pred_50
prediction_result["예측기온_100년"] = y_pred_100

prediction_result["오차_50년"] = (
    prediction_result[TEMP_COL]
    - prediction_result["예측기온_50년"]
)

prediction_result["오차_100년"] = (
    prediction_result[TEMP_COL]
    - prediction_result["예측기온_100년"]
)


print("\n==============================")
print("2006~2025년 예측 결과")
print("==============================")

print(
    prediction_result.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ------------------------------------------------------------
# 15. 예측 결과 CSV 저장
# ------------------------------------------------------------

prediction_result.to_csv(
    "temperature_regression_predictions_2006_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

results.to_csv(
    "temperature_regression_model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 16. 그래프 ① 전체 데이터 + 두 회귀선
# ------------------------------------------------------------

plt.figure(figsize=(13, 7))

# 전체 실제 데이터
plt.scatter(
    data[YEAR_COL],
    data[TEMP_COL],
    color="gray",
    alpha=0.55,
    s=25,
    label="실제 연평균 기온"
)

# 회귀선용 연도
years = np.arange(1906, 2026).reshape(-1, 1)

# 50년 회귀선
plt.plot(
    years,
    model_50.predict(years),
    color="blue",
    linewidth=2.5,
    label="1956~2005 학습 회귀선"
)

# 100년 회귀선
plt.plot(
    years,
    model_100.predict(years),
    color="red",
    linewidth=2.5,
    label="1906~2005 학습 회귀선"
)

# 테스트 구간 표시
plt.axvspan(
    2006,
    2025,
    color="orange",
    alpha=0.12,
    label="테스트 기간 (2006~2025)"
)

plt.xlabel("연도")
plt.ylabel("연평균 기온 (°C)")
plt.title("연평균 기온 선형회귀: 50년 학습 vs 100년 학습")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 17. 그래프 ② 테스트 기간 예측 성능 비교
# ------------------------------------------------------------

plt.figure(figsize=(13, 7))

plt.plot(
    test[YEAR_COL],
    y_test,
    marker="o",
    color="black",
    linewidth=2,
    label="실제 기온"
)

plt.plot(
    test[YEAR_COL],
    y_pred_50,
    marker="o",
    color="blue",
    linestyle="--",
    label="50년 학습 모델 예측"
)

plt.plot(
    test[YEAR_COL],
    y_pred_100,
    marker="o",
    color="red",
    linestyle="--",
    label="100년 학습 모델 예측"
)

plt.xlabel("연도")
plt.ylabel("연평균 기온 (°C)")
plt.title("2006~2025 테스트 데이터 예측 비교")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. MAE / MSE / R² 막대그래프
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 5)
)

models = ["50년 학습", "100년 학습"]

# MAE
axes[0].bar(
    models,
    [mae_50, mae_100],
    color=["blue", "red"]
)

axes[0].set_title("MAE")
axes[0].set_ylabel("MAE")
axes[0].grid(axis="y", alpha=0.3)

# MSE
axes[1].bar(
    models,
    [mse_50, mse_100],
    color=["blue", "red"]
)

axes[1].set_title("MSE")
axes[1].set_ylabel("MSE")
axes[1].grid(axis="y", alpha=0.3)

# R²
axes[2].bar(
    models,
    [r2_50, r2_100],
    color=["blue", "red"]
)

axes[2].set_title("R²")
axes[2].set_ylabel("R²")
axes[2].grid(axis="y", alpha=0.3)

plt.suptitle(
    "2006~2025 테스트 데이터 기준 회귀모델 성능 비교"
)

plt.tight_layout()
plt.show()
