"""Generate synthetic traffic.csv for testing (hourly data, ~3 months)."""
import numpy as np
import pandas as pd

np.random.seed(42)
periods = 24 * 90  # 90 дней × 24 часа
idx = pd.date_range("2024-01-01", periods=periods, freq="h")

# Реалистичный паттерн: суточный цикл + недельная сезонность + тренд + шум
t = np.arange(periods)
daily = 50 * np.sin(2 * np.pi * t / 24 - np.pi / 2) + 50
weekly = 15 * np.sin(2 * np.pi * t / (24 * 7))
trend = 0.005 * t
noise = np.random.normal(0, 5, periods)
intensity = np.clip(daily + weekly + trend + noise, 0, None)

# Добавляем пропуски (NaN) для проверки их заполнения
nan_idx = np.random.choice(periods, size=20, replace=False)
intensity[nan_idx] = np.nan

df = pd.DataFrame({"timestamp": idx, "intensity": intensity})
df.to_csv("data/traffic.csv", index=False)
print(f"Saved data/traffic.csv ({len(df)} rows, {df['intensity'].isna().sum()} NaNs injected)")