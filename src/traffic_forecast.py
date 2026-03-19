"""
Прогнозирование интенсивности трафика с использованием LSTM.

Использование:
    python traffic_forecast.py                  # обучение + оценка
    python traffic_forecast.py --predict        # прогноз на следующий час (требуется сохранённая модель)
    python traffic_forecast.py --csv custom.csv # путь к пользовательскому CSV
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential, load_model
except ImportError as e:
    sys.exit(f"[ERROR] TensorFlow not installed: {e}")


# Auto-create directories
Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


# ─── Constants ────────────────────────────────────────────────────────────────

LOOKBACK: int = 24          
TRAIN_SPLIT: float = 0.8
EPOCHS: int = 50
BATCH_SIZE: int = 32
MODEL_PATH: str = "models/traffic_model.keras"   
SCALER_PATH: str = "models/scaler.pkl"          
RANDOM_SEED: int = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ─── Data Layer ───────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.Series:
    """Load and validate CSV; return cleaned intensity series indexed by timestamp."""
    path = Path(csv_path)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {csv_path}")

    df = pd.read_csv(path)
    required = {"timestamp", "intensity"}
    if not required.issubset(df.columns):
        sys.exit(f"[ERROR] CSV must contain columns: {required}. Got: {set(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")

    missing = df["intensity"].isna().sum()
    if missing:
        df["intensity"] = df["intensity"].ffill().bfill()
        print(f"[WARN] Filled {missing} missing value(s) via ffill/bfill")

    return df["intensity"]


def scale(series: pd.Series) -> Tuple[np.ndarray, MinMaxScaler]:
    """Fit MinMaxScaler and return scaled array + fitted scaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    return scaled, scaler


def make_sequences(
    data: np.ndarray, lookback: int = LOOKBACK
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window sequences: X shape (n, lookback, 1), y shape (n,)."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def train_test_split_seq(
    X: np.ndarray, y: np.ndarray, split: float = TRAIN_SPLIT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split — no shuffling to preserve temporal order."""
    cut = int(len(X) * split)
    return X[:cut], X[cut:], y[:cut], y[cut:]


# ─── Model Layer ──────────────────────────────────────────────────────────────

def build_model(lookback: int = LOOKBACK) -> Sequential:
    """
    Двухслойная стекированная LSTM-сеть.
    Параметр return_sequences=True на первом слое передаёт полную последовательность во второй LSTM-слой.
    Dropout снижает риск переобучения на небольших наборах данных о трафике.
    """
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ],
        name="traffic_lstm",
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def get_callbacks() -> list:
    return [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]


def train(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tf.keras.callbacks.History:
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,  
        callbacks=get_callbacks(),
        verbose=1,
    )
    return history


# ─── Evaluation & Visualisation ───────────────────────────────────────────────

def evaluate(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
) -> dict:
    """Return MAE and RMSE in original scale."""
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"\n{'─'*40}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"{'─'*40}\n")
    return {"mae": mae, "rmse": rmse, "y_true": y_true, "y_pred": y_pred}


def plot_results(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "forecast.png") -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true, label="Actual", linewidth=1.5, color="#2196F3")
    ax.plot(y_pred, label="Predicted", linewidth=1.5, linestyle="--", color="#F44336")
    ax.set_title("Traffic Intensity — Actual vs Predicted")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/training_loss.png", dpi=150)
    #plt.show()
    print(f"[INFO] Plot saved → {save_path}")


def plot_training(history: tf.keras.callbacks.History) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"], label="Train loss")
    ax.plot(history.history["val_loss"], label="Val loss")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/training_loss.png", dpi=150)
    plt.show()


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_artifacts(model: Sequential, scaler: MinMaxScaler) -> None:
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Model → {MODEL_PATH}  |  Scaler → {SCALER_PATH}")


def load_artifacts() -> Tuple[Sequential, MinMaxScaler]:
    if not Path(MODEL_PATH).exists() or not Path(SCALER_PATH).exists():
        sys.exit("[ERROR] Saved model/scaler not found. Run training first.")
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ─── Inference ────────────────────────────────────────────────────────────────

def predict_next_hour(csv_path: str = "data/traffic.csv") -> float:
    """
    Прогнозирует интенсивность на один шаг вперёд, используя последние LOOKBACK наблюдений.
    Возвращает предсказанное значение в оригинальном (не нормализованном) диапазоне.
    """
    model, scaler = load_artifacts()
    series = load_data(csv_path)

    if len(series) < LOOKBACK:
        sys.exit(f"[ERROR] Need at least {LOOKBACK} data points, got {len(series)}")

    # Берём последние 24 часа
    last_window = series.values[-LOOKBACK:].reshape(-1, 1)
    
    # Нормализуем (используем сохранённый скалер!)
    scaled_window = scaler.transform(last_window)
    
    # Меняем форму для LSTM: (1, 24, 1)
    scaled_window = scaled_window.reshape(1, LOOKBACK, 1)
    
    # Делаем прогноз
    pred_scaled = model.predict(scaled_window, verbose=0)
    
    # ОТЛАДКА: посмотрим что возвращает модель
    print(f"[DEBUG] pred_scaled shape: {pred_scaled.shape}, value: {pred_scaled[0,0]:.4f}")
    
    # ВАЖНО: Преобразуем обратно к оригинальному масштабу
    # pred_scaled имеет форму (1, 1) - подходит для scaler
    prediction = scaler.inverse_transform(pred_scaled)[0, 0]
    
    print(f"[DEBUG] prediction after inverse: {prediction:.2f}")

    print(f"[PREDICT] Next-hour traffic intensity: {prediction:.2f}")
    return float(prediction)

# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_training_pipeline(csv_path: str) -> None:
    print("[1/6] Loading data...")
    series = load_data(csv_path)
    print(f"      {len(series)} records | {series.index.min()} → {series.index.max()}")

    print("[2/6] Scaling...")
    scaled, scaler = scale(series)

    print("[3/6] Building sequences...")
    X, y = make_sequences(scaled, LOOKBACK)
    X_train, X_test, y_train, y_test = train_test_split_seq(X, y)
    print(f"      Train: {X_train.shape}  |  Test: {X_test.shape}")

    print("[4/6] Building model...")
    model = build_model(LOOKBACK)
    model.summary()

    print("[5/6] Training...")
    history = train(model, X_train, y_train)
    plot_training(history)

    print("[6/6] Evaluating...")
    metrics = evaluate(model, X_test, y_test, scaler)
    plot_results(metrics["y_true"], metrics["y_pred"])

    save_artifacts(model, scaler)
    # Тестовый прогноз после обучения
    print("\n[TEST] Проверка модели после обучения...")
    test_window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    test_pred = model.predict(test_window, verbose=0)
    test_pred_orig = scaler.inverse_transform(test_pred.reshape(-1, 1))[0, 0]
    print(f"[TEST] Прогноз на последние данные: {test_pred_orig:.2f} машин")
    print(f"[TEST] Фактическое значение: {series.values[-1]:.2f} машин")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic LSTM Forecaster")
    parser.add_argument("--csv", default="data/traffic.csv", help="Path to CSV data file")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run predict_next_hour() using saved model (skip training)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.predict:
        predict_next_hour(args.csv)
    else:
        run_training_pipeline(args.csv)