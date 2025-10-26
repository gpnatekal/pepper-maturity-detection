from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import os
import json

app = Flask(__name__)
CORS(app)

# ---------- Load Models ----------
print("Loading YOLOv8 model...")
model = YOLO("best.pt")

print("Loading MiDaS depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

print("Loading P5 autoencoder model...")
p5_model = load_model("p5_autoencoder.h5", compile=False)
print("P5 model input shape:", p5_model.input_shape)


# ---------- Helper Functions ----------
def classify_peppercorn(depth):
    if 529 <= depth <= 569:
        return "P1"
    elif 648 <= depth <= 680:
        return "P2"
    elif 686 <= depth <= 720:
        return "P3"
    elif 722 <= depth:
        return "P4"
    else:
        return "Unknown"


def img_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def is_red_or_dark(img_bgr, red_thresh=0.02, dark_thresh=0.10):
    """
    Quick HSV-based red detection and grayscale-based dark detection.
    Returns (is_red_dark_bool, debug_dict)
    """
    h, w = img_bgr.shape[:2]
    total = h * w

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # red ranges (two ranges)
    lower1 = np.array([0, 40, 30])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 40, 30])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_count = int(np.sum(red_mask > 0))
    red_prop = red_count / float(total)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark_count = int(np.sum(gray < 50))  # pixel value < 50 considered dark
    dark_prop = dark_count / float(total)

    debug = {"red_prop": red_prop, "red_count": red_count, "dark_prop": dark_prop, "dark_count": dark_count}
    is_red_dark = (red_prop >= red_thresh) or (dark_prop >= dark_thresh)
    return is_red_dark, debug


def _prepare_input_for_model(img_bgr, target_h, target_w, channels_first=False, use_rgb=True, scale_mode="0_1"):
    """
    Prepare image numpy array to feed the autoencoder.
    - use_rgb: True -> convert to RGB, False -> keep BGR
    - scale_mode: "0_1" or "-1_1"
    - channels_first: True -> (1, C, H, W) else (1, H, W, C)
    """
    if use_rgb:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = img_bgr.copy()
    img = cv2.resize(img, (target_w, target_h))
    arr = img.astype("float32")
    if scale_mode == "0_1":
        arr = arr / 255.0
    elif scale_mode == "-1_1":
        arr = (arr / 127.5) - 1.0
    else:
        raise ValueError("Unknown scale_mode")

    if channels_first:
        arr = np.transpose(arr, (2, 0, 1))

    arr = np.expand_dims(arr, axis=0)
    return arr


def check_p5(img_bgr, color_red_thresh=0.02, color_dark_thresh=0.10, ae_threshold=0.008):
    """
    Robust P5 detection:
      1) Quick color heuristics (red/dark) to immediately detect obvious P5
      2) Otherwise use autoencoder reconstruction error with multiple preprocessing tries
    Returns: (is_p5_boolean, debug_info_dict)
    """
    debug = {}
    # Step A: simple color heuristic
    is_red_dark, color_debug = is_red_or_dark(img_bgr, red_thresh=color_red_thresh, dark_thresh=color_dark_thresh)
    debug['color_debug'] = color_debug
    if is_red_dark:
        debug['reason'] = "color_heuristic"
        debug['decision'] = True
        debug['ae_errors'] = {}
        print("[P5] Color heuristic matched:", color_debug)
        return True, debug

    # Step B: Autoencoder-based detection (fallback)
    # Determine target size & channel order expected by model
    # p5_model.input_shape: (None, H, W, C) or (None, C, H, W)
    input_shape = p5_model.input_shape
    if len(input_shape) == 4:
        _, d1, d2, d3 = input_shape
        # find which is H,W,C or C,H,W
        # naive approach: if d3 in (1,3) then likely channels_last
        if d3 in (1, 3):
            target_h, target_w, channels = d1, d2, d3
            expect_channels_first = False
        else:
            # channels might be second dimension
            target_h, target_w = d2, d3
            channels = d1
            expect_channels_first = True
    else:
        # fallback default
        target_h, target_w, channels = 128, 128, 3
        expect_channels_first = False

    # Try several reasonable preprocessings and record errors
    preprocess_options = []
    # candidate combos: rgb/bgr x 0_1/-1_1 x channels_first True/False (but prefer model's expected)
    combos = [
        (True, "0_1", expect_channels_first),
        (True, "-1_1", expect_channels_first),
        (False, "0_1", expect_channels_first),
        (False, "-1_1", expect_channels_first),
    ]
    errors = {}
    chosen_config = None
    min_error = float("inf")

    for (use_rgb, scale_mode, channels_first) in combos:
        try:
            inp = _prepare_input_for_model(img_bgr, target_h, target_w, channels_first=channels_first, use_rgb=use_rgb, scale_mode=scale_mode)
            recon = p5_model.predict(inp)
            # If model returns list, grab first output
            if isinstance(recon, list) or isinstance(recon, tuple):
                recon = recon[0]
            # Make sure recon dtype matches
            recon = recon.astype("float32")
            # If recon has shape like (1, C, H, W) and inp (1,H,W,C) -> try to transpose recon
            if recon.shape != inp.shape:
                # try to align by transposing channel order
                if recon.ndim == 4 and inp.ndim == 4:
                    # try possible transposes:
                    if recon.shape[1] in (1, 3) and recon.shape[-1] in (1, 3):
                        # try channels_first -> channels_last
                        try:
                            recon2 = np.transpose(recon, (0, 2, 3, 1))
                            if recon2.shape == inp.shape:
                                recon = recon2
                        except Exception:
                            pass
            # Compute MSE
            mse = float(np.mean((inp - recon) ** 2))
            key = f"{'RGB' if use_rgb else 'BGR'}_{scale_mode}_{'CF' if channels_first else 'CL'}"
            errors[key] = mse
            if mse < min_error:
                min_error = mse
                chosen_config = key
        except Exception as e:
            # continue but log the failure
            key = f"{'RGB' if use_rgb else 'BGR'}_{scale_mode}_{'CF' if channels_first else 'CL'}"
            errors[key] = f"error:{repr(e)}"

    debug['ae_errors'] = errors
    debug['chosen_ae'] = chosen_config
    debug['min_ae_error'] = min_error

    print("[P5 AE errors] ", errors)
    print("[P5 chosen config] ", chosen_config, "min_error=", min_error)

    # Decide: since your AE was trained on P5 (low error for P5), choose threshold accordingly
    is_p5_by_ae = False
    if isinstance(min_error, float) and not np.isinf(min_error):
        is_p5_by_ae = (min_error < ae_threshold)
    debug['ae_threshold_used'] = ae_threshold
    if is_p5_by_ae:
        debug['reason'] = "autoencoder_low_error"
        debug['decision'] = True
        return True, debug

    debug['reason'] = "no_match"
    debug['decision'] = False
    return False, debug


# ---------- Flask route ----------
@app.route('/upload', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        img_bytes = file.read()
        arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error': 'Invalid image'}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_copy_for_vis = img_bgr.copy()

        # ---------- Step 1: robust P5 detection ----------
        print("Running robust P5 detection...")
        is_p5, debug = check_p5(img_bgr, color_red_thresh=0.02, color_dark_thresh=0.10, ae_threshold=0.008)
        print("P5 detection debug:", debug)

        if is_p5:
            detections = ["Over-mature category detected: P5"]
            original_base64 = img_to_base64(img_bgr)
            return jsonify({
                'original': f'data:image/png;base64,{original_base64}',
                'depth': '',
                'detections': detections,
                'percentage': 100.0,
                'harvest_recommendation': "Ready to harvest",
                'debug': debug
            })

        # ---------- Step 2: YOLO + MiDaS (P1-P4) ----------
        print("Running YOLO detection...")
        results = model(img_rgb, conf=0.1)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 and hasattr(results[0], "boxes") else np.array([])

        if boxes.size == 0:
            print("No YOLO boxes detected; returning original image only.")
            return jsonify({
                'original': f'data:image/png;base64,{img_to_base64(img_bgr)}',
                'depth': '',
                'detections': ["No pepper detected"],
                'percentage': 0.0,
                'harvest_recommendation': "Not ready for harvest",
                'debug': debug
            })

        print("Running MiDaS depth estimation...")
        transformed = midas_transform(img_rgb)
        input_batch = transformed["image"].to(device) if isinstance(transformed, dict) else transformed.to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        depth_vis = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        detections = []
        percentages = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = min((x2 - x1), (y2 - y1)) // 2
            mask = np.zeros_like(prediction, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 1, -1)
            depth_values = prediction[mask == 1]
            if depth_values.size == 0:
                median_depth = float(np.nan)
                percentage = 0.0
            else:
                median_depth = float(np.median(depth_values))
                # ----- NEW: calculate percentage -----
                standard_depth = 734.0
                percentage = min(max((median_depth / standard_depth) * 100, 0), 100)

            # ----- harvest recommendation -----
            if percentage < 60:
                harvest = "Not ready for harvest"
            elif percentage < 85:
                harvest = "Harvest soon"
            else:
                harvest = "Ready to harvest"

            percentages.append(percentage)

            category = classify_peppercorn(median_depth) if not np.isnan(median_depth) else "Unknown"
            detections.append({
                "text": f"Median depth at ({cx},{cy}): {median_depth:.2f} → Category: {category}",
                "percentage": round(percentage, 1),
                "harvest_recommendation": harvest
            })

            cv2.circle(img_copy_for_vis, (cx, cy), r, (0, 255, 0), 2)
            cv2.putText(img_copy_for_vis, f"{category} ({median_depth:.1f})", (cx - 40, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(depth_vis, (cx, cy), r, (0, 255, 0), 2)

        # Average percentage if multiple detected
        avg_percentage = float(np.mean(percentages)) if percentages else 0.0

        # Overall harvest recommendation based on avg_percentage
        if avg_percentage < 60:
            overall_harvest = "Not ready for harvest"
        elif avg_percentage < 85:
            overall_harvest = "Harvest soon"
        else:
            overall_harvest = "Ready to harvest"

        return jsonify({
            'original': f'data:image/png;base64,{img_to_base64(img_copy_for_vis)}',
            'depth': f'data:image/png;base64,{img_to_base64(depth_vis)}',
            'detections': detections,
            'percentage': round(avg_percentage, 1),
            'harvest_recommendation': overall_harvest,
           # 'debug': debug
        })

    except Exception as e:
        print("Unexpected error:", e)
        return jsonify({'error': str(e)}), 500



# Load the trained model once at startup
disease_model = load_model("disease_model.h5")

# Define class labels
class_names = ['footrot', 'pollu_disease', 'slow_decline']

@app.route('/disease', methods=['POST'])
def predict_disease():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file found"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Save temporarily
        img_path = "temp_image.jpg"
        file.save(img_path)

        # Convert uploaded image to base64
        with open(img_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')

        # Preprocess the image for prediction
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        pred = disease_model.predict(img_array)
        predicted_class = class_names[np.argmax(pred)]
        confidence = float(np.max(pred))

        # Delete temp image
        if os.path.exists(img_path):
            os.remove(img_path)

        return jsonify({
            "predicted_disease": predicted_class,
            "confidence": round(confidence, 4),
            "original": f"data:image/jpeg;base64,{base64_image}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------
# Pepper LSTM Recommender Route (Flask version)
# --------------------------------------------------------
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import StringIO
import os
from flask import request

def scrape_pepper_prices(date_from: str, date_to: str) -> pd.DataFrame:
    base = "https://www.indianspices.com/marketing/price/domestic/current-market-price.html"
    params = {"filterSpice": "Pepper", "dateFrom": date_from, "dateTo": date_to}
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(base, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        df = pd.read_html(r.text)[0]
    else:
        df = pd.read_html(StringIO(str(table)))[0]

    # --- FIX: flatten and clean column headers ---
    df.columns = [
        ' '.join(map(str, c)).strip() if isinstance(c, tuple) else str(c).strip()
        for c in df.columns
    ]

    # --- date detection ---
    if "Date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower():
                df = df.rename(columns={c: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])

    # --- pick price column ---
    price_col = None
    for c in ["Modal Price", "ModalPrice", "MaxPrice"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            raise ValueError("No numeric columns found in scraped table.")
        price_col = num_cols[-1]

    df["Price"] = (
        df[price_col].astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([\d\.]+)")[0]
        .astype(float)
    )

    keep = ["Date", "Market", "Variety", "Price"]
    df = df[[c for c in keep if c in df.columns]].sort_values("Date").reset_index(drop=True)
    return df

def aggregate_daily(df: pd.DataFrame, market_filter: str):
    if market_filter:
        df = df[df["Market"].astype(str).str.contains(market_filter, case=False, na=False)]
        if df.empty:
            raise ValueError(f"No data for market filter '{market_filter}'")

    # Group by date and compute daily mean
    daily = df.groupby("Date", as_index=False)["Price"].mean()
    daily = daily.sort_values("Date").reset_index(drop=True)

    # Create continuous daily index only within existing range
    idx = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    daily = daily.set_index("Date").reindex(idx)
    daily.index.name = "Date"

    # Interpolate only *within* the range — no extrapolation
    daily["Price"] = daily["Price"].interpolate(method="time").ffill().bfill()

    return daily.reset_index()

def make_sequences(series: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

def build_lstm(input_seq_len: int) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(input_seq_len, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_forecast(daily, horizon, seq_len, epochs, batch_size, test_split, min_days):
    if len(daily) < max(min_days, seq_len + 2):
        raise ValueError(f"Not enough data: {len(daily)} days.")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(daily[["Price"]].values)
    split_idx = int(len(scaled) * (1 - test_split))
    train, test = scaled[:split_idx], scaled[split_idx:]
    X_train, y_train = make_sequences(train.flatten(), seq_len)
    X_test, y_test = make_sequences(test.flatten(), seq_len)
    if len(X_test) == 0:
        X_test, y_test = X_train[-seq_len:], y_train[-seq_len:]
    X_train = X_train.reshape((-1, seq_len, 1))
    X_test = X_test.reshape((-1, seq_len, 1))
    model = build_lstm(seq_len)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
        ]
    )
    preds_test = model.predict(X_test).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    preds_test_inv = scaler.inverse_transform(preds_test.reshape(-1, 1)).flatten()
    mae = float(mean_absolute_error(y_test_inv, preds_test_inv))
    rmse = float(np.sqrt(mean_squared_error(y_test_inv, preds_test_inv)))
    last_seq = scaled.flatten()[-seq_len:].tolist()
    future_scaled = []
    for _ in range(horizon):
        x_in = np.array(last_seq[-seq_len:]).reshape(1, seq_len, 1)
        y_hat = model.predict(x_in, verbose=0)[0, 0]
        future_scaled.append(y_hat)
        last_seq.append(y_hat)
    future = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    last_date = pd.to_datetime(daily["Date"].iloc[-1])
    future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    pred_df = pd.DataFrame({"Date": future_dates, "PredictedPrice": future})
    return pred_df, {"MAE": mae, "RMSE": rmse}

def make_recommendation(df, daily, pred_df, metrics):
    # Use the last real market price from scraped data
    last_price = float(df["Price"].iloc[-1])

    # For debugging — shows both
    # print(f"Actual last price: {last_price}")
    # print(f"Interpolated last price: {float(daily['Price'].iloc[-1])}")

    avg_future = float(pred_df["PredictedPrice"].mean())
    pct_change = ((avg_future - last_price) / last_price) * 100
    rmse = metrics["RMSE"]
    confidence = max(0, 1 - min(rmse / max(daily["Price"].tail(30).mean(), 1), 1))

    if pct_change >= 3:
        action = "BUY NOW (expected rise)"
    elif pct_change <= -3:
        action = "WAIT / SELL (expected drop)"
    else:
        action = "HOLD (sideways)"

    return {
        "last_price": last_price,
        "avg_future": avg_future,
        "pct_change": pct_change,
        "rmse": rmse,
        "confidence": confidence,
        "recommendation": action
    }

@app.route("/pepper-price", methods=["POST"])
def pepper_recommend():
    """
    Flask endpoint for Pepper price prediction & recommendation.
    Expects JSON body matching Angular form.
    """
    try:
        data = request.get_json()
        date_from = data.get("dateFrom", "2018-01-01")
        date_to = data.get("dateTo", datetime.today().strftime("%Y-%m-%d"))
        csv_path = data.get("csvPath", "")
        horizon = int(data.get("horizon", 7))
        seq_len = int(data.get("seqLen", 30))
        market_filter = data.get("marketFilter") or ""
        epochs = int(data.get("epochs", 40))
        batch_size = int(data.get("batchSize", 32))
        test_split = float(data.get("testSplit", 0.2))
        min_days = int(data.get("minDays", 200))

        # Load or scrape
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["Date"])
        else:
            df = scrape_pepper_prices(date_from, date_to)

        daily = aggregate_daily(df, market_filter)
        pred_df, metrics = train_forecast(
            daily, horizon, seq_len, epochs, batch_size, test_split, min_days
        )
        summary = make_recommendation(df, daily, pred_df, metrics)

        return jsonify({
            "summary": summary,
            "metrics": metrics,
            "predictions": pred_df.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
