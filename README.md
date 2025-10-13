# 🌿 Pepper Maturity Detection

An **AI-powered full-stack system** for detecting pepper maturity, predicting harvest readiness, and providing price forecasting with intelligent recommendations.

This project combines **computer vision (YOLO + MiDaS)**, **autoencoders**, and **LSTM-based time series forecasting**, integrated through an **Angular frontend** and a **Python (Flask/FastAPI) backend**.

---

## 📁 Folder Structure

```
pepper-maturity-detection/
│
├── angular/                # Frontend (Angular application)
│   ├── src/
│   ├── package.json
│   └── angular.json
;; │git reset --mixed HEAD~2

├── python/                 # Backend (Python + AI Models)
│   ├── main.py             # Main backend entry point
│   ├── main2.py            # Flask API (YOLO, MiDaS, LSTM)
│   ├── pepper_lstm_scraper_recommender.py
│   ├── requirements.txt
│   └── models/             # Place downloaded model files here
│
└── README.md
```

---

## ⚙️ Requirements

### 🧩 Frontend
- **Node.js** (v16+ recommended)
- **Angular CLI**

### 🧠 Backend
- **Python 3.8+**
- Libraries listed in [`requirements.txt`](python/requirements.txt)

---

## 🚀 Setup & Run Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/pepper-maturity-detection.git
cd pepper-maturity-detection
```

---

### 2️⃣ Set Up the Backend

```bash
cd python
pip install -r requirements.txt
```

#### 🔽 Download Required Models
Download the pre-trained AI models from Google Drive:

👉 [Download Models from Google Drive](https://drive.google.com/drive/folders/1GXd1wI6CuSky2f_0qecLd4xyl2YOuIMj?usp=drive_link)

After download, place all `.pt`, `.h5`, and related model files inside the `python/` folder.

#### ▶️ Run the Backend

```bash
python main.py
```

Backend will start at:
```
http://localhost:5000
```

---

### 3️⃣ Set Up the Frontend

```bash
cd angular
npm install
```

#### ▶️ Run the Angular App

```bash
ng serve
```

Frontend will start at:
```
http://localhost:4200
```

---

## 🌐 Access the Application

Once both servers are running:

🔗 Open in your browser:  
**[http://localhost:4200](http://localhost:4200)**

The Angular frontend communicates with the Flask backend (`http://localhost:5000`) for predictions and analytics.

---

## 🧩 Features

### 🔍 1. Pepper Maturity Detection
- Uses **YOLOv8** and **MiDaS** for depth-based maturity analysis.  
- Detects peppercorn categories (P1–P5).  
- Autoencoder-powered P5 (over-mature) classification.

### 🌡️ 2. Depth & Disease Analytics
- **MiDaS** model estimates relative depth of detected peppercorns.  
- **CNN classifier** detects common pepper diseases (e.g., footrot, pollu disease, slow decline).

### 💹 3. Price Forecasting & Recommendations
- **LSTM-based time-series forecasting** for pepper prices.  
- Predicts next 7 days of prices.  
- Provides actionable recommendations:
  - 🟢 *BUY NOW (expected rise)*
  - 🔴 *WAIT / SELL (expected drop)*
  - 🟡 *HOLD (sideways)*

### 🔄 4. Full-stack Integration
- Angular Reactive Forms for user input.  
- Flask/FastAPI backend serving AI predictions via REST API.  
- Real-time visualization of predicted trends and summaries.

---

## 🧱 Key Technologies

| Layer | Technology |
|-------|-------------|
| Frontend | Angular 17+, Bootstrap 5 |
| Backend | Python 3.8+, Flask / FastAPI |
| AI Models | YOLOv8, MiDaS, Autoencoder (P5), TensorFlow LSTM |
| Libraries | OpenCV, Torch, TensorFlow, BeautifulSoup, Pandas |
| Visualization | Chart.js, Bootstrap Tables |

---

## 🧰 Example API Endpoints

### 📸 Pepper Image Upload
```bash
POST /upload
Content-Type: multipart/form-data
```
Uploads an image and returns detected maturity levels (P1–P5).

---

### 💠 Disease Detection
```bash
POST /disease
Content-Type: multipart/form-data
```
Returns predicted disease and confidence.

---

### 📊 Pepper Price Forecast
```bash
POST /pepper-price
Content-Type: application/json

{
  "dateFrom": "2018-01-01",
  "dateTo": "2025-10-13",
  "horizon": 7,
  "seqLen": 30,
  "marketFilter": "",
  "epochs": 40,
  "batchSize": 32,
  "testSplit": 0.2,
  "minDays": 200
}
```

**Response:**
```json
{
  "summary": {
    "last_price": 695.0,
    "avg_future": 681.2,
    "pct_change": -1.98,
    "recommendation": "HOLD (sideways)"
  },
  "metrics": {"MAE": 11.27, "RMSE": 16.26},
  "predictions": [
    {"Date": "2025-10-14", "PredictedPrice": 677.5},
    {"Date": "2025-10-15", "PredictedPrice": 679.5}
  ]
}
```

---

## 📈 UI Highlights

- Responsive UI built with **Bootstrap 5**
- Clean data tables and summary cards
- Interactive chart for pepper price trends
- Real-time feedback and progress indicators

---

## 🧪 Example Workflow

1. Upload pepper image → get maturity classification (P1–P5).  
2. Predict pepper diseases if any.  
3. Use the price prediction form to forecast next-week prices.  
4. View actionable recommendation (Buy / Sell / Hold).  

---

## 🧑‍💻 Developers

- **AI & Backend**: Python, TensorFlow, PyTorch, Flask  
- **Frontend**: Angular, Bootstrap, Chart.js  
- **Integration**: REST APIs, JSON, Reactive Forms  

---

## 📜 License

This project is open-source and available for educational and research purposes.

---

### 🔗 Resources

- [📦 Download Pre-trained Models (Google Drive)](https://drive.google.com/drive/folders/1GXd1wI6CuSky2f_0qecLd4xyl2YOuIMj?usp=drive_link)
- [💡 YOLOv8 Docs](https://docs.ultralytics.com)
- [📘 MiDaS Depth Estimation](https://github.com/isl-org/MiDaS)
- [📙 TensorFlow Keras API](https://www.tensorflow.org/api_docs/python/tf/keras)

---

**🌿 Pepper Maturity Detection**  
_An AI-driven approach for precision agriculture and market intelligence._
