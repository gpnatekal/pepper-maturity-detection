# 🌿 Pepper Maturity Detection

An **AI-based system** for detecting pepper maturity, predicting harvest readiness, and providing related analytics.

---

## 📁 Folder Structure

```
pepper-maturity-detection/
├── angular/        # Angular frontend application
└── python/         # Python backend (models, APIs)
```

### `angular/`
Contains the **frontend** code (Angular application).

To set up:
```bash
cd angular
npm install
```

Start the frontend (development):
```bash
cd angular
ng serve
# or
npm start
```
Open your browser at: `http://localhost:4200`

---

### `python/`
Contains the **backend** code (Python + AI models).

1. Download the required models from the provided Google Drive and place them inside the `python/` folder.  
2. Install Python dependencies:
   ```bash
   cd python
   pip install -r requirements.txt
   ```
3. Run the backend:
   ```bash
   python main.py
   ```
By default the backend runs on `http://0.0.0.0:5000` (Flask) or `http://127.0.0.1:8000` (FastAPI / uvicorn) depending on which server you use. Check `main.py` for the actual host/port.

---

## ⚙️ Requirements

- **Frontend:** Node.js (>=14), Angular CLI  
- **Backend:** Python 3.8+  
- **Key Python libraries:** (listed in `python/requirements.txt`)  
  - fastapi, uvicorn, SQLAlchemy, pydantic, pandas, requests, beautifulsoup4, python-dotenv, apscheduler, numpy, opencv-python, torch, torchvision, pillow, ultralytics, flask, flask-cors, timm, tensorflow

---

## 🔐 SSH & GitHub (Quick Setup)

If you plan to push/pull from GitHub using SSH (Windows 11), follow these steps:

1. Open PowerShell or Git Bash.
2. Generate a new SSH key:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```
3. Start the ssh-agent and add the key:
   - PowerShell:
     ```powershell
     Start-Service ssh-agent
     ssh-add $env:USERPROFILE\.ssh\id_ed25519
     ```
   - Git Bash:
     ```bash
     eval "$(ssh-agent -s)"
     ssh-add ~/.ssh/id_ed25519
     ```
4. Copy the public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   (or `Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub` in PowerShell)
5. Add the key to GitHub: **Settings → SSH and GPG keys → New SSH key**. Paste the key and save.
6. Test:
   ```bash
   ssh -T git@github.com
   ```

---

## 🚀 How to Use (End-to-End)

1. Ensure models are placed in `python/` as described.
2. Start the backend:
   ```bash
   cd python
   python main.py
   ```
3. Start the frontend:
   ```bash
   cd angular
   ng serve
   ```
4. Use the Angular UI to upload images (for pepper maturity detection) and to request price predictions (if integrated).

---

## 🧠 Features

- Pepper maturity classification using YOLO + MiDaS depth estimation
- Pepper price forecasting using an LSTM-based time series model
- Recommendation engine suggesting BUY / HOLD / SELL based on forecast
- Full-stack integration (Angular frontend + Python backend)

---

## 🛠️ Development Tips

- Keep heavy ML models and weights in the `python/` folder and avoid committing large binaries to Git — use Git LFS or host models externally (Google Drive) and download during setup.
- For production deployment, consider:
- Serving the ML model via a separate worker or microservice.
- Using GPU-enabled instances for fast inference (CUDA).
  - Caching scraped data and predictions to avoid re-training on every request.
  - Using HTTPS and restricting CORS to your frontend domain.

---

## 📝 License & Attribution

Add a license file (e.g., `LICENSE`) if you intend to open-source the project. Cite any external models and datasets you use per their licenses.

---

If you want, I can:
- Add badges, screenshots, or a "Getting Started" checklist.
- Create a `CONTRIBUTING.md` or `LICENSE` file.
- Generate a downloadable `README.md` file for you now.
