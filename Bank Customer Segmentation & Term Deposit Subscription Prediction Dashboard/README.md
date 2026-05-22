# Bank Customer Analytics — Dash by Plotly

Interactive dashboard for bank customer segmentation and term deposit subscription prediction.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8050 in your browser.
Upload your `bank-additional-full.csv` in the sidebar, or it runs on built-in sample data.

---

## Deploy free on Render.com (public URL)

1. Push this folder to a **GitHub repository**
2. Go to https://render.com → sign in → **New Web Service**
3. Connect your GitHub repo
4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:server --bind 0.0.0.0:$PORT`
5. Click **Deploy** → get a public URL like `https://yourapp.onrender.com`

### Alternative: Railway.app
1. Go to https://railway.app → New Project → Deploy from GitHub
2. It auto-detects the Procfile and deploys instantly

---

## Dashboard pages

| Page | Content |
|---|---|
| 📊 Overview | KPIs, subscription split, job rates, monthly trend |
| 👥 Demographics | Age groups, distribution, marital status, education |
| 💰 Financial | Balance, housing/personal loans, economic indicators |
| 📣 Campaign | Contact method, previous outcome, campaign frequency |
| 🤖 ML Models | LR/RF/DT comparison, ROC curves, feature importance, confusion matrices |
| 🔵 Segments | KMeans elbow, scatter, cluster profiles + subscription rates |

## Files

```
dash_app/
├── app.py            # Full dashboard application
├── requirements.txt  # Python dependencies
├── Procfile          # For Render/Railway deployment
└── README.md
```
