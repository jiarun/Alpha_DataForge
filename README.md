# 🔥 Alpha DataForge

**An intelligent data conditioning agent with 64+ automated preprocessing operations across 14 ML pipeline categories.**

Built for data scientists and ML engineers who want a single interactive tool to clean, transform, and engineer features from any dataset — no code required.

---

## ✨ Features

- **14 Processing Categories** — from data quality to dimensionality reduction
- **64+ Operations** — all tested and validated
- **Interactive UI** — Streamlit-based, runs in browser
- **Stateful Pipeline** — chain operations, preview live, reset anytime
- **Export** — download processed dataset as CSV
- **Supports** — CSV, Excel (.xlsx/.xls), TSV

---

## 📋 Categories & Operations

| Category | Key Operations |
|----------|---------------|
| Data Quality | Missing value detection/imputation, duplicate detection/removal |
| Numeric Cleaning | Type conversion, precision standardization |
| Outlier Processing | Z-score, IQR, Isolation Forest, Local Outlier Factor |
| Distribution Conditioning | Log, Sqrt, Box-Cox, Yeo-Johnson, Quantile transforms |
| Feature Scaling | Standard, MinMax, Robust, MaxAbs, Unit Vector |
| Categorical Processing | Label encoding, One-Hot, frequency analysis, consolidation |
| Text Processing | Lowercase, whitespace cleaning, punctuation removal |
| Datetime Processing | Parsing, component extraction (year/month/day/weekday) |
| Feature Engineering | Ratios, differences, interactions, polynomial, binning, rank |
| Time-Series Features | Lag, lead, rolling mean/median/std, EWM |
| Statistical Analysis | Correlation, covariance, t-test, ANOVA, chi-square |
| Feature Selection | Variance threshold, MI, Tree importance, RFE, LASSO |
| Dimensionality Reduction | PCA, Kernel PCA, ICA, Factor Analysis, t-SNE |
| Target Conditioning | Class distribution, SMOTE, ADASYN, over/under-sampling |

---

## 🚀 Setup

### Option 1: Without Docker (Local Python)

**Prerequisites:**
- Python 3.10+

**Steps:**

```bash
# 1. Clone the repository
git clone https://github.com/jiarun/Alpha_DataForge.git
cd Alpha_DataForge

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

**Run tests:**
```bash
python test_processing.py
```

---

### Option 2: With Docker

**Prerequisites:**
- Docker & Docker Compose installed ([Get Docker](https://docs.docker.com/get-docker/))

**Steps:**

```bash
# 1. Clone the repository
git clone https://github.com/jiarun/Alpha_DataForge.git
cd Alpha_DataForge

# 2. Build and run
docker compose up --build
```

The app will open at **http://localhost:8501**

**Stop the app:**
```bash
docker compose down
```

**Run in background (detached):**
```bash
docker compose up --build -d
```

**View logs:**
```bash
docker compose logs -f
```

**Rebuild after code changes:**
```bash
docker compose up --build
```

---

## 📁 Project Structure

```
Alpha_DataForge/
├── app.py                 # Streamlit UI (5 tabs: Overview, Stats, Viz, Filter, Process)
├── processing.py          # Processing engine (64 functions)
├── requirements.txt       # Python dependencies
├── sample_data.csv        # Demo dataset with edge cases
├── test_processing.py     # 64 automated tests
├── Dockerfile             # Container image definition
├── docker-compose.yml     # One-command container orchestration
├── .dockerignore          # Files excluded from Docker build
├── presentation.pptx      # Project presentation
└── README.md              # This file
```

---

## 🧪 Testing

```bash
# Activate venv first (if not using Docker)
source venv/bin/activate

# Run all 64 tests
python test_processing.py
```

Expected output:
```
RESULTS: 64 passed, 0 failed, 64 total
🎉 All tests passed!
```

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Plotly
- **Data:** pandas, numpy, scipy
- **ML:** scikit-learn, imbalanced-learn
- **Containerization:** Docker

---

## 📊 Quick Demo

1. Run the app (either method above)
2. Upload `sample_data.csv` (included in repo)
3. Go to the **🔧 Process** tab
4. Select a category → action → columns → Apply
5. Watch the dataset transform in real-time
6. Download the processed CSV when done

---

## 📄 License

MIT

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first.
