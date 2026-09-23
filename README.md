# 🌾 Crop Yield Prediction

A full-stack machine learning web application that predicts agricultural crop yield (in tonnes/hectare) using historical Indian crop data. Built with **Python**, **Flask**, **scikit-learn**, and a pure HTML/CSS/JS frontend.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Technologies Used](#technologies-used)
- [Workflow Diagram](#workflow-diagram)

---

## Project Overview

This application allows users to predict crop yield by entering:
- **State** and **District** of cultivation
- **Crop** type (55 crop varieties)
- **Season** (Kharif, Rabi, Summer, Winter, Autumn, Whole Year)
- **Crop Year** (1997–2020)
- **Area** under cultivation (hectares)
- **Production** estimate (tonnes)

The backend uses a **Random Forest Regressor** trained on 339,240 records to return a predicted yield value instantly via a REST API.

---

## Dataset

| Property        | Value                         |
|-----------------|-------------------------------|
| File            | `Crop_data.csv`               |
| Total Records   | 339,240 (after cleaning)      |
| States          | 37                            |
| Districts       | 707                           |
| Crop Types      | 55                            |
| Seasons         | 6                             |
| Year Range      | 1997 – 2020                   |
| Target Column   | `Yield` (tonnes / hectare)    |

**Columns:** `State`, `District`, `Crop`, `Crop_Year`, `Season`, `Area`, `Production`, `Yield`

---

## Project Structure

```
Crop Yield Prediction/
│
├── Crop_data.csv            # Raw dataset
├── train_model.py           # Data preprocessing + model training script
├── app.py                   # Flask backend (REST API)
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── templates/
│   └── index.html           # Frontend UI (HTML + CSS + JS)
│
├── static/
│   ├── css/                 # (Optional static CSS)
│   └── js/                  # (Optional static JS)
│
└── model/                   # Auto-generated after training
    ├── rf_model.pkl         # Trained Random Forest model
    ├── encoders.pkl         # LabelEncoders for categorical features
    ├── unique_values.json   # Dropdown values for the UI
    ├── metrics.json         # Model evaluation metrics
    └── year_range.json      # Min/max crop year
```

---

## Installation

### 1. Clone / Download the project

```bash
cd "Crop Yield Prediction"
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Train the model

Run this **once** before starting the server:

```bash
python train_model.py
```

Expected output:
```
Training Random Forest Regressor …
  MAE  : 5.6899
  RMSE : 151.6835
  R²   : 0.9739
Artefacts saved to model/
Done!
```

### Step 2 — Start the Flask server

```bash
python app.py
```

### Step 3 — Open the app

Open your browser and go to:

```
http://127.0.0.1:5000
```

Fill in the form fields and click **Predict Yield** to get an instant prediction.

---

## ML Pipeline

```
Raw CSV
  │
  ▼
Column cleaning (strip whitespace)
  │
  ▼
Drop nulls in [Yield, Area, Production, Crop_Year]
Filter Yield > 0, Area > 0
  │
  ▼
Label Encoding: State, District, Crop, Season
  │
  ▼
Feature Matrix X = [State_enc, District_enc, Crop_enc,
                    Season_enc, Crop_Year, Area, Production]
Target y = Yield
  │
  ▼
Train/Test Split (80% / 20%, random_state=42)
  │
  ▼
Random Forest Regressor (150 trees, n_jobs=-1)
  │
  ▼
Evaluation: MAE, RMSE, R²
  │
  ▼
Persist: rf_model.pkl, encoders.pkl, unique_values.json
```

---

## Model Performance

| Metric     | Value     |
|------------|-----------|
| MAE        | 5.6899    |
| RMSE       | 151.6835  |
| R² Score   | **0.9739** |

> **R² = 0.9739** means the model explains **97.39%** of the variance in crop yield — excellent predictive power.

### Feature Importance (Random Forest)

| Feature     | Importance |
|-------------|------------|
| Production  | 59.34 %    |
| Crop        | 16.87 %    |
| Area        | 15.51 %    |
| State       | 5.91 %     |
| Crop Year   | 1.49 %     |
| District    | 0.88 %     |
| Season      | 0.00 %     |

---

## API Reference

### `GET /api/options`
Returns all dropdown values for the UI.

**Response:**
```json
{
  "states":    ["Andaman and Nicobar Island", "..."],
  "districts": ["NICOBARS", "..."],
  "crops":     ["Arecanut", "..."],
  "seasons":   ["Autumn", "Kharif", "Rabi", "Summer", "Whole Year", "Winter"],
  "year_min":  1997,
  "year_max":  2020
}
```

---

### `GET /api/metrics`
Returns model evaluation metrics.

**Response:**
```json
{ "MAE": 5.6899, "RMSE": 151.6835, "R2": 0.9739 }
```

---

### `POST /api/predict`
Predict crop yield.

**Request body (JSON):**
```json
{
  "state":      "Kerala",
  "district":   "ERNAKULAM",
  "crop":       "Coconut",
  "season":     "Whole Year",
  "crop_year":  2015,
  "area":       1200.5,
  "production": 3500
}
```

**Response:**
```json
{
  "yield": 2.9166,
  "unit":  "tonnes / hectare",
  "inputs": { ... }
}
```

---

## Technologies Used

| Layer      | Technology                          |
|------------|-------------------------------------|
| Backend    | Python 3.14, Flask 3.1              |
| ML         | scikit-learn (Random Forest)        |
| Data       | pandas, numpy                       |
| Serialization | joblib                           |
| Visualization | matplotlib, seaborn             |
| Frontend   | HTML5, CSS3, Vanilla JavaScript     |
| Server     | Flask development server            |

---

## Workflow Diagram

```
User Browser
    │
    │  GET /            → index.html (form UI)
    │  GET /api/options → dropdown data
    │  GET /api/metrics → model performance
    │  POST /api/predict→ predicted yield
    │
    ▼
Flask app.py
    │
    ├── Loads rf_model.pkl  (Random Forest)
    ├── Loads encoders.pkl  (LabelEncoders)
    └── Loads unique_values.json (dropdown lists)
            │
            ▼
      Encodes inputs → model.predict() → JSON response
```

---

## Author

Crop Yield Prediction Project — Built with Python, Flask & scikit-learn.
