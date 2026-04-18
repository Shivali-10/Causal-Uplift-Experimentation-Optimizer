# 🎯 Causal Uplift & Experimentation Optimizer

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED.svg)](https://www.docker.com/)

An enterprise-grade MLOps platform designed to solve the **"Counterfactual Problem"** in digital marketing. While traditional ML models predict who *will* buy, this platform uses **Causal Inference** to predict who will buy *specifically because of a treatment* (ad/promotion).

---

## 🚀 Deployment (One Command)

The entire ecosystem is containerized for seamless setup:

```bash
docker-compose up
```

Once running, the **Analytics Dashboard** (Streamlit) and **Inference API** (FastAPI) will be accessible on their respective local ports (8501 and 8000).

---

## 🧠 Technical Architecture

### 1. The Modeling Engine (`research/modeling.py`)
Implements sophisticated **Meta-Learners** (S-Learner, T-Learner, and Class Transformation) using the **Criteo RCT Dataset** (14M+ samples). It handles:
- **Treatment vs. Control Bias**: Correctly estimating the Individual Treatment Effect (ITE).
- **Metric Optimization**: Evaluated using **Qini Curves** and **Cumulative Gain** rather than standard accuracy.

### 2. High-Performance API (`api/main.py`)
A **FastAPI** backend that serves the best-performing causal model.
- **Latency**: Sub-50ms inference.
- **Scalability**: Designed for real-time bid-request targeting.

### 3. Strategy Dashboard (`dashboard.py`)
A "Midnight Sapphire" glassmorphism UI for business stakeholders.
- **ROI Simulator**: Estimates net profit by balancing ad spend against predicted incremental lift.
- **Global Explainability**: Native **SHAP** integration to visualize the drivers of incrementality.

---

## 🛠️ Tech Stack & MLOps

| Layer | Technologies |
| :--- | :--- |
| **Causal Modeling** | `scikit-uplift`, `xgboost`, `scikit-learn` |
| **Backend / API** | `FastAPI`, `Uvicorn`, `Pydantic` |
| **Frontend** | `Streamlit`, `Plotly`, `Seaborn` |
| **Explainability** | `SHAP` |
| **DevOps** | `Docker`, `Docker-Compose` |

---

## 📈 Core Technical Highlights

- **Causal Validation**: Validated using Qini coefficients to ensure marketing spend is directed *only* at "Persuadable" users, avoiding "Sure Things" and "Sleeping Dogs."
- **Production-Ready**: Implements health checks, modular directory structures, and decoupled microservices.
- **Industrial Scale**: Architecture optimized to handle large-scale randomized controlled trials.

---

## 👥 Contributors

- **Shivali** ([@Shivali-10](https://github.com/Shivali-10))
- **Vaishali** ([@Vaishali-1234](https://github.com/Vaishali-1234))



