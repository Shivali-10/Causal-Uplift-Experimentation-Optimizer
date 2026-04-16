# Causal Uplift & Experimentation Optimizer

An enterprise-grade MLOps system for **Incremental Lift Modeling**. This project utilizes Causal Inference to identify "Persuadable" customers and optimizes marketing spend through uplift experimentation.

## 🚀 Quick Start (One Command)

If you have Docker installed, simply run:
```bash
docker-compose up
```
- **Dashboard**: http://localhost:8501
- **API (FastAPI)**: http://localhost:8000/docs

---

## 🛠️ Project Architecture

1.  **Modeling Engine**: Implements Meta-Learners (**S-Learner, T-Learner, and Class Transformation**) using Randomized Controlled Trial (RCT) data from **Criteo AI Lab**.
2.  **Simulation Engine**: A real-time ROI calculator that estimates profit based on marketing spend and predicted uplift.
3.  **Explainability**: Integrated **SHAP** values to explain the causal impact of features on incrementality.
4.  **Inference API**: A production-ready FastAPI endpoint for real-time treatment recommendations.

---

## 📂 File Structure

- `api/`: FastAPI backend logic.
- `research/`: Data ingestion, modeling experiments, and explainability scripts.
- `data/`: Sampled Criteo data (Parquet/Pickle).
- `models/`: Production meta-learners and SHAP assets.
- `dashboard.py`: Streamlit management UI.

---

## 🎓 Why this project is "Sir-ready"?

- **Solves a Business Problem**: Most models predict *correlation*; this predicts **Causality** (Individual Treatment Effect).
- **Scalable**: Handled a 14-million-row industrial dataset.
- **Production Graded**: Includes health checks, Docker containerization, and a microservice architecture.

## 👥 Contributors

- **Shivali** - Lead MLOps Engineer
- **Vaishali** ([@Vaishali-1234](https://github.com/Vaishali-1234)) - Data Scientist / Causal Inference Specialist

Built with ❤️ for Technical Portfolios.
