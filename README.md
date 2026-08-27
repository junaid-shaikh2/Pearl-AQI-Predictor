# Pearls AQI Predictor

Real-time, serverless 72-hour Air Quality Index (AQI) forecasting pipeline and dashboard.

Developed as a core project during a Software Engineering internship at 10Pearls, this application demonstrates a complete end-to-end machine learning lifecycle—from automated data ingestion and continuous integration to model inference and UI deployment.

## 📸 Dashboard Interface

<img width="940" height="398" alt="image" src="https://github.com/user-attachments/assets/0d53b7c4-ac4f-4bdb-8600-24de3f8315f6" />


<img width="940" height="447" alt="image" src="https://github.com/user-attachments/assets/3ad2ce1c-bce2-4e22-9f65-6ca432f9e3a4" />


## 🎯 Core Features & Milestones Achieved
* **Feature Pipeline:** Automated extraction of raw weather and pollutant data from the OpenWeather API, engineering temporal and rolling-lag features.
* **Historical Backfilling:** Programmatic historical data retrieval to generate a robust foundational dataset for initial model training.
* **Model Training & Evaluation:** XGBoost regressor evaluated against RMSE, MAE, and R² metrics to ensure high-fidelity forecasting. 
* **Automated CI/CD:** Zero-touch continuous integration via GitHub Actions that handles hourly feature updates and daily model retraining.
* **Real-Time Web Dashboard:** A Streamlit front-end displaying 72-hour forecasting and hazardous AQI level indicators.
* **Advanced Analytics:** Dynamic feature importance visualization to provide model interpretability without heavy dependency bloat.

## 🏗️ System Architecture
This project abandons heavy, monolithic server setups in favor of a 100% serverless, event-driven infrastructure.

* **Data Ingestion:** Automated ETL scripts fetch live meteorological data.
* **Continuous Training:** GitHub Actions triggers a daily workflow to execute the feature pipeline, train the model, and version the `.joblib` artifacts natively in the repository.
* **Inference Engine:** An XGBoost model computes forecasts based on rolling averages, lag features, and temporal seasonality.

## ⚖️ Engineering Trade-offs & Decisions
Building for production requires balancing complexity with performance:
* **XGBoost vs. Deep Learning:** XGBoost was selected over deep learning frameworks (TensorFlow/PyTorch) due to its superior accuracy on tabular meteorological data and significantly lower compute overhead.
* **Native Feature Importance vs. SHAP:** While SHAP was evaluated, extracting native feature importances directly from XGBoost achieved the required transparency while eliminating massive container bloat and reducing load times.
* **Serverless Artifacts vs. Feature Stores:** To maintain a lightweight footprint, the architecture utilizes a stateless pipeline where historical datasets and models are versioned within the deployment ecosystem rather than relying on an expensive, always-on external Feature Store (like Hopsworks).

## 🛠️ Technologies
* **Machine Learning:** Python, Scikit-learn, XGBoost, Pandas
* **Infrastructure & DevOps:** Docker, GitHub Actions (CI/CD)
* **Frontend & Visualization:** Streamlit, Plotly
* **Data Integration:** OpenWeather API

## 🚀 Quick Start (Local Reproduction)
The application is fully containerized for immediate local testing without environment conflicts.

### 1. Clone the repository:
```bash
git clone [https://github.com/junaid-shaikh2/Pearl-AQI-Predictor.git](https://github.com/junaid-shaikh2/Pearl-AQI-Predictor.git)
cd Pearl-AQI-Predictor
```

### 2. Add your API Key:
Create a `.env` file in the root directory and configure your OpenWeather credentials:
```text
OPENWEATHER_API_KEY="your_api_key_here"
```

### 3. Build and Run via Docker:
```bash
docker-compose up --build -d
```

### 4. Access the Dashboard:
Open your browser and navigate to:
```text
http://localhost:8501
```
