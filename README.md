# Soil Prediction App 🌱

This project predicts soil parameters using Machine Learning models (Random Forest, Neural Networks, HistGBR).  
A Flask web app is built for easy interaction for different parameters like soilMoisture, soilTemp, soilInWaterpH etc. by creating a single prediction pipeline..

## Steps Performed:
- Data cleaning (removed duplicates, missing values, blank columns)
- Exploratory analysis (time series plots, mean, std, range)
- Correlation heatmap
- ML models trained and saved using joblib
- Flask app for single prediction pipeline

## Best Model:Added amongst(NN, ensembled, HistGBR)
- Random Forest (R² = 0.8852, RMSE = 1.4634)

## Run Locally
```bash
pip install -r requirements.txt
python app.py

## Dataset & Models
Due to large size, datasets (`.csv`) and trained models (`.pkl`) are excluded from this repository.
