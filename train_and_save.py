import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("your_file_no_duplicates.csv")

# Define multiple targets and their features
targets_features = {
    'soilInWaterpH': [ 'dryMassFraction','pHSoilInWaterMass', 'sampleTopDepth', 'elevation'],
    'soilTemp': ['soilMoisture', 'dryMassFraction', 'elevation', 'sampleBottomDepth', 'pHSoilInCaClMass'],
    'soilMoisture': ['dryMassBoatMass', 'pHCaClVol', 'pHSoilInCaClMass','waterpHRatio'],
    'pHCaClVol': ['waterpHRatio', 'pHWaterVol', 'soilMoisture', 'dryMassFraction'],
    'boatMass': ['dryMassBoatMass', 'sampleTopDepth', 'elevation', 'soilInCaClpH'],
    'sampleBottomDepth': ['dryMassBoatMass', 'dryMassFraction', 'pHSoilInCaClMass', 'waterpHRatio', 'caclpHRatio'],
    'dryMassFraction': ['pHSoilInWaterMass', 'waterpHRatio', 'caclpHRatio', 'sampleBottomDepth']
}

# Train, evaluate and save model for each target
for target, features in targets_features.items():
    try:
        df_model = df[features + [target]].dropna()
        X = df_model[features]
        y = df_model[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\n📌 Model for: {target}")
        print(f"   ➤ R² Score: {r2:.3f}")
        print(f"   ➤ MSE: {mse:.3f}")
        print(f"   ➤ RMSE: {rmse:.3f}")


        # Save model
        joblib.dump(model, f"{target}_model.pkl")
        print(f"✅ Saved model: {target}_model.pkl")

    except Exception as e:
        print(f"❌ Failed for {target}: {e}")
