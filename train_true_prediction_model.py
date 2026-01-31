"""
🎯 TRUE PREDICTION MODEL
Train ONLY on historical data, predict the future
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎯 TRAINING TRUE PREDICTION MODEL")
print("="*70)

# Load data
df = pd.read_csv('IEA_Global_EV_Data_2024_new__2_.csv')

# Convert value to numeric (handle European format) - FIXED VERSION
df['value'] = df['value'].astype(str).str.replace('.', '', regex=False)
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# Handle NaN and infinite values more thoroughly
df['value'] = df['value'].replace([np.inf, -np.inf], np.nan)  # Convert infinities to NaN
df['value'] = df['value'].fillna(0)  # Fill NaN with 0

# Now safe to convert to int
df['value'] = df['value'].astype(int)

print("\n📊 Dataset loaded and cleaned")

# Filter for EV sales
df_sales = df[
    (df['parameter'] == 'EV sales') &
    (df['unit'] == 'Vehicles') &
    (df['mode'] == 'Cars') &
    (df['region'] != 'World')
].copy()

print(f"Total EV sales records: {len(df_sales)}")

# CRITICAL: Split by time for true prediction
print("\n" + "="*70)
print("📅 TIME-BASED SPLITTING (KEY TO TRUE PREDICTION)")
print("="*70)

# TRAIN: Historical data ONLY (2015-2020)
df_train = df_sales[
    (df_sales['year'] >= 2015) & 
    (df_sales['year'] <= 2020) &
    (df_sales['category'] == 'Historical')
].copy()

# VALIDATION: Recent historical (2021-2022)
df_val = df_sales[
    (df_sales['year'].isin([2021, 2022])) &
    (df_sales['category'] == 'Historical')
].copy()

# TEST: Latest historical + future projections (2023-2035)
df_test = df_sales[
    (df_sales['year'] >= 2023)
].copy()

print(f"\n✓ Train set (2015-2020, Historical only): {len(df_train)} samples")
print(f"✓ Validation set (2021-2022): {len(df_val)} samples")
print(f"✓ Test set (2023+): {len(df_test)} samples")

print(f"\nTrain years: {sorted(df_train['year'].unique())}")
print(f"Val years: {sorted(df_val['year'].unique())}")
print(f"Test years: {sorted(df_test['year'].unique())}")

# One-hot encode
print("\n" + "="*70)
print("🔧 FEATURE ENGINEERING")
print("="*70)

def prepare_features(df):
    """Prepare features with one-hot encoding"""
    df_encoded = pd.get_dummies(
        df,
        columns=['region', 'category', 'powertrain'],
        drop_first=False
    )
    # Drop unnecessary columns
    df_encoded = df_encoded.drop(
        columns=['parameter', 'mode', 'unit', 'percentage'], 
        errors='ignore'
    )
    return df_encoded

df_train_enc = prepare_features(df_train)
df_val_enc = prepare_features(df_val)
df_test_enc = prepare_features(df_test)

# Align features (test might have new categories)
all_features = set(df_train_enc.columns) | set(df_val_enc.columns) | set(df_test_enc.columns)
all_features.discard('value')
feature_columns = sorted(list(all_features))

print(f"✓ Total features: {len(feature_columns)}")

# Ensure all datasets have same columns
for df_enc in [df_train_enc, df_val_enc, df_test_enc]:
    for col in feature_columns:
        if col not in df_enc.columns:
            df_enc[col] = 0

# Prepare X and y
X_train = df_train_enc[feature_columns]
y_train = df_train_enc['value']

X_val = df_val_enc[feature_columns]
y_val = df_val_enc['value']

X_test = df_test_enc[feature_columns]
y_test = df_test_enc['value']

print(f"\n✓ Training features shape: {X_train.shape}")
print(f"✓ Validation features shape: {X_val.shape}")
print(f"✓ Test features shape: {X_test.shape}")

# Save feature columns
with open('feature_columns_true_prediction.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("\n" + "="*70)
print("🤖 TRAINING MODEL")
print("="*70)

# Note: XGBoost not available in this environment
# This is the template - run this on your machine with XGBoost installed

MODEL_CODE = '''
import xgboost as xgb

# Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("Training model (this may take a few minutes)...")
model.fit(X_train, y_train)
print("✓ Model trained!")

# Save model
with open('ev_sales_true_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# VALIDATION (on 2021-2022 - data it hasn't seen)
y_val_pred = model.predict(X_val)

print("\\n" + "="*70)
print("📊 VALIDATION RESULTS (2021-2022)")
print("="*70)

val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)
val_mape = mean_absolute_percentage_error(y_val, y_val_pred)

print(f"\\nMAE: {val_mae:,.0f} vehicles")
print(f"RMSE: {val_rmse:,.0f} vehicles")
print(f"R² Score: {val_r2:.4f}")
print(f"MAPE: {val_mape*100:.2f}%")

# TEST (on 2023+ - true future prediction)
y_test_pred = model.predict(X_test)

print("\\n" + "="*70)
print("🔮 TEST RESULTS (2023-2035) - TRUE PREDICTIONS")
print("="*70)

# Separate historical test (2023) from projections (2025+)
test_2023 = df_test[df_test['year'] == 2023]
test_future = df_test[df_test['year'] >= 2025]

if len(test_2023) > 0:
    mask_2023 = df_test['year'].values == 2023
    y_test_2023 = y_test.values[mask_2023]
    y_pred_2023 = y_test_pred[mask_2023]
    
    mae_2023 = mean_absolute_error(y_test_2023, y_pred_2023)
    print(f"\\n2023 (Latest Historical) MAE: {mae_2023:,.0f} vehicles")
    print(f"  → Can be validated against actual data")

if len(test_future) > 0:
    mask_future = df_test['year'].values >= 2025
    y_test_future = y_test.values[mask_future]
    y_pred_future = y_test_pred[mask_future]
    
    mae_future = mean_absolute_error(y_test_future, y_pred_future)
    print(f"\\n2025-2035 (Projections) MAE: {mae_future:,.0f} vehicles")
    print(f"  → Comparing ML prediction vs IEA projections")
    print(f"  → Lower MAE = closer to IEA, Higher = different forecast")

# Compare some specific predictions
print("\\n" + "="*70)
print("📈 SAMPLE PREDICTIONS vs IEA PROJECTIONS")
print("="*70)

comparison_samples = df_test[
    (df_test['year'].isin([2025, 2030])) &
    (df_test['region'].isin(['China', 'USA', 'Europe']))
].head(10)

if len(comparison_samples) > 0:
    for idx, row in comparison_samples.iterrows():
        # Get prediction
        test_idx = df_test.index.get_loc(idx)
        pred = y_test_pred[test_idx]
        actual = row['value']
        diff_pct = ((pred - actual) / actual * 100) if actual > 0 else 0
        
        print(f"\\n{row['region']}, {row['year']}, {row['powertrain']}")
        print(f"  IEA Projection: {actual:>15,} vehicles")
        print(f"  ML Prediction:  {pred:>15,.0f} vehicles")
        print(f"  Difference:     {diff_pct:>14.1f}%")

print("\\n" + "="*70)
print("✅ TRUE PREDICTION MODEL COMPLETE")
print("="*70)
print("""
Key Differences from Original Model:

1. Trained ONLY on historical data (2015-2020)
   → Model learns actual growth patterns, not memorizes projections

2. Validated on recent years (2021-2022)
   → Tests if model learned real trends

3. Tested on future years (2023-2035)
   → TRUE predictions, can be compared with IEA

Now your model:
✓ Learns patterns from PAST data
✓ Predicts FUTURE based on learned trends
✓ Can be validated against IEA projections
✓ Provides independent forecast value
""")
'''

print("\n📝 Model training code generated")
print("\nTo train the model, run this code on your machine with XGBoost:")
print("-" * 70)
print(MODEL_CODE)

# Save the preprocessing
print("\n" + "="*70)
print("💾 SAVING PREPROCESSED DATA")
print("="*70)

# Save datasets
np.save('X_train_true_pred.npy', X_train.values)
np.save('y_train_true_pred.npy', y_train.values)
np.save('X_val_true_pred.npy', X_val.values)
np.save('y_val_true_pred.npy', y_val.values)
np.save('X_test_true_pred.npy', X_test.values)
np.save('y_test_true_pred.npy', y_test.values)

# Save test metadata for analysis
df_test[['region', 'year', 'category', 'powertrain', 'value']].to_csv(
    'test_metadata_true_pred.csv', 
    index=False
)

print("✓ Saved training data: X_train_true_pred.npy, y_train_true_pred.npy")
print("✓ Saved validation data: X_val_true_pred.npy, y_val_true_pred.npy")
print("✓ Saved test data: X_test_true_pred.npy, y_test_true_pred.npy")
print("✓ Saved test metadata: test_metadata_true_pred.csv")
print("✓ Saved features: feature_columns_true_prediction.pkl")

print("\n" + "="*70)
print("📋 NEXT STEPS")
print("="*70)
print("""
1. Copy all .npy and .pkl files to your local machine
2. Install XGBoost: pip install xgboost
3. Run the model training code above
4. Compare your predictions with IEA projections
5. Analyze where your model agrees/disagrees with IEA

This is TRUE prediction - your model will forecast based on
learning from historical patterns, not memorizing projections!
""")