from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained XGBoost model
model_path = os.path.join(BASE_DIR, 'ev_sales_true_prediction_model.pkl')  # UPDATED
model = pickle.load(open(model_path, 'rb'))

# Load the feature names from training
feature_path = os.path.join(BASE_DIR, 'feature_columns_true_prediction.pkl')  # UPDATED
with open(feature_path, 'rb') as f:
    feature_columns = pickle.load(f)

# Load the dataset
csv_path = os.path.join(BASE_DIR, 'IEA_Global_EV_Data_2024_new__2_.csv')
df = pd.read_csv(csv_path)

# Apply same preprocessing as training - FIXED
df['value'] = df['value'].astype(str).str.replace('.', '', regex=False)
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df['value'] = df['value'].replace([np.inf, -np.inf], np.nan)
df['value'] = df['value'].fillna(0)
df['value'] = df['value'].astype(int)

# Filter for EV sales
df_sales = df[
    (df['parameter'] == 'EV sales') &
    (df['unit'] == 'Vehicles') &
    (df['mode'] == 'Cars') &
    (df['region'] != 'World')
].copy()

# Get unique values for dropdown menus
regions = sorted(df_sales['region'].unique())
categories = sorted(df_sales['category'].unique())
years = sorted(df_sales['year'].unique())
powertrains = sorted(df_sales['powertrain'].unique())

# Create validation mapping
validation_map = {}
for _, row in df_sales.iterrows():
    key = f"{row['region']}|{int(row['year'])}"
    if key not in validation_map:
        validation_map[key] = set()
    validation_map[key].add(row['category'])

# Convert sets to lists for JSON serialization
validation_map = {k: list(v) for k, v in validation_map.items()}

@app.route('/')
def home():
    return render_template('index.html', 
                         regions=regions,
                         categories=categories,
                         years=years,
                         powertrains=powertrains,
                         validation_map=json.dumps(validation_map))

@app.route('/get_valid_categories', methods=['POST'])
def get_valid_categories():
    """API endpoint to get valid categories for a region-year combination"""
    data = request.get_json()
    region = data.get('region')
    year = data.get('year')
    
    if not region or not year:
        return jsonify({'categories': categories})
    
    key = f"{region}|{year}"
    valid_categories = validation_map.get(key, [])
    
    return jsonify({
        'categories': valid_categories,
        'has_data': len(valid_categories) > 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        year = int(request.form['year'])
        region = request.form['region']
        category = request.form['category']
        powertrain = request.form['powertrain']
        
        # Check if this combination exists in the dataset
        key = f"{region}|{year}"
        has_ground_truth = key in validation_map and category in validation_map[key]
        
        # Create a single row DataFrame with the input
        input_data = pd.DataFrame({
            'year': [year],
            'region': [region],
            'category': [category],
            'powertrain': [powertrain]
        })
        
        # Apply one-hot encoding (same as training)
        input_encoded = pd.get_dummies(
            input_data,
            columns=['region', 'category', 'powertrain'],
            drop_first=False
        )
        
        # Create DataFrame with all features initialized to 0
        input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
        
        # Copy values from encoded input to the full feature set
        for col in input_encoded.columns:
            if col in input_df.columns:
                input_df[col] = input_encoded[col].values
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get actual value if it exists
        actual_value = None
        if has_ground_truth:
            actual_row = df_sales[
                (df_sales['region'] == region) &
                (df_sales['year'] == year) &
                (df_sales['category'] == category) &
                (df_sales['powertrain'] == powertrain)
            ]
            if len(actual_row) > 0:
                actual_value = int(actual_row.iloc[0]['value'])
        
        return render_template('index.html',
                             regions=regions,
                             categories=categories,
                             years=years,
                             powertrains=powertrains,
                             validation_map=json.dumps(validation_map),
                             prediction=int(prediction),
                             actual_value=actual_value,
                             has_ground_truth=has_ground_truth,
                             selected_year=year,
                             selected_region=region,
                             selected_category=category,
                             selected_powertrain=powertrain)
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('index.html',
                             regions=regions,
                             categories=categories,
                             years=years,
                             powertrains=powertrains,
                             validation_map=json.dumps(validation_map),
                             error=str(e))

if __name__ == '__main__':
    app.run(debug=True)