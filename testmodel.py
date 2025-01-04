from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError


app = Flask(__name__)


model = load_model('lstm_demand_forecast_model.h5', custom_objects={'mse': MeanSquaredError()})


file_path = 'Inventory_DataSet.xlsx'
data = pd.read_excel(file_path, sheet_name='Product', engine='openpyxl')


features = ['Product Cateogy Name', 'Model Number', 'Supplier Name', 'Order Date']
targets = ['StockLevel', 'ReorderPoint', 'Quantity']


encoders = {}
for col in ['Product Cateogy Name', 'Model Number', 'Supplier Name']:
    encoders[col] = LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col].astype(str))

data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
mean_date = data['Order Date'].dropna().mean()
data.fillna({'Order Date': mean_date}, inplace=True)
data['Order Date'] = data['Order Date'].astype('int64') / 10**9


data = data.sort_values(by='Order Date')


scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])


@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        input_data = request.json
        year = input_data.get('year')
        month = input_data.get('month')
        day = input_data.get('day')
        product_category = input_data.get('product_category')
        model_number = input_data.get('model_number')
        supplier_name = input_data.get('supplier_name')

        
        input_date = datetime.datetime(year, month, day)
        input_timestamp = int(input_date.timestamp())

        
        try:
            product_category_encoded = encoders['Product Cateogy Name'].transform([product_category])[0]
        except ValueError:
            product_category_encoded = -1  
        try:
            model_number_encoded = encoders['Model Number'].transform([model_number])[0]
        except ValueError:
            model_number_encoded = -1  

        try:
            supplier_name_encoded = encoders['Supplier Name'].transform([supplier_name])[0]
        except ValueError:
            supplier_name_encoded = -1 

        
        input_features = np.array([[product_category_encoded, model_number_encoded, supplier_name_encoded, input_timestamp]])
        scaled_input = scaler.transform(input_features).reshape(1, 1, -1)

        
        predictions = model.predict(scaled_input)

       
        return jsonify({
            'StockLevel': float(predictions[0][0]),
            'ReorderPoint': float(predictions[0][1]),
            'Quantity': float(predictions[0][2])
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)