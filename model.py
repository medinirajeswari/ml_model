import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Data Loading
file_path = 'Inventory_DataSet.xlsx'
data = pd.read_excel(file_path, sheet_name='Product', engine='openpyxl')

# Data preprocessing
features = ['Product Cateogy Name', 'Model Number', 'Supplier Name', 'Order Date']
targets = ['StockLevel', 'ReorderPoint', 'Quantity']

# Handle the categorical variables
encoder = LabelEncoder()
for col in ['Product Cateogy Name', 'Model Number', 'Supplier Name']:
    data[col] = encoder.fit_transform(data[col].astype(str))

# Changing the date format into numerical and removed invalid dates
data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')

# Replaced the invalid dates from data with the mean or a default values
mean_date = data['Order Date'].dropna().mean()
data['Order Date'].fillna(mean_date, inplace=True)

# Dates are converted into numerical Timestamp
data['Order Date'] = data['Order Date'].astype('int64') // 10**9

# Time-series data preparation
data = data.sort_values(by='Order Date')

def create_sequences(data, target_columns, sequence_length):
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:i + sequence_length][features].values
        sequences.append(seq)
        target_row = data.iloc[i + sequence_length][target_columns].values
        targets.append(target_row)

    return np.array(sequences), np.array(targets)

# Sequence length
sequence_length = 10  

X, y = create_sequences(data, targets, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = X_train.reshape(-1, X_train.shape[-1])
X_test = X_test.reshape(-1, X_test.shape[-1])
X_train = scaler.fit_transform(X_train).reshape(-1, sequence_length, len(features))
X_test = scaler.transform(X_test).reshape(-1, sequence_length, len(features))

# Model Building
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(sequence_length, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(targets), activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Model training
history = model.fit(
    X_train,
    y_train,  
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)

results = model.evaluate(
    X_test,
    y_test  
)

print("Test Results:", results)

model.save('lstm_demand_forecast_model.h5')