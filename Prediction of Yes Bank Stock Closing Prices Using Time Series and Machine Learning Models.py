#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install numpy pandas matplotlib seaborn  scikit-learn statsmodels xgboost pmdarima


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor


# In[16]:


df = pd.read_csv('data_YesBank_StockPrices - data_YesBank_StockPrices.csv')
print(df.head())


# In[18]:


print(df.info())
print(df.isnull().sum())


# In[30]:


plt.figure(figsize=(10,6))
plt.plot(df['Date'],df['Close'],label='Close')
plt.title('yes Bank Closing price trend')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend()
plt.show()


# In[27]:


train_size=int(len(df)*0.8)
train_arima, test_arima = df['Close'][:train_size], df['Close'][train_size:]


# In[28]:


arima_model = ARIMA(train_arima, order=(5, 1, 0))  # Adjust (p, d, q) as needed
arima_result = arima_model.fit()


# In[29]:


arima_forecast = arima_result.forecast(steps=len(test_arima))


# In[31]:


arima_mae = mean_absolute_error(test_arima, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test_arima, arima_forecast))

print("ARIMA Model:")
print(f"MAE: {arima_mae}")
print(f"RMSE: {arima_rmse}")


# In[36]:


df['Price Range'] = df['High'] - df['Low']
features = ['Open', 'High', 'Low', 'Price Range']
target = 'Close'

X = df[features]
y = df[target]


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)


# In[38]:


xgb_predictions = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

print("\nXGBoost Model:")
print(f"MAE: {xgb_mae}")
print(f"RMSE: {xgb_rmse}")


# In[39]:


plt.figure(figsize=(10, 6))
plt.plot(test_arima.index, test_arima, label='Actual Closing Price', color='blue')
plt.plot(test_arima.index, arima_forecast, label='ARIMA Predicted Price', color='orange')
plt.title('ARIMA: Actual vs Predicted Closing Prices')
plt.legend()
plt.show()


# In[40]:


plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Closing Price', color='blue')
plt.plot(y_test.index, xgb_predictions, label='XGBoost Predicted Price', color='red')
plt.title('XGBoost: Actual vs Predicted Closing Prices')
plt.legend()
plt.show()


# In[41]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# For ARIMA Model
arima_mae = mean_absolute_error(test_arima, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test_arima, arima_forecast))
arima_r2 = r2_score(test_arima, arima_forecast)

print("ARIMA Model Accuracy:")
print(f"MAE: {arima_mae}")
print(f"RMSE: {arima_rmse}")
print(f"R²: {arima_r2}")

# For XGBoost Model
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_r2 = r2_score(y_test, xgb_predictions)

print("\nXGBoost Model Accuracy:")
print(f"MAE: {xgb_mae}")
print(f"RMSE: {xgb_rmse}")
print(f"R²: {xgb_r2}")


# In[42]:


# Calculate accuracy for ARIMA
arima_accuracy = (1 - (arima_mae / test_arima.mean())) * 100

# Calculate accuracy for XGBoost
xgb_accuracy = (1 - (xgb_mae / y_test.mean())) * 100

# Display accuracy in percentage
print(f"ARIMA Model Accuracy: {arima_accuracy:.2f}%")
print(f"XGBoost Model Accuracy: {xgb_accuracy:.2f}%")


# In[ ]:




