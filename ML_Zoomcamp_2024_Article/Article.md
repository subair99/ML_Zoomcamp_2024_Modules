
# Using Bagging And Xgboost To Train Large Datasets
I was recently confronted by a gigantic problem of having to train and score a dataset with 7,432,685 rows and 16 columns. The dataset is from the Kaggle [ML Zoomcamp 2024 Competition]( https://www.kaggle.com/competitions/ml-zoomcamp-2024-competition) which is aimed at rewording the model with the best root mean squared error. Before I continue I will define and explain some concepts.


## What is Bagging?
Bagging, which is also known as Bootstrap Aggregating, is an ensemble technique that combines multiple instances of a base model trained on different subsets of the training data, selected randomly with replacement. The power of ensemble methods in the ability to combine multiple models thereby improving overall prediction accuracy and model stability. Bagging stands out as a popular and widely implemented ensemble method.


## What is Ensemble Modeling?
Ensemble modeling is a process by which multiple diverse base models are used to predict an outcome, with the motivation of reducing the generalization error of the prediction. If the base models are diverse and independent, the prediction error decreases when the ensemble approach is used. The steps involved in creating an ensemble model are multiple machine learning models are trained independently, then their predictions are aggregated by voting, averaging, or weighting. This ensemble is then used to make the overall prediction.


## Advantages of Bagging
1. The chances of overfitting can be reduced which results in improved model accuracy on testing data.
2. It makes it possible to conveniently train a very large dataset with one’s desired algorithm.
3. It averages out the predictions of multiple models trained on different subsets of data leading to lower variance than a single model.
4.  There is less impact on bagged models when there is changes in the training dataset leading to a more stable overall model
5. This will be a great help for algorithms that tend to have high variance, such as decision trees.
6. It allows for parallel processing and efficient use of computational resources as each model in the ensemble can be trained independently.
7. It removes the need for complex modifications to the learning algorithm since the concept behind bagging is straightforward.
8. Noise is reduced in the final prediction because of the averaging process.
9. The performance in a scenario where the dataset is imbalanced can be increased by bagging.


## Practical Demonstration
An example of bagging will be demonstrated using the dataset of the competition mentioned at the beginning of this article. We start by importing the required modules.

```python
# Filter warnings
import warnings 
warnings.filterwarnings('ignore')

# Import required modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
%matplotlib inline
```

Then we load the data

```python
# Define the path
path = '/kaggle/input/ml-zoomcamp-2024-competition/'

# Load the data
sales = pd.read_csv(f"{path}/sales.csv")
sales.drop(columns=['Unnamed: 0'], inplace=True)
stores = pd.read_csv(f"{path}/stores.csv")
stores.drop(columns=['Unnamed: 0'], inplace=True)
catalog = pd.read_csv(f"{path}/catalog.csv")
catalog.drop(columns=['Unnamed: 0'], inplace=True)

# Merge sales with store and catalog info for feature enrichment
sales = sales.merge(stores, on="store_id", how="left")
sales = sales.merge(catalog, on="item_id", how="left")

# Add time-based features
sales["date"] = pd.to_datetime(sales["date"])
sales["year"] = sales["date"].dt.year
sales["month"] = sales["date"].dt.month
sales["day"] = sales["date"].dt.day
sales["day_of_week"] = sales["date"].dt.dayofweek

# Select features and target
features = [
    'division', 'format', 'city', 'area', 'dept_name', 'class_name', 'subclass_name', 'item_type', 
    'weight_volume', 'weight_netto', 'fatness', 'year', 'month', 'day', 'day_of_week', 'quantity'
]

sales = sales[features]
```

Show the original size of the data

```python
# Show the shape of sales
sales.shape

(7432685, 16)
```

```python
# Define function to replace null values
def replace_null(df, cat_features):
    
    # Convert categorical columns to string type and handle NaN
    for col in cat_features:
        df[col] = df[col].astype(str)
        imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        df[col] = imputer.fit_transform(df[[col]]).ravel()
    
    # Handle numerical features
    numerical_features = [col for col in df.columns if col not in cat_features + ['quantity']]
    num_imputer = SimpleImputer(strategy='mean')
    df[numerical_features] = num_imputer.fit_transform(df[numerical_features])
    
    return df
```

```python
# Define function to encode and scale
def encode_scale(df, cat_features):
    
    # Convert categorical columns to string type and handle NaN
    for col in cat_features:
        df[col] = df[col].astype(str)
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[[col]]).ravel()
    
    # Handle numerical features
    numerical_features = [col for col in df.columns if col not in cat_features + ['quantity']]
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df
```

```python
# Define categorical columns
cat_features = ['division', 'format', 'city', 'dept_name', 'class_name', 'subclass_name', 'item_type']
```

The next step is to remove all the null values in sales and the other reason is to reduce the dataset to a size that so that bare XGBRegresor can train the model so that it can be compared with the bagging model.

```python
# Remove null values from sales
sales = sales.dropna()
sales.shape

(594149, 16)
```

From above, it can be observed that the shape of sales reduced from (7432685, 16) to (594149, 16), next the data will the encoded and scaled.

```python
# Encode and scale sales
sales = encode_scale(sales, cat_features)
```

Define X and y then split with test equal to 20% of the data.

```python
# Split sales
X = sales.drop(columns=['quantity'])
y = sales['quantity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
```

```python
# Define the base XGBoost model
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=22)
```

```python
# Create the Bagging ensemble with XGBoost as the base estimator
bagging_xgb = BaggingRegressor(estimator=xgb, n_estimators=10, random_state=22)
```

Now train the bare XGBRegressor and the bagging model.

```python
# Train and evaluate the single XGBoost model
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
```

```python
# Train and evaluate the Bagging ensemble
bagging_xgb.fit(X_train, y_train)
y_pred_bagging = bagging_xgb.predict(X_test)
rmse_bagging = np.sqrt(mean_squared_error(y_test, y_pred_bagging))
```

Print the results

```python
# Print the performance metrics
print(f"Single XGBoost - RMSE: {rmse_xgb:.4f}")
print(f"Bagging XGBoost - RMSE: {rmse_bagging:.4f}")
```

```python
Single XGBoost - RMSE: 9.5957
Bagging XGBoost - RMSE: 9.6066
```

As expected the bagging model performs better than the bare XGBRegressor. The next step is to show that bagging has the ability train a data of a very large size. We start from the top.

```python
# Now train the sales data with bagging and inputting null values
# Define the path
path = '/kaggle/input/ml-zoomcamp-2024-competition/'

# Load the data
sales = pd.read_csv(f"{path}/sales.csv")
sales.drop(columns=['Unnamed: 0'], inplace=True)
stores = pd.read_csv(f"{path}/stores.csv")
stores.drop(columns=['Unnamed: 0'], inplace=True)
catalog = pd.read_csv(f"{path}/catalog.csv")
catalog.drop(columns=['Unnamed: 0'], inplace=True)

# Merge sales with store and catalog info for feature enrichment
sales = sales.merge(stores, on="store_id", how="left")
sales = sales.merge(catalog, on="item_id", how="left")

# Add time-based features
sales["date"] = pd.to_datetime(sales["date"])
sales["year"] = sales["date"].dt.year
sales["month"] = sales["date"].dt.month
sales["day"] = sales["date"].dt.day
sales["day_of_week"] = sales["date"].dt.dayofweek

# Select features and target
features = [
    'division', 'format', 'city', 'area', 'dept_name', 'class_name', 'subclass_name', 'item_type', 
    'weight_volume', 'weight_netto', 'fatness', 'year', 'month', 'day', 'day_of_week', 'quantity'
]

sales = sales[features]
```

```python
# Replace null values in sales
sales = replace_null(sales, cat_features)
```

```python
# Encode and scale sales
sales = encode_scale(sales, cat_features)
```

```python
# Split sales
X = sales.drop(columns=['quantity'])
y = sales['quantity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
```

```python
# Define the base XGBoost model
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=22)
```

```python
# Create the Bagging ensemble with XGBoost as the base estimator
bagging_xgb = BaggingRegressor(estimator=xgb, n_estimators=10, random_state=22)
```

```python
# Train and evaluate the Bagging ensemble
bagging_xgb.fit(X_train, y_train)
y_pred_bagging = bagging_xgb.predict(X_test)
rmse_bagging = np.sqrt(mean_squared_error(y_test, y_pred_bagging))
```

```python
# Print the performance metrics
print(f"Bagging XGBoost - RMSE: {rmse_bagging:.4f}")

Bagging XGBoost - RMSE: 20.7980
```

# Conclusion
In this article, I have been able to demonstarate the fact that bagging improves the result of a bare XGBRegressor model and also that it can train a very large dataset to produce a credible result