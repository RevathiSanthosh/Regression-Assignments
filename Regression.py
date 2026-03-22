# 📌 Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# 📌 Step 2: Load dataset
url = "https://raw.githubusercontent.com/RamishaRaniK/dataset/main/insurance_pre.csv"
data = pd.read_csv(url)

print("Dataset Shape:", data.shape)
print("Columns:", data.columns)
print(data.head())

# 📌 Step 3: Preprocessing
# Encode categorical variables
data_encoded = data.copy()
label_enc = LabelEncoder()

data_encoded['sex'] = label_enc.fit_transform(data_encoded['sex'])
data_encoded['smoker'] = label_enc.fit_transform(data_encoded['smoker'])
data_encoded = pd.get_dummies(data_encoded, columns=['region'], drop_first=True)

# Features and target
X = data_encoded.drop('charges', axis=1)
y = data_encoded['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 4: Try multiple models
results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
results['Linear Regression'] = r2_score(y_test, lr.predict(X_test))

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
results['Decision Tree'] = r2_score(y_test, dt.predict(X_test))

# Random Forest
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
results['Random Forest'] = r2_score(y_test, rf.predict(X_test))

# Gradient Boosting
gb = GradientBoostingRegressor(random_state=42, n_estimators=200)
gb.fit(X_train, y_train)
results['Gradient Boosting'] = r2_score(y_test, gb.predict(X_test))

# 📌 Step 5: Document results
print("\nModel Performance (R² Scores):")
for model, score in results.items():
    print(f"{model}: {score:.4f}")

# 📌 Step 6: Final Model Selection
best_model = max(results, key=results.get)
print(f"\n✅ Final Chosen Model: {best_model} with R² = {results[best_model]:.4f}")
