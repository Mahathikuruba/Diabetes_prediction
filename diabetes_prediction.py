
# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load Dataset
# Replace 'diabetes.csv' with your dataset file name
df = pd.read_csv('diabetes.csv')

print(df.columns)
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Step 3: Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing numeric values with column mean
numeric_columns = df.select_dtypes(include=np.number).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Step 4: Separate Features and Target
# Replace 'Outcome' with your target column name if different
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Step 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Predict for New Patient
print("\nEnter Patient Details:")

user_data = []

for column in X.columns:
    value = float(input(f"Enter {column}: "))
    user_data.append(value)

user_data = np.array(user_data).reshape(1, -1)

prediction = model.predict(user_data)

if prediction[0] == 1:
    print("\nPrediction: Patient is likely to have Diabetes.")
else:
    print("\nPrediction: Patient is not likely to have Diabetes.")
 
