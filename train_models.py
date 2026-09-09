# train_models.py – Retrain models on the server
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

print("🔄 Retraining models on deployment...")

# Load cleaned data
df = pd.read_csv('cleaned_tourism.csv')

# Prepare features
X = df.drop(columns=['Rating', 'VisitMode', 'UserId', 'AttractionId'], errors='ignore')
y_reg = df['Rating']
y_clf = df['VisitMode']

# Split data
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Train Regression Model
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_reg_train)

# Train Classification Model
clf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf_model.fit(X_train, y_clf_train)

# Save models
pickle.dump(reg_model, open('regressor.pkl', 'wb'))
pickle.dump(clf_model, open('classifier.pkl', 'wb'))

print("✅ Models retrained and saved successfully!")