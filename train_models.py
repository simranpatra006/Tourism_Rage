# train_models.py – With auto-install for dependencies

import subprocess
import sys
import os

# ========== AUTO-INSTALL DEPENDENCIES ==========
def install_packages():
    """Install required packages if missing."""
    required = ['pandas', 'numpy', 'scikit-learn', 'openpyxl']
    for pkg in required:
        try:
            if pkg == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(pkg)
            print(f"✅ {pkg} already installed")
        except ImportError:
            print(f"📦 Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '--quiet'])

install_packages()

# ========== NOW IMPORT REQUIRED PACKAGES ==========
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

print("="*60)
print("🔄 STARTING MODEL RETRAINING...")
print("="*60)

# ========== CHECK FILE EXISTENCE ==========
print("\n📁 Checking required files...")

required_files = ['cleaned_tourism.csv', 'encoders.pkl', 'feature_names.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f"❌ Missing files: {missing_files}")
    print("   Make sure these are in the current directory.")
    sys.exit(1)
else:
    print("   ✅ All required files found.")

# ========== LOAD DATA AND ENCODERS ==========
print("\n📊 Loading data and encoders...")
df = pd.read_csv('cleaned_tourism.csv')
print(f"   ✅ Data: {df.shape[0]} rows, {df.shape[1]} columns")

encoders = pickle.load(open('encoders.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))
print(f"   ✅ encoders.pkl loaded (keys: {list(encoders.keys())})")
print(f"   ✅ feature_names.pkl loaded ({len(feature_names)} features)")

# ========== PREPARE DATA FOR TRAINING ==========
print("\n🔧 Preparing features...")

# Drop target columns and non-feature columns
X = df.drop(columns=['Rating', 'VisitMode', 'UserId', 'AttractionId', 'TransactionId'], errors='ignore')
y_reg = df['Rating']
y_clf = df['VisitMode']

# Ensure feature order matches what was saved
X = X[feature_names]
print(f"   ✅ X shape: {X.shape}")

# ========== SPLIT DATA ==========
print("\n📊 Splitting data...")
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
print(f"   ✅ Train size: {len(X_train)}, Test size: {len(X_test)}")

# ========== TRAIN MODELS ==========
print("\n🤖 Training models...")

# Regression
print("   Training RandomForestRegressor...")
reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
reg_model.fit(X_train, y_reg_train)
pickle.dump(reg_model, open('regressor.pkl', 'wb'))
print(f"   ✅ regressor.pkl saved")

# Classification
print("   Training RandomForestClassifier...")
clf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
clf_model.fit(X_train, y_clf_train)
pickle.dump(clf_model, open('classifier.pkl', 'wb'))
print(f"   ✅ classifier.pkl saved")

# ========== VERIFY ==========
print("\n✅ Verifying saved files...")
for f in ['regressor.pkl', 'classifier.pkl']:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"   ✅ {f} ({size:.2f} MB)")
    else:
        print(f"   ❌ {f} MISSING!")

print("\n" + "="*60)
print("🎉 MODEL RETRAINING COMPLETE!")
print("="*60)