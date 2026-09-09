"""
day2_models.py
================
HOW TO RUN:
1. Place this file inside your Tourism_Project folder (same folder as cleaned_tourism.csv).
2. Open a terminal in VS Code (Terminal -> New Terminal) and make sure you're in that folder.
3. Run:  python models.py

This script trains a Regression model (predicts Rating) and a Classification
model (predicts VisitMode), then saves both models to disk using pickle so
they can be reused later in the Streamlit app (Day 4).
"""

# ========== SECTION 1: IMPORTS ==========
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    classification_report,
)
import pickle


def main():
    # ========== SECTION 2: LOAD THE CLEANED DATA ==========
    # Load the dataset we created and saved in Day 1.
    df = pd.read_csv('cleaned_tourism.csv')

    print(f"✅ Data loaded. Shape: {df.shape}")
    print("Columns available:")
    print(df.columns.tolist())

    # ========== SECTION 3: PREPARE FEATURES (X) AND TARGETS (y) ==========
    # These columns are NOT useful as predictive features, so we drop them
    # from X. We use a small helper list so we don't crash if a column
    # happens to be missing (e.g., TransactionId might not always exist).
    columns_to_drop = [
        'Rating',        # target for regression
        'VisitMode',     # target for classification
        'UserId',        # unique identifier, not predictive
        'AttractionId',  # unique identifier, not predictive
        'TransactionId', # unique identifier, not predictive (if present)
        'Attraction',    # raw attraction name, not used for prediction
    ]
    # Only drop columns that actually exist in the DataFrame.
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    X = df.drop(columns=columns_to_drop)

    # Set the two prediction targets.
    y_reg = df['Rating']
    y_clf = df['VisitMode']

    # Save the feature column names — we'll need this exact list later
    # in the Streamlit app so we feed the model features in the same order.
    feature_names = X.columns.tolist()

    print(f"✅ Features ready. Total features: {len(feature_names)}")
    print("Feature columns:", feature_names)

    # ========== SECTION 4: SPLIT DATA INTO TRAIN AND TEST SETS ==========
    # We split X against each target separately, but because random_state=42
    # is fixed both times, the row indices used for train/test will match up.
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    X_train, X_test, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )

    print(f"✅ Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # ========== SECTION 5: REGRESSION MODEL (PREDICTING ATTRACTION RATING) ==========
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_reg_train)

    y_reg_pred = reg_model.predict(X_test)

    r2 = r2_score(y_reg_test, y_reg_pred)
    mse = mean_squared_error(y_reg_test, y_reg_pred)

    print("=" * 50)
    print("📈 REGRESSION MODEL RESULTS (Predicting Rating)")
    print("=" * 50)
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print("=" * 50)

    # ========== SECTION 6: CLASSIFICATION MODEL (PREDICTING VISIT MODE) ==========
    # class_weight='balanced' helps when some visit modes (e.g., "Business")
    # have far fewer samples than others (e.g., "Family").
    clf_model = RandomForestClassifier(
    n_estimators=150,
    class_weight='balanced_subsample',
    random_state=42
)
    clf_model.fit(X_train, y_clf_train)

    y_clf_pred = clf_model.predict(X_test)

    accuracy = accuracy_score(y_clf_test, y_clf_pred)

    print("=" * 50)
    print("📊 CLASSIFICATION MODEL RESULTS (Predicting Visit Mode)")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_clf_test, y_clf_pred))
    print("=" * 50)

    # ========== SECTION 7: SAVE ALL MODELS AND FILES USING PICKLE ==========
    # Saved to the same folder as this script (relative paths).
    pickle.dump(reg_model, open('regressor.pkl', 'wb'))
    pickle.dump(clf_model, open('classifier.pkl', 'wb'))
    pickle.dump(feature_names, open('feature_names.pkl', 'wb'))

    # Note: encoders.pkl was already saved in Day 1 and will be reused
    # in the Streamlit app (Day 4) to encode new user input consistently.

    print("✅ Models saved successfully!")
    print("📁 Files created:")
    print("   - regressor.pkl")
    print("   - classifier.pkl")
    print("   - feature_names.pkl")

    # ========== SECTION 8: QUICK FEATURE IMPORTANCE CHECK ==========
    # Helps you explain, in business terms, what drives predicted ratings.
    importances = reg_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n🔎 Top 5 most important features for predicting Rating:")
    print(importance_df.head(5).to_string(index=False))

    # ========== SECTION 9: FINAL SUMMARY ==========
    print("\n🎉 Day 2 complete! Models are ready for Day 3 (Recommendation System) "
          "and Day 4 (Streamlit App).")


if __name__ == "__main__":
    main()