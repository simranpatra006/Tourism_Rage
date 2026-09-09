"""
Tourism Experience Analytics - Day 1: Data Prep, Cleaning, Feature Engineering, EDA

HOW TO RUN:
1. Open VS Code, open a terminal (Terminal > New Terminal) inside the Tourism_Project folder.
2. Make sure 'data' subfolder with the 9 CSVs sits next to this script.
3. Run:  python data_prep.py
   (You may first need: pip install pandas numpy matplotlib seaborn scikit-learn)
"""

# ========== SECTION 1: IMPORTS ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle

# Use a clean plot style
sns.set_style("whitegrid")


# ========== SECTION 2: LOAD ALL 9 CSV FILES ==========
def load_data():
    """Load all 9 Excel files from the data/ folder using relative paths.
    NOTE: your files are .xlsx (not .csv), so we use pd.read_excel here.
    This requires the 'openpyxl' package: pip install openpyxl"""
    transaction = pd.read_excel('data/Transaction.xlsx')
    # NOTE: in this file, the column called 'VisitMode' actually holds the numeric ID
    # (matching Mode.xlsx's VisitModeId column), not the text label. Rename it so the
    # merge with Mode.xlsx works, and so we don't end up with two different columns
    # both named 'VisitMode' (one numeric, one text) after merging.
    transaction = transaction.rename(columns={'VisitMode': 'VisitModeId'})
    print("✅ Transaction loaded, shape:", transaction.shape)

    user = pd.read_excel('data/User.xlsx')
    print("✅ User loaded, shape:", user.shape)

    city = pd.read_excel('data/City.xlsx')
    print("✅ City loaded, shape:", city.shape)

    country = pd.read_excel('data/Country.xlsx')
    print("✅ Country loaded, shape:", country.shape)

    region = pd.read_excel('data/Region.xlsx')
    print("✅ Region loaded, shape:", region.shape)

    continent = pd.read_excel('data/Continent.xlsx')
    print("✅ Continent loaded, shape:", continent.shape)

    item = pd.read_excel('data/Item.xlsx')
    print("✅ Item loaded, shape:", item.shape)

    attraction_type = pd.read_excel('data/Type.xlsx')
    print("✅ Type loaded, shape:", attraction_type.shape)

    # NOTE: this file is named 'Mode.xlsx' in your data folder, not 'VisitMode.xlsx'
    visit_mode = pd.read_excel('data/Mode.xlsx')
    print("✅ VisitMode loaded, shape:", visit_mode.shape)

    return transaction, user, city, country, region, continent, item, attraction_type, visit_mode


# ========== SECTION 3: MERGE ALL TABLES INTO ONE MASTER DATAFRAME ==========
def merge_data(transaction, user, city, country, region, continent, item, attraction_type, visit_mode):
    """Join Transaction -> User -> City -> Country -> Region -> Continent,
    then bring in Item -> Type, and VisitMode.

    IMPORTANT FIX: The User table already contains CityId, CountryId, RegionId,
    and ContinentId. If we merge in the full City/Country/Region tables, their
    duplicate ID columns get suffixed (e.g., CountryId_x / CountryId_y) and break
    the next merge step. To avoid this, we only pull the NEW descriptive columns
    (like CityName, Country, Region, Continent) from each lookup table, and keep
    User's original ID columns as the single source of truth for the ID chain.
    """

    # Transaction + User (on UserId) -- brings in CityId, CountryId, RegionId, ContinentId
    df = transaction.merge(user, on='UserId', how='left')

    # + City: only bring in CityName (CityId is the join key, drop City's own CountryId)
    city_slim = city[['CityId', 'CityName']]
    df = df.merge(city_slim, on='CityId', how='left')

    # + Country: only bring in Country name (drop Country's own RegionId)
    country_slim = country[['CountryId', 'Country']]
    df = df.merge(country_slim, on='CountryId', how='left')

    # + Region: only bring in Region name (drop Region's own ContinentId)
    region_slim = region[['RegionId', 'Region']]
    df = df.merge(region_slim, on='RegionId', how='left')

    # + Continent: only bring in Continent name
    continent_slim = continent[['ContinentId', 'Continent']]
    df = df.merge(continent_slim, on='ContinentId', how='left')

    # + Item (on AttractionId) -- brings in AttractionCityId, AttractionTypeId, Attraction, AttractionAddress
    df = df.merge(item, on='AttractionId', how='left')

    # + Type (on AttractionTypeId)
    df = df.merge(attraction_type, on='AttractionTypeId', how='left')

    # + VisitMode (on VisitModeId)
    df = df.merge(visit_mode, on='VisitModeId', how='left')

    print("✅ Merge complete. Final shape:", df.shape)
    print(df.head())

    return df


# ========== SECTION 4: DATA CLEANING ==========
def clean_data(df):
    """Drop unnecessary ID/address columns and handle missing values."""

    # Drop columns we no longer need after merging (we now have the human-readable names)
    columns_to_drop = [
        'CityId', 'CountryId', 'RegionId', 'ContinentId',
        'AttractionTypeId', 'VisitModeId', 'AttractionCityId', 'AttractionAddress'
    ]
    # Only drop columns that actually exist, in case of naming differences
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(columns=columns_to_drop)

    # Check for null values
    print("Null values per column BEFORE cleaning:")
    print(df.isnull().sum())

    # Fill numeric columns with median, categorical columns with "Unknown"
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                df[col] = df[col].fillna("Unknown")

    print("Null values per column AFTER cleaning:")
    print(df.isnull().sum())

    print(f"✅ Cleaned shape: {df.shape}")
    return df


# ========== SECTION 5: FEATURE ENGINEERING ==========
def get_season(month):
    """Map a numeric month (1-12) to a season string."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:  # 9, 10, 11
        return "Fall"


def feature_engineering(df):
    """Create User_Avg_Rating, Attraction_Avg_Rating, and Season columns."""

    # Average rating given by each user, merged back onto every row for that user
    user_avg = df.groupby('UserId')['Rating'].mean().reset_index()
    user_avg = user_avg.rename(columns={'Rating': 'User_Avg_Rating'})
    df = df.merge(user_avg, on='UserId', how='left')

    # Average rating received by each attraction, merged back onto every row for that attraction
    attraction_avg = df.groupby('AttractionId')['Rating'].mean().reset_index()
    attraction_avg = attraction_avg.rename(columns={'Rating': 'Attraction_Avg_Rating'})
    df = df.merge(attraction_avg, on='AttractionId', how='left')

    # Season column derived from VisitMonth
    df['Season'] = df['VisitMonth'].apply(get_season)

    print("✅ Feature engineering complete.")
    return df


# ========== SECTION 6: ENCODE CATEGORICAL VARIABLES ==========
def encode_categoricals(df):
    """Label-encode categorical columns and save the fitted encoders for later use
    (e.g., in the Streamlit app on Day 4)."""

    categorical_cols = ['Continent', 'Region', 'Country', 'CityName',
                         'AttractionType', 'VisitMode', 'Season']
    # Only encode columns that exist in the dataframe
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    encoders = {}  # dictionary to hold one LabelEncoder per column

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le  # keep the fitted encoder so we can decode/encode later

    # Save all fitted encoders to a single pickle file
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    print("✅ Encoding complete. Encoders saved.")
    return df


# ========== SECTION 7: SAVE THE CLEANED DATA ==========
def save_cleaned_data(df):
    """Save the final master DataFrame as cleaned_tourism.csv."""
    df.to_csv('cleaned_tourism.csv', index=False)
    print("✅ Cleaned data saved as 'cleaned_tourism.csv'")


# ========== SECTION 8: EXPLORATORY DATA ANALYSIS (EDA) ==========
def run_eda(df, original_visit_mode_labels=None, original_attraction_names=None):
    """Generate and save 3 EDA plots. Because VisitMode and Attraction are already
    label-encoded by this point, we plot using their encoded values but this section
    is intentionally called BEFORE encoding in main() to keep readable labels."""

    # --- Plot 1: Distribution of VisitMode ---
    plt.figure(figsize=(8, 5))
    visit_counts = df['VisitMode'].value_counts()
    sns.barplot(x=visit_counts.index, y=visit_counts.values, palette="viridis")
    plt.title("Distribution of Visit Modes")
    plt.xlabel("Visit Mode")
    plt.ylabel("Number of Visits")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("visits_mode.png")
    plt.show()
    plt.close()

    # --- Plot 2: Top 10 most popular attractions ---
    plt.figure(figsize=(10, 6))
    top_attractions = df['Attraction'].value_counts().head(10)
    sns.barplot(x=top_attractions.values, y=top_attractions.index, palette="magma")
    plt.title("Top 10 Most Popular Attractions")
    plt.xlabel("Number of Visits")
    plt.ylabel("Attraction")
    plt.tight_layout()
    plt.savefig("top_attractions.png")
    plt.show()
    plt.close()

    # --- Plot 3: Average rating by visit month ---
    plt.figure(figsize=(8, 5))
    monthly_avg = df.groupby('VisitMonth')['Rating'].mean().reindex(range(1, 13))
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', color='teal')
    plt.title("Average Rating by Visit Month")
    plt.xlabel("Month (1 = Jan, 12 = Dec)")
    plt.ylabel("Average Rating")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig("rating_by_month.png")
    plt.show()
    plt.close()

    print("✅ EDA plots saved: visits_mode.png, top_attractions.png, rating_by_month.png")


# ========== SECTION 9: SUMMARY STATISTICS ==========
def print_summary(df):
    print("\n===== df.describe() =====")
    print(df.describe())

    print("\n===== df.info() =====")
    print(df.info())

    print("\n===== df.head() =====")
    print(df.head())

    print("\n🎉 Day 1 complete! Ready for Day 2.")


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Step 1: Load
    transaction, user, city, country, region, continent, item, attraction_type, visit_mode = load_data()

    # Step 2: Merge
    df = merge_data(transaction, user, city, country, region, continent, item, attraction_type, visit_mode)

    # Step 3: Clean
    df = clean_data(df)

    # Step 4: Feature engineering (Season, avg ratings)
    df = feature_engineering(df)

    # Step 5: EDA — run this BEFORE encoding, so plots show readable text labels
    # (e.g., "Family" instead of "2") rather than numeric codes.
    run_eda(df)

    # Step 6: Encode categorical columns (after EDA, so plots stay human-readable)
    df = encode_categoricals(df)

    # Step 7: Save cleaned + encoded data
    save_cleaned_data(df)

    # Step 8: Print final summary stats
    print_summary(df)