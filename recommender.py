# ========== DAY 3: RECOMMENDATION SYSTEM (FIXED) ==========
# File: recommender.py
# Purpose: Build a Content-Based Recommendation System
# Run: Open terminal in project folder, type: python recommender.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("🚀 DAY 3: BUILDING RECOMMENDATION SYSTEM")
    print("="*60)
    
    # ========== SECTION 1: LOAD CLEANED DATA ==========
    print("\n📂 Loading cleaned data...")
    df = pd.read_csv('cleaned_tourism.csv')
    print(f"✅ Data loaded. Shape: {df.shape}")
    
    # ========== SECTION 2: EXTRACT UNIQUE ATTRACTIONS ==========
    print("\n🔍 Extracting unique attractions...")
    # IMPORTANT FIX: Reset index so it matches the similarity matrix position
    attractions_df = df[['AttractionId', 'Attraction', 'AttractionType', 'CityName']].drop_duplicates(subset=['AttractionId']).reset_index(drop=True)
    print(f"✅ Unique attractions found: {len(attractions_df)}")
    print(f"   Sample attractions: {attractions_df['Attraction'].head(3).tolist()}")
    
    # ========== SECTION 3: CREATE FEATURE VECTORS ==========
    print("\n🧮 Creating feature vectors using OneHotEncoder...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    feature_vectors = encoder.fit_transform(attractions_df[['AttractionType', 'CityName']])
    print(f"✅ Feature vectors created. Shape: {feature_vectors.shape}")
    
    # ========== SECTION 4: COMPUTE COSINE SIMILARITY ==========
    print("\n📊 Computing cosine similarity matrix...")
    similarity_matrix = cosine_similarity(feature_vectors)
    print(f"✅ Similarity matrix computed. Shape: {similarity_matrix.shape}")
    
    # ========== SECTION 5: RECOMMENDATION FUNCTION ==========
    def recommend_attractions(attraction_name, top_n=5):
        """
        Recommends top N similar attractions based on content-based filtering.
        If attraction_name is not found, returns top N most popular attractions.
        """
        # Case-insensitive check
        attraction_lower = attraction_name.lower()
        attraction_list_lower = [att.lower() for att in attractions_df['Attraction'].tolist()]
        
        if attraction_lower not in attraction_list_lower:
            print(f"⚠️ '{attraction_name}' not found. Showing popular attractions instead.")
            # Fallback: Top 5 most visited attractions
            popular = df['Attraction'].value_counts().head(top_n).index.tolist()
            return popular
        
        # Find the correct index (positional, not DataFrame index)
        # Get the attraction name exactly as it appears in the DataFrame
        exact_name = attractions_df[attractions_df['Attraction'].str.lower() == attraction_lower]['Attraction'].iloc[0]
        idx = attractions_df[attractions_df['Attraction'] == exact_name].index[0]  # This is now positional (0,1,2,...)
        
        # Get similarity scores for this attraction
        sim_scores = list(enumerate(similarity_matrix[idx]))
        
        # Sort by similarity score (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N (skip the first one because it's the attraction itself)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get the attraction names
        recommended_indices = [i[0] for i in sim_scores]
        recommended_attractions = attractions_df.iloc[recommended_indices]['Attraction'].tolist()
        
        return recommended_attractions
    
    # ========== SECTION 6: TEST THE RECOMMENDATION SYSTEM ==========
    print("\n🧪 Testing the recommendation system...")
    print("-"*60)
    
    # Get some sample attractions for testing (using the first few)
    test_attractions = attractions_df['Attraction'].head(5).tolist()
    
    for att in test_attractions:
        print(f"\n🎯 If you liked '{att}':")
        try:
            recs = recommend_attractions(att, top_n=5)
            for i, rec in enumerate(recs, 1):
                print(f"   {i}. {rec}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "-"*60)
    
    # ========== SECTION 7: SAVE ALL FILES ==========
    print("\n💾 Saving recommendation system files...")
    
    # Save similarity matrix
    pickle.dump(similarity_matrix, open('similarity_matrix.pkl', 'wb'))
    
    # Save attraction list (for dropdown in Streamlit)
    attraction_list = attractions_df['Attraction'].tolist()
    pickle.dump(attraction_list, open('attraction_list.pkl', 'wb'))
    
    # Save attractions DataFrame (for mapping IDs to names)
    pickle.dump(attractions_df, open('attractions_df.pkl', 'wb'))
    
    # Save the encoder (optional, but useful)
    pickle.dump(encoder, open('recommendation_encoder.pkl', 'wb'))
    
    print("✅ Recommendation system files saved successfully!")
    print("📁 Files created:")
    print("   - similarity_matrix.pkl")
    print("   - attraction_list.pkl")
    print("   - attractions_df.pkl")
    print("   - recommendation_encoder.pkl")
    
    # ========== SECTION 8: SUMMARY ==========
    print("\n" + "="*60)
    print("🎉 DAY 3 COMPLETE! Recommendation system is ready.")
    print("📊 Summary:")
    print(f"   - Total unique attractions: {len(attractions_df)}")
    print(f"   - Similarity matrix size: {similarity_matrix.shape}")
    print(f"   - Ready for Day 4 (Streamlit App)")
    print("="*60)

if __name__ == "__main__":
    main()