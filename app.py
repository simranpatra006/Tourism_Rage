# ========== NEO-BRUTALISM STREAMLIT APP – FINAL ==========
# File: app.py
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ========== CHECK FOR MODELS, RETRAIN IF MISSING ==========
import os
import subprocess

def ensure_models_exist():
    """Check and retrain models if missing."""
    required = ['regressor.pkl', 'classifier.pkl']
    missing = [f for f in required if not os.path.exists(f)]
    
    if missing:
        st.warning(f"⚠️ Missing model files: {missing}. Retraining now...")
        
        # Create a placeholder for progress
        status_placeholder = st.empty()
        status_placeholder.info("🔄 Training models... This may take 1-2 minutes.")
        
        try:
            # Run the training script
            result = subprocess.run(
                ['python', 'train_models.py'],
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            # Print stdout/stderr for debugging (visible in logs)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            status_placeholder.success("✅ Models retrained successfully! Reloading...")
            st.rerun()
        except subprocess.TimeoutExpired:
            status_placeholder.error("❌ Training timed out after 5 minutes.")
            st.stop()
        except Exception as e:
            status_placeholder.error(f"❌ Failed to train models: {e}")
            # Show more detailed error
            if hasattr(e, 'stderr') and e.stderr:
                st.error(e.stderr)
            st.stop()

# Run the check BEFORE any other code
ensure_models_exist()

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="🌍 TOURISM RAGE",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== NEO-BRUTALISM CSS ==========
st.markdown("""
<style>
    /* ===== RESET & BASE ===== */
    .stApp {
        background: #f0f0f0;
    }
    * {
        font-family: 'Courier New', monospace !important;
    }

    /* ===== SIDEBAR ===== */
    .css-1d391kg, .css-1kyxreq {
        background: #ffcc00 !important;
        border-right: 6px solid black !important;
        border-bottom: 6px solid black !important;
        box-shadow: 12px 12px 0 rgba(0,0,0,0.3) !important;
        margin: 0 !important;
        border-radius: 0 !important;
    }

    /* ===== CARDS ===== */
    .brutal-card {
        background: #ffffff;
        border: 5px solid black;
        border-radius: 0 !important;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 8px 8px 0 rgba(0,0,0,0.2);
        transition: all 0.1s ease;
        transform: rotate(0deg);
    }
    .brutal-card:hover {
        transform: translate(-4px, -4px);
        box-shadow: 12px 12px 0 rgba(0,0,0,0.3);
    }
    .brutal-card-yellow { background: #ffdd00; border-color: black; }
    .brutal-card-pink { background: #ff6b9d; border-color: black; }
    .brutal-card-blue { background: #4fc3f7; border-color: black; }
    .brutal-card-green { background: #81c784; border-color: black; }
    .brutal-card-orange { background: #ffb74d; border-color: black; }
    .brutal-card-purple { background: #ce93d8; border-color: black; }

    /* ===== TITLE ===== */
    .main-title {
        font-size: 4.5rem;
        font-weight: 900;
        color: #ff0033;
        text-align: center;
        text-shadow: 6px 6px 0 #00ff00, 12px 12px 0 #0000ff;
        background: #ffff00;
        padding: 0.5rem 1rem;
        border: 8px solid black;
        box-shadow: 16px 16px 0 rgba(0,0,0,0.3);
        display: inline-block;
        margin: 0 auto;
        letter-spacing: -2px;
        transform: rotate(-2deg);
    }
    .sub-title {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        background: #ff6600;
        padding: 0.8rem 2rem;
        border: 5px solid black;
        box-shadow: 10px 10px 0 rgba(0,0,0,0.2);
        display: inline-block;
        margin: 0 auto;
        transform: rotate(1deg);
    }
    .title-container {
        text-align: center;
        margin-bottom: 2rem;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: #00ffcc !important;
        color: black !important;
        font-weight: 900 !important;
        font-size: 1.2rem !important;
        padding: 0.8rem 2.5rem !important;
        border: 5px solid black !important;
        border-radius: 0 !important;
        box-shadow: 8px 8px 0 rgba(0,0,0,0.3) !important;
        transition: all 0.05s linear !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    .stButton > button:hover {
        transform: translate(-4px, -4px) !important;
        box-shadow: 12px 12px 0 rgba(0,0,0,0.4) !important;
    }
    .stButton > button:active {
        transform: translate(4px, 4px) !important;
        box-shadow: 0px 0px 0 rgba(0,0,0,0) !important;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: black;
        padding: 8px;
        border: 6px solid black;
        box-shadow: 10px 10px 0 rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab"] {
        background: #ffcc00;
        color: black;
        font-weight: 900;
        border: 4px solid black;
        border-radius: 0 !important;
        padding: 12px 28px;
        margin: 0 4px;
        font-size: 1rem;
        text-transform: uppercase;
        transition: 0s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #ff8800;
    }
    .stTabs [aria-selected="true"] {
        background: #ff0033 !important;
        color: white !important;
        border: 4px solid white !important;
    }

    /* ===== SELECT BOXES ===== */
    .stSelectbox > div > div {
        background: white !important;
        border: 5px solid black !important;
        border-radius: 0 !important;
        box-shadow: 6px 6px 0 rgba(0,0,0,0.2) !important;
        padding: 4px !important;
        font-weight: 700 !important;
    }
    .stSelectbox > div > div:focus {
        border-color: #ff0033 !important;
    }

    /* ===== SLIDERS ===== */
    .stSlider > div > div > div {
        background: black !important;
        border-radius: 0 !important;
        height: 12px !important;
        border: 2px solid white !important;
    }
    .stSlider > div > div > div > div {
        background: #ffcc00 !important;
        border-radius: 0 !important;
    }
    .stSlider > div > div > div > div > div {
        background: #ff0033 !important;
        width: 24px !important;
        height: 24px !important;
        border: 4px solid black !important;
        border-radius: 0 !important;
        margin-top: -8px !important;
        box-shadow: 4px 4px 0 rgba(0,0,0,0.2) !important;
    }
    
    .stSlider label {
        color: black !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
        font-family: 'Courier New', monospace !important;
        background: #ffcc00 !important;
        padding: 0 12px !important;
        border: 4px solid black !important;
        display: inline-block !important;
        margin-bottom: 8px !important;
        text-transform: uppercase !important;
    }
    
    .stSlider > div > div > div > div > div + div {
        color: black !important;
        font-weight: 900 !important;
        font-size: 1.4rem !important;
        font-family: 'Courier New', monospace !important;
        background: #00ffcc !important;
        padding: 2px 12px !important;
        border: 4px solid black !important;
        box-shadow: 4px 4px 0 rgba(0,0,0,0.2) !important;
        border-radius: 0 !important;
    }
    
    .stSlider > div > div > div > div > div > span,
    .stSlider .st-ae {
        color: black !important;
        font-weight: 900 !important;
        font-size: 1.3rem !important;
        font-family: 'Courier New', monospace !important;
        background: #ffff00 !important;
        padding: 2px 12px !important;
        border: 3px solid black !important;
        border-radius: 0 !important;
        box-shadow: 4px 4px 0 rgba(0,0,0,0.2) !important;
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: #00ffcc !important;
        border: 5px solid black !important;
        border-radius: 0 !important;
        box-shadow: 6px 6px 0 rgba(0,0,0,0.2) !important;
        font-weight: 900 !important;
        color: black !important;
    }

    /* ===== METRICS ===== */
    .metric-value {
        font-size: 3.5rem;
        font-weight: 900;
        color: #ff0033;
        text-shadow: 4px 4px 0 #ffff00;
    }
    .metric-label {
        font-size: 1.2rem;
        font-weight: 700;
        color: black;
        background: #00ffcc;
        padding: 0 8px;
        border: 3px solid black;
        display: inline-block;
    }

    /* ===== SIDEBAR METRICS ===== */
    .sidebar-metric {
        background: #ffffff;
        border: 5px solid black;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 6px 6px 0 rgba(0,0,0,0.2);
    }
    .sidebar-metric .value {
        font-size: 2rem;
        font-weight: 900;
        color: #ff0033;
    }
    .sidebar-metric .label {
        font-weight: 700;
        color: black;
    }

    /* ===== PROGRESS BAR ===== */
    .brutal-progress {
        margin-top: 10px;
        background: #ffffff;
        border: 5px solid black;
        height: 20px;
        box-shadow: 6px 6px 0 rgba(0,0,0,0.2);
    }
    .brutal-progress-fill {
        background: repeating-linear-gradient(45deg, #ffcc00, #ffcc00 10px, #ff0033 10px, #ff0033 20px);
        height: 100%;
        transition: width 0.2s;
    }

    /* ===== FOOTER ===== */
    .footer {
        background: #ffcc00;
        border-top: 8px solid black;
        border-bottom: 8px solid black;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        font-weight: 900;
        font-size: 1.2rem;
        color: black;
        box-shadow: 0 -12px 0 rgba(0,0,0,0.1);
        transform: rotate(-0.5deg);
    }
    .footer span {
        background: #ff0033;
        color: white;
        padding: 0 8px;
    }

    /* ===== ALERTS ===== */
    .stAlert {
        border: 5px solid black !important;
        border-radius: 0 !important;
        box-shadow: 8px 8px 0 rgba(0,0,0,0.2) !important;
    }
    .stAlert > div {
        font-weight: 700 !important;
        color: black !important;
    }

    /* ===== INPUT NUMBER ===== */
    .stNumberInput > div > div > input {
        border: 5px solid black !important;
        border-radius: 0 !important;
        box-shadow: 6px 6px 0 rgba(0,0,0,0.2) !important;
        font-weight: 700 !important;
        background: white !important;
        color: black !important;
    }

    /* ===== LOGO ===== */
    .logo-box {
        background: #ff0033;
        border: 6px solid black;
        padding: 10px;
        box-shadow: 10px 10px 0 rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 1rem;
    }
    .logo-box img {
        filter: drop-shadow(6px 6px 0 black);
        width: 70px;
    }
    .logo-box h2 {
        color: white;
        text-shadow: 4px 4px 0 black;
        margin: 0;
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== LOAD MODELS AND BUILD ENCODING MAPS ==========
@st.cache_resource
def load_models():
    try:
        # Load ML models and supporting files
        reg_model = pickle.load(open('regressor.pkl', 'rb'))
        clf_model = pickle.load(open('classifier.pkl', 'rb'))
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))
        encoders = pickle.load(open('encoders.pkl', 'rb'))
        attraction_list = pickle.load(open('attraction_list.pkl', 'rb'))
        attractions_df = pickle.load(open('attractions_df.pkl', 'rb'))
        similarity_matrix = pickle.load(open('similarity_matrix.pkl', 'rb'))
        df = pd.read_csv('cleaned_tourism.csv')

        # Load raw mapping tables
        continent_df = pd.read_excel('data/Continent.xlsx')
        region_df = pd.read_excel('data/Region.xlsx')
        country_df = pd.read_excel('data/Country.xlsx')
        city_df = pd.read_excel('data/City.xlsx')
        type_df = pd.read_excel('data/Type.xlsx')

        # Build cascading maps
        continent_names = continent_df['Continent'].tolist()
        
        region_map = {}
        for _, row in region_df.iterrows():
            cont_id = row['ContinentId']
            region = row['Region']
            if cont_id not in region_map:
                region_map[cont_id] = []
            region_map[cont_id].append(region)
        
        country_map = {}
        for _, row in country_df.iterrows():
            reg_id = row['RegionId']
            country = row['Country']
            if reg_id not in country_map:
                country_map[reg_id] = []
            country_map[reg_id].append(country)
        
        city_map = {}
        for _, row in city_df.iterrows():
            cou_id = row['CountryId']
            city = row['CityName']
            if cou_id not in city_map:
                city_map[cou_id] = []
            city_map[cou_id].append(city)
        
        type_names = type_df['AttractionType'].tolist()

        # Build encoding maps (Name -> Encoded Value)
        encoded_continent_map = {}
        for name in continent_names:
            try:
                encoded_continent_map[name] = encoders['Continent'].transform([name])[0]
            except:
                encoded_continent_map[name] = 0
        
        encoded_region_map = {}
        all_regions = region_df['Region'].tolist()
        for name in all_regions:
            try:
                encoded_region_map[name] = encoders['Region'].transform([name])[0]
            except:
                encoded_region_map[name] = 0
        
        encoded_country_map = {}
        all_countries = country_df['Country'].tolist()
        for name in all_countries:
            try:
                encoded_country_map[name] = encoders['Country'].transform([name])[0]
            except:
                encoded_country_map[name] = 0
        
        encoded_city_map = {}
        all_cities = city_df['CityName'].tolist()
        for name in all_cities:
            try:
                encoded_city_map[name] = encoders['CityName'].transform([name])[0]
            except:
                encoded_city_map[name] = 0
        
        encoded_type_map = {}
        for name in type_names:
            try:
                encoded_type_map[name] = encoders['AttractionType'].transform([name])[0]
            except:
                encoded_type_map[name] = 0
        
        season_names = ['Winter', 'Spring', 'Summer', 'Fall']
        encoded_season_map = {}
        for name in season_names:
            try:
                encoded_season_map[name] = encoders['Season'].transform([name])[0]
            except:
                encoded_season_map[name] = 0

        return (reg_model, clf_model, feature_names, encoders, attraction_list,
                attractions_df, similarity_matrix, df,
                continent_names, region_map, country_map, city_map, type_names,
                encoded_continent_map, encoded_region_map, encoded_country_map,
                encoded_city_map, encoded_type_map, encoded_season_map)
    except Exception as e:
        st.error(f"❌ Error loading: {e}")
        st.stop()

(reg_model, clf_model, feature_names, encoders, attraction_list,
 attractions_df, similarity_matrix, df,
 continent_names, region_map, country_map, city_map, type_names,
 encoded_continent_map, encoded_region_map, encoded_country_map,
 encoded_city_map, encoded_type_map, encoded_season_map) = load_models()

# ========== HELPER FUNCTIONS ==========
def get_season(month):
    if month in [12,1,2]:
        return 'Winter'
    elif month in [3,4,5]:
        return 'Spring'
    elif month in [6,7,8]:
        return 'Summer'
    else:
        return 'Fall'

def get_real_prediction(continent, region, country, city, month, year, attraction_type, user_avg, attr_avg):
    season = get_season(month)

    # Get encoded values from maps
    enc_cont = encoded_continent_map.get(continent, 0)
    enc_reg = encoded_region_map.get(region, 0)
    enc_cou = encoded_country_map.get(country, 0)
    enc_city = encoded_city_map.get(city, 0)
    enc_type = encoded_type_map.get(attraction_type, 0)
    enc_season = encoded_season_map.get(season, 0)

    # Build input in the EXACT order of feature_names
    input_values = [
        year,               # VisitYear
        month,              # VisitMonth
        enc_cont,           # Continent
        enc_reg,            # Region
        enc_cou,            # Country
        enc_city,           # CityName
        enc_type,           # AttractionType
        user_avg,           # User_Avg_Rating
        attr_avg,           # Attraction_Avg_Rating
        enc_season          # Season
    ]
    
    input_df = pd.DataFrame([input_values], columns=feature_names)

    # Make predictions
    pred_rating = reg_model.predict(input_df)[0]
    pred_mode_enc = clf_model.predict(input_df)[0]
    
    # Debug output to terminal
    print("="*60)
    print("🔍 INPUT DATA SENT TO MODEL:")
    print(input_df)
    print("-"*60)
    print(f"📍 Continent: {continent} -> {enc_cont}")
    print(f"📍 Region: {region} -> {enc_reg}")
    print(f"📍 Country: {country} -> {enc_cou}")
    print(f"📍 City: {city} -> {enc_city}")
    print(f"📍 Type: {attraction_type} -> {enc_type}")
    print(f"📍 Season: {season} -> {enc_season}")
    print("-"*60)
    
    # Get probabilities for each class
    mode_probabilities = clf_model.predict_proba(input_df)[0]
    mode_classes = encoders['VisitMode'].classes_
    print("📊 CLASS PROBABILITIES:")
    for i, (cls, prob) in enumerate(zip(mode_classes, mode_probabilities)):
        print(f"   {cls}: {prob:.4f} ({prob*100:.1f}%)")
    print("-"*60)
    print(f"📊 Predicted Rating: {pred_rating:.4f}")
    print(f"📊 Predicted Mode: {mode_classes[pred_mode_enc]}")
    print("="*60)
    
    try:
        pred_mode = encoders['VisitMode'].inverse_transform([pred_mode_enc])[0]
    except:
        pred_mode = "Unknown"
    
    return pred_rating, pred_mode

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("""
    <div class="logo-box">
        <img src="https://cdn-icons-png.flaticon.com/512/2922/2922561.png" width="70">
        <h2>TOURISM<br>RAGE</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**🔥 WELCOME, REBEL!**")
    st.markdown("Predict. Explore. Dominate.")
    st.markdown("---")
    
    st.markdown("### ⚡ QUICK STATS")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="sidebar-metric">
            <div class="value">52.9K</div>
            <div class="label">VISITS</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="sidebar-metric">
            <div class="value">4.16⭐</div>
            <div class="label">AVG</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("💡 **PRO TIP:** Click the recommendations tab for BANGERS.")

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; font-size:3rem; opacity:0.5;">
        🏝️ 🏔️ 🏛️
    </div>
    """, unsafe_allow_html=True)

# ========== MAIN TITLE ==========
st.markdown("""
<div class="title-container">
    <div class="main-title">🌍 TOURISM RAGE</div>
    <div class="sub-title">⚡ AI-POWERED CHAOS FOR YOUR NEXT TRIP ⚡</div>
</div>
""", unsafe_allow_html=True)

# ========== TABS ==========
tab1, tab2, tab3 = st.tabs(["📊 PREDICT", "🔎 RECOMMEND", "📈 DASH"])

# ========== TAB 1: PREDICT ==========
with tab1:
    st.header("🔥 PREDICT RATING & VIBE")
    st.markdown("DROP YOUR DETAILS AND LET THE MACHINE GO BRRR.")
    
    col1, col2 = st.columns(2)
    with col1:
        continent = st.selectbox("🌍 CONTINENT", continent_names, key='continent')
        
        # Get ContinentId for filtering regions
        continent_df = pd.read_excel('data/Continent.xlsx')
        cont_id = continent_df[continent_df['Continent'] == continent]['ContinentId'].iloc[0]
        regions = region_map.get(cont_id, ['No Regions'])
        region = st.selectbox("📍 REGION", regions, key='region')
        
        # Get RegionId for filtering countries
        region_df = pd.read_excel('data/Region.xlsx')
        reg_id = region_df[region_df['Region'] == region]['RegionId'].iloc[0]
        countries = country_map.get(reg_id, ['No Countries'])
        country = st.selectbox("🏳️ COUNTRY", countries, key='country')
        
        # Get CountryId for filtering cities
        country_df = pd.read_excel('data/Country.xlsx')
        cou_id = country_df[country_df['Country'] == country]['CountryId'].iloc[0]
        cities = city_map.get(cou_id, ['No Cities'])
        city = st.selectbox("🏙️ CITY", cities, key='city')

    with col2:
        month = st.slider("📅 MONTH", 1, 12, 6, key='month')
        year = st.number_input("📆 YEAR", 2020, 2025, 2023, key='year')
        attraction_type = st.selectbox("🏛️ TYPE", type_names, key='type')
        user_avg = st.slider("⭐ YOUR AVG RATING", 1.0, 5.0, 3.5, 0.1, key='user_avg')
        attr_avg = st.slider("⭐ ATTRACTION AVG RATING", 1.0, 5.0, 4.0, 0.1, key='attr_avg')

    if st.button("🔮 PREDICT NOW!", use_container_width=True):
        with st.spinner("🧠 CALCULATING MAYHEM..."):
            try:
                pred_rating, pred_mode = get_real_prediction(
                    continent, region, country, city,
                    month, year, attraction_type,
                    user_avg, attr_avg
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="brutal-card brutal-card-yellow">
                        <div class="metric-label">⭐ PREDICTED RATING</div>
                        <div class="metric-value">{pred_rating:.2f} / 5.0</div>
                        <div class="brutal-progress">
                            <div class="brutal-progress-fill" style="width:{pred_rating/5*100}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    mode_icons = {
                        'Family': '👨‍👩‍👧‍👦',
                        'Couples': '💕',
                        'Business': '💼',
                        'Friends': '🎉',
                        'Solo': '🧳'
                    }
                    icon = mode_icons.get(pred_mode, '🌟')
                    st.markdown(f"""
                    <div class="brutal-card brutal-card-pink">
                        <div class="metric-label">🎯 PREDICTED VIBE</div>
                        <div class="metric-value" style="font-size:3rem;">{icon} {pred_mode}</div>
                        <div style="background:black; color:white; padding:4px 12px; display:inline-block; font-weight:700; font-size:0.8rem;">
                            {pred_mode == 'Family' and '👨‍👩‍👧‍👦 FAMILY FUN' or ''}
                            {pred_mode == 'Couples' and '💕 ROMANTIC CHAOS' or ''}
                            {pred_mode == 'Business' and '💼 HUSTLE MODE' or ''}
                            {pred_mode == 'Friends' and '🎉 PARTY TIME' or ''}
                            {pred_mode == 'Solo' and '🧳 LONE WOLF' or ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success("✅ PREDICTION COMPLETE! AI HAS SPOKEN.")
            except Exception as e:
                st.error(f"❌ ERROR: {e}")

# ========== TAB 2: RECOMMEND ==========
with tab2:
    st.header("🔎 FIND YOUR NEXT OBSESSION")
    st.markdown("PICK A PLACE, GET CHAOS.")
    
    def recommend_attractions(attraction_name, top_n=5):
        if attraction_name not in attractions_df['Attraction'].values:
            popular = df['Attraction'].value_counts().head(top_n).index.tolist()
            return popular
        idx = attractions_df[attractions_df['Attraction'] == attraction_name].index[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        return attractions_df.iloc[[i[0] for i in sim_scores]]['Attraction'].tolist()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("🎯 CHOOSE AN ATTRACTION:", attraction_list, key='attraction_select')
    with col2:
        num_recs = st.slider("📊 COUNT", 3, 10, 5, key='num_recs')
    
    if st.button("✨ GIVE ME BANGERS!", use_container_width=True):
        with st.spinner("MINING SIMILARITIES..."):
            recs = recommend_attractions(selected, top_n=num_recs)
            st.success(f"🎉 BASED ON **{selected}**, WE SUGGEST:")
            
            colors = ['brutal-card-yellow', 'brutal-card-pink', 'brutal-card-blue', 
                      'brutal-card-green', 'brutal-card-orange', 'brutal-card-purple']
            cols = st.columns(min(3, num_recs))
            for i, rec in enumerate(recs):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="brutal-card {colors[i % len(colors)]}" style="text-align:center; min-height:150px;">
                        <div style="font-size:3rem;">🌟</div>
                        <h3 style="text-transform:uppercase; font-weight:900; margin:0;">{rec}</h3>
                        <p style="font-weight:700; background:white; padding:0 8px; border:3px solid black; display:inline-block;">SIMILAR TO {selected}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ========== TAB 3: DASH ==========
with tab3:
    st.header("📈 CHAOS DASHBOARD")
    st.markdown("DATA VIZ THAT HITS DIFFERENT.")
    
    # Visit Mode Distribution
    visit_mode_counts = df['VisitMode'].value_counts().reset_index()
    visit_mode_counts.columns = ['VisitMode', 'Count']
    fig1 = px.bar(visit_mode_counts, x='VisitMode', y='Count', color='VisitMode',
                  title="VISIT MODES (BRUTAL)", text_auto=True,
                  color_discrete_sequence=['#ff0033', '#ffcc00', '#00ffcc', '#ff6600', '#ce93d8'])
    fig1.update_layout(
        showlegend=False,
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Courier New, monospace', size=14, color='black'),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, linecolor='black', linewidth=4),
        yaxis=dict(showgrid=True, gridcolor='black', gridwidth=2, linecolor='black', linewidth=4)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Top 10 Attractions
    top_attractions = df['Attraction'].value_counts().head(10).reset_index()
    top_attractions.columns = ['Attraction', 'Visits']
    fig2 = px.bar(top_attractions, x='Attraction', y='Visits', color='Visits',
                  title="TOP 10 ATTACKS (I MEAN ATTRACTIONS)",
                  color_continuous_scale='Oranges')
    fig2.update_layout(
        height=400,
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Courier New, monospace', size=14, color='black'),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, linecolor='black', linewidth=4),
        yaxis=dict(showgrid=True, gridcolor='black', gridwidth=2, linecolor='black', linewidth=4)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Rating by Month
    monthly_rating = df.groupby('VisitMonth')['Rating'].mean().reset_index()
    fig3 = px.line(monthly_rating, x='VisitMonth', y='Rating',
                   title="RATING ROLLERCOASTER BY MONTH",
                   markers=True, line_shape='spline')
    fig3.update_traces(line_color='#ff0033', marker_color='#ffcc00', marker_size=14,
                       marker_line=dict(color='black', width=4))
    fig3.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Courier New, monospace', size=14, color='black'),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(tickmode='linear', tick0=1, dtick=1, showgrid=False, linecolor='black', linewidth=4),
        yaxis=dict(showgrid=True, gridcolor='black', gridwidth=2, linecolor='black', linewidth=4)
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Two columns
    col1, col2 = st.columns(2)
    with col1:
        fig4 = px.histogram(df, x='Rating', nbins=20,
                            title="RATING DISTRIBUTION (RAW)",
                            color_discrete_sequence=['#ff6600'])
        fig4.update_layout(
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Courier New, monospace', size=14, color='black'),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, linecolor='black', linewidth=4),
            yaxis=dict(showgrid=True, gridcolor='black', gridwidth=2, linecolor='black', linewidth=4)
        )
        st.plotly_chart(fig4, use_container_width=True)
    with col2:
        type_counts = df['AttractionType'].value_counts().reset_index()
        type_counts.columns = ['AttractionType', 'Count']
        fig5 = px.pie(type_counts, values='Count', names='AttractionType',
                      title="ATTRACTION TYPES (PIE OF CHAOS)",
                      color_discrete_sequence=['#ff0033', '#ffcc00', '#00ffcc', '#ff6600', '#ce93d8', '#4fc3f7'])
        fig5.update_traces(textposition='inside', textinfo='percent+label')
        fig5.update_layout(
            height=350,
            font=dict(family='Courier New, monospace', size=14, color='black'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Top Cities
    city_visits = df['CityName'].value_counts().head(10).reset_index()
    city_visits.columns = ['City', 'Visits']
    fig6 = px.bar(city_visits, x='City', y='Visits', color='Visits',
                  title="TOP CITIES (WHERE THE PARTY AT)",
                  color_continuous_scale='Reds')
    fig6.update_layout(
        height=400,
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Courier New, monospace', size=14, color='black'),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, linecolor='black', linewidth=4),
        yaxis=dict(showgrid=True, gridcolor='black', gridwidth=2, linecolor='black', linewidth=4)
    )
    st.plotly_chart(fig6, use_container_width=True)

# ========== FOOTER ==========
st.markdown("""
<div class="footer">
    🌍 TOURISM RAGE · BUILT WITH <span>💔</span> AND <span>☕</span> · DATA GO BRRR<br>
    <span style="font-size:0.8rem; background:none; color:black;">NO BORING ALLOWED</span>
</div>
""", unsafe_allow_html=True)