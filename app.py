import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="FIFA 20 Analytics & Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Styling for glassmorphism and modern aesthetics */
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #1E1E24;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #333;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(26, 115, 232, 0.2);
        border-color: #1a73e8;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(26, 115, 232, 0.4);
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
    }
    .title-gradient {
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    df_raw = pd.read_csv('players_20.csv', low_memory=False)
    
    cols = [
        'short_name', 'age', 'height_cm', 'weight_kg', 'overall', 'potential', 'value_eur', 'wage_eur', 'club', 'nationality',
        'preferred_foot', 'work_rate', 'player_positions', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',
        'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
        'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
        'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
        'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties',
        'mentality_composure', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle'
    ]
    df = df_raw.copy()
    existing_cols = [c for c in cols if c in df.columns]
    
    # Process numeric columns
    numeric_cols = df[existing_cols].select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Create required features
    df['main_pos'] = df['player_positions'].str.split(',').str[0].str.strip()
    
    return df_raw, df

try:
    df_raw, df_processed = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# --- ML Model Training ---
@st.cache_resource
def train_value_model(df):
    # Prepare data for model: we want to predict value_eur based on stats
    features = ['age', 'overall', 'potential', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    # ensure predictors exist and value_eur isn't 0
    df_train = df[(df['value_eur'] > 0)][features + ['value_eur']].dropna()
    
    X = df_train[features]
    y = df_train['value_eur']
    
    # Using small estimators for speed
    rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42)
    rf.fit(X, y)
    return rf, features

rf_model, predictor_features = train_value_model(df_processed)

# --- Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/FIFA_logo_without_slogan.svg/1200px-FIFA_logo_without_slogan.svg.png", width=150)
st.sidebar.markdown("<h2 style='text-align: center;'>FIFA Engine Max</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("Navigation", 
    ["🏠 Dashboard overview", "🔍 Player EDA & Clubs", "⚔️ Player Comparison", "🕵️‍♂️ Scouting System", "🧠 Skills Clustering (3D)", "🔮 Market Value Predictor"]
)

st.sidebar.markdown("---")
st.sidebar.info("The premier analytical dashboard for FIFA player stats, machine learning estimation, and interactive discovery.")

# --- Main App ---
if app_mode == "🏠 Dashboard overview":
    st.markdown("<h1 class='title-gradient'>Dashboard Overview</h1>", unsafe_allow_html=True)
    st.markdown("Explore the extensive FIFA 20 dataset featuring thousands of players, detailed attributes, and global statistics.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>Total Players</h3><h2>{len(df_raw):,}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>Nationalities</h3><h2>{df_raw['nationality'].nunique()}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>Avg Age</h3><h2>{df_raw['age'].mean():.1f}</h2></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h3>Avg Rating</h3><h2>{df_raw['overall'].mean():.1f}</h2></div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top Players Matrix
    st.subheader("🌟 Top Players by Overall Rating")
    top_10 = df_raw.nlargest(10, 'overall')[['short_name', 'age', 'nationality', 'club', 'overall', 'wage_eur']]
    st.dataframe(top_10, use_container_width=True)

elif app_mode == "🔍 Player EDA & Clubs":
    st.markdown("<h1 class='title-gradient'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Demographics", "📈 Age & Performance", "💰 Financials", "🏟️ Club Analysis"])
    
    with tab1:
        st.subheader("Top 10 Countries by Player Count")
        top_countries = df_raw['nationality'].value_counts().head(10).reset_index()
        top_countries.columns = ['Nationality', 'Count']
        
        fig = px.bar(top_countries, x='Count', y='Nationality', orientation='h',
                     color='Count', color_continuous_scale='Blues',
                     title="Global Player Distribution")
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Age vs Overall Rating")
        age_overall = df_raw.groupby('age')['overall'].mean().reset_index()
        
        fig = px.line(age_overall, x='age', y='overall', markers=True,
                      title="Age Progression of Overall Ratings",
                      labels={'age': 'Age', 'overall': 'Average Overall Rating'})
        fig.update_traces(line_color='#00f2fe', line_width=3, marker_size=8)
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.subheader("Average Wages by Position (ST vs RW vs LW)")
        pos_df = df_processed[df_processed['main_pos'].isin(['ST', 'RW', 'LW'])]
        wage_by_pos = pos_df.groupby('main_pos')['wage_eur'].mean().reset_index()
        
        fig = px.bar(wage_by_pos, x='main_pos', y='wage_eur', color='main_pos',
                     title="Average Weekly Wage (€) by Attacking Position",
                     labels={'main_pos': 'Position', 'wage_eur': 'Avg Wage (€)'},
                     color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99'])
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab4:
        st.subheader("Club Overview & Best Starting XI")
        clubs = sorted(df_processed['club'].dropna().unique())
        selected_club = st.selectbox("Select Club", clubs, index=clubs.index('FC Barcelona') if 'FC Barcelona' in clubs else 0)
        
        club_df = df_processed[df_processed['club'] == selected_club].sort_values(by='overall', ascending=False)
        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Squad Size", len(club_df))
        col_c2.metric("Average Overall Rating", f"{club_df['overall'].mean():.1f}")
        col_c3.metric("Total Weekly Wage Bill", f"€ {club_df['wage_eur'].sum():,}")
        
        st.markdown(f"#### Top 10 Best Rated Players at {selected_club}")
        st.dataframe(club_df[['short_name', 'main_pos', 'overall', 'potential', 'wage_eur']].head(10), use_container_width=True)

elif app_mode == "⚔️ Player Comparison":
    st.markdown("<h1 class='title-gradient'>Player Face-to-Face</h1>", unsafe_allow_html=True)
    st.markdown("Compare the technical, mental, and physical attributes of two players side-by-side.")
    
    players_list = df_processed.sort_values(by='overall', ascending=False)['short_name'].tolist()
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p1 = st.selectbox("Select Player 1", players_list, index=0) # Messi usually
    with col_p2:
        p2 = st.selectbox("Select Player 2", players_list, index=1) # Ronaldo usually
        
    df_p1 = df_processed[df_processed['short_name'] == p1].iloc[0]
    df_p2 = df_processed[df_processed['short_name'] == p2].iloc[0]
    
    categories = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    val1 = df_p1[categories].fillna(50).values.flatten().tolist()
    val2 = df_p2[categories].fillna(50).values.flatten().tolist()
    
    # Complete loop for radar
    val1 += val1[:1]
    val2 += val2[:1]
    cat_loop = categories + categories[:1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=val1, theta=[c.capitalize() for c in cat_loop],
        fill='toself', name=p1, line_color='#00f2fe'
    ))
    fig.add_trace(go.Scatterpolar(
        r=val2, theta=[c.capitalize() for c in cat_loop],
        fill='toself', name=p2, line_color='#ff9999'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        template='plotly_dark',
        title=f"{p1} vs {p2}"
    )
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(f"🔵 {p1}")
        st.write(f"**Overall**: {df_p1['overall']}\n\n**Potential**: {df_p1['potential']}\n\n**Value**: €{df_p1['value_eur']:,.0f}")
        st.subheader(f"🔴 {p2}")
        st.write(f"**Overall**: {df_p2['overall']}\n\n**Potential**: {df_p2['potential']}\n\n**Value**: €{df_p2['value_eur']:,.0f}")
    with col2:
        st.plotly_chart(fig, use_container_width=True)

elif app_mode == "🕵️‍♂️ Scouting System":
    st.markdown("<h1 class='title-gradient'>Scouting Network (Wonderkids)</h1>", unsafe_allow_html=True)
    st.markdown("Adjust parameters to find the perfect signing for your club.")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        max_age = st.slider("Maximum Age", 16, 45, 23)
    with col_s2:
        min_pot = st.slider("Minimum Potential", 50, 99, 85)
    with col_s3:
        max_val = st.slider("Maximum Value (€)", 0, int(df_processed['value_eur'].max()), 40000000, step=1000000, format="%d")
    with col_s4:
        pos_filter = st.selectbox("Preferred Position", ["Any"] + sorted(df_processed['main_pos'].unique().tolist()))
        
    filtered = df_processed[
        (df_processed['age'] <= max_age) & 
        (df_processed['potential'] >= min_pot) & 
        (df_processed['value_eur'] <= max_val)
    ]
    if pos_filter != "Any":
        filtered = filtered[filtered['main_pos'] == pos_filter]
        
    filtered = filtered.sort_values(by='potential', ascending=False)
    
    st.success(f"Scouts found {len(filtered)} players matching your criteria!")
    st.dataframe(filtered[['short_name', 'age', 'main_pos', 'overall', 'potential', 'club', 'value_eur', 'wage_eur']], use_container_width=True)

elif app_mode == "brains Skills Clustering (3D)":
    st.markdown("<h1 class='title-gradient'>K-Means Skills Clustering</h1>", unsafe_allow_html=True)
    st.markdown("Group players based on their technical skills and physical attributes, visualized in incredible 3D.")
    
    st.sidebar.subheader("Clustering Configuration")
    n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 8, 4)
    
    skill_cols = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',
                  'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
                  'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
                  'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
                  'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties',
                  'mentality_composure', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle']
    
    skill_cols = [c for c in skill_cols if c in df_processed.columns]
    
    with st.spinner("Training K-Means Model..."):
        clustering_df = df_processed[skill_cols].dropna()
        original_indices = clustering_df.index
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_df)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        df_display = df_raw.loc[original_indices, ['short_name', 'overall', 'player_positions'] + skill_cols]
        df_display['Cluster'] = clusters
        df_display['Cluster Name'] = "Cluster " + df_display['Cluster'].astype(str)
        
    st.success(f"Successfully clustered {len(clustering_df):,} players into {n_clusters} groups!")
    
    st.subheader("3D Cluster Distribution Map (PCA Reduced)")
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_data)
    
    df_display['PCA1'] = pca_result[:, 0]
    df_display['PCA2'] = pca_result[:, 1]
    df_display['PCA3'] = pca_result[:, 2]
    
    fig = px.scatter_3d(df_display, x='PCA1', y='PCA2', z='PCA3', color='Cluster Name', 
                     hover_name='short_name', hover_data=['overall', 'player_positions'],
                     title="Player Clusters mapped to 3D Space",
                     template="plotly_dark",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'))
    st.plotly_chart(fig, use_container_width=True)

elif app_mode == "🔮 Market Value Predictor":
    st.markdown("<h1 class='title-gradient'>Market Value AI Predictor</h1>", unsafe_allow_html=True)
    st.markdown("Use our custom-trained Random Forest model to estimate a player's valuation on the transfer market based on their attributes!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Design Your Player")
        input_age = st.slider("Age", 16, 45, 25)
        input_ovr = st.slider("Overall Rating", 40, 99, 75)
        input_pot = st.slider("Potential Rating", 40, 99, 80)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        input_pac = col_s1.number_input("Pace", 1, 99, 70)
        input_sho = col_s2.number_input("Shooting", 1, 99, 65)
        input_pas = col_s3.number_input("Passing", 1, 99, 70)
        
        col_s4, col_s5, col_s6 = st.columns(3)
        input_dri = col_s4.number_input("Dribbling", 1, 99, 72)
        input_def = col_s5.number_input("Defending", 1, 99, 50)
        input_phy = col_s6.number_input("Physicality", 1, 99, 68)
        
    with col2:
        st.subheader("AI Recommendation")
        st.info("The model learned from thousands of real FIFA 20 variations.")
        
        # Build the feature vector
        try:
            x_input = pd.DataFrame([[input_age, input_ovr, input_pot, input_pac, input_sho, input_pas, input_dri, input_def, input_phy]], columns=predictor_features)
            predicted_value = rf_model.predict(x_input)[0]
            
            st.markdown(f"<div class='metric-card' style='text-align: center; margin-top: 2rem; border-color: #00f2fe;'>"
                        f"<h3>Estimated Transfer Value</h3>"
                        f"<h1 style='color: #4facfe; font-size: 3.5rem;'>€{predicted_value:,.0f}</h1>"
                        f"</div>", unsafe_allow_html=True)
                        
            st.write("")
            st.markdown(f"**Insight**: An {input_age}-year-old player with {input_ovr} Overall and {input_pot} Potential commands this tag in the market.")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
