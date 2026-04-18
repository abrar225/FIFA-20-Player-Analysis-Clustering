import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="FIFA 20 Analytics & Clustering",
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
    df_raw = pd.read_csv('players_20.csv')
    
    cols = [
        'short_name', 'age', 'height_cm', 'weight_kg', 'overall', 'potential',
        'preferred_foot', 'work_rate', 'player_positions',
        'crossing', 'finishing', 'heading_accuracy', 'short_passing',
        'volleys', 'dribbling', 'curve', 'fk_accuracy', 'long_passing',
        'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions',
        'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots',
        'aggression', 'interceptions', 'positioning', 'vision', 'penalties',
        'composure', 'marking', 'standing_tackle', 'sliding_tackle'
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

# --- Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/FIFA_logo_without_slogan.svg/1200px-FIFA_logo_without_slogan.svg.png", width=150)
st.sidebar.markdown("<h2 style='text-align: center;'>FIFA 20 Engine</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("Navigation", 
    ["🏠 Dashboard overview", "🔍 Player EDA", "🧠 Skills Clustering"]
)

st.sidebar.markdown("---")
st.sidebar.info("Developed for analyzing FIFA 20 player stats, discovering insights, and performing K-Means clustering.")

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

elif app_mode == "🔍 Player EDA":
    st.markdown("<h1 class='title-gradient'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🌍 Demographics", "📈 Age & Performance", "💰 Financials"])
    
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

elif app_mode == "🧠 Skills Clustering":
    st.markdown("<h1 class='title-gradient'>K-Means Skills Clustering</h1>", unsafe_allow_html=True)
    st.markdown("Group players based on their technical skills and physical attributes.")
    
    # Feature selection
    st.sidebar.subheader("Clustering Configuration")
    n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 8, 4)
    
    # Prep data for clustering
    skill_cols = ['crossing', 'finishing', 'heading_accuracy', 'short_passing',
                  'volleys', 'dribbling', 'curve', 'fk_accuracy', 'long_passing',
                  'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions',
                  'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots',
                  'aggression', 'interceptions', 'positioning', 'vision', 'penalties',
                  'composure', 'marking', 'standing_tackle', 'sliding_tackle']
    
    # Ensure columns exist
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
        
        # Mapping clusters to rough playstyles (heuristic)
        # We can analyze center of clusters or just leave them as numbers
        df_display['Cluster Name'] = "Cluster " + df_display['Cluster'].astype(str)
        
    st.success(f"Successfully clustered {len(clustering_df):,} players into {n_clusters} groups!")
    
    # Clustering Visualization (PCA for 2D or 3D)
    st.subheader("Cluster Distribution Map (PCA Reduced)")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    df_display['PCA1'] = pca_result[:, 0]
    df_display['PCA2'] = pca_result[:, 1]
    
    fig = px.scatter(df_display, x='PCA1', y='PCA2', color='Cluster Name', 
                     hover_data=['short_name', 'overall', 'player_positions'],
                     title="Player Clusters mapped to 2D Space",
                     template="plotly_dark",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample from each cluster
    st.subheader("Sample Players from Each Cluster")
    cols = st.columns(n_clusters)
    for i in range(n_clusters):
        with cols[i]:
            st.markdown(f"**Cluster {i}**")
            sample = df_display[df_display['Cluster'] == i].nlargest(5, 'overall')[['short_name', 'player_positions']]
            st.dataframe(sample, hide_index=True)
