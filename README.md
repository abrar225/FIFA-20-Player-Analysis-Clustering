# ⚽ FIFA 20 Analytics & Predictor Engine

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

An advanced, interactive data science application built for deep exploration of the FIFA 20 player dataset. This project goes beyond simple analysis, incorporating **Machine Learning (Random Forest)** for value prediction and **Unsupervised Learning (K-Means)** for player style clustering.

---

## 🚀 Key Features

### 1. 🏠 Dashboard Overview
Get instant access to high-level global metrics:
- Total player count (18,000+).
- Global nationality distribution.
- Real-time leaderboard of the world's top-rated players.

### 2. 🔍 Advanced EDA & Club Analysis
Deep dive into the trends that shape world football:
- **Demographics**: Visualization of the powerhouse nations.
- **Performance vs Age**: Track how ratings peak and decline over a career.
- **Financial Insights**: Wage distribution across different field positions.
- **Club deep-dive**: Select any club to view their squad size, total wage bill, and generate a **Best Starting XI**.

### 3. ⚔️ Player Face-to-Face Comparison
Settle the "Messi vs Ronaldo" debate once and for all.
- Select any two players to overlap their stats on a **Dynamic Radar Chart**.
- Compare Pace, Shooting, Passing, Dribbling, Defending, and Physicality side-by-side.

### 4. 🕵️‍♂️ Smart Scouting Network
Acting like a real football director:
- Filter through 18,000+ players using sliders for **Age, Potential, and Market Value**.
- Instantly identify "Wonderkids" (high potential, low value) for your team.

### 5. 🧠 K-Means Skills Clustering (3D)
Uncover hidden player archetypes using AI:
- Uses **K-Means Clustering** to group players based on 29+ technical and physical attributes.
- Visualized in a stunning **3D PCA space**, allowing you to rotate and explore the "planet of players" to see which stars truly play alike.

### 6. 🔮 Market Value AI Predictor
Wondering how much a custom-designed player would cost?
- Powered by a **Random Forest Regressor** trained on real market data.
- Input custom stats (Pace, Overall, Shooting, etc.) and receive an instantaneous AI valuation.

---

## 🛠️ How It Works (Technical Breakdown)

### Machine Learning
- **Predictor**: Uses a `RandomForestRegressor` with 30 estimators and depth constraints for a balance between accuracy and blazing-fast web responsiveness. It analyzes correlations between performance metrics and market value.
- **Clustering**: Employs `KMeans` with the **Elbow Method** logic to group players into distinct playstyles.

### Dimensionality Reduction
- Uses **PCA (Principal Component Analysis)** to condense 29 dimensional technical skill-vectors into a 3D coordinate system ($X, Y, Z$) for visualization without losing the core variance of the data.

### Data Processing
- **Pandas Pipeline**: Handles missing values (filling with medians), cleans position strings, and calculates aggregate club metrics.
- **Standardization**: All skills are scaled using `StandardScaler` before clustering to ensure physical attributes (like weight) don't overpower technical ones.

---

## 📥 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abrar225/FIFA-20-Player-Analysis-Clustering.git
   cd Fifa20
   ```

2. **Create a virtual environment (Recommended)**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   # .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the engine**:
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Tech Stack
- **Frontend**: Streamlit (Modern Glassmorphism UI)
- **Visuals**: Plotly Express & Graph Objects
- **Analysis**: Pandas, NumPy
- **ML/AI**: Scikit-Learn (KMeans, PCA, Random Forest)

---

*Developed with ❤️ for the intersection of Football and Data Science.*
