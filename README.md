# 🌌 E-Commerce Customer Intelligence Dashboard

A high-fidelity, AI-powered customer segmentation platform developed for **Data Warehousing and Data Mining (DWDM)**. This application uses advanced machine learning (K-Means/DBSCAN) and high-performance inference (Groq/Llama-3) to transform raw transactional data into actionable business strategy.

## 🚀 Key Features
- **Intelligent Preprocessing**: Automated handling of ARFF telemetry with standard scaling and PCA-driven dimensionality reduction.
- **Dual-Clustering Architecture**: Select between density-based (DBSCAN) and centroid-based (K-Means) segmentation.
- **Live Market Visualization**: 2D Projection mapping and market share analysis using interactive Plotly charts.
- **Llama-3 Strategy Hub**: Near-instant generation of business playbooks using the Groq high-speed inference engine.
- **Industrial SaaS UI**: Dark-themed glassmorphism interface with premium Poppins typography.

## 🛠 Tech Stack
- **Data Engine**: Pandas, NumPy, Scikit-Learn.
- **Visuals**: Plotly Express.
- **Frontend**: Streamlit (SaaS Config).
- **AI Inference**: Groq SDK (Llama-3.3-70b).

## 📥 Getting Started
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   Create a `.env` file and add your Groq API key:
   ```env
   GROQ_API_KEY=your_key_here
   ```
4. **Launch Application**:
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset Requirement
The application specifically consumes **ARFF (Attribute-Relation File Format)** datasets containing transactional customer telemetry.

---