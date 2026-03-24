import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.io import arff
import io
import groq
import time
import os
from dotenv import load_dotenv

# --- INITIALIZE ENVIRONMENT ---
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Intelligence - Customer Hub",
    page_icon="🌌",
    layout="wide",
)

# --- STYLING: POPPINS & DARK THEME ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
    /* Global Background and Text */
    .stApp {
        background: radial-gradient(circle at top left, #1e293b, #0f172a) !important;
        color: #f8fafc !important;
    }
    
    /* Targeted Poppins: PROTECTS ICONS by avoiding span/div/* overrides */
    html, body, .stMarkdown, p, label, h1, h2, h3, .stMetricValue, .stButton {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Metric/K-means Label Colors */
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }

    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(to right, #38bdf8, transparent);
        margin: 2.5rem 0;
        opacity: 0.2;
    }

    /* AI Executive Report Container */
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .ai-report-card {
        background: rgba(56, 189, 248, 0.05) !important;
        border: 2px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 20px !important;
        padding: 40px !important;
        margin-top: 2rem !important;
        animation: slideUpFade 1s cubic-bezier(0.23, 1, 0.32, 1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .ai-report-card h1, .ai-report-card h2, .ai-report-card h3 {
        color: #38bdf8 !important;
        margin-bottom: 1.5rem;
    }
    
    .ai-report-card p, .ai-report-card li {
        font-size: 1.1rem;
        line-height: 1.7;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🌌 Intelligence")
    st.markdown("---")

    st.markdown("### 🛠 High-Performance Engine")
    cluster_tech = st.radio("Strategy", ["K-Means", "DBSCAN"])

    if cluster_tech == "K-Means":
        k_val = st.slider("Target Clusters (K)", 2, 8, 4)
    else:
        eps_val = st.slider("Radius (Epsilon)", 0.1, 2.0, 0.5)
        min_samp = st.slider("Min Cluster Density", 2, 10, 5)

    st.markdown("---")
    st.caption("Industrial Data Mining System v6.0")
    st.info("💡 Powered by Groq (Llama-3 Architecture)")

# --- DATA LOADING ---
@st.cache_data
def process_data(file_obj):
    try:
        if file_obj is None: return None
        file_obj.seek(0)
        content = file_obj.read().decode("utf-8")
        data, meta = arff.loadarff(io.StringIO(content))
        df = pd.DataFrame(data)

        # decode bytes and clean for Arrow
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x))
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) if df[col].dtype.name != 'category' else df[col]
        
        return df.convert_dtypes()
    except Exception as e:
        st.error(f"Engine Fault: {str(e)}")
        return None

# --- MAIN DASHBOARD AREA ---
st.title("📊 Customer Segmentation Hub")
uploaded_file = st.file_uploader("📂 Drop your ARFF transactional data here to begin analysis", type=["arff"])

if uploaded_file:
    df = process_data(uploaded_file)
    
    if df is not None:
        st.markdown(f"*Success: Analyzed {len(df)} customer records.*")

        # KPIs
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        kp1, kp2, kp3 = st.columns(3)
        kp1.metric("Total Records", f"{len(df):,}")
        if "total_sales" in df.columns: kp2.metric("Gross Sales", f"${df['total_sales'].sum():,.0f}")
        if "total_profit" in df.columns: kp3.metric("Avg Profit/Customer", f"${df['total_profit'].mean():.2f}")

        # Data Preview
        with st.expander("🔍 View Raw Dataset Inventory", expanded=False):
            st.dataframe(df.head(20), width='stretch')
            st.dataframe(df.describe().T, width='stretch')

        # Feature Selection
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 1:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[num_cols])

            # Clustering
            if cluster_tech == "K-Means":
                model = KMeans(n_clusters=k_val, random_state=42, n_init="auto")
            else:
                model = DBSCAN(eps=eps_val, min_samples=min_samp)

            labels = model.fit_predict(scaled_data)
            df["Cluster"] = labels

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.subheader("🟣 Segmentation Map")
            st.success(f"Algorithm successfully identified {len(set(labels))} unique archetypes.")

            # Charts
            chart_c1, chart_c2 = st.columns(2)
            with chart_c1:
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(scaled_data)
                p_df = pd.DataFrame(reduced, columns=["X", "Y"])
                p_df["Cluster"] = labels.astype(str)
                fig_map = px.scatter(p_df, x="X", y="Y", color="Cluster", 
                                    template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Prism)
                fig_map.update_traces(marker=dict(size=14, opacity=0.8, line=dict(width=1, color='white')))
                st.plotly_chart(fig_map, width='stretch')
            
            with chart_c2:
                share = df['Cluster'].value_counts().reset_index()
                share.columns = ['Segment', 'Volume']
                fig_pie = px.pie(share, values='Volume', names='Segment', hole=0.5,
                                template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Prism)
                st.plotly_chart(fig_pie, width='stretch')

            # 🚀 STRATEGIC AI CENTER (GROQ LLAMA-3)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.subheader("🔥 Strategic Business Discovery")
            
            summary = df.groupby("Cluster")[num_cols].mean()
            
            if GROQ_KEY and GROQ_KEY != "your_actual_groq_key_here":
                if st.button("🚀 Unlock AI-Powered Market Insights", use_container_width=True):
                    with st.status("🤖 AI Engine is decoding patterns...", expanded=True) as status:
                        st.write("Reading high-dimensional matrices...")
                        time.sleep(0.5)
                        st.write("Synthesizing behavioral playbooks...")
                        
                        try:
                            client = groq.Groq(api_key=GROQ_KEY)
                            
                            prompt = f"""
                            You are a Senior Data Mining Consultant. 
                            Analyze these e-commerce customer clusters based on their feature averages:
                            {summary.to_string()}
                            
                            Required Output (Strictly Markdown):
                            1. **Segment Names & Behavioral DNA**: Give each cluster a creative business name.
                            2. **Monetization Roadmap**: 2 specific strategies for each cluster to increase revenue.
                            3. **Churn Prevention**: Which cluster is high-risk?
                            4. **Final Recommendation**: One global strategic move for the CEO.
                            
                            Tone: Professional.
                            """
                            
                            completion = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.5,
                                max_tokens=2048
                            )
                            
                            status.update(label="✅ Success: Executive Insights Ready", state="complete", expanded=False)
                            st.markdown('<div class="ai-report-card">', unsafe_allow_html=True)
                            st.markdown(completion.choices[0].message.content)
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            status.update(label="❌ Engine Fault", state="error")
                            st.error(f"Groq API Error: {str(e)}")
            else:
                st.error("🔑 System Critical: GROQ_API_KEY NOT FOUND in .env file.")
                st.info("Action: Open the `.env` file and paste your key into `GROQ_API_KEY=...` (Get it from console.groq.com)")
                st.dataframe(summary, width='stretch')

            # EXPORT
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.download_button(
                label="📥 Export Segments to CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="segmented_data.csv",
                mime="text/csv",
                width='stretch'
            )
else:
    st.info("🌌 Standby. Awaiting ARFF data stream... (Use the uploader above)")