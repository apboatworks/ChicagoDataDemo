
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from io import BytesIO
from fpdf import FPDF
import datetime

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Boat Issues Demo", layout="wide")

DEFAULT_FILE = "chicago_boat_issues.xlsx"  # Preloaded file

# -----------------------------
# FUNCTIONS
# -----------------------------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["boat_name"] = df["boat_name"].str.strip().str.lower()
    df["boat_issue"] = df["boat_issue"].str.strip()
    df["boat_issue_class"] = df["boat_issue_(class)"].str.strip().str.title()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def ai_cluster_issues(df, n_clusters=6):
    issue_texts = df["boat_issue"].dropna().tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(issue_texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    df.loc[df["boat_issue"].notna(), "ai_category"] = clusters
    return df

def plot_top_issues(df):
    counts = df["boat_issue_class"].value_counts().head(10)
    fig, ax = plt.subplots()
    counts.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Reports")
    ax.set_ylabel("Issue Type")
    ax.set_title("Top 10 Recurring Boat Issue Types")
    st.pyplot(fig)

def plot_top_boats(df):
    counts = df["boat_name"].value_counts().head(10)
    fig, ax = plt.subplots()
    counts.plot(kind="barh", color="orange", ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Reports")
    ax.set_ylabel("Boat Name")
    ax.set_title("Top 10 Boats by Reported Issues")
    st.pyplot(fig)

def plot_timeline(df):
    issues_over_time = df.groupby(df["date"].dt.to_period("M")).size()
    fig, ax = plt.subplots()
    issues_over_time.plot(kind="line", marker="o", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Issues")
    ax.set_title("Reported Issues Over Time")
    st.pyplot(fig)

def export_pdf(df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Boat Issues Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

    pdf.cell(200, 10, txt="Top Issues:", ln=True)
    top_issues = df["boat_issue_class"].value_counts().head(10)
    for issue, count in top_issues.items():
        pdf.cell(200, 8, txt=f"{issue}: {count}", ln=True)

    pdf.cell(200, 10, txt="Top Boats:", ln=True)
    top_boats = df["boat_name"].value_counts().head(10)
    for boat, count in top_boats.items():
        pdf.cell(200, 8, txt=f"{boat}: {count}", ln=True)

    # Convert to BytesIO
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# -----------------------------
# APP LAYOUT
# -----------------------------
st.title("🚤 Boat Maintenance Insights Demo")
st.write("This demo shows how AI can instantly turn messy maintenance logs into actionable intelligence.")

# File upload
uploaded_file = st.file_uploader("Upload a maintenance report Excel file", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data(DEFAULT_FILE)

# AI categorization
df = ai_cluster_issues(df)

# Charts
col1, col2 = st.columns(2)
with col1:
    plot_top_issues(df)
with col2:
    plot_top_boats(df)

plot_timeline(df)

# AI category browser
st.subheader("🔍 AI-Categorized Issues")
category = st.selectbox("Select AI category", sorted(df["ai_category"].dropna().unique()))
st.write(df[df["ai_category"] == category][["boat_name", "boat_issue_class", "boat_issue"]])

# PDF export
pdf_file = export_pdf(df)
st.download_button(
    label="📄 Download PDF Report",
    data=pdf_file,
    file_name="boat_issues_report.pdf",
    mime="application/pdf"
)
