import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Data Agent Dashboard", layout="wide")

st.title("📊 Data Analysis Agent")

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV/TSV)", type=["csv", "tsv"])

if uploaded_file:
    # Detect separator
    sep = "\t" if uploaded_file.name.endswith("tsv") else ","

    df = pd.read_csv(uploaded_file, sep=sep)

    # Sampling large dataset
    if len(df) > 10000:
        st.warning("Dataset is large. Sampling 10,000 rows for performance.")
        df = df.sample(10000, random_state=42)

    st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ---------------------------
    # SIDEBAR OPTIONS
    # ---------------------------
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose Action", [
        "Preview Data",
        "Structure",
        "Summary",
        "Missing Values",
        "Visualizations",
        "Ask Questions"
    ])

    # ---------------------------
    # PREVIEW
    # ---------------------------
    if option == "Preview Data":
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100))

    # ---------------------------
    # STRUCTURE
    # ---------------------------
    elif option == "Structure":
        st.subheader("Dataset Info")
        buffer = []
        df.info(buf=buffer)
        s = "\n".join(buffer)
        st.text(s)

    # ---------------------------
    # SUMMARY
    # ---------------------------
    elif option == "Summary":
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include='all'))

    # ---------------------------
    # MISSING VALUES
    # ---------------------------
    elif option == "Missing Values":
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing)

        st.bar_chart(missing)

    # ---------------------------
    # VISUALIZATION
    # ---------------------------
    elif option == "Visualizations":
        st.subheader("Create Visualization")

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        chart_type = st.selectbox("Select Chart Type", [
            "Histogram", "Boxplot", "Correlation Heatmap"
        ])

        if chart_type == "Histogram":
            col = st.selectbox("Select Column", numeric_cols)

            fig, ax = plt.subplots()
            ax.hist(df[col])
            st.pyplot(fig)

        elif chart_type == "Boxplot":
            col = st.selectbox("Select Column", numeric_cols)

            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)

        elif chart_type == "Correlation Heatmap":
            fig, ax = plt.subplots()
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, ax=ax)
            st.pyplot(fig)

    # ---------------------------
    # Q&A (AI)
    # ---------------------------
    elif option == "Ask Questions":
        st.subheader("Ask Questions About Data")

        question = st.text_input("Enter your question")

        if st.button("Ask"):
            if question:
                client = OpenAI()

                sample = df.head(50).to_string()

                prompt = f"""
                You are a data analyst.

                Dataset sample:
                {sample}

                Question:
                {question}

                Answer clearly.
                """

                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": prompt}]
                )

                st.write(response.choices[0].message.content)
