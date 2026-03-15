"""
Lending Club Loan Analysis - Streamlit Dashboard
Run with: streamlit run streamlit_app/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Lending Club Loan Analysis",
    page_icon="💰",
    layout="wide",
)

st.title("Lending Club Loan Default Analysis")
st.markdown(
    """
    This dashboard presents our analysis of the Lending Club loan dataset (2007-2018).
    Use the sidebar to navigate between different sections.
    """
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Feature Analysis", "Model Performance", "Prediction"],
)

if page == "Overview":
    st.header("Project Overview")
    st.markdown(
        """
        ### Dataset
        - **Source**: Lending Club (2007 - 2018 Q4)
        - **Accepted loans**: ~2.26 million records, 150+ features
        - **Rejected loans**: ~27.6 million records

        ### Objective
        Predict whether a borrower will **default** on their loan based on
        information available at the time of loan application.

        ### Approach
        1. Exploratory Data Analysis (EDA)
        2. Feature Engineering & Selection
        3. Baseline Models (Logistic Regression, Decision Tree)
        4. Advanced Models (XGBoost, LightGBM)
        5. Model Interpretation (SHAP)
        """
    )

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    st.info("EDA visualizations will be added here after notebook analysis is complete.")

elif page == "Feature Analysis":
    st.header("Feature Analysis")
    st.info("Feature importance and SHAP analysis will be displayed here.")

elif page == "Model Performance":
    st.header("Model Performance Comparison")
    st.info("Model comparison metrics and charts will be shown here.")

elif page == "Prediction":
    st.header("Loan Default Prediction")
    st.info("Interactive prediction interface will be available after model training.")
