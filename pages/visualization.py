import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Model Visualization",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Model Visualization and Analysis")
st.write("This page shows various visualizations and analysis of the cardiovascular disease prediction model.")

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('cleaned_cardio_data.csv')
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load model
@st.cache_resource
def load_model():
    try:
        import joblib
        model = joblib.load('final_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load data and model
data = load_data()
model = load_model()

if data is not None and model is not None:
    # Feature Importance
    st.header("Feature Importance")
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': data.columns[:-1],  # Exclude target column
            'Importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Feature Importance in the Model')
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Distribution
    st.header("Data Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution
        fig = px.histogram(data, 
                          x='age',
                          color='cardio',
                          title='Age Distribution by Cardiovascular Disease',
                          labels={'age': 'Age', 'cardio': 'Cardiovascular Disease'},
                          color_discrete_sequence=['#00CC96', '#EF553B'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender Distribution
        gender_counts = data.groupby(['gender', 'cardio']).size().reset_index(name='count')
        fig = px.bar(gender_counts,
                    x='gender',
                    y='count',
                    color='cardio',
                    title='Gender Distribution by Cardiovascular Disease',
                    labels={'gender': 'Gender', 'count': 'Count', 'cardio': 'Cardiovascular Disease'},
                    color_discrete_sequence=['#00CC96', '#EF553B'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Matrix
    st.header("Feature Correlations")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix,
                   labels=dict(color="Correlation"),
                   title="Correlation Matrix of Features",
                   color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)
    
    # Blood Pressure Analysis
    st.header("Blood Pressure Analysis")
    fig = px.scatter(data,
                    x='ap_hi',
                    y='ap_lo',
                    color='cardio',
                    title='Systolic vs Diastolic Blood Pressure',
                    labels={'ap_hi': 'Systolic BP', 'ap_lo': 'Diastolic BP'},
                    color_discrete_sequence=['#00CC96', '#EF553B'])
    st.plotly_chart(fig, use_container_width=True)
    
    # BMI Analysis
    st.header("BMI Analysis")
    data['bmi'] = data['weight'] / ((data['height']/100) ** 2)
    fig = px.box(data,
                x='cardio',
                y='bmi',
                title='BMI Distribution by Cardiovascular Disease',
                labels={'cardio': 'Cardiovascular Disease', 'bmi': 'BMI'},
                color='cardio',
                color_discrete_sequence=['#00CC96', '#EF553B'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk Factors
    st.header("Risk Factors Analysis")
    risk_factors = ['smoke', 'alco', 'active']
    risk_data = data[risk_factors + ['cardio']].melt(id_vars=['cardio'],
                                                    var_name='Risk Factor',
                                                    value_name='Value')
    
    fig = px.bar(risk_data,
                x='Risk Factor',
                y='Value',
                color='cardio',
                title='Risk Factors Distribution',
                labels={'Value': 'Count'},
                color_discrete_sequence=['#00CC96', '#EF553B'])
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Please make sure both the data file and model file are available.")