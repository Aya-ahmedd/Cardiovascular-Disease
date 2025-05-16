import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Cardiovascular Disease Visualization",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Cardiovascular Disease Data Visualization")
st.write("""
Explore the relationships between different factors and cardiovascular disease through interactive visualizations.
""")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_cardio_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Age range filter
    age_range = st.sidebar.slider(
        "Select Age Range",
        min_value=int(df['age'].min()),
        max_value=int(df['age'].max()),
        value=(int(df['age'].min()), int(df['age'].max()))
    )
    
    # Gender filter
    gender_filter = st.sidebar.multiselect(
        "Select Gender",
        options=["Male", "Female"],
        default=["Male", "Female"]
    )
    
    # Apply filters
    filtered_df = df[
        (df['age'] >= age_range[0]) &
        (df['age'] <= age_range[1]) &
        (df['gender'].map({1: "Male", 2: "Female"}).isin(gender_filter))
    ]
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Correlation Analysis", "Risk Factors"])
    
    with tab1:
        st.header("Distribution Analysis")
        
        # Age distribution
        fig_age = px.histogram(
            filtered_df,
            x='age',
            color='cardio',
            title='Age Distribution by Cardiovascular Disease Status',
            labels={'age': 'Age', 'cardio': 'Cardiovascular Disease'},
            color_discrete_map={0: 'green', 1: 'red'}
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
        # BMI distribution
        fig_bmi = px.box(
            filtered_df,
            x='cardio',
            y='bmi',
            title='BMI Distribution by Cardiovascular Disease Status',
            labels={'cardio': 'Cardiovascular Disease', 'bmi': 'BMI'}
        )
        st.plotly_chart(fig_bmi, use_container_width=True)
    
    with tab2:
        st.header("Correlation Analysis")
        
        # Correlation matrix
        numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title='Correlation Matrix of Numeric Features',
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter plot
        x_axis = st.selectbox("Select X-axis", numeric_cols, index=0)
        y_axis = st.selectbox("Select Y-axis", numeric_cols, index=1)
        
        fig_scatter = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color='cardio',
            title=f'{x_axis} vs {y_axis} by Cardiovascular Disease Status',
            color_discrete_map={0: 'green', 1: 'red'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.header("Risk Factors Analysis")
        
        # Create age groups with more granular bins
        filtered_df['age_group'] = pd.cut(
            filtered_df['age'],
            bins=[0, 30, 40, 50, 60, 70, 80, 100],
            labels=['0-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
        )
        
        # Calculate risk factors by age group
        risk_factors = ['smoke', 'alco', 'active']
        
        # Create a more detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk factors by age group (stacked bar chart)
            risk_by_age = filtered_df.groupby(['age_group', 'cardio'])[risk_factors].mean().reset_index()
            risk_by_age_melted = pd.melt(
                risk_by_age,
                id_vars=['age_group', 'cardio'],
                value_vars=risk_factors,
                var_name='Risk Factor',
                value_name='Percentage'
            )
            
            fig_risk = px.bar(
                risk_by_age_melted,
                x='age_group',
                y='Percentage',
                color='Risk Factor',
                barmode='group',
                title='Risk Factors Distribution by Age Group',
                labels={
                    'age_group': 'Age Group',
                    'Percentage': 'Percentage of Population',
                    'Risk Factor': 'Risk Factor'
                },
                color_discrete_map={
                    'smoke': '#FF9999',
                    'alco': '#66B2FF',
                    'active': '#99FF99'
                }
            )
            fig_risk.update_layout(
                xaxis_title="Age Group",
                yaxis_title="Percentage",
                legend_title="Risk Factor"
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            # Disease prevalence by age group
            disease_by_age = filtered_df.groupby('age_group')['cardio'].mean().reset_index()
            fig_disease = px.bar(
                disease_by_age,
                x='age_group',
                y='cardio',
                title='Cardiovascular Disease Prevalence by Age Group',
                labels={
                    'age_group': 'Age Group',
                    'cardio': 'Disease Prevalence'
                },
                color='cardio',
                color_continuous_scale='Reds'
            )
            fig_disease.update_layout(
                xaxis_title="Age Group",
                yaxis_title="Disease Prevalence",
                showlegend=False
            )
            st.plotly_chart(fig_disease, use_container_width=True)
        
        # Add detailed statistics
        st.subheader("Detailed Risk Factor Statistics")
        
        # Calculate statistics for each age group
        stats_df = filtered_df.groupby('age_group').agg({
            'smoke': ['mean', 'count'],
            'alco': ['mean', 'count'],
            'active': ['mean', 'count'],
            'cardio': 'mean'
        }).round(3)
        
        # Format the statistics
        stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
        stats_df = stats_df.reset_index()
        
        # Display the statistics in a more readable format
        st.write("""
        The table below shows the percentage of people in each age group who:
        - Smoke
        - Consume alcohol
        - Are physically active
        - Have cardiovascular disease
        """)
        
        # Format the statistics for display
        display_stats = pd.DataFrame({
            'Age Group': stats_df['age_group'],
            'Smoking Rate': (stats_df['smoke_mean'] * 100).round(1).astype(str) + '%',
            'Alcohol Consumption': (stats_df['alco_mean'] * 100).round(1).astype(str) + '%',
            'Physical Activity': (stats_df['active_mean'] * 100).round(1).astype(str) + '%',
            'Disease Prevalence': (stats_df['cardio_mean'] * 100).round(1).astype(str) + '%',
            'Sample Size': stats_df['smoke_count']
        })
        
        st.dataframe(display_stats, use_container_width=True)

# Add footer
st.write("---")
st.write("Developed for Cardiovascular Disease Analysis") 