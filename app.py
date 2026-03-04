import streamlit as st
import pandas as pd
import numpy as np
import wbgapi as wb
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# 3. Pre-defined Indicator Dictionary
INDICATORS = {
    "GDP per capita (Current US$)": "NY.GDP.PCAP.CD",
    "GNI per capita (PPP)": "NY.GNP.PCAP.PP.CD",
    "Life Expectancy at Birth": "SP.DYN.LE00.IN",
    "Infant Mortality Rate": "SP.DYN.IMRT.IN",
    "Literacy Rate (Adult)": "SE.ADT.LITR.ZS",
    "School Enrollment (Secondary %)": "SE.SEC.ENRR",
    "CO2 Emissions (Metric Tons per capita)": "EN.ATM.CO2E.PC",
    "Population, total": "SP.POP.TOTL",
    "Gini Index (Income Inequality)": "SI.POV.GINI"
}

st.set_page_config(layout="wide", page_title="World Bank Country Clustering")

st.title("World Bank Country Clustering App")

# 4. UI Architecture (Split Screen Layout)
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configuration & Hyperparameters")
    
    # Feature Selection
    selected_features_names = st.multiselect(
        "Select Indicators (Features):",
        options=list(INDICATORS.keys()),
        default=["GDP per capita (Current US$)", "Life Expectancy at Birth"]
    )
    
    # Year Selection
    year = st.slider("Select Year:", min_value=1960, max_value=2023, value=2022)
    
    # Algorithm Selection
    algo = st.radio("Select Algorithm:", options=["K-Means", "Gaussian Mixture Model (GMM)"])
    
    # Hyperparameters
    k = st.slider("Number of Clusters / Components (k):", min_value=2, max_value=12, value=4)
    
    covariance_type = "full"
    if algo == "Gaussian Mixture Model (GMM)":
        covariance_type = st.selectbox(
            "Covariance Type:", 
            options=["spherical", "tied", "diag", "full"], 
            index=3
        )

with col2:
    st.header("Visualization & Analysis")
    
    if not selected_features_names:
        st.warning("Please select at least one indicator to proceed.")
        st.stop()
        
    # Get Indicator IDs
    selected_indicators = [INDICATORS[name] for name in selected_features_names]
    
    with st.spinner("Fetching data from World Bank API..."):
        try:
            # 1. Data Retrieval
            df_raw = wb.data.DataFrame(selected_indicators, time=year, economy='all')
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
            
    if df_raw.empty:
        st.warning("No data returned for the selected year and indicators.")
        st.stop()

    try:
        economies_df = wb.economy.DataFrame()
        # Filter out regional aggregates directly based on regions not being empty
        countries = economies_df[economies_df['region'] != ''].index.tolist()
        df = df_raw.loc[df_raw.index.intersection(countries)].copy()
    except Exception as e:
        # Fallback if economy list fetching fails
        df = df_raw.copy()

    rename_mapping = {ind_code: ind_name for ind_name, ind_code in INDICATORS.items()}
    df.rename(columns=rename_mapping, inplace=True)
    
    # 3. Preprocessing
    available_features = [f for f in selected_features_names if f in df.columns]
    
    if not available_features:
        st.warning("No data retrieved for the selected features in the selected year.")
        st.stop()
        
    X = df[available_features].copy()
    
    # Drop columns that are completely NaN
    X.dropna(axis=1, how='all', inplace=True)
    available_features = X.columns.tolist()
    
    if not available_features:
        st.warning("All selected features have no data for the selected year.")
        st.stop()
        
    # Handle missing values using median imputation
    X = X.fillna(X.median())
    
    # Drop any rows that still have NaNs
    X.dropna(axis=0, how='any', inplace=True)
    
    if X.empty:
        st.warning("No countries have sufficient data for the selected features and year.")
        st.stop()
        
    # Align df with the cleaned X
    df = df.loc[X.index].copy()
    
    # Mandatory StandardScaler before clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Execution
    if algo == "K-Means":
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)
    else:  # GMM
        model = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42)
        labels = model.fit_predict(X_scaled)
        
    df['Cluster'] = labels.astype(str)

    # 5. Mapping
    df_plot = df.reset_index().rename(columns={'economy': 'Country Code', 'index': 'Country Code'})
    
    if 'name' in economies_df.columns:
        df_plot['Country Name'] = df_plot['Country Code'].map(economies_df['name'])
    else:
        df_plot['Country Name'] = df_plot['Country Code']
        
    # Standardizing cluster names by adding 'Cluster ' prefix (optional, maybe cleaner on legend)
    df_plot['Cluster_Label'] = "Cluster " + df_plot['Cluster']
    df_plot = df_plot.sort_values("Cluster_Label")
    
    color_seq = px.colors.qualitative.Plotly
    
    map_fig = px.choropleth(
        df_plot,
        locations="Country Code",
        color="Cluster_Label",
        hover_name="Country Name",
        hover_data=available_features,
        title=f"Country Clusters ({year})",
        color_discrete_sequence=color_seq
    )
    map_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(map_fig, use_container_width=True)
    
    # Cluster Scatter Plot
    if len(available_features) == 2:
        scatter_fig = px.scatter(
            df_plot,
            x=available_features[0],
            y=available_features[1],
            color="Cluster_Label",
            hover_name="Country Name",
            title="2D Scatter Plot of Clusters",
            color_discrete_sequence=color_seq
        )
    elif len(available_features) > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df_plot['PCA 1'] = pca_result[:, 0]
        df_plot['PCA 2'] = pca_result[:, 1]
        
        scatter_fig = px.scatter(
            df_plot,
            x='PCA 1',
            y='PCA 2',
            color="Cluster_Label",
            hover_name="Country Name",
            title="PCA 2D Projection of Clusters",
            color_discrete_sequence=color_seq
        )
    else:
        # Fallback to 1D scatter with jitter
        df_plot['Y_Jitter'] = np.random.normal(0, 0.1, size=len(df_plot))
        scatter_fig = px.scatter(
            df_plot,
            x=available_features[0],
            y="Y_Jitter",
            color="Cluster_Label",
            hover_name="Country Name",
            title="1D Scatter Plot of Clusters",
            color_discrete_sequence=color_seq
        )
        scatter_fig.update_yaxes(visible=False, showticklabels=False)
    
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    with st.expander("Data Table (Assigned Clusters)"):
        display_cols = ['Country Name', 'Country Code', 'Cluster_Label'] + available_features
        st.dataframe(df_plot[display_cols], hide_index=True)
