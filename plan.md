# Implementation Plan: World Bank Country Clustering App

## 1. Project Overview
Build a Streamlit web application that performs unsupervised machine learning (Clustering) on global development indicators. The app will fetch data from the World Bank API and visualize how countries group together based on socio-economic features.

## 2. Technical Requirements
-**Environment** python virtual environment
- **Language:** Python 3.x
- **UI Framework:** `streamlit` (Two-column/panel layout)
- **Data Source:** `wbgapi` (World Bank Integration)
- **Machine Learning:** `scikit-learn` (Clustering & Preprocessing)
- **Visualization:** `plotly` (Choropleth maps and Scatter plots)
- **Data Handling:** `pandas`, `numpy`

## 3. Pre-defined Indicator Dictionary
The agent must use this mapping for the feature selection tool:
- **GDP per capita (Current US$):** `NY.GDP.PCAP.CD`
- **GNI per capita (PPP):** `NY.GNP.PCAP.PP.CD`
- **Life Expectancy at Birth:** `SP.DYN.LE00.IN`
- **Infant Mortality Rate:** `SP.DYN.IMRT.IN`
- **Literacy Rate (Adult):** `SE.ADT.LITR.ZS`
- **School Enrollment (Secondary %):** `SE.SEC.ENRR`
- **CO2 Emissions (Metric Tons per capita):** `EN.ATM.CO2E.PC`
- **Population, total:** `SP.POP.TOTL`
- **Gini Index (Income Inequality):** `SI.POV.GINI`

## 4. UI Architecture (Split Screen Layout)
The app must be divided into two distinct sections using `st.columns([1, 2])` or a sidebar:

### Left Panel: Configuration & Hyperparameters
- **Feature Selection:** Multi-select dropdown using the Indicator Names above.
- **Year Selection:** Slider for the data year (Default: 2022).
- **Algorithm Selection:** Radio buttons to choose:
    - `K-Means`
    - `Gaussian Mixture Model (GMM)`
- **Hyperparameters:**
    - Slider for `k` (Number of Clusters / Components), range 2-12.
    - If GMM: Dropdown for `covariance_type` (spherical, tied, diag, full).

### Right Panel: Visualization & Analysis
- **World Map:** Plotly Choropleth map coloring countries by their assigned Cluster ID.
- **Cluster Scatter Plot:** - If 2 features are selected: 2D Scatter Plot of those features.
    - If >2 features are selected: Use PCA to reduce data to 2 dimensions for the plot.
- **Data Table:** Expandable section showing the processed dataframe and Cluster assignments.

## 5. Implementation Logic
1. **Data Retrieval:** Use `wbgapi.data.DataFrame(indicators, time=year, economy='all')`.
2. **Filtering:** Filter out regional aggregates (e.g., "World", "High Income") to focus only on individual countries.
3. **Preprocessing:** - Handle missing values using median imputation.
    - **Mandatory:** Use `StandardScaler` on all features before clustering.
4. **Execution:** Run the selected `scikit-learn` model based on the left-panel inputs.
5. **Mapping:** Join cluster labels back to the country names/codes for Plotly visualization.

## 6. Constraints & Safety
- Do not calculate or display evaluation metrics (like Silhouette Score).
- Handle cases where no data is returned for a specific year/indicator combination with a `st.warning`.
- Use a consistent color palette for clusters across both the map and the scatter plot.