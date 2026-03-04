# World Bank Country Clustering Visualizing App

A Streamlit web application that performs unsupervised machine learning (Clustering) on global development indicators. The app fetches data from the World Bank API and visualizes how countries group together based on socio-economic features.

Vibe coded with Antigravity.

## Features

- **World Bank API Integration**: Fetches development indicators dynamically using `wbgapi`.
- **Socio-Economic Indicators**: Choose from a predefined list of features including GDP per capita, Life Expectancy, Infant Mortality Rate, Literacy Rate, CO2 Emissions, Population, Gini Index, and more.
- **Clustering Algorithms**:
  - K-Means
  - Gaussian Mixture Model (GMM)
- **Interactive Visualizations**:
  - **Choropleth World Map**: Visualizes the cluster assignments for each country.
  - **Interactive Scatter Plots**: 2D scatter plots of the clusters. Utilizes PCA for dimensionality reduction if more than two features are selected.
- **Parameter Tuning**: Adjust the number of clusters ($k$) and GMM covariance types.
- **Historical Data**: Analyze historical data by selecting any year from 1960 to 2023.

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NarezIn/clustering-visual-FML.git
   cd clustering-visual-FML
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     .venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:
   Make sure you have the required packages installed.
   ```bash
   pip install streamlit pandas numpy wbgapi plotly scikit-learn
   ```

### Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The application will open automatically in your web browser.

## Usage

1. **Configuration (Left Panel)**: 
   - Select your desired socio-economic indicators.
   - Choose the year for the data.
   - Select the clustering algorithm (K-Means or GMM) and configure its hyperparameters (like the number of clusters).
2. **Visualization & Analysis (Right Panel)**: 
   - Explore the generated Choropleth map to see global clustering patterns.
   - Use the scatter plot to understand how the features correlate and separate into clusters.
   - Expand the "Data Table" section to view the raw data and exact cluster assignments for each country.

## Project Structure

- `app.py`: Main Streamlit application and entry point.
- `plan.md`: The development plan and technical requirements used for building the app.

## License

This project is licensed under the MIT License.
