# Predicting Demand for Perishable Goods – Forecasting & Pricing Analysis

## Short Overview
This project analyses perishable goods sales data to forecast future demand and understand how price changes affect sales. It brings together historical sales, product attributes, store information, and external factors to build predictive models and visualise demand patterns.

## Business Problem
Retailers dealing with perishable goods face a constant trade‑off: order too little and they miss sales, order too much and they increase spoilage and waste. Effective inventory and pricing decisions require a clear understanding of how demand responds to price, promotions, store characteristics and external conditions such as holidays or weather.

## Solution / Project Purpose
The project builds forecasting models and analyses price elasticity to provide decision makers with actionable insights. It combines data cleaning, exploratory analysis and statistical modelling to predict demand and reveal key drivers, enabling retailers to plan inventory and pricing strategies that reduce waste and maximise revenue.

## Key Features
- Uses multiple structured datasets including sales history, product details, store information and external factors.
- Creates time‑series forecasts of demand at product‑category and store level.
- Analyses price elasticity of demand to understand how price changes impact sales.
- Generates visualisations to highlight demand trends and price‑sales relationships.
- Provides a reproducible workflow in Python with notebooks and scripts.

## Tools and Technologies
- Python with Pandas and NumPy for data manipulation.
- Statsmodels and Scikit‑learn for statistical modelling and machine learning.
- Matplotlib and Seaborn for visualisation.
- Jupyter Notebook for interactive analysis.

## Workflow / Method
1. **Data loading and validation:** Load and merge weekly sales, product details, store and holiday data.
2. **Exploratory analysis:** Analyse demand patterns by product, store and time; assess price distributions and relationships.
3. **Feature engineering:** Create lagged variables, moving averages and holiday indicators to capture demand drivers.
4. **Model building:** Train time‑series models and regression models to forecast demand and estimate price elasticity.
5. **Evaluation:** Compare model performance using appropriate error metrics and visual diagnostics.
6. **Visualisation:** Produce charts to communicate demand trends and price vs sales relationships.

## Key Insights or Outcomes
The analysis highlights seasonal patterns and product‑level trends in perishable goods demand. Price sensitivity varies across categories; certain products show strong elasticity while others are less responsive. These insights help optimise ordering policies and promotional strategies.

## Business or Practical Impact
By forecasting demand and understanding price elasticity, retailers can align inventory with expected sales and adjust pricing to drive revenue. This reduces waste from over‑ordering, prevents lost sales due to stockouts and supports data‑driven decision making in pricing and promotions.

## Visual Outputs
### Demand Trend
<p align="center">
  <img src="outputs/demand_trend_chart.png" width="700"/>
</p>

### Price vs Sales
<p align="center">
  <img src="outputs/price_vs_sales_chart.png" width="700"/>
</p>

## Project Structure
```
predicting_demand_for_perishable_goods/
├── data/                  # Raw datasets (not included here)
├── notebooks/             # Jupyter notebooks for analysis and modelling
├── outputs/
│   ├── demand_trend_chart.png
│   ├── price_vs_sales_chart.png
│   └── validation_predictions.csv
├── src/                   # Python scripts for loading data, training models and evaluation
├── tools/                 # Utility scripts or helper functions
└── README.md
```

## How to Run or View
1. Clone the repository and install the dependencies listed in `requirements.txt`.
2. Open the notebooks in the `notebooks` folder to follow the analysis and model building process.
3. Run the Python scripts in `src` to reproduce the forecasts and elasticity estimates.
4. View the charts saved in the `outputs` folder for quick insights.

## Future Improvements
- Incorporate external factors such as weather data, promotions and marketing campaigns to improve forecasts.
- Explore more advanced forecasting techniques like Prophet or gradient boosted trees.
- Build an interactive dashboard to allow stakeholders to explore demand scenarios and pricing impacts.
