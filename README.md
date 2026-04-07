# Predicting Demand for Perishable Goods

## Business Problem
Retailers dealing with perishable goods face a constant trade-off: order too little and they lose sales, order too much and they increase spoilage and waste. To make better stock decisions, they need a clearer view of how demand changes across product categories, pricing, promotions, stores and external conditions.

## Objective
The objective of this project is to analyse historical perishable-goods sales data and identify the main factors affecting demand, wastage and stock efficiency. The project also explores simple forecasting logic that could support better ordering decisions.

## Data Used
The analysis uses multiple structured datasets to build a fuller view of demand:

- weekly sales data
- product details
- store information
- supplier information
- weather and holiday data

These datasets allow demand to be analysed in the context of price, marketing, shelf life, store characteristics and external conditions.

## Approach
The project follows a practical analytics workflow:

1. **Data loading and validation**
   - Loaded the datasets into Python
   - Checked for missing values, duplicates and incorrect data types
   - Joined the datasets into a single analysis-ready view

2. **Exploratory analysis**
   - Analysed sales trends over time
   - Compared performance across product categories and locations
   - Examined wastage patterns on short shelf-life products

3. **Driver analysis**
   - Explored the relationship between price and units sold
   - Reviewed how marketing spend related to demand
   - Looked at the effect of weather and holidays on weekly sales

4. **Forecast exploration**
   - Used simple baseline forecasting logic to estimate future demand
   - Assessed how demand patterns could support better stock planning

## Tools
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Key Results
- Demand varied significantly by product category, with some categories showing more consistent weekly patterns than others.  
  **Business decision:** Stock planning should be category-specific rather than using one blanket ordering approach.

- Higher prices generally aligned with lower sales volumes, though the strength of that relationship differed by category.  
  **Business decision:** Pricing changes should be targeted, because not all products respond in the same way to price movement.

- Marketing spend appeared to support demand up to a point, but the relationship was not unlimited.  
  **Business decision:** Promotional spend should be allocated selectively rather than increased uniformly.

- Short shelf-life categories showed higher wastage risk when demand was overestimated.  
  **Business decision:** These products need closer forecasting attention and tighter replenishment logic.

## Business Impact
This project shows how operational and commercial data can support better inventory decisions in a retail setting. The value is not just in describing sales patterns, but in helping a business decide:

- what to stock more carefully
- where demand is predictable
- where pricing needs closer review
- which products carry the highest waste risk

In practice, this kind of analysis can help reduce spoilage, improve stock availability, and support more disciplined demand planning.
