## Exploratory Data Analysis (EDA) Report

This project performs an in-depth exploratory data analysis (EDA) on transaction and behavioral data to uncover risk patterns relevant to credit scoring.

### Steps Performed

1. **Data Loading**  
   The dataset is loaded from `data/data.csv` using a custom utility.

2. **Dataset Overview**

   - The shape, columns, data types, and a preview of the first few rows are displayed.
   - This provides an initial understanding of the dataset structure.

3. **Summary Statistics**

   - Descriptive statistics (mean, std, min, max, quartiles) are generated for numerical features.
   - Helps identify data ranges, central tendencies, and potential anomalies.

4. **Numerical Features Distribution**

   - Distributions of key numerical columns (e.g., `Amount`, `Value`) are visualized.
   - Histograms and KDE plots reveal skewness, modality, and outliers.

5. **Categorical Features Distribution**

   - Count plots for categorical variables (e.g., `CurrencyCode`, `CountryCode`, `ProviderId`, etc.) show class balance and frequency.

6. **Correlation Analysis**

   - A heatmap visualizes correlations between numerical features.
   - Useful for detecting multicollinearity and relationships between variables.

7. **Missing Value Analysis**

   - The amount and pattern of missing data are assessed.
   - Visualizations help identify if missingness is systematic or random.

8. **Outlier Detection**
   - Boxplots are used to detect outliers in numerical columns.
   - Outliers are flagged for further investigation or preprocessing.

### Key Insights

- The dataset contains both numerical and categorical features relevant to credit risk.
- Visualizations highlight the distribution and relationships of features.
- Missing values and outliers are present and require appropriate handling in subsequent modeling steps.
