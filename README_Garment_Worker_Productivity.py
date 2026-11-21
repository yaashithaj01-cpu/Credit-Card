"""
README: Garment Worker Productivity Prediction
==============================================

PROJECT OVERVIEW
----------------
This project uses the Garments Worker Productivity dataset to explore
production-related information and build machine learning regression models
that estimate actual_productivity.

PROJECT OBJECTIVE
-----------------
- Explore the factors and production characteristics in the dataset.
- Prepare the data for machine learning.
- Train and compare regression models.
- Identify the best-performing model using evaluation metrics.

DATASET
-------
Required file: garments_worker_productivity.csv

Keep the CSV file in the same folder as the Jupyter Notebook.

Target column:
- actual_productivity: the productivity value predicted by the models.

The dataset contains production and workforce information such as date,
quarter, department, team, targeted productivity, standard minute value,
work in progress, overtime, incentives, idle time, number of workers, and
actual productivity.

TECHNOLOGIES AND LIBRARIES
--------------------------
- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

PROJECT WORKFLOW
----------------
1. Import the required Python libraries.
2. Load garments_worker_productivity.csv into a Pandas DataFrame.
3. Explore the data using head(), tail(), info(), shape, columns, and
   descriptive statistics.
4. Clean the data by checking missing values and duplicates, converting
   the date column, handling missing values, and removing rows without
   the target value.
5. Detect possible outliers using boxplots and the IQR method.
6. Create date-based features such as month, day of month, and day of week.
7. Encode categorical columns into numerical features.
8. Separate independent variables (X) and the target variable (y).
9. Split the data into training and testing sets.
10. Scale the input features.
11. Select useful features using SelectKBest and f_regression.
12. Train regression models and generate predictions.
13. Evaluate models using MAE, MSE, RMSE, and R-squared.
14. Compare actual and predicted productivity and inspect model results.

MACHINE LEARNING ALGORITHMS
---------------------------
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Neighbors Regressor (KNN)

EVALUATION METRICS
------------------
- MAE (Mean Absolute Error): average absolute prediction error. Lower is
  better.
- MSE (Mean Squared Error): average squared prediction error. Lower is
  better.
- RMSE (Root Mean Squared Error): square root of MSE. Lower is better.
- R-squared (R2): proportion of target variation explained by the model.
  Higher is generally better.

The notebook selects the model with the lowest RMSE as the best model
for the test set. Actual scores are generated when the notebook is run.

VISUALIZATIONS
--------------
The notebook includes:
- Numeric-feature boxplots for outlier detection.
- Model performance comparison.
- Actual versus predicted productivity plot.
- Residual distribution plot.
- Random Forest feature-importance chart.

HOW TO RUN THE PROJECT
---------------------
1. Place the notebook and garments_worker_productivity.csv in the same
   folder.
2. Install the required packages:

   pip install numpy pandas matplotlib seaborn scikit-learn jupyter

3. Start Jupyter Notebook:

   jupyter notebook

4. Open the project notebook.
5. Run the notebook cells from top to bottom.

EXPECTED OUTPUT
---------------
After running the notebook, you should obtain:
- A cleaned and prepared dataset.
- Exploratory data analysis and visualizations.
- Selected feature scores.
- Predictions from the regression models.
- A model comparison table with MAE, MSE, RMSE, and R2.
- The best model based on RMSE.
- Actual-versus-predicted and residual visualizations.
- Random Forest feature-importance results.

NOTES AND LIMITATIONS
---------------------
- Results depend on the supplied dataset and train-test split.
- Feature selection is performed as an analysis step; the model-training
  process should be checked to confirm whether selected features are used.
- For a stricter production workflow, fit preprocessing steps only on the
  training data and apply the fitted transformations to the test data to
  reduce data leakage.
- Correlations and feature importance show patterns in the data; they do
  not prove that a variable causes productivity to change.

CONCLUSION
----------
This project demonstrates a machine learning workflow for garment worker
productivity prediction. It explores and prepares the dataset, trains
regression algorithms, and compares their predictions using standard
regression metrics.
"""

# This file contains project documentation as a Python module docstring.
# It is intended to be opened and read, not executed as a machine learning script.
