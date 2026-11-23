# Student Performance Regression

Predicting student exam scores with **Linear Regression** and **Polynomial Regression**, from raw data to model comparison in a single Jupyter notebook.

## Overview

This project analyzes how different factors in a student's life (study habits, attendance, motivation, family background, and more) relate to their final exam score, and then trains two regression models to predict that score.

The notebook walks through the full machine learning workflow step by step, with every decision explained in plain language:

1. **Data Understanding** - load the data, check types, missing values and duplicates.
2. **Exploratory Data Analysis (EDA)** - distributions, boxplots, count plots, a correlation heatmap, scatter plots and category comparisons, each with a written observation.
3. **Data Cleaning & Preprocessing** - remove one invalid exam score (101/100), impute missing categorical values with the mode, one-hot encode categorical features, and scale numerical features. All preprocessing is fit on the training data only to avoid leakage.
4. **Linear Regression** - train, predict and evaluate.
5. **Polynomial Regression (degree 2)** - expand 27 features into 405 polynomial terms, train and evaluate.
6. **Model Comparison** - MAE, MSE, RMSE and R2 side by side, overfitting check (train vs test R2), plus comparison charts, actual-vs-predicted plots and residual plots.

## Dataset

`StudentPerformanceFactors.csv` - 6,607 students, 20 columns (19 features + target).

- **Target:** `Exam_Score` (final exam score)
- **Numerical features:** Hours_Studied, Attendance, Sleep_Hours, Previous_Scores, Tutoring_Sessions, Physical_Activity
- **Categorical features:** Parental_Involvement, Access_to_Resources, Extracurricular_Activities, Motivation_Level, Internet_Access, Family_Income, Teacher_Quality, School_Type, Peer_Influence, Learning_Disabilities, Parental_Education_Level, Distance_from_Home, Gender

Key EDA finding: **Attendance (correlation 0.58)** and **Hours_Studied (0.45)** have the strongest relationship with the exam score, while Sleep_Hours and Physical_Activity are close to zero.

## Results

Evaluated on a 20% held-out test set (1,321 students):

| Model | MAE | MSE | RMSE | R2 |
|---|---|---|---|---|
| **Linear Regression** | **0.416** | **2.314** | **1.521** | **0.825** |
| Polynomial Regression (degree 2) | 0.632 | 2.651 | 1.628 | 0.800 |

**Linear Regression wins.** The relationship between the features and the exam score is mostly linear, so the 405 polynomial terms added complexity without accuracy, and showed signs of overfitting (higher train R2, lower test R2). On average the linear model's prediction is off by only ~0.42 score points and explains ~82.5% of the variance in exam scores.

## Project structure

```
├── Student_Performance_Regression.ipynb   # the full analysis, modeling and results
├── StudentPerformanceFactors.csv          # the dataset
└── README.md
```

## How to run

1. Make sure Python 3 is installed, then install the requirements:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

2. Clone this repository and open the notebook:

   ```bash
   git clone https://github.com/rahmahesham3/student-performance-regression.git
   cd student-performance-regression
   jupyter notebook Student_Performance_Regression.ipynb
   ```

3. Run the cells top to bottom. The notebook expects `StudentPerformanceFactors.csv` in the same folder (it is included in this repo).

## Requirements

- Python 3
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook (or JupyterLab / VS Code with the Jupyter extension)
