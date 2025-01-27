People Analytics: Well-Being and Turnover Analysis
Overview

This repository contains Python scripts, datasets, and insights derived from analyzing employee well-being and turnover at FAU Clinic. It leverages machine learning and data visualization to identify factors influencing work-life balance and voluntary employee turnover, providing actionable recommendations for improving organizational practices.
Table of Contents

    Project Description
    Code Overview
        Well-Being Analysis
        Turnover Analysis
    Key Results
        Well-Being
        Turnover
    Technologies Used
    How to Run the Code
    Acknowledgments

Project Description

The project investigates:

    Employee Well-Being: Understanding the factors influencing work-life balance (WLB) and daily stress using statistical analysis and predictive modeling.
    Employee Turnover: Identifying drivers of voluntary turnover through theoretical models and predictive analytics.

Code Overview
Well-Being Analysis

The wellbeing.py script analyzes the FAU Clinic Employee Well-Being Dataset to:

    Map categorical variables like AGE and GENDER to numeric values.
    Visualize trends such as daily stress by gender and age.
    Compute a correlation matrix for features related to WORK_LIFE_BALANCE_SCORE.
    Train a Linear Regression model to predict WLB scores with an R² score of 0.853.
    Evaluate model performance using scatter plots and summary statistics.

Turnover Analysis

The turnover.py script analyzes the FAU Clinic Turnover Dataset to:

    Explore patterns such as job satisfaction and salary distribution by role.
    Preprocess data by encoding categorical variables and creating combined features.
    Visualize correlations with turnover (left) and compute key metrics (e.g., satisfaction levels of employees who left).
    Train a Random Forest Classifier, achieving 99% accuracy.
    Analyze feature importance, highlighting significant turnover drivers.

Key Results
Well-Being

    Top Correlated Factors:
        Achievement, to-do completion, and time for hobbies showed high correlation with WLB.
    Daily Stress Trends:
        Females reported slightly higher stress than males.
        Age group 36–50 exhibited the highest daily stress levels.
    Predictive Model:
        The Linear Regression model demonstrated strong predictive capabilities for WLB scores.

Turnover

    Key Findings:
        Employees in high-stress roles with low satisfaction were more likely to leave.
        The average time spent at FAU Clinic before leaving was 4 years.
    Predictive Model:
        The Random Forest Classifier provided reliable predictions for employee turnover.
        Feature importance highlighted dissatisfaction and workload as primary turnover drivers.

Technologies Used

    Programming Language: Python
    Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Statsmodels
    Tools: Jupyter Notebook, Visualization APIs
