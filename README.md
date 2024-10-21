# Project Title: Insurance-Claim-Classification
This project is a Text Classification model to predict Coverage Code and Accident Source from the Claim Description

# Project Description
This repository contains a web application for predicting Coverage Code and Accident Source using machine learning techniques. Built with Streamlit, the application allows users to upload datasets and select from various models to generate predictions. The app evaluates model performance and visualizes results, providing an intuitive interface for users to interact with predictive analytics in the insurance domain.

# Problem statement
In this classification problem, you need address the following using data science techniques:
1.	Address class imbalance issue and select the best technique.
2.	Create a model to predict “Coverage Code” & “Accident Source”.
3.	Design a GUI that take the dataset file as input and will have a “Run” button to execute the model you created.
4.	After executing the model, the GUI will summarize the evaluation results on the screen and store the excel file in a folder.

# Dataset Description
The dataset contains 190,000+ claim records with only one feature i.e., Claim description. The target columns are Coverage Code and Accident Source.

# Features
Model Training: Supports Random Forest and XGBoost algorithms.

User-Friendly Interface: Simple file upload and model selection process.

Performance Metrics: Displays precision and recall for both training and testing datasets.

Visualizations: Bar charts representing model performance metrics.

Export Functionality: Save results in Excel format for further analysis.


# Technologies Used
Python: The main programming language.

Streamlit: For creating the web interface.

Pandas: For data manipulation and handling.

Scikit-learn: For machine learning model training and evaluation.

XGBoost: For the gradient boosting model.

Matplotlib: For data visualization.


