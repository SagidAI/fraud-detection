# 🔍 Credit Card Fraud Detection

## Problem
Banks lose billions of dollars annually due to fraudulent transactions. 
This project builds a machine learning model to detect fraud in real-time.

## Demo
[🚀 Live App](https://fraud-detection-sagid.streamlit.app)

## Technologies Used
- Python
- XGBoost
- Streamlit
- Pandas & NumPy
- Scikit-learn

## Model Performance
- Accuracy: 99.9%
- Recall: 83%
- Fraud Probability Threshold: 0.3

## Key Insights
- V14 is the most important feature for fraud detection
- Dataset is highly imbalanced (492 fraud vs 284,315 normal)
- Used scale_pos_weight to handle imbalanced data

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
[Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Author
Sagid Elhag - Machine Learning Engineer