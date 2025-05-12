# Customer Churn Prediction Using Big Data and Machine Learning

This project focuses on predicting customer churn using machine learning algorithms applied to a large-scale version of the IBM Telco Customer Churn dataset. The goal is to identify customers at risk of leaving a service and support business decision-making for retention strategies.

## 📂 Project Structure

```
.
├── data/                   # Data files (original and expanded)
├── Preprocessing and EDA/  # Code for preprocessing and EDA
├── models/                 # Saved model files
├── src/                    # Source code for training and evaluation
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── Customer_Churn_Prediction_with_Big_Data.docx  # Final report
```

## 🚀 Features

- Data preprocessing with missing value handling, encoding, outlier removal
- Feature engineering from demographics, service usage, and billing
- Model training using:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Hyperparameter tuning with Grid Search
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC
- Scalable data augmentation to simulate big data

## 📊 Dataset

- **Source**: IBM Sample Dataset [Telco Customer Churn](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)
- **Size**: Expanded from ~7,000 to ~1,000,000 rows

## 🛠️ Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/churn-prediction-bigdata.git
   cd churn-prediction-bigdata
   ```

2. Run preprocessing:
   ```bash
   python src/preprocessing.py
   ```

3. Train model:
   ```bash
   python src/train_model.py --model xgboost
   ```

4. Evaluate model:
   ```bash
   python src/evaluate_model.py --model xgboost
   ```

## 📄 License

This project is for educational and research use only. Please cite appropriately if reused in publications.

## ✍️ Author

Ke Xu – University of Maryland, College Park  
Email: kxu233@umd.edu
