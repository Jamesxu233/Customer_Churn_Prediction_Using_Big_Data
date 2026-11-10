# Customer Churn Prediction Using Big Data and Machine Learning

This project focuses on predicting customer churn using machine learning algorithms applied to a large-scale version of the IBM Telco Customer Churn dataset. The goal is to identify customers at risk of leaving a service and support business decision-making for retention strategies.

**Author:** Ke Xu  
**Course:** Big Data Analysis  
**Institution:** University of Maryland, College Park  
**Language:** Python

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
- Run EDA.py and Feature_Importance_Analysis.py to get feature insightful plots

## 📊 Dataset

- **Source**: IBM Sample Dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)
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
   python src/Preprocessing and EDA/Data_Augmentation.py
   ```
   
   ```bash
   python src/Preprocessing and EDA/Data_Preprocessing.py
   ```

3. Train model:
   ```bash
   python src/models/XGboost.py
   ```
   
   ```bash
   python src/models/Random_forest.py
   ```
   
   ```bash
   python src/models/Logistic_Regression.py
   ```
   
4. Evaluate model:
   ```bash
   python src/Model_Comparsion.py
   ```

## 📄 License

This project is for educational and research use only. Please cite appropriately if reused in publications.

## ✍️ Author

Ke Xu – University of Maryland, College Park  
Email: kxu233@umd.edu
