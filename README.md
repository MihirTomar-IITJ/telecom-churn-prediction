# Telecom Customer Churn Prediction

A comprehensive machine learning project for predicting customer churn in the telecommunications industry. This project analyzes customer data to identify patterns and predict which customers are likely to leave the service.

## 📊 Project Overview

Customer churn is one of the biggest challenges in the telecom industry, with average monthly churn rates of 1.9% - 2% among top wireless carriers in the US. This project uses machine learning techniques to predict customer churn and help businesses take proactive measures to retain customers.

## 🎯 Objectives

- Perform exploratory data analysis (EDA) on telecom customer data
- Build and compare multiple machine learning models for churn prediction
- Identify key factors contributing to customer churn
- Provide actionable insights for customer retention strategies

## 📁 Dataset

The project uses the **WA_Fn-UseC_-Telco-Customer-Churn.csv** dataset containing:
- **7,043 customers** (after cleaning)
- **21 features** including:
  - Customer demographics (gender, senior citizen status, partner, dependents)
  - Account information (tenure, contract type, payment method)
  - Services subscribed (phone service, internet service, streaming services, etc.)
  - Billing information (monthly charges, total charges)
  - Target variable: **Churn** (Yes/No)

## 🔍 Key Features

### Data Preprocessing
- Handling missing values (11 missing values in TotalCharges column)
- Data type conversions
- Feature encoding for categorical variables
- Data normalization and scaling

### Exploratory Data Analysis
- Distribution analysis of customer demographics
- Churn rate analysis across different customer segments
- Correlation analysis between features
- Visualization of key patterns and trends

### Machine Learning Models

The project implements and compares **5 different classification algorithms**:

1. **Logistic Regression**
   - Baseline model for binary classification
   - Feature importance analysis using coefficients

2. **Random Forest Classifier**
   - Accuracy: ~80.8%
   - 1000 estimators with optimized hyperparameters
   - Feature importance visualization

3. **Support Vector Machine (SVM)**
   - Accuracy: ~82.0%
   - Linear kernel implementation
   - Confusion matrix analysis

4. **AdaBoost Classifier**
   - Accuracy: ~81.7%
   - Ensemble learning approach
   - Default DecisionTree base estimator

5. **XGBoost Classifier**
   - Accuracy: ~80.1%
   - Gradient boosting implementation
   - Advanced ensemble technique

## 📈 Results

### Model Performance Comparison

| Model | Accuracy |
|-------|----------|
| SVM | 82.0% |
| AdaBoost | 81.7% |
| Random Forest | 80.8% |
| XGBoost | 80.1% |
| Logistic Regression | 80.75% |

### Key Insights

- **Contract Type**: Month-to-month contracts show higher churn rates
- **Tenure**: Customers with shorter tenure are more likely to churn
- **Services**: Certain service combinations correlate with higher churn
- **Payment Method**: Electronic check users show different churn patterns

## 🛠️ Technologies Used

- **Python 3.13+**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning algorithms
  - `xgboost` - Gradient boosting
  - `seaborn` - Statistical data visualization
  - `matplotlib` - Plotting and visualization

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Telecom churn prediction"
```

2. Install required dependencies:
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib jupyter
```

## 🚀 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook telecom-churn-prediction.ipynb
```

2. Run all cells to:
   - Load and preprocess the data
   - Perform exploratory data analysis
   - Train multiple machine learning models
   - Evaluate and compare model performance

## 📊 Project Structure

```
Telecom churn prediction/
│
├── telecom-churn-prediction.ipynb  # Main Jupyter notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── README.md  # Project documentation
├── pyproject.toml  # Project configuration
└── .gitignore  # Git ignore file
```

## 🔬 Methodology

1. **Data Loading & Cleaning**
   - Import dataset
   - Handle missing values
   - Convert data types

2. **Exploratory Data Analysis**
   - Statistical summary
   - Distribution analysis
   - Correlation analysis
   - Visualization

3. **Feature Engineering**
   - Encoding categorical variables
   - Feature scaling
   - Train-test split (80-20)

4. **Model Training**
   - Train multiple classifiers
   - Hyperparameter tuning
   - Cross-validation

5. **Model Evaluation**
   - Accuracy metrics
   - Confusion matrices
   - Feature importance analysis
   - Model comparison

## 📝 Future Improvements

- [ ] Implement deep learning models (Neural Networks)
- [ ] Add more feature engineering techniques
- [ ] Perform hyperparameter optimization using GridSearchCV
- [ ] Implement SMOTE for handling class imbalance
- [ ] Create a web dashboard for predictions
- [ ] Add model deployment pipeline
- [ ] Include cost-benefit analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Mihir Tomar

## 🙏 Acknowledgments

- Dataset source: IBM Sample Data Sets
- Inspired by real-world telecom industry challenges
- Thanks to the open-source community for the amazing tools and libraries

---

**Note**: This project is for educational and research purposes. The models and insights should be validated with domain experts before production use.
