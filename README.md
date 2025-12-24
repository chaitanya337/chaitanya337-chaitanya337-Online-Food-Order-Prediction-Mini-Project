# Online Food Order Prediction

A machine learning web application that predicts whether a customer will order food online based on various demographic and behavioral features.

## 📋 Overview

This project uses machine learning algorithms to predict online food ordering behavior. The application features a Flask web interface where users can input customer data and receive instant predictions.

## 🚀 Features

- **Multiple ML Models**: Evaluates 8 different machine learning algorithms
- **Web Interface**: User-friendly Flask web application
- **Real-time Predictions**: Instant prediction results
- **Model Comparison**: Automatically selects the best-performing model

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: 
  - scikit-learn
  - XGBoost
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors
  - Support Vector Machine
  - Decision Tree
  - Naive Bayes
  - Gradient Boosting
- **Data Processing**: Pandas, NumPy
- **Model Serialization**: Joblib
- **Frontend**: HTML, CSS

## 📁 Project Structure

```
mini project/
│
├── app.py                                  # Flask web application
├── model.py                                # ML model training and evaluation
├── model.pkl                               # Saved trained model
├── onlinefoods1.csv                        # Dataset
├── Online_Food_Order_Prediction-checkpoint.ipynb  # Jupyter notebook
├── static/
│   └── styles.css                          # CSS styling
└── templates/
    └── index.html                          # Web interface
```

## 📊 Features Used for Prediction

The model uses the following features to make predictions:

1. **Age**: Customer's age
2. **Gender**: Male (1) or Female (0)
3. **Marital Status**: Single (1), Married (2), Prefer not to say (0)
4. **Occupation**: Student (1), Employee (2), Self-Employed (3), Housewife (4)
5. **Monthly Income**: Income range in rupees
6. **Educational Qualifications**: Graduate (1), Post Graduate (2), Ph.D (3), School (4), Uneducated (5)
7. **Family Size**: Number of family members
8. **Pin Code**: Location pin code
9. **Feedback**: Positive (1) or Negative (0)

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Setup

1. Clone or download the project:
```bash
cd "d:\mini project"
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

3. Install required packages:
```bash
pip install flask pandas numpy scikit-learn xgboost joblib
```

## 🎯 Usage

### Training the Model

1. Ensure `onlinefoods1.csv` is in the project directory
2. Run the model training script:
```bash
python model.py
```
3. This will:
   - Train 8 different ML models
   - Compare their accuracies
   - Save the best model as `model.pkl`

### Running the Web Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. Fill in the customer information form
4. Click "Predict" to get the prediction result

## 📈 Model Performance

The application evaluates multiple models and automatically selects the best performer based on accuracy. Models evaluated include:

- Random Forest Classifier
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Naive Bayes
- Gradient Boosting
- XGBoost

## 🎨 Web Interface

The web interface provides:
- Clean, user-friendly form for data input
- Clear instructions for each field
- Instant prediction results (Yes/No)
- Responsive design with CSS styling

## 📝 Data Preprocessing

The model includes the following preprocessing steps:

- **Categorical Encoding**: Converts categorical variables to numerical values
- **Missing Value Handling**: Drops rows with missing values in critical columns
- **Feature Selection**: Uses relevant features for prediction
- **Train-Test Split**: 90% training, 10% testing

## ⚠️ Error Handling

The application includes robust error handling for:
- Missing form fields
- Invalid input data
- Model loading errors
- Prediction errors

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## 📄 License

This project is available for educational and research purposes.

Name: Chaitanya Vardhineedi

Email: chaitanyavardhineedi@gmail.com

GitHub: @chaitanya337
