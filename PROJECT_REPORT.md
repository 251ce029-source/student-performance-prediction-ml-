# PROJECT REPORT
## Student Performance Prediction Using Machine Learning

### Abstract
This project demonstrates a complete supervised machine-learning workflow for classifying student performance into Low, Medium, or High categories. It includes a synthetic dataset, preprocessing, a Random Forest classifier, evaluation metrics, and two user interfaces built with Flask and Streamlit.

### 1. Introduction
Educational performance is influenced by several academic and contextual factors. This project is designed as a learning demonstration of how tabular data can be prepared and used in a classification workflow. It is not intended to judge students or replace teachers.

### 2. Problem definition
The task is a three-class classification problem. Input features include attendance, study hours, previous score, assignment completion, sleep hours, internet access, and extracurricular participation. The output is a broad performance category.

### 3. Dataset description
The project contains 800 generated rows. The dataset is synthetic and generated using a fixed random seed so that the exercise is reproducible. It does not contain personal information and is not a real institutional dataset.

### 4. Data preprocessing and EDA
- Inspect rows, columns, data types and target distribution.
- Separate input features and target label.
- Impute numerical missing values using the median.
- Impute categorical missing values using the most frequent value.
- Standardize numerical features.
- One-hot encode categorical features.
- Use a stratified 80:20 train/test split.

### 5. Model development
A Random Forest classifier is used because it can model non-linear relationships and works with mixed feature types after preprocessing. The preprocessing and classifier are combined in a scikit-learn Pipeline to ensure the same transformations are applied during training and prediction.

### 6. Evaluation
The training script calculates accuracy, macro precision, macro recall, macro F1, a classification report, and a confusion matrix. The test set is held out from model fitting. Metrics are saved in `reports/metrics.json`.

### 7. Application development
The Flask application provides a simple HTML form for entering one student's feature values. The Streamlit application provides interactive controls and a class-distribution chart. Both applications load the saved model pipeline.

### 8. Critical analysis
The dataset is synthetic, so its metrics only show that the software workflow operates on the generated examples. They do not demonstrate that the model can predict real student outcomes. Real educational data can contain missingness, measurement errors, cultural and socioeconomic differences, and historical bias. A model could wrongly label a student and affect confidence or opportunity. Any real research would require consent, data minimization, security, fairness evaluation, external validation, and educator oversight.

### 9. Conclusion
The project demonstrates the main stages of a machine-learning classification system and two ways to make it accessible to users. The work can be extended with responsibly collected real data, model comparison, explainability, and fairness analysis.

### 10. References
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Pandas documentation: https://pandas.pydata.org/docs/
- Flask documentation: https://flask.palletsprojects.com/
- Streamlit documentation: https://docs.streamlit.io/
