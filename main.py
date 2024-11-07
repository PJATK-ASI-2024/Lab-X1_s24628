import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import logging

warnings.filterwarnings("ignore")

logging.basicConfig(filename="log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Script started.")

try:
    data = pd.read_csv("CollegeDistance.csv")
    logging.info("Data loaded successfully.")
except FileNotFoundError:
    logging.error("Data file not found.")
    raise

logging.info("Dataset information and statistical summary:")
logging.info(data.info())
logging.info(data.describe())

numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix (Numeric Columns Only)")
plt.show()
logging.info("Correlation matrix plotted.")

data['score_category'] = pd.qcut(data['score'], 5, labels=['very low', 'low', 'medium', 'high', 'very high'])
logging.info("Score categories created based on quintiles.")
logging.info("Distribution of score categories:\n%s", data['score_category'].value_counts())

data = pd.get_dummies(data, columns=['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'income', 'region'], drop_first=True)
logging.info("Categorical variables encoded using one-hot encoding.")

X = data.drop(columns=['rownames', 'score', 'score_category', 'education'])
y = data['score_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logging.info("Data split into training and test sets.")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logging.info("Feature scaling applied to training and test sets.")

model = RandomForestClassifier(max_depth=10, min_samples_split=10, n_estimators=150)
model.fit(X_train, y_train)
logging.info("Random Forest model trained.")

y_pred = model.predict(X_test)
logging.info("Model predictions completed.")

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

logging.info("Model Accuracy: %s", accuracy)
logging.info("Confusion Matrix:\n%s", conf_matrix)
logging.info("Classification Report:\n%s", class_report)

importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances['importance'], y=feature_importances['feature'], palette="viridis")
plt.title("Feature Importance in Random Forest Model")
plt.show()
logging.info("Feature importance plot generated.")

# Comparison of actual vs predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
actual_counts = results['Actual'].value_counts().sort_index()
predicted_counts = results['Predicted'].value_counts().sort_index()
comparison_df = pd.DataFrame({'Actual': actual_counts, 'Predicted': predicted_counts})

comparison_df.plot(kind='bar', figsize=(10, 6))
plt.title("Comparison of Actual vs Predicted Categories")
plt.xlabel("Score Category")
plt.ylabel("Frequency")
plt.xticks(rotation=0)
plt.legend(title="Legend")
plt.show()
logging.info("Comparison of actual vs predicted categories plotted.")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix of Predicted vs Actual")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
logging.info("Confusion matrix heatmap generated.")

logging.info("Script completed.")
