import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

data = pd.read_csv("CollegeDistance.csv")

correlation_matrix = data.corr(numeric_only=True)

data['score_category'] = pd.cut(data['score'], bins=[0, 50, 100], labels=["niski", "wysoki"])

numeric_features = ["unemp", "wage", "distance", "tuition", "education"]
categorical_features = ["gender", "ethnicity", "fcollege", "mcollege", "home", "urban", "income", "region"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

smote = SMOTE(random_state=42)

pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", smote),  # Adding SMOTE here
    ("classifier", RandomForestClassifier(random_state=42))
])

X = data.drop(columns=["score", "score_category"])
y = data["score_category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique())
plt.title("Confusion Matrix: Predicted vs Actual")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
plt.figure(figsize=(10, 6))
sns.countplot(data=comparison_df, x="Actual", hue="Predicted", palette="Set2")
plt.title("Comparison of Actual vs Predicted Categories")
plt.xlabel("Actual Category")
plt.ylabel("Count")
plt.legend(title="Predicted Category")
plt.show()
