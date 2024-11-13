import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Załadowanie danych
diabetes_data = pd.read_csv('data/diabetes.csv')

# Rozdzielmy dane na cechy
X = diabetes_data.drop(['Diabetic'], axis=1)
y = diabetes_data['Diabetic']

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Definiowanie cech numerycznych i kategorycznych
num_features = ['Age', 'BMI']
cat_features = ['SerumInsulin']

# Przygotowanie wartości numerycznych
num_preparation = Pipeline(steps=[
    ('fill_missings', SimpleImputer(strategy='mean'))
])

# Przygotowanie wartości kategorycznych
cat_preparation = Pipeline(steps=[
    ('fill_missings', SimpleImputer(strategy='most_frequent')),
])

# Transformer dla wartości numerycznych i kategorycznych
data_preparation = ColumnTransformer(
    transformers=[
        ('numeric_preprocessing', num_preparation, num_features),
        ('categorical_preprocessing', cat_preparation, cat_features)
    ]
)

# Modelowa Pipeline z LogisticRegression
model_pipeline = Pipeline(steps=[
    ('preprocessor', data_preparation),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=10000))
])

# Siatka hiperparametrów
param_grid = {
    'model__penalty': ['l2'],
    'model__C': [0.001, 0.01, 0.1, 1, 10],
    'model__max_iter': [500, 1000, 2000]
}

# Wykonanie GridSearchCV
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
try:
    grid_search.fit(X_train, y_train)
except Exception as e:
    print(f"Błąd podczas wykonywania GridSearchCV: {e}")

print("Best parameters:", grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy on test set: {accuracy}")
print("\nClassification Report:")
print(report)
