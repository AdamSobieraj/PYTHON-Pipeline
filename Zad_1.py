import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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
num_features = ['Age',  'BMI']
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

# Modelowa Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', data_preparation),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=10000))
])

# Trening modelu i ocena na zbiorze testowym
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(report)
