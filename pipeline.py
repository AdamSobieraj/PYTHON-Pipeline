# manipulacja danymi
import numpy as np
import pandas as pd

# wizualizacja
import matplotlib.pyplot as plt
import seaborn as sns

# podział danych na zbiory treningowe/walidacyjne/testowe
from sklearn.model_selection import train_test_split, GridSearchCV

# budowa Pipeline
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, PowerTransformer

# redukcja wymiarowości
from sklearn.decomposition import PCA

# model
from sklearn.linear_model import LogisticRegression

# ewaluacja
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, f1_score, roc_auc_score

from FilterName import FilterName

dataset = pd.read_csv("data/titanic.csv")
print(dataset)

# przygotowanie dnych treningowych
# X (Features):
#   The variable dataset likely contains all features and the target variable 'Survived'.
#   We use drop() to remove the 'Survived' column from dataset.
#   The axis=1 parameter specifies that we're dropping columns.
#   .copy() is used to create a copy of the resulting DataFrame, ensuring we don't modify the original dataset.
# y (Target):
#   This creates a separate variable containing only the 'Survived' column from the original dataset.
#   Again, .copy() is used to ensure we have a distinct copy of this column.
X = dataset.drop(['Survived'], axis=1).copy()
y = dataset['Survived'].copy()

# train_test_split() is a function from scikit-learn, commonly used for splitting datasets.
# It takes four parameters:
#   X and y: Our prepared features and target variables.
#   test_size=0.25: This sets aside 25% of the data for testing (75% for training).
#   random_state=42: This ensures reproducibility of the split.
#   stratify=y: This preserves the proportion of positive ('Survived') cases in both the training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# dane numeryczne
num_features = ['Age', 'SibSp', 'Parch', 'Fare']

# Transformer - usupełnienie danych na podstawie danych treningoeych
# przygotowanie wartości numerycznych
num_preparation = Pipeline(steps=[('fill_missings', SimpleImputer(strategy='mean'))])

# It prints a message indicating that we're looking at the raw test dataset.
# X_test[num_features] selects only the numerical features from the test dataset.
# .isnull().mean() calculates the proportion of null values in each numerical feature.
print('Surowy zbiór danych - zbiór treningowy:')
print(X_train[num_features].isnull().mean())
# num_preparation.transform(X_test[num_features]) applies the previously defined transformations to the numerical features of the test set.
# The result is stored in X_test_transformed.
# We convert the transformed array back into a DataFrame for easier handling.
X_train_trasnformed = num_preparation.fit_transform(X_train[num_features])
X_train_trasnformed = pd.DataFrame(X_train_trasnformed, columns=num_features)
# It prints a message indicating we're looking at the output of the pipeline (transformed test dataset).
# X_test_trasnformed[num_features] selects only the numerical features from the transformed test dataset.
# Again, .isnull().mean() calculates the proportion of null values in each numerical feature.
print('\nWyjście Pipeline - zbiór treningowy')
print(X_train_trasnformed[num_features].isnull().mean())

print('Surowy zbiór danych - zbiór testowy:')
print(X_test[num_features].isnull().mean())
X_test_trasnformed = num_preparation.transform(X_test[num_features])
X_test_trasnformed = pd.DataFrame(X_test_trasnformed, columns=num_features)
print('\nWyjście Pipeline - zbiór testowy')
print(X_test_trasnformed[num_features].isnull().mean())

# dla data_preparation
num_features = ['Age', 'SibSp', 'Parch', 'Fare']

# przygotowanie wartości numerycznych
num_preparation = Pipeline(steps=[
    ('fill_missings', SimpleImputer(strategy='mean'))
])

# transformer = wartości numeryczne oraz kategoryczne
data_preparation = ColumnTransformer(transformers=[
    ('numeric_preprocessing', num_preparation, num_features)
])

print(data_preparation.fit_transform(X_train))

## Trasnsformer FilterName
transformer_filter_name = FilterName(column='Name')
print(X_train[['Name']])
print(transformer_filter_name.fit_transform(X_train[['Name']]))

cat_features = ['Name', 'Sex', 'Embarked']

# przygotowanie wartości kategorycznych
cat_preparation = Pipeline(steps=[
    ('filter_name', FilterName(column='Name'))
])

print('Przed')
print(X_train[cat_features])
print('Po')
print(cat_preparation.fit_transform(X_train[cat_features]))

# przygotowanie wartości kategorycznych - najczęściej występującą kategorie
cat_preparation = Pipeline(steps=[
    ('filter_name', FilterName(column='Name')),
    ('fill_missings', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
cat_preparation.fit_transform(X_train[cat_features])

print(cat_preparation.fit_transform(X_train[cat_features]).shape)

## Połączenie 2 pipelinów

num_features = ['Age', 'SibSp', 'Parch', 'Fare']
cat_features = ['Name', 'Sex', 'Embarked']

# przygotowanie wartości numerycznych
num_preparation = Pipeline(steps=[
    ('fill_missings', SimpleImputer(strategy='mean'))
])

# przygotowanie wartości kategorycznych
cat_preparation = Pipeline(steps=[
    ('filter_name', FilterName(column='Name')),
    ('fill_missings', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse=False ,handle_unknown='ignore'))
])

# transformer = wartości numeryczne oraz kategoryczne
data_preparation = ColumnTransformer(transformers=[
    ('numeric_preprocessing', num_preparation, num_features),
    ('categorical_preprocessing', cat_preparation, cat_features)
])

print(data_preparation.fit_transform(X_train))

# Kształt przetworzonych danych:
print(data_preparation.fit_transform(X_train).shape)

#########################################################################
# połączenie z modelem
###############################################################################
num_features = ['Age', 'SibSp', 'Parch', 'Fare']
cat_features = ['Name', 'Sex', 'Embarked']

# przygotowanie wartości numerycznych
num_preparation = Pipeline(steps=[
    ('fill_missings', SimpleImputer(strategy='mean'))
])

# przygotowanie wartości kategorycznych
cat_preparation = Pipeline(steps=[
    ('filter_name', FilterName(column='Name')),
    ('fill_missings', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# transformer = wartości numeryczne oraz kategoryczne
data_preparation = ColumnTransformer(transformers=[
    ('numeric_preprocessing', num_preparation, num_features),
    ('categorical_preprocessing', cat_preparation, cat_features)
])

model_pipeline_v1 = Pipeline(steps=[('preprocessor', data_preparation),
                                    ('model', LogisticRegression(max_iter=10000))])

print(model_pipeline_v1.fit(X_train, y_train))

metrics_dataframe = pd.DataFrame(columns = ['Model', 'F1_score', 'AUC'])
print(metrics_dataframe)
models = []
models_names = []
predictions_proba_list = []

def calculate_metrics(model, name, X_checked, y_checked):
    models.append(model)
    models_names.append(name)
    global metrics_dataframe
    predictions = model.predict(X_checked)
    predictions_proba = model.predict_proba(X_checked)
    predictions_proba_list.append(predictions_proba[:,1])

    ############## metryki dla sprawdzanego modelu ################

    # Precision, Recall, F1, Accuracy
    print(classification_report(y_checked, predictions))

    # Confusion matrix
    plt.figure()
    cm = confusion_matrix(y_checked, predictions)
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.0f')
    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    plt.show()

    # plot ROC curve
    fig = plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    for model_selected, name_selected, pred_proba in zip(models, models_names, predictions_proba_list):
        fpr, tpr, thresholds = roc_curve(y_checked, pred_proba)
        plt.plot(fpr, tpr, label=name_selected)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    f1_metric = f1_score(y_checked, predictions)
    auc_metric = roc_auc_score(y_checked, predictions_proba[:,1])
    metrics_dataframe = metrics_dataframe.append({'Model': name, 'F1_score': f1_metric, 'AUC': auc_metric},
                                                 ignore_index=True)
    return metrics_dataframe

calculate_metrics(model_pipeline_v1, 'Logistic Regression', X_test, y_test)

####################################################################################
num_features = ['Age', 'SibSp', 'Parch', 'Fare']
cat_features = ['Name', 'Sex', 'Embarked']

# przygotowanie wartości numerycznych
num_preparation = Pipeline(steps=[
    ('fill_missings', SimpleImputer(strategy='mean')),
    ('polynomial_features', PolynomialFeatures(degree=3)),
    ('scaler_1', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('scaler_2', StandardScaler())
])

# przygotowanie wartości kategorycznych
cat_preparation = Pipeline(steps=[
    ('filter_name', FilterName(column='Name')),
    ('fill_missings', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# transformer = wartości numeryczne oraz kategoryczne
data_preparation = ColumnTransformer(transformers=[
    ('numeric_preprocessing', num_preparation, num_features),
    ('categorical_preprocessing', cat_preparation, cat_features)
])

model_pipeline_v2 = Pipeline(steps=[('preprocessor', data_preparation),
                                    ('model', LogisticRegression(max_iter=10000))])

print(model_pipeline_v2.fit(X_train, y_train))

model_pipeline_v3 = Pipeline(steps=[('preprocessor', data_preparation),
                                    ('model', LogisticRegression(max_iter=10000))])

list(model_pipeline_v3.get_params().keys())

params = {
    'preprocessor__numeric_preprocessing__fill_missings__strategy': ['mean', 'median'],
    'preprocessor__numeric_preprocessing__polynomial_features__degree': [1, 2, 3, 4],
    'preprocessor__numeric_preprocessing__pca__n_components': [0.85, 0.90, 0.95, 0.99, 0.99999],
    'model__C': np.logspace(-4, 4, 50)
}

grid_search = GridSearchCV(model_pipeline_v3, params, cv=10, n_jobs=-1, verbose=10, scoring='f1_macro')
grid_search.fit(X_train, y_train)
print('Wybrane hiperparametry: ', grid_search.best_params_)
model_v3 = grid_search.best_estimator_

calculate_metrics(model_v3, 'Logistic Regression - wybór hiperparametrów', X_test, y_test)