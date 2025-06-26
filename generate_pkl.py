# Skrip ini akan dijalankan di lingkungan virtual yang sudah benar
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

print(f"--- Membuat model dengan scikit-learn versi: {sklearn.__version__} ---")

# Muat data dari CSV di folder ini
df = pd.read_csv('horse_colic_dataset.csv')

# Proses data
df['outcome_binary'] = df['outcome'].apply(lambda x: 1 if x == 1.0 else 0)
df = df.drop('outcome', axis=1)
X = df.drop('outcome_binary', axis=1)
y = df['outcome_binary']
numerical_features = ['rectal_temperature', 'pulse', 'respiratory_rate', 'packed_cell_volume', 'total_protein']
categorical_features = [col for col in X.columns if col not in numerical_features]
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("--- Data berhasil diproses ---")

# Latih ulang model Decision Tree
dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(random_state=42))])
dt_pipeline.fit(X_train, y_train)
print("--- Model Decision Tree berhasil dilatih ulang ---")

# Simpan file .pkl yang baru dan 100% kompatibel
joblib.dump(dt_pipeline, 'horse_colic_pipeline.pkl')
print("\nBERHASIL! File 'horse_colic_pipeline.pkl' baru yang kompatibel telah dibuat.")
