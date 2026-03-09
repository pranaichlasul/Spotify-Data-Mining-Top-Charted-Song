import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from scipy.spatial.distance import cdist

# Load and clean data
df = pd.read_excel('Dataset Spotify ver2.xlsx')

# Fitur yang digunakan
features = [
    'Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
    'Liveness', 'Tempo', 'Duration (ms)', 'Valence'
]
target = 'Genre'

# Pastikan semua fitur numerik
for col in features:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

df = df.dropna(subset=features + [target])

# Extract first genre only
def extract_first_genre(val):
    if isinstance(val, list):
        return val[0] if len(val) > 0 else 'Other'
    if isinstance(val, str) and val.startswith('['):
        try:
            import ast
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
            else:
                return 'Other'
        except:
            return 'Other'
    return val

df[target] = df[target].apply(extract_first_genre)

# Hapus baris dengan genre 'Other'
df_no_other = df[df[target] != 'Other'].copy()

# Hitung centroid untuk setiap genre yang valid
X = df_no_other[features]
y = df_no_other[target]
genre_counts = y.value_counts()
centroids = {}
for g in genre_counts.index:
    centroids[g] = X[y == g].mean().values

# Untuk baris dengan genre 'Other', tetapkan ke genre terdekat berdasarkan centroid
df_other = df[df[target] == 'Other'].copy()
def nearest_genre(row):
    row_feat = row[features].values.reshape(1, -1)
    centroid_matrix = np.array(list(centroids.values()))
    dists = cdist(row_feat, centroid_matrix)
    nearest_idx = np.argmin(dists)
    return list(centroids.keys())[nearest_idx]

if not df_other.empty:
    df_other[target] = df_other.apply(nearest_genre, axis=1)
    df_final = pd.concat([df_no_other, df_other], ignore_index=True)
else:
    df_final = df_no_other

X = df_final[features]
y = df_final[target]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Gradient Boosting Classifier with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gbc = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=cv,
    n_jobs=-1,
    scoring='accuracy'
)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting accuracy: {acc:.3f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model, scaler, features, and label_encoder
joblib.dump(gbc.best_estimator_, 'genre_model.pkl')
joblib.dump(scaler, 'genre_scaler.pkl')
joblib.dump(features, 'genre_features.pkl')
joblib.dump(label_encoder, 'genre_label_encoder.pkl')
