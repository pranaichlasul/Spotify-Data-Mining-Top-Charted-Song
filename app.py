import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model and label encoder
with open('genre_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('genre_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# List of features used in the improved model
feature_names = [
    'Energy', 'Acousticness', 'Tempo',
    'Danceability', 'Loudness', 'Speechiness', 'Valence',
    'Energy_to_Loudness', 'Is_Fast_Tempo', 'Danceability_Squared'
]

st.title("🎵 Spotify Genre Prediction Dashboard (Improved Model)")

# --- Pie chart for genre distribution ---
# You must provide the genre distribution used in the model.
# If you have the training data available, use it. Otherwise, use the label_encoder classes and counts.
try:
    # Try to load the training data genre distribution
    df = pd.read_excel("Dataset_Spotify_Clean.xlsx")
    # Clean and group genres as in your model training
    mask_clean = ~df['Genre'].astype(str).str.startswith("[")
    df_clean = df[mask_clean]
    genre_counts = df_clean['Genre'].where(
        df_clean['Genre'].isin(label_encoder.classes_), 'Other'
    ).value_counts().reindex(label_encoder.classes_, fill_value=0)
except Exception:
    # Fallback: use label_encoder classes with equal counts
    genre_counts = pd.Series([1]*len(label_encoder.classes_), index=label_encoder.classes_)

st.subheader("Genre Distribution Used in Model")
fig, ax = plt.subplots()
ax.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
ax.axis('equal')
st.pyplot(fig)

# Input sliders for all features
energy = st.slider("Energy", min_value=0, max_value=1000, value=500)
acousticness = st.slider("Acousticness", min_value=0.0, max_value=1000.0, value=250.0)
tempo = st.slider("Tempo", min_value=0, max_value=250000, value=120000)
danceability = st.slider("Danceability", min_value=0, max_value=1000, value=500)
loudness = st.slider("Loudness", min_value=-12000, max_value=0, value=-6000)
speechiness = st.slider("Speechiness", min_value=0, max_value=1000, value=200)
valence = st.slider("Valence", min_value=0, max_value=1000, value=500)

# Engineered features
energy_to_loudness = energy / (loudness if loudness != 0 else 1)
is_fast_tempo = int(tempo > 120000)  # Use median from your data if available
danceability_squared = danceability ** 2

input_features = np.array([[
    energy, acousticness, tempo, danceability, loudness, speechiness, valence,
    energy_to_loudness, is_fast_tempo, danceability_squared
]])

if st.button("Predict Genre"):
    pred_encoded = model.predict(input_features)
    pred_genre = label_encoder.inverse_transform(pred_encoded)[0]
    pred_probs = model.predict_proba(input_features)[0]
    st.success(f"Predicted Genre: {pred_genre}")

    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Genre": label_encoder.classes_,
        "Probability": pred_probs
    })
    st.bar_chart(prob_df.set_index("Genre"))
