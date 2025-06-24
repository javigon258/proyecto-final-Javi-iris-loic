import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors
import ast
import spacy
import subprocess
import sys

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
# ...resto de tu código...

def clean_text_spacy(text, pos_tags):
    doc = nlp(text.lower())
    return " ".join(
        dict.fromkeys(  # elimina duplicados
            token.lemma_
            for token in doc
            if token.pos_ in pos_tags and token.is_alpha and not token.is_stop
        )
    )

def normalize_list(values):
    # Convierte listas tipo ['Drama', 'Action'] en string limpio
    return " ".join(sorted(set([v.lower().strip() for v in values if v.strip()])))

# --- Función para limpiar valores de columnas tipo lista ---
def clean_column_values(column):
    cleaned = set()
    for val in df[column].dropna():
        try:
            parsed = ast.literal_eval(val) if isinstance(val, str) and val.startswith('[') else val.split(',')
        except:
            parsed = val.split(',')
        for item in parsed:
            item_cleaned = item.strip().strip("'\"[] ")  # Mejora: eliminar corchetes y espacios residuales
            if item_cleaned:
                cleaned.add(item_cleaned)
    return sorted(cleaned)

# --- Cargar datos ---
@st.cache_data
def load_data():
    df = pd.read_csv("../data/data2_streamlit.csv")
    model = joblib.load("../models/knn_model.pkl")

    vectores = [
        joblib.load(f"../models/{col}_vect.pkl") for col in [
            'title_clean', 'tagline_clean', 'production_companies_clean', 'overview_clean', 'keywords_clean']
    ]

    escalado = joblib.load("../models/numeric_minmax_scaler.pkl")
    movie_vectors = load_npz("../data/matrix_combinada.npz")

    actors_vectorizer = joblib.load('../models/top_actors_clean_vect.pkl')
    director_vectorizer = joblib.load('../models/director_clean_vect.pkl')
    genre_vectorizer = joblib.load("../models/genres_clean_vect.pkl")
    countries_vectorizer = joblib.load("../models/production_countries_clean_vect.pkl")
    languages_vectorizer = joblib.load("../models/spoken_languages_clean_vect.pkl")
    original_lang_vectorizer = joblib.load("../models/original_language_clean_vect.pkl")

    return df, model, vectores, escalado, movie_vectors, actors_vectorizer, director_vectorizer, genre_vectorizer, countries_vectorizer, languages_vectorizer, original_lang_vectorizer

# --- Cargar todo ---
df, model, vectores, escalador, movie_vectors, actors_vect, director_vect, genre_vect, country_vect, lang_vect, orig_lang_vect = load_data()


# --- UI ---
st.title("🎬 Recomendador de Películas")
st.markdown("Rellena los campos para obtener recomendaciones precisas:")

# --- Entradas ---
user_query = st.text_area("Describe lo que buscas (puede incluir título, palabras clave, etc.)")

unique_actors = clean_column_values('top_actors')
unique_directors = clean_column_values('director')
unique_genres = clean_column_values('genres')
unique_countries = clean_column_values('production_countries')
unique_languages = clean_column_values('spoken_languages')
unique_original_langs = sorted(df['original_language'].dropna().unique())
unique_years = sorted(df['release_year'].dropna().unique())

sel_actors = st.multiselect("Actores", unique_actors)
sel_director = st.selectbox("Director", [""] + unique_directors)
sel_genres = st.multiselect("🎭 Géneros", unique_genres)
sel_countries = st.multiselect("🌍 Países", unique_countries)
sel_languages = st.multiselect("🗣 Idiomas hablados", unique_languages)
sel_original_lang = st.selectbox("📌 Idioma original", [""] + unique_original_langs)
sel_year = st.selectbox("📅 Año de estreno", [""] + list(map(str, unique_years)))
adult = st.selectbox("🔞 ¿Solo películas para adultos?", ["No", "Sí"])

vote = st.slider("⭐ Puntuación", float(df['vote_average'].min()), float(df['vote_average'].max()), float(df['vote_average'].mean()), step=0.1)
vote_count = st.slider("Número de votos:", int(df['vote_count'].min()), int(df['vote_count'].max()), int(df['vote_count'].mean()), step=1)
runtime = st.slider("⏱ Duración (min)", 30.0, 300.0, 120.0, step=1.0)
budget = st.slider("💰 Presupuesto (millones)", 0.0, 500.0, 100.0, step=1.0)
popularity = st.slider("Popularidad:", float(df['popularity'].min()), float(df['popularity'].max()), float(df['popularity'].mean()), step=1.0)



# --- Botón de predicción ---
if st.button("🔎 Obtener recomendaciones"):
   
    # --- Limpieza y normalización de entradas ---
    title_clean = clean_text_spacy(user_query, ['NOUN', 'VERB', 'PROPN', 'ADJ'])
    tagline_clean = clean_text_spacy(user_query, ['NOUN', 'VERB', 'PROPN', 'ADJ'])
    overview_clean = clean_text_spacy(user_query, ['NOUN', 'VERB', 'PROPN', 'ADJ'])

    actors_clean = normalize_list(sel_actors)
    director_clean = sel_director.lower().strip()
    genres_clean = normalize_list(sel_genres)
    countries_clean = normalize_list(sel_countries)
    languages_clean = normalize_list(sel_languages)
    orig_lang_clean = sel_original_lang.lower().strip()
    keywords_clean = user_query.lower().strip()
    prod_companies_clean = user_query.lower().strip()

    # --- Vectorizar ---
    vectores_usuario = [
        vectores[0].transform([title_clean]),
        vectores[1].transform([tagline_clean]),
        vectores[2].transform([prod_companies_clean]),
        vectores[3].transform([overview_clean]),
        vectores[4].transform([keywords_clean])
    ]

    actors_vec = actors_vect.transform([actors_clean])
    director_vec = director_vect.transform([director_clean])
    genre_vec = genre_vect.transform([genres_clean])
    country_vec = country_vect.transform([countries_clean])
    lang_vec = lang_vect.transform([languages_clean])
    orig_lang_vec = orig_lang_vect.transform([orig_lang_clean])

    user_numeric = pd.DataFrame([{
        'vote_average': vote,
        'vote_count': vote_count,
        'adult': 1 if adult == "Sí" else 0,
        'popularity': popularity,
        'runtime_final': runtime,
        'budget_final': budget,
        'release_year': sel_year if sel_year != "" else 0
    }])
    user_numeric_scaled = escalador.transform(user_numeric)
    user_numeric_sparse = csr_matrix(user_numeric_scaled)

    user_vector = hstack([
    vectores_usuario[0], #title_clean
    vectores_usuario[1], #tagline_clean
    actors_vec,
    vectores_usuario[2], #production_companies
    vectores_usuario[3], #overview_clean
    vectores_usuario[4], #keywords
    director_vec,
    genre_vec,
    orig_lang_vec,
    country_vec,
    lang_vec,
    user_numeric_sparse
    ])

    # --- Buscar recomendaciones ---
    dists, indices = model.kneighbors(user_vector, n_neighbors=10)

    # --- Visualización completa de las recomendaciones ---
    st.markdown("---")
    st.subheader("📽 Resultados detallados")

    TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

    try:
        recomendadas = df.iloc[indices[0]]
    except:
        recomendadas = pd.DataFrame()

    if recomendadas.empty:
        st.warning("No se encontraron coincidencias.")
    else:
        # Configuración
        image_width = 130
        image_height = 190
        column_gap = 5

        # Mostrar recomendaciones
        num_cols = 5
        for i in range(0, len(recomendadas), num_cols):
            fila = recomendadas.iloc[i:i+num_cols]
            cols = st.columns(num_cols)
            for j, (_, row) in enumerate(fila.iterrows()):
                poster_path = row.get('poster_path', '')
                with cols[j]:
                    margin = f"{column_gap}px" if j != num_cols - 1 else "0px"
                    st.markdown(
                        f"""
                        <div style="width:{image_width}px; height:{image_height}px; overflow:hidden; display:flex; align-items:center; justify-content:center; 
                                    border-radius:10px; margin-right:{margin};">
                            {f'<img src="{TMDB_IMAGE_BASE}{poster_path}" style="width:100%; height:100%; object-fit:cover; border-radius:10px;" />' 
                            if pd.notnull(poster_path) and poster_path != '' 
                            else f'<div style="width:100%; height:100%; background-color:#eee; color:#555; display:flex; align-items:center; justify-content:center; font-size:14px; border-radius:10px;">🎞️ Sin imagen</div>'}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(f"**{row['title']}**")
                    st.caption(f"🎬 {row['genres']}, 🌍 {row['production_countries']}")
                    st.caption(f"⭐ {row['vote_average']} - ⏱ {row['runtime_final']} mins")