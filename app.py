import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from collections import Counter
import numpy as np
import ast

from data_loader import download_movies_dataset
from transformers import pipeline, extract_year, filter_recent, extract_categorical_values
from analysis import COLUMN_DESCRIPTIONS, movies_per_year, avg_rating_per_year, top_genres, cluster_budget_revenue


st.set_page_config(page_title="Analiza The Movies Dataset", layout="wide")
st.title("Analiza The Movies Dataset (Kaggle)")


def load_data():
    df = download_movies_dataset()
    if df is None:
        st.error("Nie udało się wczytać danych.")
        st.stop()
    return df

if "df" not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df


if "original_cols" not in st.session_state:
    st.session_state.original_cols = df.columns.tolist()

original_cols = st.session_state.original_cols


df_clean = pipeline(df, [extract_year, filter_recent])

max_rows = st.slider(
    "Ile wierszy brać pod uwagę dla wszystkich analiz?",
    min_value=100, max_value=len(df_clean), value=5000, step=100
)
df_clean_limited = df_clean.head(max_rows)


tab1, tab2, tab3, tab4 = st.tabs(["Podstawowa eksploracja danych", "Filtry i wykresy", "Klasteryzacja", "Najlepiej oceniane filmy"])

#region Zakładka 1

with tab1:
    st.header("Kolumny w zbiorze")
    n_cols = 4
    cols = st.columns(n_cols)
    for i, col_name in enumerate(original_cols):
        cols[i % n_cols].write(col_name)
    
    column_descriptions = COLUMN_DESCRIPTIONS
    
    st.divider()

    st.header("Opis kolumn")
    with st.expander("Opis kolumn (kliknij, aby rozwinąć)"):
        for col in original_cols:
            if col in column_descriptions:
                desc = column_descriptions[col]
            else:
                desc = "Brak przygotowanego opisu"

            st.markdown(f"**{col}** – {desc}")

    st.divider()

    st.header("Pierwsze wiersze danych")
    st.dataframe(df_clean_limited.head(5))

    numeric_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "vote_count"]
    num_col = st.selectbox("Wybierz kolumnę numeryczną do wykresu gęstości", numeric_cols)

    col_data = pd.to_numeric(df_clean_limited[num_col], errors='coerce').dropna()
    if len(col_data) == 0:
        st.warning(f"Brak danych numerycznych w kolumnie {num_col}")
    else:
        stats = col_data.describe()[["mean", "std", "min", "max"]].to_frame().T.rename(index={0:num_col}).round(2)
        st.table(stats)

        fig, ax = plt.subplots(figsize=(4,2))
        sns.kdeplot(col_data, fill=True, color="skyblue", bw_adjust=0.5)
        ax.set_xlabel(num_col)
        ax.set_ylabel("Gęstość")

        col1, col2, col3 = st.columns([1,5,1])
        with col2:
            st.pyplot(fig, bbox_inches='tight')

    categorical_cols = ["original_language","genres","production_companies","production_countries","status","video"]
    cat_col = st.selectbox("Wybierz kolumnę kategoryczną", categorical_cols)
    cat_data = extract_categorical_values(df_clean_limited, cat_col)

    max_bars = 20
    counts = Counter(cat_data)
    top_counts = dict(counts.most_common(max_bars))

    df_top = pd.DataFrame({"Kategoria": list(top_counts.keys()), "Liczba": list(top_counts.values())})
    fig = px.bar(df_top, x="Kategoria", y="Liczba", text="Liczba", color="Liczba",
                 color_continuous_scale="Blues", title=f"Top {max_bars} wartości w kolumnie {cat_col}")
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, height=600)
    st.plotly_chart(fig)

#endregion

#region Zakładka 2

with tab2:
    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        year_slider = st.slider("Minimalny rok wydania filmu:", int(df_clean_limited["year"].min()), int(df_clean_limited["year"].max()))
        rating_slider = st.slider("Minimalna ocena:", 0.0, 10.0, 0.0)
        title_filter = st.text_input("Filtr tytułów (zawiera):", "")
        top_genre_count = st.selectbox("Ile najpopularniejszych gatunków pokazać?", [5,10,15,20], index=1)
    with col2:
        df_filtered = df_clean_limited[df_clean_limited["year"] >= year_slider]
        df_filtered = df_filtered[df_filtered["vote_average"] >= rating_slider] if "vote_average" in df_filtered.columns else df_filtered
        if title_filter:
            df_filtered = df_filtered[df_filtered["title"].str.contains(title_filter, case=False, na=False)]

        st.subheader("Liczba filmów w danym roku")
        counts = movies_per_year(df_filtered)
        if not counts.empty:
            st.bar_chart(counts)

        st.subheader("Średnia ocena filmów w danym roku")
        avg_r = avg_rating_per_year(df_filtered)
        if not avg_r.empty:
            st.line_chart(avg_r)

        st.subheader(f"Najpopularniejsze {top_genre_count} gatunków filmów")
        genres = top_genres(df_filtered, top_n=top_genre_count)
        if genres:
            df_genres = pd.DataFrame({"Gatunek": list(genres.keys()), "Liczba filmów": list(genres.values())})
            fig = px.bar(df_genres, x="Gatunek", y="Liczba filmów", text="Liczba filmów",
                         title=f"Top {top_genre_count} gatunków filmów")
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig)

#endregion

#region Zakładka 3

with tab3:
    st.header("Klasteryzacja filmów: Budget vs Revenue")
    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        n_clusters = st.slider("Liczba klastrów:", 2,6,4)
        remove_zero = st.checkbox("Usuń z analizy filmy z budget = 0 lub revenue = 0", value=True)
    with col2:
        df_cluster = cluster_budget_revenue(df_clean_limited, n_clusters=n_clusters, remove_zero=remove_zero)
        if df_cluster.empty:
            st.warning("Brak danych do klasteryzacji.")
        else:
            fig = px.scatter(df_cluster, x="budget", y="revenue", color="cluster",
                             hover_data=["title","year"], title=f"Klasteryzacja filmów (k = {n_clusters})")
            st.plotly_chart(fig, width='stretch')

#endregion

#region Zakładka 4

with tab4:
    st.header("Top 10 filmów wg roku i gatunku")
    years = sorted(df_clean_limited['year'].dropna().unique().astype(int))
    selected_year = st.selectbox("Wybierz rok", years, index=years.index(max(years)))

    all_genres = extract_categorical_values(df_clean_limited, "genres")
    flat_genres = sorted(set(all_genres))
    selected_genres = st.multiselect("Wybierz gatunek(i)", flat_genres)

    df_filtered_genres = df_clean_limited.copy()

    df_filtered_genres['genres_name'] = df_filtered_genres['genres'].dropna().apply(
        lambda x: [g['name'] for g in ast.literal_eval(x)] if x != '[]' else []
    )
    df_filtered_genres = df_filtered_genres[df_filtered_genres['year']==selected_year]

    if selected_genres:
        df_filtered_genres = df_filtered_genres[df_filtered_genres['genres_name'].apply(lambda gs: any(g in gs for g in selected_genres))]

    df_top10 = df_filtered_genres.sort_values(['vote_average','vote_count'], ascending=[False,False]).head(10)
    if df_top10.empty:
        st.write("Brak filmów spełniających wybrane kryteria")
    else:
        st.subheader(f"Top 10 filmów w roku {selected_year} dla wybranego gatunku/gatunków")
        st.dataframe(df_top10[['title','year','genres_name','vote_average','vote_count']])
        
#endregion