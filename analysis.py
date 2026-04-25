import pandas as pd
import ast
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

COLUMN_DESCRIPTIONS = {
    "adult": ("informuje, czy film jest przeznaczony tylko dla dorosłych"),
    "belongs_to_collection": ("informacja, czy film należy do serii lub kolekcji"),
    "budget": ("budżet produkcji filmu w dolarach"),
    "genres": ("lista gatunków filmu"),
    "homepage": ("oficjalna strona internetowa filmu"),
    "id": ("unikalny identyfikator filmu w bazie TMDB"),
    "imdb_id": ("identyfikator filmu w serwisie IMDb"),
    "original_language": ("język oryginalny filmu"),
    "original_title": ("oryginalny tytuł filmu"),
    "overview": ("krótki opis fabuły filmu"),
    "popularity": ("wskaźnik popularności filmu"),
    "poster_path": ("ścieżka do plakatu filmu"),
    "production_companies": ("firmy produkujące film"),
    "production_countries": ("kraje produkcji filmu"),
    "release_date": ("data premiery filmu"),
    "revenue": ("przychód filmu w dolarach"),
    "runtime": ("długość filmu w minutach"),
    "spoken_languages": ("języki używane w filmie"),
    "status": ("status produkcji filmu"),
    "tagline": ("hasło promocyjne filmu"),
    "title": ("oficjalny tytuł filmu"),
    "video": ("informacja, czy wpis zawiera materiał wideo"),
    "vote_average": ("średnia ocena filmu"),
    "vote_count": ("liczba oddanych głosów"),
}

def movies_per_year(df):
    if "year" not in df.columns:
        return pd.Series(dtype=int)
    return df["year"].value_counts().sort_index()

def avg_rating_per_year(df):
    if "year" not in df.columns or "vote_average" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("year")["vote_average"].mean()

def top_genres(df, top_n=10):
    if "genres" not in df.columns:
        return {}
    genres_list = df["genres"].dropna().apply(lambda x: [g['name'] for g in ast.literal_eval(x)] if x != '[]' else [])
    all_genres = [g for sublist in genres_list for g in sublist]
    return dict(Counter(all_genres).most_common(top_n))

def cluster_budget_revenue(df, n_clusters=4, remove_zero=True):

    df_cluster = df.copy()
    df_cluster["budget"] = pd.to_numeric(df_cluster["budget"], errors="coerce")
    df_cluster["revenue"] = pd.to_numeric(df_cluster["revenue"], errors="coerce")
    
    if remove_zero:
        df_cluster = df_cluster[(df_cluster["budget"]>0) & (df_cluster["revenue"]>0)]
    
    df_cluster = df_cluster.dropna(subset=["budget","revenue"])
        
    if df_cluster.empty:
        return df_cluster
    
    X = df_cluster[["budget","revenue"]]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cluster["cluster"] = kmeans.fit_predict(X_scaled)
    
    return df_cluster
