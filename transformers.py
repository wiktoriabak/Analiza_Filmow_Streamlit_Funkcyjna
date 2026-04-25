from functools import reduce
import pandas as pd
import ast

def extract_year(df):
    
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

        df["year"] = df["release_date"].dt.year.astype("Int64")
    else:
        df["year"] = pd.NA
    return df


def filter_recent(df, year_threshold=None):

    if "year" not in df.columns:
        return df
    if year_threshold is None:
        year_threshold = int(df["year"].min())
    return df[df["year"] >= year_threshold]


def extract_categorical_values(df, column):

    if column not in df.columns:
        return []

    if column == "genres":
        values = df["genres"].dropna().apply(lambda x: [g['name'] for g in ast.literal_eval(x)] if x != '[]' else [])
    elif column == "production_companies":
        values = df["production_companies"].dropna().apply(lambda x: [c['name'] for c in ast.literal_eval(x)] if x != '[]' else [])
    elif column == "production_countries":
        values = df["production_countries"].dropna().apply(lambda x: [c['name'] for c in ast.literal_eval(x)] if x != '[]' else [])
    elif column == "year":
        return df["year"].dropna().astype(int).tolist()
    else:
        return df[column].dropna().astype(str).tolist()

    flat_values = [v for sublist in values for v in sublist]
    return flat_values


def pipeline(df, functions):
    return reduce(lambda acc, f: f(acc), functions, df)
