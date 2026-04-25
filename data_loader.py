import os
import pandas as pd
import streamlit as st
from decorators import measure_time, with_spinner


@with_spinner("Pobieranie i wczytywanie danych z Kaggle...")
@measure_time
def download_movies_dataset(
    dataset="rounakbanik/the-movies-dataset",
    file_name="movies_metadata.csv",
):
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, file_name)

    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")


    if os.path.exists(kaggle_json_path):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(dataset, path=data_dir, unzip=True)

            df = pd.read_csv(csv_path, low_memory=False)
            st.success("Pobrano dane z Kaggle i wczytano CSV")
            return df

        except Exception as e:
            st.warning(
                f"Nie udało się pobrać danych z Kaggle, przechodzę w tryb offline: {e}"
            )

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            st.info("Używam lokalnego pliku CSV (tryb offline)")
            return df
        except Exception as e:
            st.error(f"Błąd przy wczytywaniu lokalnego CSV: {e}")
            return None

    st.error("Brak tokena Kaggle i brak lokalnego pliku CSV.")
    return None
