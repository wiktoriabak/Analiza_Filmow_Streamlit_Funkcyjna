import time
import streamlit as st
from functools import wraps

def with_spinner(text="Ładowanie danych..."):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with st.spinner(text):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} wykonało się w {end-start:.2f}s")
        return result
    return wrapper
