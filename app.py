import streamlit as st
import cv2
import tempfile
import os
from detector import ObjektumDetektor
from utils import video_informacio
from torch import cuda  # GPU támogatás ellenőrzése

# Streamlit oldal konfigurálása
st.set_page_config(page_title="Valós idejű Objektumdetektáló", layout="wide")

# Oldal címe
st.title("Valós idejű Objektumdetektáló")
st.markdown("### Egy mesterséges intelligencia alapú alkalmazás valós idejű objektumdetektálásra videókban.")
st.sidebar.header("Beállítások")

# GPU elérhetőség ellenőrzése
eszkozok = ["CPU"]
if cuda.is_available():
    eszkozok.append("GPU")

# GPU/CPU választási lehetőség
feldolgozo_eszkoz = st.sidebar.radio("Válaszd ki a feldolgozási módot", options=eszkozok, index=0)
