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

# Objektumosztályok betöltése
OSZTALY_NEVEK = {
    0: 'ember',
    1: 'bicikli',
    2: 'autó',
    3: 'motorkerékpár',
    4: 'repülőgép',
    5: 'busz',
    6: 'vonat',
    7: 'teherautó',
    8: 'csónak',
    9: 'közlekedési lámpa',
    10: 'tűzcsap',
    # További osztályok hozzáadhatók ide
}

# Felhasználói beállítások
biztonsagi_kuszob = st.sidebar.slider("Detektálási küszöbérték", 0.0, 1.0, 0.5, 0.05)
kivalasztott_osztalyok = st.sidebar.multiselect("Objektumok kiválasztása", options=list(OSZTALY_NEVEK.values()), default=list(OSZTALY_NEVEK.values()))

# Osztályok azonosítóinak lekérése
kivalasztott_osztaly_idk = [kulcs for kulcs, ertek in OSZTALY_NEVEK.items() if ertek in kivalasztott_osztalyok]

# Videó feltöltése
feltoltott_fajl = st.file_uploader("Tölts fel egy videót (.mp4, .avi formátumban)", type=["mp4", "avi"])

if feltoltott_fajl is not None:
    # Ideiglenes fájl mentése
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ideiglenes_fajl:
        ideiglenes_fajl.write(feltoltott_fajl.read())
        video_utvonal = ideiglenes_fajl.name

    # Objektumdetektáló inicializálása
    detektor = ObjektumDetektor(biztonsagi_kuszob=biztonsagi_kuszob, osztalyok=kivalasztott_osztaly_idk)

    # Videó feldolgozása
    st.markdown("### Feldolgozás alatt...")
    video = cv2.VideoCapture(video_utvonal)
    szelesseg, magassag, fps, osszes_kocka = video_informacio(video)

    # Kimeneti videó fájl beállítása
    kimeneti_utvonal = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    kimenet = cv2.VideoWriter(kimeneti_utvonal, cv2.VideoWriter_fourcc(*"mp4v"), fps, (szelesseg, magassag))

    haladasjelzo = st.progress(0)
    stframe = st.empty()
    kockaszam = 0

    while video.isOpened():
        ret, kocka = video.read()
        if not ret:
            break

        # Objektumdetektálás
        annotalt_kocka = detektor.detektal(kocka)

        # Kimenet írása
        kimenet.write(annotalt_kocka)

        # Eredmények megjelenítése
        stframe.image(annotalt_kocka, channels="BGR", use_container_width=True)

        # Haladásjelző frissítése
        kockaszam += 1
        haladas = kockaszam / osszes_kocka
        haladasjelzo.progress(haladas)

    video.release()
    kimenet.release()

    # Feldolgozás sikeres üzenet
    st.success("Feldolgozás befejezve!")

    # Videó megjelenítése
    with open(kimeneti_utvonal, "rb") as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)

    # Eredmény letöltése
    with open(kimeneti_utvonal, 'rb') as fajl:
        st.download_button('Feldolgozott videó letöltése', fajl, file_name='eredmeny.mp4')

# Footer
st.markdown("---")
st.markdown("&copy; 2024 | Mesterséges Intelligencia Projekt")
