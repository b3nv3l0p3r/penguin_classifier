import streamlit as st
import pandas as pd
import pickle
from PIL import Image

st.set_page_config(page_title="Pinguin Klassifikator")

st.title("🐧 Pinguin Klassifikator")

st.write(
    "Diese App nutzt ein Machine-Learning-Modell, um die Spezies von Pinguinen "
    "anhand ihrer Merkmale vorherzusagen. Gib die Daten in der Seitenleiste ein."
)

# Lade das Pinguin-Bild, ohne eine Warnung anzuzeigen, wenn es fehlt
try:
    penguin_image = Image.open('penguin.png')
    st.image(penguin_image, width=300)
except FileNotFoundError:
    pass

# --- HIER IST DIE GEWÜNSCHTE FEHLERMELDUNG IMPLEMENTIERT ---
# Versuche, das trainierte Modell und die Klassen zu laden
try:
    with open('rfc_penguin.pkl', 'rb') as f:
        rfc = pickle.load(f)
    with open('output_penguin.pkl', 'rb') as f:
        unique_penguin_mapping = pickle.load(f)
# Wenn die Dateien nicht gefunden werden, zeige eine Fehlermeldung an
except FileNotFoundError:
    st.error(
        "Modell-Dateien nicht gefunden! 😱\n"
        "Bitte gehe zuerst zur Seite **'2_Datenerstellung'** und trainiere das Modell."
    )
    # Stoppe die Ausführung der Seite, da ohne Modell nichts funktioniert
    st.stop()
# -----------------------------------------------------------------


# Eingabefelder in der Sidebar
st.sidebar.header("Pinguin-Merkmale eingeben")

island = st.sidebar.selectbox('Insel', ('Biscoe', 'Dream', 'Torgersen'))
sex = st.sidebar.selectbox('Geschlecht', ('MALE', 'FEMALE'))

culmen_length = st.sidebar.slider('Schnabellänge (mm)', 30.0, 60.0, 43.9)
culmen_depth = st.sidebar.slider('Schnabeltiefe (mm)', 13.0, 22.0, 17.2)
flipper_length = st.sidebar.slider('Flossenlänge (mm)', 170.0, 235.0, 201.0)
body_mass = st.sidebar.slider('Körpermasse (g)', 2700.0, 6300.0, 4200.0)

# Erstelle einen DataFrame aus den User-Eingaben
user_input = {
    'island': island,
    'sex': sex,
    'culmen_length_mm': culmen_length,
    'culmen_depth_mm': culmen_depth,
    'flipper_length_mm': flipper_length,
    'body_mass_g': body_mass,
}
input_df = pd.DataFrame([user_input])

# Wende One-Hot-Encoding auf die Eingabedaten an
input_df_encoded = pd.get_dummies(input_df)

# Sorge dafür, dass die Spalten exakt mit denen des Trainings übereinstimmen.
final_df = input_df_encoded.reindex(columns=rfc.feature_names_in_, fill_value=0)


# Mache die Vorhersage, wenn der Button geklickt wird
if st.sidebar.button('Vorhersage starten'):
    prediction = rfc.predict(final_df)
    predicted_species = unique_penguin_mapping[prediction][0]

    st.subheader(f"Die vorhergesagte Pinguin-Spezies ist:")
    st.success(f"**{predicted_species}**")

    st.subheader("Wahrscheinlichkeitsverteilung")

    # Nutze predict_proba, um die Wahrscheinlichkeiten zu erhalten
    probabilities = rfc.predict_proba(final_df)

    # Erstelle einen DataFrame für eine saubere Darstellung
    prob_df = pd.DataFrame(
        data=probabilities[0],
        index=unique_penguin_mapping,
        columns=["Wahrscheinlichkeit"]
    )
    
    # Formatiere die Wahrscheinlichkeit als Prozentzahl und sortiere die Tabelle
    prob_df = prob_df.sort_values(by="Wahrscheinlichkeit", ascending=False)
    prob_df["Wahrscheinlichkeit"] = prob_df["Wahrscheinlichkeit"].apply(lambda x: f"{x:.2%}")

    st.dataframe(prob_df)
else:
    st.info("Passe die Merkmale in der Seitenleiste an und klicke auf 'Vorhersage starten'.")