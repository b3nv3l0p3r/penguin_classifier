import streamlit as st
import pandas as pd
import pickle
from PIL import Image

st.set_page_config(page_title="Pinguin Klassifikator", layout="wide")

st.title("🐧 Pinguin Klassifikator")

# --- MODELLE LADEN ---
def load_model(path):
    """Lädt ein Pickle-Modell und gibt None zurück, wenn es nicht gefunden wird."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Lade beide Modelle und ihre Mappings zu Beginn
rfc_model = load_model('datasets/rfc_penguin.pkl')
rfc_uniques = load_model('datasets/output_penguin.pkl')
dtc_model = load_model('datasets/dtc_penguin.pkl')
dtc_uniques = load_model('datasets/output_penguin_dtc.pkl')


st.write(
    "Diese App nutzt ein Machine-Learning-Modell, um die Spezies von Pinguinen "
    "anhand ihrer Merkmale vorherzusagen. Gib die Daten in der Seitenleiste ein."
)

# Lade das Pinguin-Bild, ohne eine Warnung anzuzeigen, wenn es fehlt
try:
    penguin_image = Image.open('penguin.png')
    st.image(penguin_image, width=300, use_column_width='auto')
except FileNotFoundError:
    pass

st.header("Modelltraining")
st.write("Trainiere die Modelle neu, um Vorhersagen zu treffen.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Random Forest Modell trainieren"):
        st.switch_page("pages/2_Random_Forrest.py")


with col2:
    if st.button("Decision Tree Modell trainieren"):
        st.switch_page("pages/3_Entscheidungsbaum.py")


# Modell auswählen
st.sidebar.header("Modell auswählen")
model_choice = st.sidebar.selectbox("Wähle ein Modell für die Vorhersage", ("Random Forest", "Decision Tree"))


# --- HIER IST DIE GEWÜNSCHTE FEHLERMELDUNG IMPLEMENTIERT ---
# Versuche, das trainierte Modell und die Klassen zu laden
try:
    if model_choice == "Random Forest":
        # Diese Variablen werden am Anfang des Skripts geladen
        model = rfc_model
        unique_penguin_mapping = rfc_uniques
    else: # Decision Tree
        model = dtc_model
        unique_penguin_mapping = dtc_uniques
    
    # Prüfen, ob das Modell erfolgreich geladen wurde
    if model is None:
        raise FileNotFoundError

# Wenn die Dateien nicht gefunden werden, zeige eine Fehlermeldung an
except FileNotFoundError:
    st.error(
        f"Modell-Dateien für '{model_choice}' nicht gefunden! 😱\n"
        f"Bitte gehe zuerst zur entsprechenden Seite und trainiere das Modell."
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

# Holen der Feature-Namen aus dem geladenen Modell
# Sicherstellen, dass die Spalten exakt mit denen des Trainings übereinstimmen.
final_df = input_df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)


# Mache die Vorhersage, wenn der Button geklickt wird
if st.sidebar.button('Vorhersage starten'):
    prediction = model.predict(final_df)
    predicted_species = unique_penguin_mapping[prediction][0]

    st.subheader(f"Die vorhergesagte Pinguin-Spezies ist:")
    st.success(f"**{predicted_species}**")

    st.subheader("Wahrscheinlichkeitsverteilung")

    # Nutze predict_proba, um die Wahrscheinlichkeiten zu erhalten
    probabilities = model.predict_proba(final_df)

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


# --- BEREICH: MODELLVERGLEICH MIT BEISPIELDATEN (JETZT AM ENDE) ---
st.divider()
st.header("Beispieldaten & Modellvergleich")
st.write("Hier siehst du, wie die trainierten Modelle drei typische Pinguine aus dem Datensatz klassifizieren.")

try:
    # Lade und bereite die Daten für die Beispiele vor
    penguin_df = pd.read_csv('penguins_size.csv')
    penguin_df.dropna(inplace=True)
    penguin_df.reset_index(drop=True, inplace=True)

    # Wähle je ein Beispiel pro Spezies
    adelie_sample = penguin_df[penguin_df['species'] == 'Adelie'].iloc[0]
    chinstrap_sample = penguin_df[penguin_df['species'] == 'Chinstrap'].iloc[0]
    gentoo_sample = penguin_df[penguin_df['species'] == 'Gentoo'].iloc[0]
    
    samples_df = pd.DataFrame([adelie_sample, chinstrap_sample, gentoo_sample])
    samples_features = samples_df.drop(columns='species')
    samples_encoded = pd.get_dummies(samples_features)
    
    # Vorhersagen für die Beispiele machen
    rfc_pred, dtc_pred = None, None
    if rfc_model is not None and rfc_uniques is not None:
        rfc_samples_final = samples_encoded.reindex(columns=rfc_model.feature_names_in_, fill_value=0)
        rfc_pred_indices = rfc_model.predict(rfc_samples_final)
        rfc_pred = rfc_uniques[rfc_pred_indices]

    if dtc_model is not None and dtc_uniques is not None:
        dtc_samples_final = samples_encoded.reindex(columns=dtc_model.feature_names_in_, fill_value=0)
        dtc_pred_indices = dtc_model.predict(dtc_samples_final)
        dtc_pred = dtc_uniques[dtc_pred_indices]

    # Spalten-Layout
    col_ex1, col_ex2, col_ex3 = st.columns(3, gap="large")

    # Funktion zur Anzeige einer Vorhersage
    def show_prediction(model_name, prediction, true_species):
        if prediction is None:
            st.warning(f"_{model_name}-Modell nicht trainiert._")
            return
        
        is_correct = (prediction == true_species)
        emoji = "✅" if is_correct else "❌"
        st.metric(label=f"Vorhersage: {model_name}", value=prediction, delta=f"Korrekt {emoji}", delta_color="off")

    # Mapping für schönere Spaltennamen
    column_name_mapping = {
        'island': 'Insel',
        'culmen_length_mm': 'Schnabellänge (mm)',
        'culmen_depth_mm': 'Schnabeltiefe (mm)',
        'flipper_length_mm': 'Flossenlänge (mm)',
        'body_mass_g': 'Körpermasse (g)',
        'sex': 'Geschlecht'
    }

    with col_ex1:
        st.subheader("Adelie Pinguin")
        display_adelie = adelie_sample.drop('species').rename(index=column_name_mapping)
        st.write(display_adelie.astype(str))
        st.divider()
        show_prediction("Random Forest", rfc_pred[0] if rfc_pred is not None else None, "Adelie")
        show_prediction("Decision Tree", dtc_pred[0] if dtc_pred is not None else None, "Adelie")

    with col_ex2:
        st.subheader("Chinstrap Pinguin")
        display_chinstrap = chinstrap_sample.drop('species').rename(index=column_name_mapping)
        st.write(display_chinstrap.astype(str))
        st.divider()
        show_prediction("Random Forest", rfc_pred[1] if rfc_pred is not None else None, "Chinstrap")
        show_prediction("Decision Tree", dtc_pred[1] if dtc_pred is not None else None, "Chinstrap")

    with col_ex3:
        st.subheader("Gentoo Pinguin")
        display_gentoo = gentoo_sample.drop('species').rename(index=column_name_mapping)
        st.write(display_gentoo.astype(str))
        st.divider()
        show_prediction("Random Forest", rfc_pred[2] if rfc_pred is not None else None, "Gentoo")
        show_prediction("Decision Tree", dtc_pred[2] if dtc_pred is not None else None, "Gentoo")

except FileNotFoundError:
    st.error("`penguins_size.csv` nicht gefunden, um Beispieldaten zu laden.")
except Exception as e:
    st.error(f"Ein Fehler beim Laden der Beispieldaten ist aufgetreten: {e}")