# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Datenvorverarbeitungsfunktion ---
def preprocess_penguin_data(file_path):
    """Lädt und verarbeitet den Pinguin-Datensatz."""
    penguin_df = pd.read_csv(file_path)
    penguin_df.dropna(inplace=True)

    # One-Hot-Encoding für kategoriale Spalten
    penguin_df = pd.get_dummies(penguin_df, columns=['island', 'sex'], drop_first=False) # drop_first=False um alle Spalten zu erhalten

    # Zielvariable (y) und Merkmale (x) definieren
    y = penguin_df['species']
    x = penguin_df.drop('species', axis=1)

    return x, y, x.columns # Spaltennamen zurückgeben

# --- Streamlit App Layout ---

# Titel der App festlegen
st.title('🐧 Pinguin-Arten-Klassifikator')
st.write("Diese App verwendet ein **Random Forest**-Modell, um die Art von Pinguinen basierend auf ihren körperlichen Merkmalen vorherzusagen.")

# --- Sidebar für Benutzereingaben ---
st.sidebar.header('Pinguin-Merkmale anpassen')

# Schieberegler für die Anzahl der Bäume im Random Forest
n_estimators = st.sidebar.slider('Anzahl der Bäume (n_estimators)',
                                 min_value=50,
                                 max_value=500,
                                 value=100,
                                 step=50,
                                 help="Passe die Anzahl der Bäume im Forest an. Mehr Bäume können zu einer besseren Genauigkeit führen, aber auch zu einer längeren Trainingszeit.")

# Schieberegler und Auswahlboxen für Pinguin-Merkmale
culmen_length = st.sidebar.slider('Schnabellänge (mm)', 32.1, 59.6, 43.9)
culmen_depth = st.sidebar.slider('Schnabeltiefe (mm)', 13.1, 21.5, 17.2)
flipper_length = st.sidebar.slider('Flossenlänge (mm)', 172.0, 231.0, 200.0)
body_mass = st.sidebar.slider('Körpermasse (g)', 2700.0, 6300.0, 4200.0)
island = st.sidebar.selectbox('Insel', ('Biscoe', 'Dream', 'Torgersen'))
sex = st.sidebar.selectbox('Geschlecht', ('MALE', 'FEMALE'))


# --- Hauptseiteninhalt ---

# 1. Daten laden und anzeigen
st.header('1. Datenübersicht')

try:
    x, y, feature_names = preprocess_penguin_data("penguins_size.csv")
    st.success('Daten erfolgreich geladen und vorverarbeitet.')
    st.write('**Datenvorschau:**')
    # Merkmale und Ziel für die Anzeige kombinieren
    preview_df = x.join(y)
    st.dataframe(preview_df.head())
except FileNotFoundError:
    st.error('Fehler: Die Datei "penguins_size.csv" wurde nicht gefunden. Bitte stelle sicher, dass sie sich im selben Verzeichnis wie dieses Skript befindet.')
    st.stop() # Skript anhalten, wenn die Daten nicht geladen werden können

# Daten in Trainings- und Testsets aufteilen
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 2. Random Forest-Modell trainieren
st.header('2. Modelltraining')
with st.spinner(f'Trainiere das Random Forest-Modell mit {n_estimators} Bäumen...'):
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    rf.fit(x_train, y_train)
st.success('Modelltraining abgeschlossen!')


# --- Pinguin-Identifikation ---
st.header('Identifiziere einen Pinguin')
if st.button('Pinguin identifizieren'):
    # Benutzereingaben sammeln
    island_biscoe = 1 if island == 'Biscoe' else 0
    island_dream = 1 if island == 'Dream' else 0
    island_torgersen = 1 if island == 'Torgersen' else 0

    sex_male = 1 if sex == 'MALE' else 0
    sex_female = 1 if sex == 'FEMALE' else 0

    # Einen DataFrame aus den Eingaben erstellen
    # Die Reihenfolge der Spalten muss mit den Trainingsdaten übereinstimmen
    input_data = pd.DataFrame([[culmen_length, culmen_depth, flipper_length, body_mass,
                                island_biscoe, island_dream, island_torgersen,
                                sex_female, sex_male]],
                              columns=[ 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g',
                                        'island_Biscoe', 'island_Dream', 'island_Torgersen', 'sex_FEMALE', 'sex_MALE'])

    # Sicherstellen, dass alle Spalten aus den Trainingsdaten vorhanden sind
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    # Spalten neu anordnen, um der Reihenfolge des Trainingsdatensatzes zu entsprechen
    input_data = input_data[feature_names]


    # Vorhersage
    prediction = rf.predict(input_data)
    prediction_proba = rf.predict_proba(input_data)

    st.subheader('Vorhersage')
    st.write(f'Die vorhergesagte Pinguinart ist: **{prediction[0]}**')

    st.subheader('Vorhersagewahrscheinlichkeit')
    st.write(pd.DataFrame(prediction_proba, columns=rf.classes_, index=['Wahrscheinlichkeit']))


# 3. Modell bewerten
st.header('3. Modellbewertung')
y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Die Genauigkeit des Modells auf den Testdaten beträgt **{accuracy:.2%}**.")

# --- Detaillierte Metriken anzeigen ---
st.subheader('Klassifikationsbericht')
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.subheader('Konfusionsmatrix')
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)

# Eine Abbildung für den Plot erstellen
fig, ax = plt.subplots()
cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
fig.colorbar(cax)

# Über die Daten dimensionen loopen und Textanmerkungen erstellen
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='large', color='black')

# Beschriftungen und Titel festlegen
ax.set_xticks(np.arange(len(rf.classes_)))
ax.set_xticklabels(rf.classes_, rotation=45)
ax.set_yticks(np.arange(len(rf.classes_)))
ax.set_yticklabels(rf.classes_)
plt.xlabel('Vorhergesagte Art', fontsize=12)
plt.ylabel('Tatsächliche Art', fontsize=12)
plt.title('Konfusionsmatrix', fontsize=15)

# Den Plot in Streamlit anzeigen
st.pyplot(fig)