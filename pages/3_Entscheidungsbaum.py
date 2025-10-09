import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

st.set_page_config(page_title="Modelltraining Decision Tree")

st.title("🧪 Datenerstellung & Modelltraining (Decision Tree)")

st.write(
    "Auf dieser Seite kannst du das Machine-Learning-Modell mit dem Decision Tree Algorithmus neu trainieren. "
    "Passe die Modell-Parameter an und klicke auf den Button, um den Prozess zu starten."
)

# --- PARAMETER ---
st.subheader("Modell-Parameter")
max_depth = st.slider(
    'Maximale Tiefe des Baumes (max_depth)',
    min_value=1,
    max_value=50,
    value=10,
    step=1,
    help="Wähle die maximale Tiefe des Entscheidungsbaums. Ein tieferer Baum kann komplexere Muster lernen, neigt aber auch zu Overfitting."
)

min_samples_leaf = st.slider(
    'Minimale Anzahl an Samples pro Blatt (min_samples_leaf)',
    min_value=1,
    max_value=50,
    value=1,
    step=1,
    help="Die minimale Anzahl an Samples, die in einem Blattknoten erforderlich sind. Erhöhen dieses Wertes kann Overfitting reduzieren."
)
# ---------------------------------------------

# Button, um den Trainingsprozess zu starten
if st.button("Modell jetzt trainieren"):
    try:
        st.info(f"Lade Pinguin-Daten und trainiere Decision Tree Modell mit max_depth={max_depth} und min_samples_leaf={min_samples_leaf}...")

        penguin_df = pd.read_csv('penguins_size.csv')
        penguin_df.dropna(inplace=True)

        output = penguin_df['species']
        features = penguin_df[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
        features = pd.get_dummies(features)
        output, uniques = pd.factorize(output)

        st.success("Daten erfolgreich vorbereitet.")

        # Teile die Daten in Trainings- und Testsets auf (KORRIGIERT)
        x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.2, random_state=42)

        st.info("Trainiere das Decision Tree Modell...")

        dtc = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=15)
        dtc.fit(x_train, y_train)

        y_pred = dtc.predict(x_test)
        score = round(accuracy_score(y_pred, y_test), 2)

        st.success(f"Modell erfolgreich trainiert! Genauigkeit: {score*100} %")

        st.info("Speichere das trainierte Modell und die Output-Klassen...")

        with open('datasets/dtc_penguin.pkl', 'wb') as f:
            pickle.dump(dtc, f)

        with open('datasets/output_penguin_dtc.pkl', 'wb') as f:
            pickle.dump(uniques, f)

        st.success("Modell und Klassen wurden erfolgreich gespeichert! ✅")
        st.balloons()

    except FileNotFoundError:
        st.error("Fehler: Die Datei `penguins_size.csv` wurde nicht im Hauptverzeichnis gefunden.")
    except KeyError as e:
        st.error(f"Fehler bei der Spaltenauswahl: Die Spalte {e} wurde in der CSV-Datei nicht gefunden. "
                 f"Bitte überprüfe die Spaltennamen in `penguins_size.csv`.")