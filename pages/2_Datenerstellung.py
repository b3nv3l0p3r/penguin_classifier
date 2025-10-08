import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

st.set_page_config(page_title="Modelltraining")

st.title("🧪 Datenerstellung & Modelltraining")

st.write(
    "Auf dieser Seite kannst du das Machine-Learning-Modell neu trainieren. "
    "Passe die Modell-Parameter an und klicke auf den Button, um den Prozess zu starten."
)

# --- NEU: SLIDER FÜR DIE ANZAHL DER BÄUME ---
st.subheader("Modell-Parameter")
n_estimators = st.slider(
    'Anzahl der Bäume (n_estimators)',
    min_value=10, 
    max_value=1000, 
    value=100, 
    step=10,
    help="Wähle die Anzahl der Entscheidungsbäume im Random Forest. Mehr Bäume können das Modell genauer, aber auch langsamer machen."
)
# ---------------------------------------------

# Button, um den Trainingsprozess zu starten
if st.button("Modell jetzt trainieren"):
    try:
        st.info(f"Lade Pinguin-Daten und trainiere Modell mit **{n_estimators} Bäumen**...")
        
        # Lese die korrekte CSV-Datei
        penguin_df = pd.read_csv('penguins_size.csv')
        
        # Entferne Zeilen mit fehlenden Werten
        penguin_df.dropna(inplace=True)

        # Trenne Features (X) und Zielvariable (y)
        output = penguin_df['species']
        features = penguin_df[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
        
        # Wandle kategorische Features mittels One-Hot-Encoding um
        features = pd.get_dummies(features)
        
        # Wandle die Zielvariable in Zahlen um und speichere die Zuordnung
        output, uniques = pd.factorize(output)
        
        st.success("Daten erfolgreich vorbereitet.")

        # Teile die Daten in Trainings- und Testsets auf
        x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.8, random_state=42)
        
        st.info("Trainiere das Random Forest Modell...")
        
        # Initialisiere und trainiere das Modell
        # HIER WIRD DER WERT DES SLIDERS VERWENDET
        rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=15)
        rfc.fit(x_train, y_train)
        
        # Teste das Modell und erhalte die Genauigkeit
        y_pred = rfc.predict(x_test)
        score = round(accuracy_score(y_pred, y_test), 2)
        
        st.success(f"Modell erfolgreich trainiert! Genauigkeit: {score}")

        st.info("Speichere das trainierte Modell und die Output-Klassen...")
        
        # Speichere das Modell
        with open('rfc_penguin.pkl', 'wb') as f:
            pickle.dump(rfc, f)
            
        # Speichere die Zuordnung der Pinguin-Arten
        with open('output_penguin.pkl', 'wb') as f:
            pickle.dump(uniques, f)
            
        st.success("Modell und Klassen wurden erfolgreich gespeichert! ✅")
        st.balloons()

    except FileNotFoundError:
        st.error("Fehler: Die Datei `penguins_size.csv` wurde nicht im Hauptverzeichnis gefunden.")
    except KeyError as e:
        st.error(f"Fehler bei der Spaltenauswahl: Die Spalte {e} wurde in der CSV-Datei nicht gefunden. "
                 f"Bitte überprüfe die Spaltennamen in `penguins_size.csv`.")