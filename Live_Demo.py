import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import matplotlib.pyplot as plt
import graphviz
import io

from one_hot_encoder import preprocess_penguin_data

# Seitenkonfiguration auf "wide" setzen
st.set_page_config(layout="wide")
st.title("🐧 Pinguin-Arterkennung")
st.write("Identifizieren Sie Pinguinarten anhand von körperlichen Messwerten.")

# Initialisierung des Session State für die App-Logik und Beispieldaten

# Flag, um zu steuern, ob die Analyse-Logik ausgeführt wurde
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
    
# Standardwerte für die Eingabefelder im Session State speichern
if 'culmen_length' not in st.session_state:
    st.session_state.culmen_length = 40.0
if 'culmen_depth' not in st.session_state:
    st.session_state.culmen_depth = 18.0
if 'flipper_length' not in st.session_state:
    st.session_state.flipper_length = 200.0
if 'body_mass' not in st.session_state:
    st.session_state.body_mass = 4000.0
if 'island' not in st.session_state:
    st.session_state.island = "Biscoe"
if 'sex' not in st.session_state:
    st.session_state.sex = "MALE"
    
# Standardwerte für die Modellparameter
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Random Forest"
if 'n_estimators' not in st.session_state:
    st.session_state.n_estimators = 100
if 'max_depth' not in st.session_state:
    st.session_state.max_depth = 5

# Beispieldatensätze für die Buttons in der Sidebar
example_data = {
    "Adelie": {
        'culmen_length': 38.8, 'culmen_depth': 18.3, 'flipper_length': 190.0,
        'body_mass': 3700.0, 'island': "Torgersen", 'sex': "MALE"
    },
    "Gentoo": {
        'culmen_length': 47.5, 'culmen_depth': 14.9, 'flipper_length': 217.0,
        'body_mass': 5000.0, 'island': "Biscoe", 'sex': "FEMALE"
    },
    "Chinstrap": {
        'culmen_length': 48.8, 'culmen_depth': 18.4, 'flipper_length': 196.0,
        'body_mass': 3730.0, 'island': "Dream", 'sex': "MALE"
    }
}

# Funktion, um die Eingabefelder (im Session State) mit Beispieldaten zu füllen
def load_example(species):
    data = example_data[species]
    st.session_state.culmen_length = data['culmen_length']
    st.session_state.culmen_depth = data['culmen_depth']
    st.session_state.flipper_length = data['flipper_length']
    st.session_state.body_mass = data['body_mass']
    st.session_state.island = data['island']
    st.session_state.sex = data['sex']

# Laden und Vorverarbeiten der Pinguin-Daten beim Start
try:
    X, y = preprocess_penguin_data('penguins_size.csv')
except FileNotFoundError:
    # Kritischer Fehler, wenn die Datendatei fehlt
    st.error("FEHLER: Die Datei 'penguins_size.csv' wurde nicht gefunden. Stellen Sie sicher, dass sie sich im selben Verzeichnis befindet.")
    st.stop()

# Definition der Streamlit Sidebar (Eingabefelder und Buttons)

st.sidebar.header("🚀 Beispiel-Pinguine")
# Buttons zum Laden von Beispielen
st.sidebar.button("Adelie-Beispiel laden", on_click=load_example, args=("Adelie",), width="stretch")
st.sidebar.button("Gentoo-Beispiel laden", on_click=load_example, args=("Gentoo",), width="stretch")
st.sidebar.button("Chinstrap-Beispiel laden", on_click=load_example, args=("Chinstrap",), width="stretch")
st.sidebar.divider()

# Ein Formular bündelt alle Eingaben und verhindert einen Rerun bei jeder Änderung
with st.sidebar.form(key='eingabe_formular'):
    st.header("📊 Pinguin-Messwerte")

    island_options = ["Biscoe", "Dream", "Torgersen"]
    sex_options = ["MALE", "FEMALE"]

    # Die Widgets verwenden die Werte aus dem Session State als Standard
    culmen_length_form = st.slider("Schnabellänge (mm)", 30.0, 60.0, st.session_state.culmen_length, 0.1)
    culmen_depth_form = st.slider("Schnabeltiefe (mm)", 13.0, 22.0, st.session_state.culmen_depth, 0.1)
    flipper_length_form = st.slider("Flossenlänge (mm)", 170.0, 230.0, st.session_state.flipper_length, 1.0)
    body_mass_form = st.slider("Körpermasse (g)", 2500.0, 6500.0, st.session_state.body_mass, 50.0)
    island_form = st.selectbox("Insel", island_options, index=island_options.index(st.session_state.island))
    sex_form = st.selectbox("Geschlecht", sex_options, index=sex_options.index(st.session_state.sex))

    st.header("🤖 Modell-Einstellungen")
    model_type_form = st.selectbox("Modell-Typ", ["Random Forest", "Entscheidungsbaum"], index=0 if st.session_state.model_type == "Random Forest" else 1)

    # Bedingte Anzeige von Hyperparametern je nach Modellwahl
    if model_type_form == "Random Forest":
        n_estimators_form = st.slider("Anzahl der Bäume", 10, 200, st.session_state.n_estimators, 10)
        max_depth_form = st.slider("Maximale Baumtiefe", 3, 10, st.session_state.max_depth, 1)
    else:
        n_estimators_form = 100 # Dummy-Wert, wird nicht verwendet
        max_depth_form = st.slider("Maximale Baumtiefe", 3, 10, st.session_state.max_depth, 1)
    
    submitted = st.form_submit_button("Starte Analyse und Vorhersage")

# Logik, die ausgeführt wird, wenn der 'Submit'-Button gedrückt wird
if submitted:
    # Alle Werte aus dem Formular werden in den Session State übertragen
    st.session_state.culmen_length = culmen_length_form
    st.session_state.culmen_depth = culmen_depth_form
    st.session_state.flipper_length = flipper_length_form
    st.session_state.body_mass = body_mass_form
    st.session_state.island = island_form
    st.session_state.sex = sex_form
    st.session_state.model_type = model_type_form
    st.session_state.n_estimators = n_estimators_form
    st.session_state.max_depth = max_depth_form
    
    # Setzt das Flag, um den Haupt-Analyseblock unten auszulösen
    st.session_state.analysis_run = True

# Hilfsfunktion, um die Benutzereingaben in einen DataFrame mit One-Hot-Encoding umzuwandeln
def create_input_dataframe(culmen_length, culmen_depth, flipper_length, 
                           body_mass, island, sex, feature_columns):
    # Erstellt einen leeren DataFrame mit den korrekten Spalten (aus den Trainingsdaten)
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns, dtype=float)
    input_data['culmen_length_mm'] = culmen_length
    input_data['culmen_depth_mm'] = culmen_depth
    input_data['flipper_length_mm'] = flipper_length
    input_data['body_mass_g'] = body_mass
    
    # Setzt die entsprechende One-Hot-Spalte für 'island'
    island_col = f'island_{island}'
    if island_col in feature_columns:
        input_data[island_col] = 1
    
    # Setzt die entsprechende One-Hot-Spalte für 'sex'
    sex_col = f'sex_{sex}'
    if sex_col in feature_columns:
        input_data[sex_col] = 1
    
    return input_data

# Hauptteil der Anwendung: Wird nur ausgeführt, wenn das Formular abgeschickt wurde

# Dieser Block wird nur ausgeführt, wenn 'analysis_run' (durch Formular-Submit) True ist
if st.session_state.analysis_run:
    
    # Modell-Instanziierung basierend auf der Auswahl im Session State
    if st.session_state.model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=st.session_state.n_estimators, max_depth=st.session_state.max_depth, random_state=42)
        n_estimators = st.session_state.n_estimators 
        model_type = "Random Forest"
    else:
        model = DecisionTreeClassifier(max_depth=st.session_state.max_depth, random_state=42)
        n_estimators = 1 # Dummy-Wert
        model_type = "Entscheidungsbaum"

    # Training des Modells mit den geladenen Daten (X, y)
    with st.spinner("Modell wird trainiert..."):
        model.fit(X, y)

    # Erstellen des One-Hot-kodierten Input-DataFrames für die Vorhersage
    input_data = create_input_dataframe(
        st.session_state.culmen_length, st.session_state.culmen_depth, st.session_state.flipper_length, 
        st.session_state.body_mass, st.session_state.island, st.session_state.sex, 
        X.columns
    )

    # Ein ausklappbarer Bereich zur Überprüfung der Feature-Namen und Input-Werte
    with st.expander("🔍 Debug-Info - Feature-Spalten"):
        st.write("**Trainings-Features:**", X.columns.tolist())
        st.write("**Input-Daten-Spalten:**", input_data.columns.tolist())
        st.write("**Input-Daten:**")
        st.dataframe(input_data)

    # Durchführung der Vorhersage
    prediction = model.predict(input_data)
    # Abrufen der Wahrscheinlichkeiten für jede Klasse
    prediction_proba = model.predict_proba(input_data)

    st.header("🎯 Vorhersage-Ergebnis")
    
    species_names = model.classes_
    predicted_species = prediction[0]

    # Anzeige der Ergebnisse in Metriken
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vorhergesagte Art", predicted_species)
    with col2:
        st.metric("Konfidenz", f"{max(prediction_proba[0]) * 100:.1f}%")
    with col3:
        st.metric("Modell-Typ", model_type)

    # Visualisierung der Vorhersagewahrscheinlichkeiten als Balkendiagramm
    st.subheader("Wahrscheinlichkeitsverteilung")
    prob_df = pd.DataFrame({
        'Art': species_names,
        'Wahrscheinlichkeit': prediction_proba[0]
    })
    st.bar_chart(prob_df.set_index('Art'))

    st.header("🌳 Visualisierung des Entscheidungsbaums")
    st.write("Verstehen Sie, wie das Modell seine Vorhersagen trifft, indem Sie die Baumstruktur erkunden:")

    tab1, tab2, tab3 = st.tabs(["📊 Baumstruktur", "🔍 Graphviz-Ansicht", "🌲 Bäume erkunden"])

    with tab1:
        st.subheader("Baumstruktur (Matplotlib)")
        
        # Beim Random Forest wird standardmäßig nur der erste Baum angezeigt
        if model_type == "Random Forest":
            tree_to_plot = model.estimators_[0]
            st.info(f"Zeige den ersten Baum des Random Forest (von {n_estimators} Bäumen)")
        else:
            tree_to_plot = model
        
        # Zeichnen des Baums mit Matplotlib (plot_tree)
        fig, ax = plt.subplots(figsize=(25, 15))
        plot_tree(tree_to_plot,
                  feature_names=X.columns.tolist(),
                  class_names=species_names,
                  filled=True,
                  rounded=True,
                  fontsize=10,
                  ax=ax,
                  impurity=True,
                  proportion=True)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Erstellen eines Download-Buttons für das Baum-Bild
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label="📥 Baum-Bild herunterladen",
            data=buf,
            file_name="entscheidungsbaum.png",
            mime="image/png"
        )

    with tab2:
        st.subheader("Graphviz-Visualisierung")
        st.write("Professionelle Baumvisualisierung mit besserem Layout:")
        
        # Auswahl des Baums (erster für RF, sonst der einzelne Baum)
        if model_type == "Random Forest":
            tree_to_export = model.estimators_[0]
        else:
            tree_to_export = model
        
        # Exportieren der Baumstruktur in das DOT-Format (für Graphviz)
        dot_data = export_graphviz(
            tree_to_export,
            out_file=None,
            feature_names=X.columns.tolist(),
            class_names=species_names,
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=4, 
            impurity=True,
            proportion=True
        )
        
        # Rendern des DOT-Strings mit Graphviz
        graph = graphviz.Source(dot_data)
        st.graphviz_chart(graph, width="stretch")
        
        st.info("💡 Tipp: Rechtsklick und 'Bild in neuem Tab öffnen' für eine größere Ansicht")

    with tab3:
        # Dieser Tab ist nur für Random Forest aktiv
        if model_type == "Random Forest":
            st.subheader("Einzelne Bäume des Forests erkunden")
            st.write(f"Ihr Random Forest enthält {n_estimators} Entscheidungsbäume. Jeder Baum sieht die Daten leicht unterschiedlich.")
            
            # Slider zur Auswahl eines einzelnen Baums aus dem Forest
            tree_index = st.slider(
                "Baum zur Visualisierung auswählen", 
                0, 
                len(model.estimators_) - 1, 
                0,
                key="tree_index_slider", # Der Key stellt sicher, dass der Wert des Sliders erhalten bleibt
                help="Jeder Baum im Forest kann unterschiedliche Entscheidungen treffen"
            )
            
            st.write(f"**Zeige Baum #{tree_index + 1}**")
            
            # Zeichnen des ausgewählten Baums
            fig, ax = plt.subplots(figsize=(25, 15))
            plot_tree(model.estimators_[tree_index],
                      feature_names=X.columns.tolist(),
                      class_names=species_names,
                      filled=True,
                      rounded=True,
                      fontsize=10,
                      ax=ax,
                      impurity=True)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Anzeigen von Metriken für den ausgewählten Baum
            tree = model.estimators_[tree_index].tree_
            st.write(f"**Baum-Statistiken:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Baumtiefe", tree.max_depth)
            with col2:
                st.metric("Anzahl der Blätter", tree.n_leaves)
            with col3:
                st.metric("Anzahl der Knoten", tree.node_count)
            
        else:
            st.info("🌳 Diese Option ist nur für Random Forest-Modelle verfügbar. Wechseln Sie in der Seitenleiste zu 'Random Forest', um mehrere Bäume zu erkunden.")

    # Anzeige der Feature Importances des trainierten Modells
    st.header("📊 Merkmalswichtigkeit")
    st.write("Welche Merkmale sind am wichtigsten für die Vorhersage der Pinguinart?")

    feature_importance = model.feature_importances_
    feature_names = X.columns.tolist()

    importance_df = pd.DataFrame({
        'Merkmal': feature_names,
        'Wichtigkeit': feature_importance
    }).sort_values('Wichtigkeit', ascending=False)

    # Aufteilung in Diagramm und Tabelle
    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(importance_df.set_index('Merkmal'))
    with col2:
        st.dataframe(importance_df.style.format({'Wichtigkeit': '{:.4f}'}))

else:
    # Info-Text, der angezeigt wird, bevor das Formular abgeschickt wurde
    st.info("Passen Sie die Werte in der Seitenleiste an (oder laden Sie ein Beispiel) und klicken Sie auf 'Starte Analyse und Vorhersage', um ein Ergebnis zu erhalten.")


# Hilfssektion am Ende der Seite
st.header("📖 Wie man den Entscheidungsbaum liest")

# Ein ausklappbarer Bereich (Expander) mit Erklärungen
with st.expander("Klicken Sie hier, um zu erfahren, wie man die Baumvisualisierung interpretiert"):
    st.markdown("""
    ### Die Baumstruktur verstehen
    Jeder **Knoten** (Box) im Baum repräsentiert einen Entscheidungspunkt oder ein Ergebnis:
    #### 🔵 Interne Knoten (Entscheidungspunkte)
    - **Bedingung**: Zeigt das Merkmal und den Schwellenwert (z.B. "flipper_length_mm <= 206.5")
    - **Gini/Impurity (Unreinheit)**: Misst, wie gemischt die Klassen an diesem Knoten sind (niedriger = reiner)
    - **Samples**: Anzahl der Trainingsbeispiele, die diesen Knoten erreicht haben
    - **Value**: Verteilung der Klassen (z.B. [Adelie, Gentoo, Chinstrap])
    - **Class**: Die Mehrheitsklasse an diesem Knoten
    #### 🍃 Blattknoten (Endgültige Entscheidungen)
    - Endknoten, an denen Vorhersagen getroffen werden
    - Die Farbintensität zeigt die Konfidenz der Vorhersage an
    - Stärkere Farbe = sicherere Vorhersage
    #### 🎨 Farbkodierung
    - Die Farben werden den Klassen (z.B. Adelie, Gentoo, Chinstrap) zugeordnet, um die Mehrheit in jedem Knoten visuell darzustellen.
    #### 📊 Entscheidungspfad
    1. Beginnen Sie am **Wurzelknoten** (oben)
    2. Folgen Sie den Ästen basierend auf Ihren Merkmalswerten
    3. **Linker Ast**: Bedingung ist WAHR
    4. **Rechter Ast**: Bedingung ist FALSCH
    5. Fahren Sie fort, bis Sie einen **Blattknoten** (unten) erreichen
    6. Die Klasse des Blattknotens ist die Vorhersage
    #### 💡 Beispiel
    Wenn Flossenlänge ≤ 206.5 mm → gehe nach links  
    Wenn Flossenlänge > 206.5 mm → gehe nach rechts
    ### Merkmalswichtigkeit
    Merkmale, die weiter oben im Baum erscheinen, sind im Allgemeinen wichtiger für die Klassifizierung.
    """)