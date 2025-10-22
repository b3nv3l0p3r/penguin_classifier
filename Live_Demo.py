import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import matplotlib.pyplot as plt
import graphviz
import io

from one_hot_encoder import preprocess_penguin_data

st.set_page_config(layout="wide")
st.title("🐧 Pinguin-Arterkennung")
st.write("Identifizieren Sie Pinguinarten anhand von körperlichen Messwerten.")

# =========================================================================
# Session State & Beispieldaten
# =========================================================================

# FIX 1: 'analysis_run' merkt sich, ob die Analyse angezeigt werden soll.
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
    
# Initialisiere Session State für die Eingabefelder
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
    
# Initialisiere State für Modellparameter
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Random Forest"
if 'n_estimators' not in st.session_state:
    st.session_state.n_estimators = 100
if 'max_depth' not in st.session_state:
    st.session_state.max_depth = 5

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

def load_example(species):
    data = example_data[species]
    st.session_state.culmen_length = data['culmen_length']
    st.session_state.culmen_depth = data['culmen_depth']
    st.session_state.flipper_length = data['flipper_length']
    st.session_state.body_mass = data['body_mass']
    st.session_state.island = data['island']
    st.session_state.sex = data['sex']

try:
    X, y = preprocess_penguin_data('penguins_size.csv')
except FileNotFoundError:
    st.error("FEHLER: Die Datei 'penguins_size.csv' wurde nicht gefunden. Stellen Sie sicher, dass sie sich im selben Verzeichnis befindet.")
    st.stop()

# =========================================================================
# Sidebar
# =========================================================================

st.sidebar.header("🚀 Beispiel-Pinguine")
st.sidebar.button("Adelie-Beispiel laden", on_click=load_example, args=("Adelie",), width="stretch")
st.sidebar.button("Gentoo-Beispiel laden", on_click=load_example, args=("Gentoo",), width="stretch")
st.sidebar.button("Chinstrap-Beispiel laden", on_click=load_example, args=("Chinstrap",), width="stretch")
st.sidebar.divider()

with st.sidebar.form(key='eingabe_formular'):
    st.header("📊 Pinguin-Messwerte")

    island_options = ["Biscoe", "Dream", "Torgersen"]
    sex_options = ["MALE", "FEMALE"]

    # Widgets lesen ihren Standardwert aus dem session_state
    culmen_length_form = st.slider("Schnabellänge (mm)", 30.0, 60.0, st.session_state.culmen_length, 0.1)
    culmen_depth_form = st.slider("Schnabeltiefe (mm)", 13.0, 22.0, st.session_state.culmen_depth, 0.1)
    flipper_length_form = st.slider("Flossenlänge (mm)", 170.0, 230.0, st.session_state.flipper_length, 1.0)
    body_mass_form = st.slider("Körpermasse (g)", 2500.0, 6500.0, st.session_state.body_mass, 50.0)
    island_form = st.selectbox("Insel", island_options, index=island_options.index(st.session_state.island))
    sex_form = st.selectbox("Geschlecht", sex_options, index=sex_options.index(st.session_state.sex))

    st.header("🤖 Modell-Einstellungen")
    model_type_form = st.selectbox("Modell-Typ", ["Random Forest", "Entscheidungsbaum"], index=0 if st.session_state.model_type == "Random Forest" else 1)

    if model_type_form == "Random Forest":
        n_estimators_form = st.slider("Anzahl der Bäume", 10, 200, st.session_state.n_estimators, 10)
        max_depth_form = st.slider("Maximale Baumtiefe", 3, 10, st.session_state.max_depth, 1)
    else:
        n_estimators_form = 100 # Standardwert, wird nicht verwendet
        max_depth_form = st.slider("Maximale Baumtiefe", 3, 10, st.session_state.max_depth, 1)
    
    submitted = st.form_submit_button("Starte Analyse und Vorhersage")

# FIX 2: Wenn das Formular abgeschickt wird...
if submitted:
    # ...speichern wir alle Werte aus dem Formular im session_state...
    st.session_state.culmen_length = culmen_length_form
    st.session_state.culmen_depth = culmen_depth_form
    st.session_state.flipper_length = flipper_length_form
    st.session_state.body_mass = body_mass_form
    st.session_state.island = island_form
    st.session_state.sex = sex_form
    st.session_state.model_type = model_type_form
    st.session_state.n_estimators = n_estimators_form
    st.session_state.max_depth = max_depth_form
    
    # ...und setzen das Flag, dass die Analyse laufen soll.
    st.session_state.analysis_run = True


def create_input_dataframe(culmen_length, culmen_depth, flipper_length, 
                           body_mass, island, sex, feature_columns):
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns, dtype=float)
    input_data['culmen_length_mm'] = culmen_length
    input_data['culmen_depth_mm'] = culmen_depth
    input_data['flipper_length_mm'] = flipper_length
    input_data['body_mass_g'] = body_mass
    
    island_col = f'island_{island}'
    if island_col in feature_columns:
        input_data[island_col] = 1
    
    sex_col = f'sex_{sex}'
    if sex_col in feature_columns:
        input_data[sex_col] = 1
    
    return input_data

# =========================================================================
# Haupt-Analyseblock
# =========================================================================

# FIX 3: Die Analyse wird jetzt ausgeführt, wenn das Flag im session_state True ist.
# Das bleibt auch True, wenn der Slider in Tab 3 bewegt wird.
if st.session_state.analysis_run:
    
    # Modell-Objekt basierend auf Werten im session_state erstellen
    if st.session_state.model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=st.session_state.n_estimators, max_depth=st.session_state.max_depth, random_state=42)
        n_estimators = st.session_state.n_estimators # für Anzeige-Logik
        model_type = "Random Forest"
    else:
        model = DecisionTreeClassifier(max_depth=st.session_state.max_depth, random_state=42)
        n_estimators = 1 # Dummy-Wert
        model_type = "Entscheidungsbaum"

    with st.spinner("Modell wird trainiert..."):
        model.fit(X, y)

    # Input-Daten basierend auf Werten im session_state erstellen
    input_data = create_input_dataframe(
        st.session_state.culmen_length, st.session_state.culmen_depth, st.session_state.flipper_length, 
        st.session_state.body_mass, st.session_state.island, st.session_state.sex, 
        X.columns
    )

    with st.expander("🔍 Debug-Info - Feature-Spalten"):
        st.write("**Trainings-Features:**", X.columns.tolist())
        st.write("**Input-Daten-Spalten:**", input_data.columns.tolist())
        st.write("**Input-Daten:**")
        st.dataframe(input_data)

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.header("🎯 Vorhersage-Ergebnis")
    
    species_names = model.classes_
    predicted_species = prediction[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vorhergesagte Art", predicted_species)
    with col2:
        st.metric("Konfidenz", f"{max(prediction_proba[0]) * 100:.1f}%")
    with col3:
        st.metric("Modell-Typ", model_type)

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
        
        if model_type == "Random Forest":
            tree_to_plot = model.estimators_[0]
            st.info(f"Zeige den ersten Baum des Random Forest (von {n_estimators} Bäumen)")
        else:
            tree_to_plot = model
        
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
        
        if model_type == "Random Forest":
            tree_to_export = model.estimators_[0]
        else:
            tree_to_export = model
        
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
        
        graph = graphviz.Source(dot_data)
        st.graphviz_chart(graph, width="stretch")
        
        st.info("💡 Tipp: Rechtsklick und 'Bild in neuem Tab öffnen' für eine größere Ansicht")

    with tab3:
        if model_type == "Random Forest":
            st.subheader("Einzelne Bäume des Forests erkunden")
            st.write(f"Ihr Random Forest enthält {n_estimators} Entscheidungsbäume. Jeder Baum sieht die Daten leicht unterschiedlich.")
            
            # FIX 4: Dieser Slider löst jetzt einen Rerun aus,
            # aber da 'analysis_run' True bleibt, wird die Seite
            # einfach mit dem neuen Slider-Wert neu gezeichnet.
            tree_index = st.slider(
                "Baum zur Visualisierung auswählen", 
                0, 
                len(model.estimators_) - 1, 
                0,
                key="tree_index_slider", # Wichtig: Ein key speichert den Wert
                help="Jeder Baum im Forest kann unterschiedliche Entscheidungen treffen"
            )
            
            st.write(f"**Zeige Baum #{tree_index + 1}**")
            
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

    st.header("📊 Merkmalswichtigkeit")
    st.write("Welche Merkmale sind am wichtigsten für die Vorhersage der Pinguinart?")

    feature_importance = model.feature_importances_
    feature_names = X.columns.tolist()

    importance_df = pd.DataFrame({
        'Merkmal': feature_names,
        'Wichtigkeit': feature_importance
    }).sort_values('Wichtigkeit', ascending=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(importance_df.set_index('Merkmal'))
    with col2:
        st.dataframe(importance_df.style.format({'Wichtigkeit': '{:.4f}'}))

else:
    st.info("Passen Sie die Werte in der Seitenleiste an (oder laden Sie ein Beispiel) und klicken Sie auf 'Starte Analyse und Vorhersage', um ein Ergebnis zu erhalten.")


# =========================================================================
# Hilfetext
# =========================================================================
st.header("📖 Wie man den Entscheidungsbaum liest")

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
