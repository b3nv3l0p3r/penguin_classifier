import streamlit as st

# === Seitenkonfiguration (muss ganz oben stehen) ===
st.set_page_config(layout="wide")


# === Folien-Inhalte ===
# Jede Folie ist eine eigene Funktion, die den Inhalt rendert.
# Der Inhalt wurde aus Ihrer index.html-Datei extrahiert.

def render_slide_1():
    st.image(
        "https://pplx-res.cloudinary.com/image/upload/v1761051204/pplx_project_search_images/a5db955fa9cca9f3baf6dc0e207a8b545b8b9d1e.png",
        width=300
    )
    st.title("Machine Learning Workflow am Beispiel der Artenklassifikation")
    st.header("Pinguinarten-Klassifikation mit Random Forest und Decision Tree")

    with st.container(border=True):
        st.subheader("Projektteam:")
        st.markdown("""
        - Bendix Greiner
        - Maurice Baumann
        - Pascal Grimm
        """)

def render_slide_2():
    st.divider()
    st.title("Kontext und Rolle")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("🔬 Unsere Rolle")
            st.write("Forschungsgruppe als Anwender von Machine Learning")
        with st.container(border=True):
            st.subheader("📚 Projektumfang")
            st.write("Vermittlung eines ML-Projekts von der Problemdefinition über Modellierung bis zur Evaluation")
    with col2:
        with st.container(border=True):
            st.subheader("🎯 Projektziel")
            st.write("Automatische Klassifikation von Pinguinarten")
        with st.container(border=True):
            st.subheader("📊 Datenbasis")
            st.write("Palmer Penguins Datensatz zur praktischen Veranschaulichung")

def render_slide_3():
    st.divider()
    st.title("Theoretischer Teil: Entscheidungsbaum (Decision Tree)")

    st.info("Definition: ML-Modell mit Baumstruktur für Vorhersagen")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Struktur")
        st.markdown("""
        - Knoten repräsentieren Entscheidungsregeln basierend auf Feature-Werten
        - Blätter stehen für Zielklassen (Kategorien)
        """)

        st.subheader("✅ Vorteile")
        st.markdown("""
        - Einfach interpretierbar
        - Visualisierbar
        - Transparent
        """)

        st.subheader("⚠️ Nachteile")
        st.markdown("""
        - Neigung zu Overfitting bei komplexen Daten
        - Erfordert Pruning zur Regularisierung
        """)

    with col2:
        st.subheader("Beispiel-Diagramm (vereinfacht)")
        # Wir erstellen den Baum aus der HTML mit Graphviz nach
        st.graphviz_chart('''
        digraph {
            node [shape=box, style="filled,rounded", fillcolor="#21808D", fontcolor="white"]
            leaf [shape=ellipse, style=filled, fillcolor="#D5F5E3"]

            node1 [label="Flipper Length < 206.5mm?"]
            node2 [label="Culmen Length < 43.35mm?"]
            leaf_adelie [label="Adelie", fillcolor="#f0f8ff"]
            leaf_chinstrap [label="Chinstrap", fillcolor="#fff0f5"]
            leaf_gentoo [label="Gentoo", fillcolor="#e6ffe6"]

            node1 -> node2 [label="  Ja"]
            node1 -> leaf_gentoo [label="  Nein"]
            node2 -> leaf_adelie [label="  Ja"]
            node2 -> leaf_chinstrap [label="  Nein"]
        }
        ''')

def render_slide_4():
    st.divider()
    st.title("Theoretischer Teil: Random Forest")

    st.info("Definition: Ensemble-Verfahren aus vielen Entscheidungsbäumen")

    st.subheader("Ensemble-Prozess")
    st.markdown("""
    🌲 Tree 1 &nbsp;&nbsp; 🌲 Tree 2 &nbsp;&nbsp; 🌲 Tree 3 &nbsp;&nbsp; ... &nbsp;&nbsp; 🌲 Tree n

    ⬇️ ⬇️ ⬇️ ⬇️ ⬇️

    🗳️ **Mehrheitsentscheidung**

    ⬇️

    🏆 **Finale Vorhersage**
    """)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prozess")
        st.markdown("""
        - Jeder Baum wird auf unterschiedlichen Datenstichproben trainiert (Bootstrapping)
        - Aggregation der Baumvorhersagen zu robusterer Gesamtausgabe
        """)
    with col2:
        st.subheader("✅ Vorteile")
        st.markdown("""
        - Höhere Genauigkeit
        - Robustheit gegen Overfitting
        - Schätzung von Feature-Wichtigkeiten
        """)

def render_slide_5():
    st.divider()
    st.title("CRISP-DM: Standardisierter ML-Prozess")
    st.subheader("Cross-Industry Standard Process for Data Mining")

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("<h3><span style='background-color:#21808D; color:white; border-radius:50%; padding: 5px 12px;'>1</span> Business Understanding</h3>", unsafe_allow_html=True)
            st.write("Problemstellung und Ziele definieren")
        with st.container(border=True):
            st.markdown("<h3><span style='background-color:#21808D; color:white; border-radius:50%; padding: 5px 12px;'>4</span> Modeling</h3>", unsafe_allow_html=True)
            st.write("Modelle wählen und trainieren")
    with col2:
        with st.container(border=True):
            st.markdown("<h3><span style='background-color:#21808D; color:white; border-radius:50%; padding: 5px 12px;'>2</span> Data Understanding</h3>", unsafe_allow_html=True)
            st.write("Daten explorieren und verstehen")
        with st.container(border=True):
            st.markdown("<h3><span style='background-color:#21808D; color:white; border-radius:50%; padding: 5px 12px;'>5</span> Evaluation</h3>", unsafe_allow_html=True)
            st.write("Modelle bewerten mit Metriken")
    with col3:
        with st.container(border=True):
            st.markdown("<h3><span style='background-color:#21808D; color:white; border-radius:50%; padding: 5px 12px;'>3</span> Data Preparation</h3>", unsafe_allow_html=True)
            st.write("Datenaufbereitung und Feature Engineering")
        with st.container(border=True):
            st.markdown("<h3><span style='background-color:#21808D; color:white; border-radius:50%; padding: 5px 12px;'>6</span> Deployment</h3>", unsafe_allow_html=True)
            st.write("Modell in Anwendung überführen")

    st.warning("💡 **Wichtig:** CRISP-DM ist ein iterativer Prozess - Phasen werden oft wiederholt und verfeinert")

def render_slide_6():
    st.divider()
    st.title("Der Palmer Penguins Datensatz")

    col1, col2, col3 = st.columns(3)
    col1.metric("Pinguine gesamt", "344")
    col2.metric("Arten", "3")
    col3.metric("Inseln", "3")

    st.subheader("Artenverteilung")
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("<h4>Adelie</h4>", unsafe_allow_html=True)
            st.markdown("**152** (44.2%)")
    with col2:
        with st.container(border=True):
            st.markdown("<h4>Gentoo</h4>", unsafe_allow_html=True)
            st.markdown("**124** (36.0%)")
    with col3:
        with st.container(border=True):
            st.markdown("<h4>Chinstrap</h4>", unsafe_allow_html=True)
            st.markdown("**68** (19.8%)")

    st.subheader("Features und Inseln")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Inseln**")
        st.markdown("- Biscoe\n- Dream\n- Torgersen")
    with col2:
        st.markdown("**Features**")
        st.code("""
- culmen_length_mm
- culmen_depth_mm
- flipper_length_mm
- body_mass_g
- island
- sex
        """)
    st.caption("Quelle: Palmer Station LTER, Antarctica")

def render_slide_7():
    st.divider()
    st.title("Datenvorbereitung (Data Preparation)")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height=200):
            st.subheader("1. Bereinigung")
            st.write("Missing Values entfernt")
            st.info("Datensatz von 344 → 333 Pinguine")
        with st.container(border=True, height=200):
            st.subheader("3. Feature-Transformation")
            st.write("Von 6 Features zu 10 Spalten nach Encoding")

    with col2:
        with st.container(border=True, height=200):
            st.subheader("2. Encoding")
            st.write("One-Hot-Encoding für kategoriale Variablen")
            st.code("island → island_Biscoe, island_Dream, ...\nsex → sex_Female, sex_Male")
        with st.container(border=True, height=200):
            st.subheader("4. Train-Test-Split")
            st.write("Aufteilung für valide Evaluation")
            st.info("80% Training (266) / 20% Test (67)")

    st.subheader("Detail: Feature-Transformation")
    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**Vorher (6 Features)**")
        st.code("""
- culmen_length_mm
- culmen_depth_mm
- flipper_length_mm
- body_mass_g
- island
- sex
        """)
    with col_after:
        st.markdown("**Nachher (10 Spalten)**")
        st.code("""
- culmen_length_mm
- culmen_depth_mm
- flipper_length_mm
- body_mass_g
- island_Biscoe
- island_Dream
- island_Torgersen
- sex_Female
- sex_Male
        """)

def render_slide_8():
    st.divider()
    st.title("ML-Workflow: Vom Datensatz zum Modell")

    st.markdown("<h3>1. 📊 Daten laden</h3>", unsafe_allow_html=True)
    st.code("X, y = preprocess_penguin_data(df)")
    st.markdown("<h1 style='text-align:center;'>⬇</h1>", unsafe_allow_html=True)

    st.markdown("<h3>2. ✂️ Train-Test-Split</h3>", unsafe_allow_html=True)
    st.code("X_train, X_test, y_train, y_test = train_test_split(...)")
    st.markdown("<h1 style='text-align:center;'>⬇</h1>", unsafe_allow_html=True)

    st.markdown("<h3>3. 🤖 Modelltraining</h3>", unsafe_allow_html=True)
    st.code("rf_model.fit(X_train, y_train)\ndt_model.fit(X_train, y_train)")
    st.markdown("<h1 style='text-align:center;'>⬇</h1>", unsafe_allow_html=True)

    st.markdown("<h3>4. 🎯 Vorhersagen</h3>", unsafe_allow_html=True)
    st.code("y_pred = model.predict(X_test)")
    st.markdown("<h1 style='text-align:center;'>⬇</h1>", unsafe_allow_html=True)

    st.markdown("<h3>5. 📈 Evaluation</h3>", unsafe_allow_html=True)
    st.code("accuracy_score(y_test, y_pred)")
    st.markdown("<h1 style='text-align:center;'>⬇</h1>", unsafe_allow_html=True)

    st.markdown("<h3>6. 📊 Visualisierung</h3>", unsafe_allow_html=True)
    st.code("confusion_matrix(y_test, y_pred)")


def render_slide_9():
    st.divider()
    st.title("Ergebnisse und Erkenntnisse")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("🌲 Random Forest")
            st.metric("Accuracy", "~99%")
            st.write("Sehr hohe Klassifikationsgenauigkeit durch Ensemble-Methode.")
            st.markdown("""
            - **Precision:** 0.99
            - **Recall:** 0.99
            - **F1-Score:** 0.99
            """)
    with col2:
        with st.container(border=True):
            st.subheader("🌳 Decision Tree")
            st.metric("Accuracy", "~95%")
            st.write("Einfachere Interpretierbarkeit bei leicht reduzierter Genauigkeit.")
            st.markdown("""
            - **Precision:** 0.95
            - **Recall:** 0.94
            - **F1-Score:** 0.95
            """)

    st.subheader("Confusion Matrix Beispiel")
    st.markdown("""
    | | Predicted Adelie | Predicted Chinstrap | Predicted Gentoo |
    |---|---|---|---|
    | **Actual Adelie** | <span style="background-color:#e6ffe6; color:green;">25</span> | <span style="background-color:#ffe6e6; color:red;">0</span> | <span style="background-color:#ffe6e6; color:red;">1</span> |
    | **Actual Chinstrap** | <span style="background-color:#ffe6e6; color:red;">0</span> | <span style="background-color:#e6ffe6; color:green;">15</span> | <span style="background-color:#ffe6e6; color:red;">0</span> |
    | **Actual Gentoo** | <span style="background-color:#ffe6e6; color:red;">0</span> | <span style="background-color:#ffe6e6; color:red;">0</span> | <span style="background-color:#e6ffe6; color:green;">26</span> |
    """, unsafe_allow_html=True)

    st.success("💡 **Fazit:** Praxisbeispiel zeigt die Leistungsfähigkeit von ML-Algorithmen bei Klassifikationsaufgaben")

def render_slide_10():
    st.divider()
    st.title("Praktische Umsetzung: Streamlit-App")

    st.link_button("🔗 GitHub Repository", "https://github.com/b3nv3l0p3r/penguin_classifier")

    st.subheader("🚀 App-Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🎯 **Interaktive Vorhersage:** Eingabe von Pinguinmaßen für Live-Klassifikation")
        st.markdown("🤖 **Modelltraining:** Live-Training und Vergleich verschiedener Modelle")
    with col2:
        st.markdown("📊 **Datenvisualisierung:** Scatter Plots und Verteilungen der Features")
        st.markdown("🎨 **Confusion Matrix:** Detaillierte Darstellung der Klassifikationsergebnisse")

    st.subheader("🛠️ Technologie-Stack")
    with st.container(border=True):
        st.markdown("""
        - **Python 3.12.4:** Programmiersprache
        - **scikit-learn:** Machine Learning Framework
        - **pandas & numpy:** Datenverarbeitung
        - **matplotlib:** Datenvisualisierung
        - **Streamlit:** Web App Framework
        """)

def render_slide_11():
    st.divider()
    st.title("Kritische Reflexion")
    st.subheader("Wichtige Überlegungen zum ML-Projekt")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("<h3>🔍 Datenqualität</h3>", unsafe_allow_html=True)
            st.write("**Aspekt:** Bedeutung von sauberer Vorverarbeitung")
            with st.expander("Details anzeigen"):
                st.markdown("""
                - Missing Values können Modellleistung stark beeinflussen
                - Feature Engineering ist entscheidend für Erfolg
                - Datenvalidierung und -bereinigung sind zeitaufwändig aber kritisch
                """)
        with st.container(border=True):
            st.markdown("<h3>⚠️ Grenzen</h3>", unsafe_allow_html=True)
            st.write("**Aspekt:** Generalisierbarkeit, Overfitting, Datenabhängigkeit")
            with st.expander("Details anzeigen"):
                st.markdown("""
                - Modelle nur so gut wie die Trainingsdaten
                - Bias in Daten führt zu bias in Vorhersagen
                - Übertragbarkeit auf andere Kontexte fraglich
                """)
    with col2:
        with st.container(border=True):
            st.markdown("<h3>📊 Evaluation</h3>", unsafe_allow_html=True)
            st.write("**Aspekt:** Notwendigkeit der Interpretation der Modellleistung")
            with st.expander("Details anzeigen"):
                st.markdown("""
                - Accuracy allein reicht nicht aus
                - Precision, Recall, F1-Score für vollständiges Bild
                - Cross-Validation für robuste Bewertung
                """)
        with st.container(border=True):
            st.markdown("<h3>✅ Best Practices</h3>", unsafe_allow_html=True)
            st.write("**Aspekt:** Reproduzierbarkeit, Dokumentation, transparente Methodik")
            with st.expander("Details anzeigen"):
                st.markdown("""
                - Code-Dokumentation und Versionskontrolle
                - Reproduzierbare Experimente mit festen Seeds
                - Transparente Berichterstattung über Methodik
                """)

def render_slide_12():
    st.divider()
    st.title("Zusammenfassung & Ausblick")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Zusammenfassung")
        st.markdown("""
        - 🎯 ML ist leistungsfähig für komplexe Klassifikationsprobleme
        - 🌳 Decision Trees und Random Forests bieten unterschiedliche Stärken
        - ⚙️ Strukturierte Methodik (CRISP-DM) essentiell für erfolgreiche Projekte
        - 🛠️ Open-Source-Tools erleichtern Zugang zu ML-Technologien
        """)
    with col2:
        st.subheader("🔮 Ausblick")
        with st.container(border=True):
            st.markdown("🧬 **Weitere Anwendungen**")
            st.write("Biologie und Ökologie: Artenerkennung, Biodiversitätsmonitoring, Habitatanalyse")
        with st.container(border=True):
            st.markdown("⚡ **Modelloptimierung**")
            st.write("Hyperparameter-Tuning, Neural Networks, Deep Learning für Bilderkennung")
        with st.container(border=True):
            st.markdown("🚀 **Production Deployment**")
            st.write("Cloud-basierte APIs, Echtzeit-Klassifikation, mobile Anwendungen")

    st.divider()
    with st.container(border=True):
        st.header("Vielen Dank für Ihre Aufmerksamkeit!")
        st.subheader("Fragen und Diskussion sind herzlich willkommen")


# === Haupt-Skript: Alle Folien nacheinander rendern ===
render_slide_1()
render_slide_2()
render_slide_3()
render_slide_4()
render_slide_5()
render_slide_6()
render_slide_7()
render_slide_8()
render_slide_9()
render_slide_10()
render_slide_11()
render_slide_12()