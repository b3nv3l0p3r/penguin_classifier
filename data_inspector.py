# Importiert das Modul zur Datenvorverarbeitung
import one_hot_encoder as enc

# Führt die Vorverarbeitung aus, um X (Features) und y (Zielvariable) zu erhalten
X, y = enc.preprocess_penguin_data('penguins_size.csv')

# Gibt die ersten 5 Zeilen der verarbeiteten Features aus
print(X.head())
# Gibt die ersten 5 Zeilen der Zielvariable aus
print(y.head())