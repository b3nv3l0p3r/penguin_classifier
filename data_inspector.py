#data_inspector.py
import one_hot_encoder as enc


X, y = enc.preprocess_penguin_data('penguins_size.csv')
print(X.head())
print(y.head())