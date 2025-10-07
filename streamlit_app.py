# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np # <-- Added this import
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Data Preprocessing Function ---
# This function loads the data, handles missing values, and performs one-hot encoding.
def preprocess_penguin_data(file_path):
    """Loads and preprocesses the penguin dataset."""
    penguin_df = pd.read_csv(file_path)
    penguin_df.dropna(inplace=True)
    
    # One-Hot-Encoding for categorical columns
    penguin_df = pd.get_dummies(penguin_df, columns=['island', 'sex'], drop_first=True)
    
    # Define target (y) and features (x)
    y = penguin_df['species']
    x = penguin_df.drop('species', axis=1)
    
    return x, y

# --- Streamlit App Layout ---

# Set the title of the app
st.title('🐧 Penguin Species Classifier')
st.write("This app uses a **Random Forest** model to predict the species of penguins based on their physical measurements.")

# --- Sidebar for User Inputs ---
st.sidebar.header('Model Parameters')

# Slider for the number of trees in the Random Forest
n_estimators = st.sidebar.slider('Number of Trees (n_estimators)', 
                                 min_value=50, 
                                 max_value=500, 
                                 value=100, 
                                 step=50,
                                 help="Adjust the number of trees in the forest. More trees can lead to better accuracy but longer training time.")

# --- Main Page Content ---

# 1. Load and display data
st.header('1. Data Overview')

try:
    x, y = preprocess_penguin_data("penguins_size.csv")
    st.success('Data successfully loaded and preprocessed.')
    st.write('**Data Preview:**')
    # Combine features and target for display
    preview_df = x.join(y)
    st.dataframe(preview_df.head())
except FileNotFoundError:
    st.error('Error: The file "penguins_size.csv" was not found. Please make sure it is in the same directory as this script.')
    st.stop() # Stop the script if the data can't be loaded

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 2. Train the Random Forest model
st.header('2. Model Training')
with st.spinner(f'Training the Random Forest model with {n_estimators} trees...'):
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    rf.fit(x_train, y_train)
st.success('Model training complete!')

# 3. Evaluate the model
st.header('3. Model Evaluation')
y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"The model's accuracy on the test data is **{accuracy:.2%}**.")

# --- Display detailed metrics ---
st.subheader('Classification Report')
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.subheader('Confusion Matrix')
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)

# Create a figure for the plot
fig, ax = plt.subplots()
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)

# Loop over data dimensions and create text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='large', color='black')

# Set labels and title
ax.set_xticks(np.arange(len(rf.classes_)))
ax.set_xticklabels(rf.classes_, rotation=45)
ax.set_yticks(np.arange(len(rf.classes_)))
ax.set_yticklabels(rf.classes_)
plt.xlabel('Predicted Species', fontsize=12)
plt.ylabel('Actual Species', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)

# Display the plot in Streamlit
st.pyplot(fig)