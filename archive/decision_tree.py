import one_hot_encoder as enc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# loading data from encoder
x, y = enc.preprocess_penguin_data("penguins_size.csv")

# splitting data into training and test data (70%/30%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# initializing decision tree classifier
dt = DecisionTreeClassifier(random_state=42)

# training
dt.fit(x_train, y_train)

# predictions
y_pred = dt.predict(x_test)

# evaluation
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# creating confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=dt.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Decision Tree Penguin Classification')
plt.show()
