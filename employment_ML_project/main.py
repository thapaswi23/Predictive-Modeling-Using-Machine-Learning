import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
df = pd.read_csv("employment_dataset.csv")
print("\nFIRST 5 ROWS OF DATASET:\n")
print(df.head())
df = df.dropna()
df["Experience"] = df["Experience"].str.replace(" years", "")
df["Experience"] = df["Experience"].astype(int)
df["Hired"] = df["Hired"].astype(str)
df["Hired"] = df["Hired"].str.strip()
df["Hired"] = df["Hired"].replace({
    "Yes": 1,
    "Y": 1,
    "No": 0,
    "N": 0
})
df["Hired"] = pd.to_numeric(df["Hired"], errors="coerce")
df = df.dropna(subset=["Hired"])
df["Hired"] = df["Hired"].astype(int)
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])
df["Department"] = encoder.fit_transform(df["Department"])
df["Location"] = encoder.fit_transform(df["Location"])
X = df[[
    "Age",
    "Salary",
    "Experience",
    "Performance_Score",
    "Gender",
    "Department",
    "Location"
]]
y = df["Hired"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nMODEL ACCURACY:\n")
print(accuracy)
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.title("Confusion Matrix")
plt.show()