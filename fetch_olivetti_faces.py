import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import fetch_olivetti_faces
from sklearn.linear_model import LogisticRegression

data = fetch_olivetti_faces()

print('data  - shape ', data.data.shape)

X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.2, random_state=42)

lg = LogisticRegression()
lg.fit(X_train,y_train)

y_pred = lg.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print('Accuracy Score: ', accuracy)

target_names = [str(i) for i in range(36)] 

print('Classification Report: \n', classification_report(y_test,y_pred,target_names=target_names))

# Confusion matrix heatmap
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(" Confusion Matrix - Iris")
plt.show()

input('wait...')
