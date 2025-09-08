# practice

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
col_names = ['Age', 'Sex', 'BP', 'Cholestrol', 'Na_to_k', 'Drug']

drug = pd.read_csv("drug200.csv", header=1, names=col_names)

print(drug.head())

feature_cols = ['Age', 'Sex', 'BP', 'Cholestrol', 'Na_to_k']
X = drug[feature_cols]
y = drug.Drug # Target variable

X.loc[:, 'BP'] = X['BP'].map({'HIGH': 0, 'LOW': 1, 'NORMAL': 2})
X.loc[:, 'Sex'] = X['Sex'].map({'M': 0, 'F': 1})
X.loc[:, 'Cholestrol'] = X['Cholestrol'].map({'HIGH': 0, 'LOW': 1, 'NORMAL': 2})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

# Predict
y_pred = clf.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

accuracy = accuracy_score(y_test,y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
import numpy as np

class_names = np.unique(y).astype(str)
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=class_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('drugs1.png')
Image(graph.create_png())

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from six import StringIO 
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
class_names = np.unique(y).astype(str)
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=class_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('drugs2.png')
Image(graph.create_png())


input("Wait for me...")