from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

#load iris dataset
dataset = datasets.load_iris()

tree = DecisionTreeClassifier()#classifer
tree.fit(dataset.data, dataset.target)#train the model (x,y)  x

expected = dataset.target
predicted = tree.predict(dataset.data)

#summary
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))


