from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import numpy
import onnxruntime as rt

import pickle

# Slit the dataset in a training and testing dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, Y_test = train_test_split(X, y)

# Define the model and fit the model with training data and print information about the model
clr = DecisionTreeClassifier()
clr.fit(X_train, y_train)
print(clr)

#Convert the model from sklearn format to ONNX (Open Neural Network Exchange)
initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("decision_tree_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())