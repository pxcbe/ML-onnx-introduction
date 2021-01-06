import numpy as np
import onnxruntime as rt

X_test = np.array([[5.8,4.0,1.2,0.2],[7.7,3.8,6.7,2.2,]])

sess = rt.InferenceSession("decision_tree_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(np.float32)})[0]
print(pred_onx)