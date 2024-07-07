import tensorflow as tf
from sklearn.svm import SVC
import joblib

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_and_preprocess_data()

model = SVC(kernel='linear')
model.fit(x_train, y_train)

joblib.dump(model, 'svm_mnist_model.pkl')
