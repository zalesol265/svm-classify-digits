import tkinter as tk
from tkinter import *
import numpy as np
from PIL import Image, ImageDraw
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# Load and preprocess data (for input shape reference)
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_and_preprocess_data()

# Load the trained model
model = joblib.load('svm_mnist_model.pkl')

# GUI setup
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Digit Recognizer")
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()
        self.button_predict = tk.Button(self, text="Predict", command=self.predict_digit)
        self.button_predict.pack()
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()
        self.label_result = tk.Label(self, text="Draw a digit and click Predict")
        self.label_result.pack()

        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280, 280), 255)  # Adjusted size to match canvas size
        self.draw_img = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        self.draw_img.ellipse([x-r, y-r, x+r, y+r], fill='black')

    def predict_digit(self):
        img = self.image.resize((28, 28))
        img = np.array(img)
        
        # Debugging: show the image
        plt.imshow(img, cmap='gray')
        plt.title("Preprocessed Image")
        plt.show()

        img = img.reshape(1, -1) / 255.0
        prediction = model.predict(img)[0]
        self.label_result.config(text=f"Predicted Digit: {prediction}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)  # Adjusted size to match canvas size
        self.draw_img = ImageDraw.Draw(self.image)

if __name__ == "__main__":
    app = App()
    app.mainloop()
