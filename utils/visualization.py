import matplotlib.pyplot as plt

def visualize_predictions(model, x_test, y_test, num_samples=5):
    for i in range(num_samples):
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[i]}, Predicted: {model.predict([x_test[i]])[0]}")
        plt.show()
