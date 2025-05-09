import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
tf.get_logger().setLevel('ERROR')

import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import matplotlib.pyplot as plt

from custom_conv2D import CustomConv2D
from custom_Dense import CustomDense
import networkx as nx

# -------------------- Graph Visualization ------------------
def visualize_model_architecture():
    print("[MODE] Visualizing Classic Neural Network Architecture...")

    model_path = "custom_cnn_model.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")

    model = load_model(model_path, custom_objects={
        'CustomConv2D': CustomConv2D,
        'CustomDense': CustomDense
    })

    G = nx.DiGraph()

    layer_neurons = []  # List of neuron names for each layer
    layer_sizes = []    # Number of neurons per layer

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            input_dims = 1
            for dim in model.input_shape[1:]:  # skip batch dimension (None)
                input_dims *= dim
            neurons = [f"Input_{i}" for i in range(input_dims)]
        elif isinstance(layer, CustomConv2D):
            neurons = [f"{layer.name}_F{i}" for i in range(layer.filters)]
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            neurons = [f"Pool_{layer.name}"]
        elif isinstance(layer, tf.keras.layers.Flatten):
            neurons = [f"Flatten_{layer.name}"]
        elif isinstance(layer, CustomDense):
            neurons = [f"{layer.name}_N{i}" for i in range(layer.units)]
        else:
            neurons = [f"{layer.name}"]

        layer_neurons.append(neurons)
        layer_sizes.append(len(neurons))

    # Now add nodes
    for layer in layer_neurons:
        for neuron in layer:
            G.add_node(neuron)

    # Add edges between layers
    for i in range(len(layer_neurons) - 1):
        for src in layer_neurons[i]:
            for dst in layer_neurons[i + 1]:
                G.add_edge(src, dst)

    # Now calculate positions (left to right)
    pos = {}
    x_gap = 3
    y_gap = 1.5

    for layer_idx, neurons in enumerate(layer_neurons):
        x = layer_idx * x_gap
        total_neurons = len(neurons)
        y_start = -(total_neurons - 1) * y_gap / 2
        for neuron_idx, neuron in enumerate(neurons):
            y = y_start + neuron_idx * y_gap
            pos[neuron] = (x, y)

    # Draw
    plt.figure(figsize=(18, 10))
    nx.draw(G, pos, with_labels=False, arrows=False, node_size=100, node_color='skyblue')

    # Draw neuron labels
    for neuron, (x, y) in pos.items():
        plt.text(x, y, neuron, fontsize=5, ha='center', va='center')

    plt.title("Custom CNN - Classic Neural Network View", fontsize=16)
    plt.axis('off')
    plt.show()

# ------------------ Load Dataset ------------------
def load_npy_dataset(split="test"):
    x = np.load(os.path.join("resized_mnist", f"x_{split}.npy"))
    y = np.load(os.path.join("resized_mnist", f"y_{split}.npy"))
    x = x.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return x, y

# ------------------ Build the Model ------------------
def build_custom_cnn(input_shape=(28, 28, 1)):
    inputs = layers.Input(shape=input_shape)
    x = CustomConv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = CustomConv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = CustomConv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = CustomDense(256, activation='relu')(x)
    outputs = CustomDense(10, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# ------------------ Main Logic ------------------
def train_model():
    model_path = "custom_cnn_model.h5"
    print("[MODE] Training...")

    model = build_custom_cnn()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    x_train, y_train = load_npy_dataset("test")  # Right now using test set for training

    model.fit(x_train, y_train, epochs=3, batch_size=8, validation_split=0.1)
    model.save(model_path)
    print(f"[SAVED] Model saved at {model_path}")

def inference_model(digit_to_find=None, run_mode='predict'):
    model_path = "custom_cnn_model.h5"
    print("[MODE] Inference...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")

    model = load_model(model_path, custom_objects={
        'CustomConv2D': CustomConv2D,
        'CustomDense': CustomDense
    })

    x_test, y_test = load_npy_dataset("test")

    if digit_to_find is not None:
        print(f"[INFO] Looking for a sample of digit '{digit_to_find}'...")
        matching_indices = np.where(y_test == digit_to_find)[0]
        if len(matching_indices) == 0:
            print(f"[ERROR] No samples found for digit {digit_to_find}. Exiting.")
            sys.exit(1)
        sample_index = random.choice(matching_indices)
        print(f"[INFO] Randomly picked index {sample_index} with label {digit_to_find}")
    else:
        sample_index = 0
        print("[INFO] No digit specified, defaulting to first sample (index 0).")

    x_sample = np.expand_dims(x_test[sample_index], axis=0)
    true_label = y_test[sample_index]

    # Core inference
    if run_mode == 'predict':
        print("[INFO] Running in predict mode...")
        preds = model.predict(x_sample)
    else:
        print("[INFO] Running in eager mode...")
        preds = model(x_sample, training=False)  # <-- Pure eager mode call

    predicted_digit = tf.argmax(preds, axis=-1).numpy()

    # plt.imshow(np.squeeze(x_sample), cmap='gray')
    # plt.title(f"Input Image - True Label {true_label}")
    # plt.axis('off')
    # plt.show()

    print(f"[GROUND TRUTH] True label: {true_label}")
    print(f"[PREDICTION] Predicted label: {predicted_digit}")

    if predicted_digit == true_label:
        print("[RESULT] ✅ Prediction matches ground truth!")
    else:
        print("[RESULT] ❌ Prediction does NOT match ground truth.")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    digit_to_find = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if mode == 'train':
        train_model()
    elif mode == 'inference':
        inference_model(digit_to_find)
    elif mode == 'show':
        visualize_model_architecture()
    else:
        print("[ERROR] Invalid mode. Use 'train' or 'inference' or 'show'.")
