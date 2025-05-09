import numpy as np
import struct
import os

# Paths to your downloaded files
image_file_path = "mnist/t10k-images.idx3-ubyte"
label_file_path = "mnist/t10k-labels.idx1-ubyte"

# Output directory to save .npy files
save_dir = "resized_mnist"
os.makedirs(save_dir, exist_ok=True)

# ------------------ Load MNIST Images ------------------
def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        print(f"[INFO] Magic: {magic}, Number of Images: {num}, Size: {rows}x{cols}")
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape((num, rows, cols))
    return images

# ------------------ Load MNIST Labels ------------------
def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        print(f"[INFO] Magic: {magic}, Number of Labels: {num}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# ------------------ Save without Resizing ------------------
def save_raw(images, labels, split_name="test"):
    images = np.expand_dims(images, -1)  # Add channel dimension (num, 28, 28, 1)
    np.save(os.path.join(save_dir, f"x_{split_name}.npy"), images)
    np.save(os.path.join(save_dir, f"y_{split_name}.npy"), labels)
    print(f"[SAVED] {split_name}: {len(images)} images saved.")

# ------------------ Main Run ------------------
if __name__ == "__main__":
    images = load_mnist_images(image_file_path)
    labels = load_mnist_labels(label_file_path)

    save_raw(images, labels, split_name="test")

    print("[DONE] Raw images and labels saved into:", save_dir)
