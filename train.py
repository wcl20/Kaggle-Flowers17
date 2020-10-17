import argparse
import cv2
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from core.datasets import SimpleDatasetLoader
from core.preprocessing import ResizeWithAspectRatio
from core.preprocessing import ToArray
from core.nn import MiniVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Path to dataset")
    args = parser.parse_args()

    preprocessors = [ResizeWithAspectRatio(64, 64), ToArray()]
    dataset_loader = SimpleDatasetLoader(preprocessors)

    img_paths = glob.glob(f"{args.dataset}/images/*/*.jpg")
    data, labels = dataset_loader.load(img_paths)
    data = data.astype("float") / 255.
    label_encoder = LabelBinarizer()
    labels = label_encoder.fit_transform(labels)
    print(f"[INFO] Data: {data.shape}. Labels: {labels.shape}")

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    # Data augmentation
    augmentation = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    optimizer = SGD(lr=0.05)
    model = MiniVGGNet.build(height=64, width=64, channels=3, classes=len(label_encoder.classes_))
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    H = model.fit(
        augmentation.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        validation_data=(X_test, y_test),
        epochs=100,
        verbose=1
    )

    preds = model.predict(X_test, batch_size=32)
    report = classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=label_encoder.classes_)
    print(report)

    plt.style.use("ggplot")
    plt.figure()
    N = np.arange(0, 100)
    plt.plot(N, H.history["loss"], label="training loss")
    plt.plot(N, H.history["val_loss"], label="validation loss")
    plt.plot(N, H.history["accuracy"], label="training accuracy")
    plt.plot(N, H.history["val_accuracy"], label="validation accuracy")
    plt.title("Training Loss/Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig("loss_augmentation.png")



if __name__ == '__main__':
    main()
