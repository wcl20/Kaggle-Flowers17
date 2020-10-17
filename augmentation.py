import argparse
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Path to input image")
    args = parser.parse_args()

    augmentation = ImageDataGenerator(
        # Randomly rotate +/- 30 degrees
        rotation_range=30,
        # Randomly shift 10% of width
        width_shift_range=0.1,
        # Randomly shift 10% of height
        height_shift_range=0.1,
        shear_range=0.2,
        # Randomly zoom by scale factor [0.8, 1.2]
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    image = load_img(args.image)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    os.makedirs("output", exist_ok=True)
    generator = augmentation.flow(image, batch_size=1, save_to_dir="output", save_prefix="image", save_format="jpg")
    generator = iter(generator)

    for _ in range(10):
        image = next(generator)



if __name__ == '__main__':
    main()
