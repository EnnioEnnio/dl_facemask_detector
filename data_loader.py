"""
This file is typically responsible for loading and preprocessing your data. It
can handle tasks such as reading data from files, performing data augmentation,
data normalization, and creating data loaders for efficient batching of the
data during training and evaluation.

Obserservations:
    - Images from the dataset are of different sizes.
"""

import os
import argparse
from PIL import Image


def resize_image(image_dir, output_dir, target_size=(256, 256)):
    """
    Resize images to target size and save them to output directory.
    Currently only supports .jpg and .png images.
    It's worth noting that resize will stretch and sometimes rotate Images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path)

            resized_image = image.resize(target_size)

            output_path = os.path.join(output_dir, image_name)
            resized_image.save(output_path)


def process_dataset(input_dir, output_dir, target_size):
    for _, dir, _ in os.walk(input_dir):
        for d in dir:
            in_dir = os.path.abspath(os.path.join(input_dir, d))
            out_dir = os.path.abspath(os.path.join(output_dir, d))
            resize_image(in_dir, out_dir, target_size=target_size)


def main(args):
    process_dataset(args.input, args.output, tuple(args.size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="input directory"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./out",
        help="output directory (default: ./out)",
    )
    parser.add_argument(
        "-s",
        "--size",
        nargs=2,
        type=int,
        default=[256, 256],
        help="transformed image dimensions (w/h) (default: (256, 256))",
    )
    main(parser.parse_args())
