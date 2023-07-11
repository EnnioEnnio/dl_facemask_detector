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
from pathlib import Path


def resize_images(image_dir, output_dir, target_size=(256, 256), rotation=0):
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
            image = image.resize(target_size)

            prefix = Path(image_name).with_suffix("")
            suffix = Path(image_name).suffix
            if rotation:
                image = image.rotate(rotation)
                suffix = f"-rot-{rotation}{suffix}"
            output_path = os.path.join(output_dir, (f"{prefix}{suffix}"))
            image.save(output_path)


def process_dataset(input_dir, output_dir, target_size, rotate):
    rotations = [90, 180, 270]
    for _, dir, _ in os.walk(input_dir):
        for d in dir:
            in_dir = os.path.abspath(os.path.join(input_dir, d))
            out_dir = os.path.abspath(os.path.join(output_dir, d))
            resize_images(in_dir, out_dir, target_size=target_size)
            if rotate:
                for rotation in rotations:
                    resize_images(
                        in_dir,
                        out_dir,
                        target_size=target_size,
                        rotation=rotation,
                    )


def main(args):
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size=tuple(args.size),
        rotate=args.rotate,
    )


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
    parser.add_argument(
        "-r",
        "--rotate",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="save additional rotated copies of source images (default: True)",
    )
    main(parser.parse_args())