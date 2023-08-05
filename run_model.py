from architecture import Model1
from util import log, Config
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import argparse
from PIL import Image
from data_loader import process_single_image


def run_model(model, image_path: str):
    image = process_single_image(image_path)
    print("masked" if torch.sigmoid(
        model(image)).item() > 0.5 else "unmasked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict masked/unmasked label from an image using a trained model as defined in the config.ini.")
    parser.add_argument("--image", type=str, help="Path to the input image.")
    args = parser.parse_args()

    model = Model1()
    config = Config()

    # load pre-trained model
    model_path = os.path.abspath(
        os.getenv("MODEL_PATH") or config.get(
            "Paths", "model") or "./trained.pt"
    )
    log.info(f"Model path: {model_path}")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    run_model(model, args.image)
