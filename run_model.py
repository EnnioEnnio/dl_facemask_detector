from architecture import Model1, load_and_modify_resnet18
from util import log, Config, get_device
import os
import torch
import argparse
from data_loader import process_single_image


def run_model(model, image_path: str):
    # using device to run model on machines with & without GPU
    device = get_device()

    image = process_single_image(image_path).to(device)

    neural_net = model.to(device)
    print("unmasked" if torch.sigmoid(neural_net(image)).item() > 0.5 else "masked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict masked/unmasked label from an image using a trained model as defined in the config.ini.")
    parser.add_argument(
        "--image", type=str, help="Path to the input image.", required=True)
    args = parser.parse_args()

    model = load_and_modify_resnet18()
    config = Config()

    #load pre-trained model
    model_path = os.path.abspath(
        os.getenv("MODEL_PATH") or config.get("Paths", "model") or "./trained.pt")
    log.info(f"Model path: {model_path}")

    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(get_device())))
    model.eval()
    run_model(model, args.image)
