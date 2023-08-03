from PIL import Image
from architecture import Model1
from config import Config
from cv2 import cv2
from torchvision import transforms
import logging as log
import os
import torch

log.basicConfig(level=log.INFO, format="[%(levelname)s] %(message)s")


def eval_with_webcam(model):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
        ]
    )
    # Load the saved model
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()

        im = Image.fromarray(frame, "RGB")
        data = transform(im).unsqueeze(0)

        # color-code frame based on prediction
        out = torch.sigmoid(model(data)).item()
        prediction = 1 if out > 0.5 else 0
        log.info(f"Prediction: {prediction} ({out}")
        if prediction == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = Model1()
    config = Config()

    # load pre-trained model
    model_path = os.path.abspath(
        os.getenv("MODEL") or config.get("Paths", "model") or "./trained.pt"
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    eval_with_webcam(model)
