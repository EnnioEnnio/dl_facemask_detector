from PIL import Image
from architecture import Model1
from util import log, Config, get_device
from cv2 import cv2
from torchvision import transforms
import os
import torch


def eval_with_webcam(model):
    device = get_device()
    neural_net = model.to(device)
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
        data = transform(im).unsqueeze(0).to(device)

        # color-code frame based on prediction
        out = torch.sigmoid(neural_net(data)).item()
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
        os.getenv("MODEL_PATH") or config.get("Paths", "model") or "./trained.pt"
    )
    log.info(f"Model path: {model_path}")

    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(get_device()))
    )
    model.eval()
    eval_with_webcam(model)
