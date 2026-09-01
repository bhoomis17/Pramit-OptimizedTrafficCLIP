import argparse
import os
import tempfile
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms

from src.models.traffic_clip import TrafficCLIP


def make_dummy_image():
    tmp_dir = tempfile.mkdtemp()
    img_path = os.path.join(tmp_dir, "dummy_flow.png")
    arr = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    PILImage.fromarray(arr, mode="L").save(img_path)
    return img_path


def main():
    parser = argparse.ArgumentParser(description="Predict traffic class for a single image")
    parser.add_argument("--image", default=None, help="Path to a traffic image (28x28 grayscale)")
    parser.add_argument("--classes", nargs="+", default=["Skype", "Zoom", "BitTorrent"],
                         help="List of possible class names")
    args = parser.parse_args()

    if args.image is None or not os.path.exists(args.image):
        print("No real image provided — using a dummy generated image for a pipeline smoke test.")
        image_path = make_dummy_image()
    else:
        image_path = args.image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrafficCLIP(embed_dim=1024).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    image = PILImage.open(image_path).convert("L")
    image_tensor = transform(image).unsqueeze(0).to(device)

    text_prompts = [f"a network traffic photo of {c}" for c in args.classes]

    with torch.no_grad():
        logits_per_image, _ = model(image_tensor, text_prompts)
        probs = logits_per_image.softmax(dim=1).squeeze(0)
        pred_idx = probs.argmax().item()

    print(f"Predicted class: {args.classes[pred_idx]}")
    print(f"Confidence scores: {dict(zip(args.classes, probs.cpu().tolist()))}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.txt", "a") as f:
        f.write(f"\n--- predict.py run ---\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Predicted: {args.classes[pred_idx]}\n")
        f.write(f"Scores: {dict(zip(args.classes, probs.cpu().tolist()))}\n")


if __name__ == "__main__":
    main()