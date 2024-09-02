import io
import json

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

LABELS = ["gender", "articleType", "baseColour", "season", "usage"]


class Handler(BaseHandler):
    def __init__(self):
        super(Handler, self).__init__()

        # Define the dictionary to map each model's output to labels
        self.label_dict = {label: Handler.load_dict(label) for label in LABELS}

        # Define the transformation to resize the image
        self.image_processing = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.models = dict()

        # We'll load models in the `initialize` method

    @staticmethod
    def load_dict(label):
        with open(f"{label}.json", "r") as f:
            return json.load(f)

    def initialize(self, context):
        """Initialize and load all models."""
        properties = context.system_properties

        # Load all five models
        for label in LABELS:
            self.models[label] = self.load_model(f"../../models/{label}/traced_model.pt")
            self.models[label].eval()

    def load_model(self, model_path):
        """Helper method to load a model from a serialized file."""
        model = torch.jit.load(model_path)
        return model

    def preprocess(self, data):
        """Preprocess the input data."""
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            else:
                image = Image.open(image).convert("RGB")

            # Apply the transformation
            image = self.image_processing(image)
            images.append(image)

        return torch.stack(images)

    def inference(self, data):
        """Perform inference on all models."""
        # Perform inference on each model
        outputs = dict()
        for label in LABELS:
            outputs[label] = self.models[label](data)

        # Return  all model outputs
        return outputs.values()

    def postprocess(self, inference_output):
        """Postprocess the output data to return the human-readable labels."""
        results = dict()
        for model_output, label in zip(inference_output, LABELS):
            _, predicted = torch.max(model_output, 1)
            results[label] = self.label_dict[label][predicted.item()]

        return {label: results[label] for label in LABELS}
