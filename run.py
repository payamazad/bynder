import os

import numpy as np
import pandas as pd
from PIL import Image

from src.data import BynderData
from src.model import BynderModel

IMAGES_DIR = "../dataset/images/"
ORIGINAL_META_FILE = "../dataset/styles.csv"
META_FILE = "../dataset/cleaned_styles.csv"
BATCH_SIZE = 32
LABELS = ["gender", "articleType", "baseColour", "season", "usage"]
N_EPOCHS = 10


def main():
    for label in LABELS:
        data = BynderData(label_name=label, images_dir=IMAGES_DIR, metadata_file_path=META_FILE, bath_size=BATCH_SIZE)
        model = BynderModel(
            label_name=label,
            label_map=data.label_map,
            log_path="logs",
            model_path="models",
        )
        train_logs = model.train(
            train_dataloader=data.train_dataloader,
            val_dataloader=data.val_dataloader,
            n_epochs=N_EPOCHS,
        )
        print(train_logs)
        test_metrics = model.test(data.test_dataloader)
        print(test_metrics)
        _ = model.save()


def clean_data():
    """
    function to clean data
    """
    metadata = pd.read_csv(ORIGINAL_META_FILE)
    path = IMAGES_DIR
    wrong_shape_images = []
    missing_files = []
    for image_id in metadata.id.items():
        filepath = path + str(image_id[1]) + ".jpg"
        if not os.path.isfile(filepath):
            missing_files.append(image_id[1])
            continue
        img = np.array(Image.open(filepath))
        if img.ndim != 3:
            # print(f"{filename} :: {img.shape}")
            wrong_shape_images.append(image_id[1])
    metadata = metadata[~metadata["id"].isin(wrong_shape_images + missing_files)]
    metadata = metadata.fillna("missing")
    metadata.to_csv(META_FILE)


if __name__ == "__main__":
    clean_data()
    main()
