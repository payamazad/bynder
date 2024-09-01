from src.model import BynderModel
from src.data import BynderData

IMAGES_DIR = "../dataset/images/"
ORIGINAL_META_FILE = "../dataset/styles.csv"
META_FILE = "../dataset/cleaned_styles.csv"
BATCH_SIZE = 32
LABELS = ["baseColour", "season", "usage"]
N_EPOCHS = 10


def main():
    for label in LABELS:
        data = BynderData(label_name=label,
                          images_dir=IMAGES_DIR,
                          metadata_file_path=META_FILE,
                          bath_size=BATCH_SIZE)
        model = BynderModel(label_name=label,
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


if __name__ == "__main__":
    main()
