import os.path
from src.Cream.TinyViT.models.tiny_vit import tiny_vit_21m_224
import torch.optim as optim
import pandas as pd
import os
import time
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import json


class BynderModel:
    def __init__(self, label_name: str, label_map: dict, log_path: str, model_path: str, train_last_layer: bool = True):
        self.n_classes = len(label_map)
        self.train_last_layer = train_last_layer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_name = label_name
        self.classifier = self.create_model()
        self.log_path = log_path
        self.model_path = model_path

    def create_model(self):
        # Build model
        model = tiny_vit_21m_224(pretrained=True)
        if self.train_last_layer:
            for param in model.parameters():
                param.requires_grad = False
        model.head = torch.nn.Linear(model.head.in_features, self.n_classes)
        for param in model.head.parameters():
            param.requires_grad = True
        return model

    def train(self, train_dataloader, val_dataloader, n_epochs) -> pd.DataFrame:
        self.classifier.to(self.device)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)

        # Initialize logs
        best_val_loss = float('inf')
        logs = {}

        # Training loop
        for epoch in range(n_epochs):
            epoch_log = {}
            running_loss = 0.0
            start_time = time.time()
            print(f"Epoch {epoch + 1}/{n_epochs}")

            # Training phase
            self.classifier.train()
            correct_train = 0
            total_train = 0

            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.classifier(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Track training loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            epoch_log["train_loss"] = running_loss / len(train_dataloader.dataset)
            epoch_log["train_acc"] = correct_train / total_train

            print(f"Training Loss: {epoch_log['train_loss']:.4f} | Training Accuracy: {epoch_log['train_acc']:.4f}")

            # Validation phase
            self.classifier.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.classifier(inputs)
                    loss = criterion(outputs, labels)

                    running_val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())

            epoch_log["val_loss"] = running_val_loss / len(val_dataloader.dataset)
            epoch_log["val_acc"] = correct_val / total_val
            print(f"Validation Loss: {epoch_log['val_loss']:.4f} | Validation Accuracy: {epoch_log['val_acc']:.4f}")
            # Calculate precision, recall, and F1-score
            epoch_log["val_precision"] = precision_score(all_labels, all_preds, average='weighted', zero_division=0.0)
            epoch_log["val_recall"] = recall_score(all_labels, all_preds, average='weighted', zero_division=0.0)
            epoch_log["val_f1s"] = f1_score(all_labels, all_preds, average='weighted', zero_division=0.0)

            # Save the model if validation loss improves
            if epoch_log["val_loss"] < best_val_loss:
                best_val_loss = epoch_log["val_loss"]
                root_path = f"{self.model_path}/{self.label_name}"
                if not os.path.exists(root_path):
                    os.makedirs(root_path)
                torch.save(self.classifier.state_dict(), f'{root_path}/best_model.pth')
                print("Best model saved.")

            logs[epoch] = epoch_log
        # Total time logging
        total_time = time.time() - start_time
        print(f"Training completed in {total_time / 60:.2f} minutes.")
        df = pd.DataFrame.from_dict(logs)
        log_path = f"{self.log_path}/{self.label_name}"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        df.to_csv(f"{log_path}/train_log.csv")
        return df

    def test(self, test_dataloader) -> dict:
        # load best model
        best_model_path = f"{self.model_path}/{self.label_name}/best_model.pth"
        if os.path.exists(best_model_path):
            self.classifier.load_state_dict(torch.load(best_model_path), strict=True)
        self.classifier.to(self.device)
        self.classifier.eval()  # Set the model to evaluation mode

        # Define loss criterion
        criterion = nn.CrossEntropyLoss()

        test_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.classifier(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

                # Get predictions and append them for metric calculation
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        metrics = dict()

        # Average loss
        metrics["avg_test_loss"] = test_loss / len(test_dataloader.dataset)

        # Calculate precision, recall, and F1-score
        metrics["precision"] = float(precision_score(all_labels, all_preds, average='weighted', zero_division=0))
        metrics["recall"] = float(recall_score(all_labels, all_preds, average='weighted', zero_division=0))
        metrics["f1"] = float(f1_score(all_labels, all_preds, average='weighted', zero_division=0))

        print(f"Test Loss: {metrics['avg_test_loss']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")

        with open(f"{self.log_path}/{self.label_name}/test_metrics.json", "w") as f:
            json.dump(metrics, f)
        return metrics

    def save(self):
        root_path = f"{self.model_path}/{self.label_name}/"
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        model_path = f"{root_path}/saved_model.pt"
        # model_scripted = torch.jit.script(self.classifier)  # Export to TorchScript
        # model_scripted.save(model_path)  # Save
        torch.save(self.classifier, model_path)
        return model_path

    def load(self, model_path):
        # self.classifier = torch.jit.load(model_path)
        self.classifier = torch.load(model_path, weights_only=False)

    def infer(self, image):
        return self.classifier(image)
