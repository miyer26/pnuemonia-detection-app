import peft

import timm
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
import matplotlib.pyplot as plt


class PeftModel:
    def __init__(self, model_id_timm):
        self._model_id_timm = model_id_timm
        self.model = self._get_timm_model()

    def _get_timm_model(self):
        return timm.create_model(self._model_id_timm, pretrained=True, num_classes=2)

    @property
    def conv_layers(self):
        layers = []
        for n, m in self.model.named_modules():
            if type(m) is torch.nn.modules.conv.Conv2d:
                layers.append(n)

        return layers

    def check_final_layers(self):
        return [(n, type(m)) for n, m in self.model.named_modules()][-5:]

    def get_peft_lora_model(self, r=8, modules_to_save=None, device="cuda"):
        config = peft.LoraConfig(
            r, target_modules=self.conv_layers, modules_to_save=modules_to_save
        )
        return peft.get_peft_model(self.model, config).to(device)


class Trainer:
    def __init__(
        self,
        model=None,
        train_dataloader=None,
        eval_dataloader=None,
        optimizer=None,
        criterion=None,
        device=None,
        patience=3,
        num_epochs=10,
        lr_scheduler=None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.num_epochs = num_epochs
        self.lr_scheduler = lr_scheduler  # Assigning the lr_scheduler attribute
        self.optimizer = optimizer

    def print_trainable_parameters(self):
        grouped_params = {}
        total_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                group = name.split(".")[
                    0
                ]  # Assume the first part of the name is the group
                if group not in grouped_params:
                    grouped_params[group] = []
                grouped_params[group].append(name)
                total_params += param.numel()

        # Print parameters grouped by their respective groups
        for group, params in grouped_params.items():
            print(f"Group: {group}")
            for param in params:
                print(param)

        print(f"Total Trainable Parameters: {total_params}")

    def compute_classwise_precision(self, predictions, targets, class_label=1):

        return precision_score(
            targets, predictions, pos_label=class_label, average="binary"
        )

    def train(self):
        best_metric = float("-inf")
        best_epoch = 0
        no_improvement_counter = 0

        self.model.train()

        for epoch in range(self.num_epochs):
            train_loss = 0
            train_progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                unit="batch",
            )
            for batch in train_progress_bar:
                x, y = batch["x"].to(self.device), batch["y"].to(self.device)
                outputs = self.model(x)
                loss = self.criterion(
                    outputs, y
                )  # Accessing criterion from the class attribute
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += loss.item()
                train_progress_bar.set_postfix({"loss": loss.item()})

                # Update learning rate using scheduler
                # if self.lr_scheduler:
                #   self.lr_scheduler.step()  # Update the learning rate

            train_progress_bar.close()

            # Evaluation
            self.model.eval()
            correct = 0
            n_total_0 = 0
            n_total_1 = 0
            precision_sum_1 = 0.0
            precision_weighed = 0.0
            f1_score_batch = 0.0
            total_samples = 0
            eval_loss = 0

            train_loss_tracker = []
            valid_loss_tracker = []

            eval_progress_bar = tqdm(
                self.eval_dataloader, desc="Evaluation", unit="batch"
            )
            for batch in eval_progress_bar:
                x, y = batch["x"].to(self.device), batch["y"].to(self.device)
                with torch.no_grad():
                    outputs = self.model(x)
                loss = self.criterion(
                    outputs, y
                )  # Accessing criterion from the class attribute
                eval_loss += loss.item()

                # Compute precision
                predictions = torch.argmax(outputs, dim=-1).cpu().tolist()
                targets = y.cpu().tolist()
                precision_1 = self.compute_classwise_precision(
                    predictions, targets, class_label=1
                )
                precision_weighed += precision_score(
                    targets, predictions, average="weighted"
                )
                f1_score_batch += f1_score(targets, predictions)
                precision_sum_1 += precision_1

                # Correctness check
                correct += (torch.argmax(outputs, dim=-1) == y).sum().item()
                n_total_0 += (y == 0).sum()
                n_total_1 += (y == 1).sum()
                total_samples += len(predictions)

            eval_progress_bar.close()

            # Calculate metrics
            average_precision_1 = (
                precision_sum_1 / len(self.eval_dataloader)
                if total_samples > 0
                else 0.0
            )
            average_precision_weighed = (
                precision_weighed / len(self.eval_dataloader)
                if total_samples > 0
                else 0.0
            )
            average_f1_score = (
                f1_score_batch / len(self.eval_dataloader) if total_samples > 0 else 0.0
            )
            train_loss_total = train_loss / len(self.train_dataloader)
            valid_loss_tracker.append(train_loss_total)
            valid_loss_total = eval_loss / len(self.eval_dataloader)
            train_loss_tracker.append(valid_loss_total)
            valid_acc_total = correct / (n_total_0 + n_total_1)

            # update learning rate if needed
            self.lr_scheduler.step(average_precision_1)
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")

            if epoch == 0:
                print(
                    f"Number of samples in Class 1: {n_total_1} \t Class 0: {n_total_0}"
                )

            print(
                f"{epoch=:<2}  {train_loss_total=:.4f}  {valid_loss_total=:.4f}  {valid_acc_total=:.4f}"
            )
            print(
                f"Epoch {epoch + 1}: Average precision for class 1: {average_precision_1}"
            )
            print(
                f"Epoch {epoch + 1}: Average weighed precision: {average_precision_weighed}"
            )
            print(f"Epoch {epoch + 1}: f1 score for class 1: {average_f1_score}")

            # Check for improvement
            if average_precision_1 > best_metric:
                best_metric = average_precision_1
                best_epoch = epoch
                # Save the full model state
                torch.save(self.model, "best_model.pth")
                print(f"Best model weights saved at epoch {epoch + 1}")
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            # Early stopping
            if no_improvement_counter >= self.patience:
                print(
                    f"No improvement in precision for {self.patience} epochs. Early stopping..."
                )
                break

        print(
            f"Training finished. Best average precision: {best_metric} at epoch {best_epoch + 1}"
        )

        plt.figure()
        epochs = range(1, len(valid_loss_tracker) + 1)
        # Plot validation loss over epochs
        plt.plot(epochs, valid_loss_tracker, label="Validation Loss")
        plt.plot(epochs, train_loss_tracker, label="Train Loss")
        plt.title("Losses Over Training Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def test(self, test_dataloader, best_model_dir):

        model = torch.load(best_model_dir)
        if torch.cuda.is_available():
            model = model.cuda()

        model.eval()

        true_labels = []
        predicted_labels = []

        test_progress_bar = tqdm(test_dataloader, desc="Testing", unit="batch")
        # Run inference on the test dataset
        for batch in test_progress_bar:
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)

            # Forward pass
            outputs = model(x)

            # Convert logits to probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get predicted labels
            _, predicted = torch.max(probabilities, 1)

            # Move predicted labels to CPU if necessary
            predicted = predicted.cpu().numpy()

            # Append true and predicted labels
            true_labels.extend(y.cpu().numpy())
            predicted_labels.extend(predicted)

        # Calculate accuracy and precision
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average="weighted")
        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Normal", "Pneumonia"]
        )
        disp.plot()

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
