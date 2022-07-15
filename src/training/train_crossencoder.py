import os
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.utils.utils import save_yaml

from transformers import (  # isort:skip
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)


class CrossEncoderModelTrainer:
    def __init__(
        self, cfg, train_dataloader, val_dataloader, do_lower_case=True, device="cuda"
    ):
        self.cfg = cfg
        self.model_name_or_path = self.cfg["CROSS_ENCODER_TRAINING"]["MODEL_NAME"]
        self.tokenizer_name_or_path = self.cfg["CROSS_ENCODER_TRAINING"][
            "TOKENIZER_NAME"
        ]
        self.do_lower_case = do_lower_case
        self.device = torch.device(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path
        )
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.cfg["CROSS_ENCODER_TRAINING"]["LEARNING_RATE"]),
        )
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader

    def train(self):
        num_training_steps = self.cfg["CROSS_ENCODER_TRAINING"]["NUM_ITERATIONS"]
        SCHEDULER = self.cfg["CROSS_ENCODER_TRAINING"]["SCHEDULER"]
        TRAIN_OUTPUT_DIR = (
            self.cfg["CROSS_ENCODER_TRAINING"]["TRAIN_OUTPUT_DIR"] + "cross_encoder.pt"
        )
        LAYER_TO_UNFREEZE = self.cfg["CROSS_ENCODER_TRAINING"][
            "BERT_LAYERS_TO_UNFREEZE"
        ]

        steps_done = 0
        if os.path.isfile(TRAIN_OUTPUT_DIR):
            return TRAIN_OUTPUT_DIR

        lr_scheduler = get_scheduler(
            SCHEDULER,
            optimizer=self.optimizer,
            num_warmup_steps=num_training_steps * 0.1,
            num_training_steps=num_training_steps,
        )

        # Freeze BERT layers
        layers_to_train = {
            f"bert.encoder.layer.{LAYER_TO_UNFREEZE}": self.model.bert.encoder.layer[
                int(LAYER_TO_UNFREEZE)
            ],
            "bert.pooler": self.model.bert.pooler,
            "classifier": self.model.classifier,
        }

        print(f"**** Freezing model layers ****\n")
        params = list(self.model.named_parameters())
        for idx, (name, param) in enumerate(params):
            if not name.startswith(tuple(layers_to_train.keys())):
                print(f"Freezing {name} layer")
                param.requires_grad = False

        progress_bar = tqdm(range(num_training_steps))

        running_loss = 0.0
        running_correct = 0

        while True:
            self.model.train()
            for batch in self.train_dl:
                if steps_done >= 20000:
                    break
                batch[0]["labels"] = batch[1]
                batch[0].to(torch.device("cuda"))
                outputs = self.model(**batch[0])
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                steps_done += 1

                running_loss += loss.item()
                predicted = torch.argmax(outputs.logits, 1)
                running_correct += (predicted == batch[0]["labels"]).sum().item()

                if steps_done % 100 == 0:
                    # print(f"Steps Done: {steps_done}")
                    # print(f"training_loss: {running_loss/100}")
                    # print(f"training_accuracy: {running_correct/100}")
                    running_loss = 0.0
                    running_correct = 0

            if steps_done >= 20000:
                break

            with torch.no_grad():
                labels = []
                preds = []
                n_samples = 0
                n_correct = 0
                val_running_loss = 0
                len_data = len(self.val_dl)
                self.model.eval()
                print("Running validations")

                for batch in self.val_dl:
                    batch[0]["labels"] = batch[1]
                    batch[0].to(torch.device("cuda"))
                    outputs = self.model(**batch[0])

                    val_running_loss += outputs.loss.item()
                    n_samples += batch[0]["labels"].shape[0]
                    predicted = torch.argmax(outputs.logits, 1)
                    n_correct += (predicted == batch[0]["labels"]).sum().item()

                    class_predictions = [
                        F.softmax(output, dim=0) for output in outputs.logits
                    ]

                    preds.append(class_predictions)
                    labels.append(batch[0]["labels"])

                preds = torch.cat([torch.stack(pred) for pred in preds])
                labels = torch.cat(labels)

                acc = (n_correct / n_samples) * 100.0

                print(f"val_accuracy = {acc:.4f}")
                print(f"validation_loss: {val_running_loss/len_data}")
                print(f"validation_accuracy: {acc}")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps": steps_done,
        }
        torch.save(checkpoint, f"{TRAIN_OUTPUT_DIR}")

        save_yaml(self.cfg, f"{TRAIN_OUTPUT_DIR}")

        return TRAIN_OUTPUT_DIR
