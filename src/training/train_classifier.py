import math
import time

import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm

from src.evaluate import evaluate

from src.utils.utils import (  # isort:skip
    get_dataloader_class,
    run_evaluation_metrics,
    save_yaml,
)

from transformers import (  # isort:skip
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


set_seed(123)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open("src/config/config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


class BertBasedClassifier:
    def __init__(self, model_name, num_labels):
        self.model_name = model_name
        self.num_labels = num_labels
        self.load_model()
        self.layers_to_train = ["pooler", "classifier"]

    def load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

    def train(self, config, train_dataloader, eval_dataloader=None):
        LEARNING_RATE = float(config["TRAINING"]["LEARNING_RATE"])
        NUM_ITERATIONS = config["TRAINING"]["NUM_ITERATIONS"]
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
        STEPS_PER_EPOCH = math.ceil(len(train_dataloader))
        NUM_TRAIN_EPOCHS = math.ceil(NUM_ITERATIONS / STEPS_PER_EPOCH)
        TRAIN_OUTPUT_DIR = (
            config["TRAINING"]["TRAIN_OUTPUT_DIR"] + str(int(time.time())) + "/"
        )
        if STEPS_PER_EPOCH * NUM_TRAIN_EPOCHS > NUM_ITERATIONS:
            STEPS_PER_EPOCH = math.ceil(NUM_ITERATIONS / NUM_TRAIN_EPOCHS)

        # Freeze weights
        LAYERS_TO_UNFREEZE = config["TRAINING"]["LAYERS_TO_UNFREEZE"]
        for layer in LAYERS_TO_UNFREEZE:
            self.layers_to_train.append(f"bert.encoder.layer.{layer}")

        params = list(self.model.named_parameters())

        for idx, (name, param) in enumerate(params):
            if not name.startswith(tuple(self.layers_to_train)):
                param.requires_grad = False
            else:
                param.requires_grad = True

        warmup_steps = math.ceil(
            len(train_dataloader) * NUM_TRAIN_EPOCHS * 0.1
        )  # 10% of train data for warm-up

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=STEPS_PER_EPOCH
        )
        all_loss = {"train_loss": [], "val_loss": []}
        all_acc = {"train_acc": [], "val_acc": []}
        self.model.to(device)

        for epoch in tqdm(range(NUM_TRAIN_EPOCHS)):
            self.model.train()
            prediction_labels = []
            true_labels = []
            n = 0
            total_loss = 0
            progress_bar = tqdm(range(STEPS_PER_EPOCH))
            for batch in train_dataloader:
                batch[0]["labels"] = batch[1]
                batch[0].to(torch.device("cuda"))
                true_labels += batch[1].detach().cpu().flatten().tolist()
                outputs = self.model(**batch[0])
                loss = outputs.loss
                loss, logits = outputs[:2]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                optimizer.zero_grad()
                predicted = torch.argmax(outputs.logits, 1)

                prediction_labels += predicted.detach().cpu().flatten().tolist()
                n += 1
                if n > STEPS_PER_EPOCH:
                    break

            avg_epoch_loss = total_loss / STEPS_PER_EPOCH
            train_acc = accuracy_score(true_labels, prediction_labels)
            all_loss["train_loss"].append(avg_epoch_loss)
            all_acc["train_acc"].append(train_acc)

            if eval_dataloader:
                self.model.eval()
                total_loss_eval = 0
                prediction_labels_eval = []
                true_labels_eval = []
                for batch in eval_dataloader:
                    true_labels_eval += batch[1].numpy().flatten().tolist()
                    with torch.no_grad():
                        batch[0]["labels"] = batch[1]
                        batch[0].to(torch.device("cuda"))
                        outputs = self.model(**batch[0])
                        loss, logits = outputs[:2]
                        logits = logits.detach().cpu().numpy()
                        total_loss_eval += loss.item()
                        predicted = torch.argmax(outputs.logits, 1)
                        prediction_labels_eval += (
                            predicted.detach().cpu().flatten().tolist()
                        )

                avg_epoch_eval_loss = total_loss_eval / len(eval_dataloader)
                eval_acc = accuracy_score(true_labels_eval, prediction_labels_eval)
                all_acc["val_acc"].append(eval_acc)
                all_loss["val_loss"].append(avg_epoch_eval_loss)

        print(all_acc, all_loss)
        # Save trained model
        self.model.save_pretrained(TRAIN_OUTPUT_DIR)
        save_yaml(config, f"{TRAIN_OUTPUT_DIR}/")

        return TRAIN_OUTPUT_DIR

    def predict(self, test_dataloader):
        self.model.eval()
        prediction_labels_test = []
        true_labels_test = []
        pred_probs = []
        for batch in test_dataloader:
            true_labels_test += batch[1].numpy().flatten().tolist()
            with torch.no_grad():
                batch[0].to(torch.device("cuda"))
                outputs = self.model(**batch[0])
                prediction_probs = F.softmax(outputs.logits, dim=1)
                predicted = torch.argsort(outputs.logits, 1, descending=True)
                predicted = predicted.detach().cpu().numpy()
                prediction_probs = prediction_probs.detach().cpu().numpy()
                pred_probs_ = [
                    list(i[j])[0] for i, j in zip(prediction_probs, predicted)
                ]
                pred_probs += pred_probs_
                prediction_labels_test += list(predicted)
        return true_labels_test, prediction_labels_test, pred_probs


# def evaluate_bert_classifier(config):
#     data_source = config["DATASETS"]["DATASET_SOURCE"]
#     dataset_name = config["DATASETS"]["DATASET_NAME"]
#     data_subset = config["DATASETS"]["DATA_SUBSET"]
#     num_labels = config["DATASETS"]["N_LABELS"]

#     model_name = config["TRAINING"]["MODEL_NAME"]
#     batch_size = config["TRAINING"]["BATCH_SIZE"]

#     dataloader = get_dataloader_class(config)
#     dl_train = dataloader(
#         data_source=data_source,
#         dataset_name=dataset_name,
#         data_type="train",
#         data_subset=data_subset,
#     )

#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     train_dataloader, val_dataloader = dl_train.get_dataloader(
#         batch_size=batch_size, tokenizer=tokenizer, val_split_pct=0.2
#     )

#     bert_classifier = BertBasedClassifier(model_name=model_name, num_labels=num_labels)

#     finetuned_model_name = bert_classifier.train(
#         config, train_dataloader, val_dataloader
#     )
#     print(finetuned_model_name)

#     dl_test = dataloader(
#         data_source=data_source,
#         dataset_name=dataset_name,
#         data_type="test",
#         intent_label_to_idx=dl_train.dataset.intent_label_to_idx,
#     )

#     test_dataloader, _ = dl_test.get_dataloader(
#         batch_size=batch_size, tokenizer=tokenizer
#     )

#     true_labels_test, prediction_labels_test, pred_probs = bert_classifier.predict(
#         test_dataloader
#     )

#     true_labels_test = [[i] for i in true_labels_test]

#     oos_label_indx = None
#     if config["EVALUATION"]["CHECK_OOS_ACCURACY"]:
#         oos_label_indx = dl_train.dataset.intent_label_to_idx[
#             config["DATASETS"]["OOS_CLASS_NAME"]
#         ]

#     eval_metrics = run_evaluation_metrics(
#         config, true_labels_test, prediction_labels_test, pred_probs, oos_label_indx
#     )

#     return eval_metrics


# def evaluate_bert_classifier_embeddings(config):
#     data_source = config["DATASETS"]["DATASET_SOURCE"]
#     dataset_name = config["DATASETS"]["DATASET_NAME"]
#     data_subset = config["DATASETS"]["DATA_SUBSET"]
#     num_labels = config["DATASETS"]["N_LABELS"]

#     model_name = config["TRAINING"]["MODEL_NAME"]
#     batch_size = config["TRAINING"]["BATCH_SIZE"]

#     dataloader = get_dataloader_class(config)
#     dl_train = dataloader(
#         data_source=data_source,
#         dataset_name=dataset_name,
#         data_type="train",
#         data_subset=data_subset,
#     )

#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     train_dataloader, val_dataloader = dl_train.get_dataloader(
#         batch_size=batch_size, tokenizer=tokenizer, val_split_pct=0.2
#     )

#     bert_classifier = BertBasedClassifier(model_name=model_name, num_labels=num_labels)

#     finetuned_model_name = bert_classifier.train(
#         config, train_dataloader, val_dataloader
#     )
#     print(finetuned_model_name)
#     config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
#     config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
#     config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"
#     config["EVALUATION"]["MODEL_NAME"] = finetuned_model_name
#     eval_metrics = evaluate(config)

#     return eval_metrics
