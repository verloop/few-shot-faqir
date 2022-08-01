# Training script for bi-encoder
from sentence_transformers import (  # isort:skip
    SentenceTransformer,
    losses,
    models,
)
import math
import time

import torch
from sentence_transformers.cross_encoder import CrossEncoder

from src.utils.utils import save_yaml

from sentence_transformers.cross_encoder.evaluation import (  # isort:skip
    CEBinaryClassificationEvaluator,
)


class SbertCrossEncoderModelTrainer:
    def __init__(self, config, do_lower_case=True, device="cuda"):
        self.config = config
        self.model_name_or_path = self.config["TRAINING"]["MODEL_NAME"]
        self.do_lower_case = do_lower_case
        self.device_str = device
        self.device = torch.device(device)
        self.model = CrossEncoder(
            self.model_name_or_path,
            num_labels=1,
            device=self.device_str,
        )
        self.layers_to_train = ["classifier"]

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]
        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels = labels + [example.label]

        sentence_pairs = list(zip(texts[0], texts[1]))
        return sentence_pairs, labels

    def train(self, train_dataloader, val_dataloader=None):

        NUM_ITERATIONS = self.config["TRAINING"]["NUM_ITERATIONS"]
        LEARNING_RATE = float(self.config["TRAINING"]["LEARNING_RATE"])
        SCHEDULER = self.config["TRAINING"]["SCHEDULER"]
        TRAIN_OUTPUT_DIR = self.config["TRAINING"]["TRAIN_OUTPUT_DIR"] + str(
            int(time.time())
        )
        OPTIMIZER = torch.optim.AdamW
        STEPS_PER_EPOCH = math.ceil(len(train_dataloader))
        NUM_TRAIN_EPOCHS = math.ceil(NUM_ITERATIONS / STEPS_PER_EPOCH)
        if STEPS_PER_EPOCH * NUM_TRAIN_EPOCHS > NUM_ITERATIONS:
            STEPS_PER_EPOCH = math.ceil(NUM_ITERATIONS / NUM_TRAIN_EPOCHS)

        # self.model.train()

        LAYERS_TO_UNFREEZE = self.config["TRAINING"]["LAYERS_TO_UNFREEZE"]
        for layer in LAYERS_TO_UNFREEZE:
            self.layers_to_train.append(f"roberta.encoder.layer.{layer}")

        # Freeze weights
        params = list(self.model.model.named_parameters())
        for idx, (name, param) in enumerate(params):
            if not name.startswith(tuple(self.layers_to_train)):
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("Parameter gradients frozen")
        warmup_steps = math.ceil(
            len(train_dataloader) * NUM_TRAIN_EPOCHS * 0.1
        )  # 10% of train data for warm-up
        evaluator = None
        if val_dataloader:
            val_dataloader.collate_fn = self.smart_batching_collate
            sentence_pairs = []

            labels = []
            for x in val_dataloader:
                sentence_pairs = sentence_pairs + x[0]
                labels = labels + x[1]
            evaluator = CEBinaryClassificationEvaluator(
                sentence_pairs=sentence_pairs, labels=labels
            )
        print("Setup losses and warmup steps")
        t1 = time.time()
        self.model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=NUM_TRAIN_EPOCHS,
            evaluation_steps=500,
            warmup_steps=warmup_steps,
            output_path=TRAIN_OUTPUT_DIR,
            optimizer_params={"lr": LEARNING_RATE},
            scheduler=SCHEDULER,
            optimizer_class=OPTIMIZER,
            save_best_model=True,
        )

        print(f"Time to train {str(time.time() - t1)}")
        save_yaml(self.config, TRAIN_OUTPUT_DIR + "/")

        del warmup_steps, train_dataloader, params
        return TRAIN_OUTPUT_DIR
