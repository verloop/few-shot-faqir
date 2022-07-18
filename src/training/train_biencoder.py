# Training script for bi-encoder
from sentence_transformers import (  # isort:skip
    SentenceTransformer,
    losses,
    models,
)
import math
import time

import torch
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from src.utils.utils import save_yaml


class BiEncoderModelTrainer:
    def __init__(self, config, do_lower_case=True, device="cuda"):
        self.config = config
        self.model_name_or_path = self.config["TRAINING"]["MODEL_NAME"]
        self.do_lower_case = do_lower_case
        self.device_str = device
        self.device = torch.device(device)
        self.word_embedding_model = models.Transformer(
            self.model_name_or_path, do_lower_case=self.do_lower_case
        )
        self.word_embedding_model.auto_model = self.word_embedding_model.auto_model.to(
            self.device
        )
        self.pooling_model = models.Pooling(
            self.word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        self.model = SentenceTransformer(
            modules=[self.word_embedding_model, self.pooling_model],
            device=self.device_str,
        )
        self.layers_to_train = {
            "pooler": self.word_embedding_model.auto_model.pooler,
        }

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

        sentences_1 = texts[0]
        sentences_2 = texts[1]

        return sentences_1, sentences_2, labels

    def train(self, train_dataloader, val_dataloader=None):

        NUM_ITERATIONS = self.config["TRAINING"]["NUM_ITERATIONS"]
        LEARNING_RATE = float(self.config["TRAINING"]["LEARNING_RATE"])
        SCHEDULER = self.config["TRAINING"]["SCHEDULER"]
        TRAIN_OUTPUT_DIR = (
            self.config["TRAINING"]["TRAIN_OUTPUT_DIR"] + str(int(time.time())) + "/"
        )
        OPTIMIZER = torch.optim.AdamW
        STEPS_PER_EPOCH = math.ceil(len(train_dataloader))
        NUM_TRAIN_EPOCHS = math.ceil(NUM_ITERATIONS / STEPS_PER_EPOCH)

        if STEPS_PER_EPOCH * NUM_TRAIN_EPOCHS > NUM_ITERATIONS:
            STEPS_PER_EPOCH = math.ceil(NUM_ITERATIONS / NUM_TRAIN_EPOCHS)

        LOSS_METRIC = self.config["TRAINING"]["LOSS_METRIC"]
        self.model.train()
        LAYERS_TO_UNFREEZE = self.config["TRAINING"]["LAYERS_TO_UNFREEZE"]
        for layer in LAYERS_TO_UNFREEZE:
            self.layers_to_train.update(
                {
                    f"encoder.layer.{layer}": self.word_embedding_model.auto_model.encoder.layer[
                        layer
                    ]
                }
            )

        # Freeze weights
        params = list(self.word_embedding_model.auto_model.named_parameters())
        for idx, (name, param) in enumerate(params):
            if not name.startswith(tuple(self.layers_to_train.keys())):
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("Parameter gradients frozen")
        if LOSS_METRIC == "CosineSimilarityLoss":
            train_loss = losses.CosineSimilarityLoss(model=self.model)
        elif LOSS_METRIC == "ContrastiveLoss":
            train_loss = losses.ContrastiveLoss(model=self.model)
        else:
            print("Loss metric not supported")
            raise
        warmup_steps = math.ceil(
            len(train_dataloader) * NUM_TRAIN_EPOCHS * 0.1
        )  # 10% of train data for warm-up
        evaluator = None
        if val_dataloader:
            val_dataloader.collate_fn = self.smart_batching_collate
            sentences1 = []
            sentences2 = []
            scores = []
            for x in val_dataloader:
                sentences1 = sentences1 + x[0]
                sentences2 = sentences2 + x[1]
                scores = scores + x[2]
            evaluator = EmbeddingSimilarityEvaluator(
                sentences1=sentences1, sentences2=sentences2, scores=scores
            )
        print("Setup losses and warmup steps")
        t1 = time.time()
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=NUM_TRAIN_EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=TRAIN_OUTPUT_DIR,
            optimizer_params={"lr": LEARNING_RATE},
            scheduler=SCHEDULER,
            optimizer_class=OPTIMIZER,
        )

        print(f"Time to train {str(time.time() - t1)}")
        save_yaml(self.config, TRAIN_OUTPUT_DIR)

        del warmup_steps, train_loss, train_dataloader, params
        return TRAIN_OUTPUT_DIR
