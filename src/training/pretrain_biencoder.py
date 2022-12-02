# Training script for bi-encoder
import math
import os
import time
from unittest import TextTestRunner

import numpy as np
import torch
from sklearn.preprocessing import normalize

from src.utils.utils import save_yaml

from sentence_transformers.evaluation import (  # isort:skip
    EmbeddingSimilarityEvaluator,
    TripletEvaluator,
)


from sentence_transformers import (  # isort:skip
    SentenceTransformer,
    losses,
    models,
)


class BiEncoderModelPreTrainer:
    def __init__(self, model_name, do_lower_case=True, device="cuda"):
        """
        Pretrainer class for Sentence BERT Bi-encoder model with Triplets/Question pairs.
        """
        self.model_name_or_path = model_name
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

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]
        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        if self.loss_metric == "ContrastiveLoss":
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
        else:
            num_texts = len(batch[0].texts)
            texts = [[] for _ in range(num_texts)]
            labels = []

            for example in batch:
                for idx, text in enumerate(example.texts):
                    texts[idx].append(text)

            anchors = texts[0]
            positives = texts[1]
            negatives = texts[2]

            return anchors, positives, negatives

    def train(self, config, train_dataloader, val_dataloader=None):

        LEARNING_RATE = float(config["PRETRAINING"]["LEARNING_RATE"])
        SCHEDULER = config["PRETRAINING"]["SCHEDULER"]
        TRAIN_OUTPUT_DIR = "./models/" + str(int(time.time())) + "/"
        OPTIMIZER = torch.optim.AdamW
        STEPS_PER_EPOCH = config["PRETRAINING"]["STEPS_PER_EPOCH"]
        NUM_TRAIN_EPOCHS = config["PRETRAINING"]["NUM_TRAIN_EPOCHS"]

        LOSS_METRIC = config["PRETRAINING"]["LOSS_METRIC"]
        self.loss_metric = LOSS_METRIC
        self.model.train()
        # Freeze weights
        params = list(self.word_embedding_model.auto_model.named_parameters())

        if LOSS_METRIC == "ContrastiveLoss":
            train_loss = losses.ContrastiveLoss(model=self.model)
        elif LOSS_METRIC == "TripletLoss":
            train_loss = losses.TripletLoss(
                model=self.model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=0.15,
            )
        else:
            print("Loss metric not supported")
            raise
        warmup_steps = math.ceil(
            STEPS_PER_EPOCH * NUM_TRAIN_EPOCHS * 0.1
        )  # 10% of train data for warm-up
        evaluator = None
        if val_dataloader:
            if LOSS_METRIC == "ContrastiveLoss":
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
            else:
                val_dataloader.collate_fn = self.smart_batching_collate
                anchors = []
                positives = []
                negatives = []
                scores = []
                for x in val_dataloader:
                    anchors = anchors + x[0]
                    positives = positives + x[1]
                    negatives = negatives + x[2]
                evaluator = TripletEvaluator(
                    anchors=anchors, positives=positives, negatives=negatives
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
            checkpoint_save_steps=20000,
            output_path=TRAIN_OUTPUT_DIR,
            checkpoint_path="./models/",
            optimizer_params={"lr": LEARNING_RATE},
            scheduler=SCHEDULER,
            optimizer_class=OPTIMIZER,
        )

        print(f"Time to train.. {str(time.time() - t1)}")

        del warmup_steps, train_loss, train_dataloader, params
        return TRAIN_OUTPUT_DIR
