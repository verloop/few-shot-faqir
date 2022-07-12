# Training script for bi-encoder
from sentence_transformers import (  # isort:skip
    SentenceTransformer,
    losses,
    models,
)
import math
import time

import torch

from src.utils.utils import save_yaml


class BiEncoderModelTrainer:
    def __init__(self, model_name, do_lower_case=True, device="cuda"):
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
        self.layers_to_train = {
            "pooler": self.word_embedding_model.auto_model.pooler,
        }

    def train(self, train_dataloader, cfg):

        NUM_ITERATIONS = cfg["TRAINING"]["NUM_ITERATIONS"]
        LEARNING_RATE = float(cfg["TRAINING"]["LEARNING_RATE"])
        SCHEDULER = cfg["TRAINING"]["SCHEDULER"]
        TRAIN_OUTPUT_DIR = (
            cfg["TRAINING"]["TRAIN_OUTPUT_DIR"] + str(int(time.time())) + "/"
        )
        OPTIMIZER = torch.optim.AdamW
        STEPS_PER_EPOCH = math.ceil(len(train_dataloader))
        NUM_TRAIN_EPOCHS = math.ceil(NUM_ITERATIONS / STEPS_PER_EPOCH)

        if STEPS_PER_EPOCH * NUM_TRAIN_EPOCHS > NUM_ITERATIONS:
            STEPS_PER_EPOCH = math.ceil(NUM_ITERATIONS / NUM_TRAIN_EPOCHS)

        LOSS_METRIC = cfg["TRAINING"]["LOSS_METRIC"]
        self.model.train()
        LAYERS_TO_UNFREEZE = cfg["TRAINING"]["SENTBERT_LAYERS_TO_UNFREEZE"]
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
        print("Setup losses and warmup steps")
        t1 = time.time()
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
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
        save_yaml(cfg, TRAIN_OUTPUT_DIR)

        del warmup_steps, train_loss, train_dataloader, params
        return TRAIN_OUTPUT_DIR
