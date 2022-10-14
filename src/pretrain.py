import yaml
from transformers import AutoTokenizer

from src.data.dataloaders import PretrainDataLoader
from src.evaluate import evaluate
from src.training.pretrain_biencoder import BiEncoderModelPreTrainer


def pretrain():

    dl_train = PretrainDataLoader()
    train_dataloader = dl_train.get_triplets_sbert_dataloader(batch_size=32)
    trainer = BiEncoderModelPreTrainer()

    model_folder = trainer.train(train_dataloader)
    print(model_folder)
    return model_folder


if __name__ == "__main__":
    pretrain()
