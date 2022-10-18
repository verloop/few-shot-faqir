import yaml
from transformers import AutoTokenizer

from src.data.dataloaders import PretrainDataLoader
from src.evaluate import evaluate
from src.training.pretrain_biencoder import BiEncoderModelPreTrainer


def pretrain(config):

    dl_train = PretrainDataLoader(data_set="train")
    dl_val = PretrainDataLoader(data_set="val")
    train_dataloader = dl_train.get_triplets_sbert_dataloader(
        batch_size=config["PRETRAINING"]["BATCH_SIZE"]
    )
    val_dataloader = dl_val.get_triplets_sbert_dataloader(
        batch_size=config["PRETRAINING"]["BATCH_SIZE"]
    )
    trainer = BiEncoderModelPreTrainer()
    model_folder = trainer.train(config, train_dataloader, val_dataloader)
    print(model_folder)
    return model_folder


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    pretrain(config)
