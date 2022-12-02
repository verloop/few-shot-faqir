import yaml

from src.data.dataloaders import PretrainDataLoader
from src.training.pretrain_biencoder import BiEncoderModelPreTrainer


def pretrain(config):
    """
    Wrapper for pre-training Sentence Bert bi-encoder model as per parameters from config.yaml
    """
    dl_train = PretrainDataLoader(data_set="train")
    dl_val = PretrainDataLoader(data_set="val")
    train_dataloader = dl_train.get_triplets_sbert_dataloader(
        batch_size=config["PRETRAINING"]["BATCH_SIZE"]
    )
    val_dataloader = dl_val.get_triplets_sbert_dataloader(
        batch_size=config["PRETRAINING"]["BATCH_SIZE"]
    )
    trainer = BiEncoderModelPreTrainer(model_name=config["PRETRAINING"]["MODEL_NAME"])
    model_folder = trainer.train(config, train_dataloader, val_dataloader)
    return model_folder


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    pretrain(config)
