import yaml
from transformers import AutoTokenizer

from src.data.dataloaders import HaptikDataLoader
from src.evaluate import evaluate
from src.training.train_biencoder import BiEncoderModelTrainer
from src.training.train_classifier import BertBasedClassifier
from src.training.train_crossencoder import CrossEncoderModelTrainer
from src.training.train_sbert_crossencoder import SbertCrossEncoderModelTrainer
from src.utils.utils import get_dataloader_class


def train(config):
    dataloader = get_dataloader_class(config)
    data_source = config["DATASETS"]["DATASET_SOURCE"]
    dataset_name = config["DATASETS"]["DATASET_NAME"]
    data_subset = config["DATASETS"]["DATA_SUBSET"]
    batch_size = config["TRAINING"]["BATCH_SIZE"]
    num_labels = config["DATASETS"]["N_LABELS"]

    dl_train = dataloader(
        data_source=data_source,
        dataset_name=dataset_name,
        data_type="train",
        data_subset=data_subset,
    )

    if config["TRAINING"]["MODEL_TYPE"] == "BI_ENCODER":
        print("Training and evaluation with Bi-Encoder")
        trainer = BiEncoderModelTrainer(config)
        train_dataloader, val_dataloader = dl_train.get_qp_sbert_dataloader(
            batch_size=batch_size, val_split_pct=config["TRAINING"]["VALIDATION_SPLIT"]
        )
        model_folder = trainer.train(train_dataloader, val_dataloader)
        print(model_folder)
        return model_folder

    if config["TRAINING"]["MODEL_TYPE"] == "CROSS_ENCODER":
        print("Training and evaluation with Cross Encoder")
        tokenizer = AutoTokenizer.from_pretrained(config["TRAINING"]["TOKENIZER_NAME"])
        train_dataloader, val_dataloader = dl_train.get_qp_dataloader(
            tokenizer=tokenizer,
            batch_size=batch_size,
            val_split_pct=config["TRAINING"]["VALIDATION_SPLIT"],
        )
        trainer = CrossEncoderModelTrainer(config)
        model_folder = trainer.train(train_dataloader, val_dataloader)
        print(model_folder)
        return model_folder

    if config["TRAINING"]["MODEL_TYPE"] == "SBERT_CROSS_ENCODER":
        print("Training and evaluation with sbert cross encoder")
        trainer = SbertCrossEncoderModelTrainer(config)
        train_dataloader, val_dataloader = dl_train.get_qp_sbert_dataloader(
            batch_size=batch_size, val_split_pct=config["TRAINING"]["VALIDATION_SPLIT"]
        )
        model_folder = trainer.train(train_dataloader, val_dataloader)
        print(model_folder)
        return model_folder

    if config["TRAINING"]["MODEL_TYPE"] == "CLASSIFIER":
        print("Training with classifier")
        tokenizer = AutoTokenizer.from_pretrained(config["TRAINING"]["TOKENIZER_NAME"])
        train_dataloader, val_dataloader = dl_train.get_dataloader(
            batch_size=batch_size, tokenizer=tokenizer, val_split_pct=0.2
        )
        bert_classifier = BertBasedClassifier(
            model_name=config["TRAINING"]["MODEL_NAME"], num_labels=num_labels
        )
        model_folder = bert_classifier.train(config, train_dataloader, val_dataloader)
        print(model_folder)
        return model_folder


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    train(config)
