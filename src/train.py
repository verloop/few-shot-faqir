import yaml
from transformers import AutoTokenizer

from src.data.dataloaders import HaptikDataLoader
from src.evaluate import evaluate
from src.training.train_biencoder import BiEncoderModelTrainer
from src.training.train_crossencoder import CrossEncoderModelTrainer
from src.training.train_sbert_crossencoder import SbertCrossEncoderModelTrainer
from src.utils.utils import get_dataloader_class


def train(config):
    dataloader = get_dataloader_class(config)
    data_source = config["DATASETS"]["DATASET_SOURCE"]
    dataset_name = config["DATASETS"]["DATASET_NAME"]
    data_subset = config["DATASETS"]["DATA_SUBSET"]
    batch_size = config["TRAINING"]["BATCH_SIZE"]
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
        config["EMBEDDINGS"]["MODEL_NAME"] = model_folder
        config["EVALUATION"]["EVALUATION_METHOD"] == "EMBEDDINGS"
        eval_metrics = evaluate(config)
        return eval_metrics

    if config["TRAINING"]["MODEL_TYPE"] == "CROSS_ENCODER":
        print("Training and evaluation with Cross Encoder")
        tokenizer = AutoTokenizer.from_pretrained(config["TRAINING"]["TOKENIZER_NAME"])
        train_dataloader, val_dataloader = dl_train.get_qp_dataloader(
            tokenizer=tokenizer,
            batch_size=batch_size,
            val_split_pct=config["TRAINING"]["VALIDATION_SPLIT"],
        )
        val_dataloader = None
        trainer = CrossEncoderModelTrainer(config)
        model_folder = trainer.train(train_dataloader, val_dataloader)
        print(model_folder)
        config["TRAINING"]["MODEL_NAME"] = model_folder
        config["EVALUATION"]["EVALUATION_METHOD"] == "CROSS_ENCODER"
        eval_metrics = evaluate(config)
        return eval_metrics

    if config["TRAINING"]["MODEL_TYPE"] == "SBERT_CROSS_ENCODER":
        print("Training and evaluation with sbert cross encoder")
        trainer = SbertCrossEncoderModelTrainer(config)
        train_dataloader, val_dataloader = dl_train.get_qp_sbert_dataloader(
            batch_size=batch_size, val_split_pct=config["TRAINING"]["VALIDATION_SPLIT"]
        )
        model_folder = trainer.train(train_dataloader, val_dataloader)
        # model_folder="models/1658252948_clinc_train_5"
        print(model_folder)
        config["TRAINING"]["MODEL_NAME"] = model_folder
        config["EVALUATION"]["EVALUATION_METHOD"] == "SBERT_CROSS_ENCODER"
        eval_metrics = evaluate(config)
        return eval_metrics


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        train(config)
