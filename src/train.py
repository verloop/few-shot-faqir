import yaml
from transformers import AutoTokenizer

from src.data.dataloaders import HaptikDataLoader
from src.evaluate import evaluate
from src.training.train_biencoder import BiEncoderModelTrainer
from src.training.train_crossencoder import CrossEncoderModelTrainer
from src.utils.utils import get_dataloader_class


def train(config):

    if not config["CROSS_ENCODER_TRAINING"]["USE_CROSS_ENCODER"]:
        trainer = BiEncoderModelTrainer(config["TRAINING"]["MODEL_NAME"])
        dataloader = get_dataloader_class(config)
        data_source = config["DATASETS"]["DATASET_SOURCE"]
        dataset_name = config["DATASETS"]["DATASET_NAME"]
        batch_size = config["TRAINING"]["BATCH_SIZE"]
        dl_train = dataloader(
            data_source=data_source, dataset_name=dataset_name, data_type="train"
        )
        train_dataloader = dl_train.get_qp_sbert_dataloader(batch_size=batch_size)
        model_folder = trainer.train(train_dataloader, config)
        config["EMBEDDINGS"]["MODEL_NAME"] = model_folder
        eval_metrics = evaluate(config)
        return eval_metrics

    else:
        dataloader = get_dataloader_class(config)
        data_source = config["DATASETS"]["DATASET_SOURCE"]
        dataset_name = config["DATASETS"]["DATASET_NAME"]
        batch_size = config["CROSS_ENCODER_TRAINING"]["BATCH_SIZE"]
        tokenizer = AutoTokenizer.from_pretrained(
            config["CROSS_ENCODER_TRAINING"]["TOKENIZER_NAME"]
        )
        dl_class = dataloader(
            data_source=data_source, dataset_name=dataset_name, data_type="train"
        )
        train_dataloader = dl_class.get_qp_train_dataloader(
            tokenizer=tokenizer, batch_size=batch_size
        )
        val_dataloader = dl_class.get_qp_val_dataloader(
            tokenizer=tokenizer, batch_size=batch_size
        )
        trainer = CrossEncoderModelTrainer(config, train_dataloader, val_dataloader)
        model_folder = trainer.train()
        config["EVALUATION"]["CROSS_ENCODER_MODEL_NAME"] = model_folder
        eval_metrics = evaluate(config)
        return eval_metrics


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        train(config)
