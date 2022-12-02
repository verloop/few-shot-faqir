import yaml

from src.inference.predictor_biencoder import BiEncoderModelPredictor


def predict(config):
    """
    Method to showcase tentant weight switching for sample tenants configured in the config. The respective tenant needs to be trained first so that the tenant specific weights are saved
    """
    predictor = BiEncoderModelPredictor(config)

    tenants = config["INFERENCE"]["TENANT_NAMES"]
    inference_file_path = config["INFERENCE"]["INFERENCE_FILE_PATH"]

    with open(inference_file_path, "r") as f:
        inference_texts = f.read().splitlines()
    for tenant_id in tenants:
        for text in inference_texts:
            print("-" * 50)
            print(f'Utterance is: "{text}"')
            topn_sents, topn_labels, topn_sims = predictor.predict(tenant_id, text)
            print(f"Top n sentences are: {topn_sents}")
            print(f"Top n labels are: {topn_labels}")
            print(f"Top n similarity scores are: {topn_sims}")


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    predict(config)
