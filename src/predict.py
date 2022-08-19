import yaml

from src.inference.predictor_biencoder import BiEncoderModelPredictor


def predict(config):
    predictor = BiEncoderModelPredictor(config)

    clients = config["INFERENCE"]["CLIENT_NAMES"]
    inference_file_path = config["INFERENCE"]["INFERENCE_FILE_PATH"]

    with open(inference_file_path, "r") as f:
        inference_texts = f.read().splitlines()
    for client_id in clients:
        for text in inference_texts:
            print("-" * 50)
            print(f'Utterance is: "{text}"')
            topn_sents, topn_labels, topn_sims = predictor.predict(client_id, text)
            print(f"Top n sentences are: {topn_sents}")
            print(f"Top n labels are: {topn_labels}")
            print(f"Top n similarity scores are: {topn_sims}")


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    predict(config)
