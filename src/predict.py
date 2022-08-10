import yaml

from src.inference.predictor_biencoder import BiEncoderModelPredictor


def predict(config):
    predictor = BiEncoderModelPredictor(config)

    clients = config["INFERENCE"]["CLIENT_NAMES"]
    for client_id in clients:
        print(predictor.predict(client_id))


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    predict(config)
