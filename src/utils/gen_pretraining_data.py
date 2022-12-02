import csv
import glob
import itertools
import math
import random

import pandas as pd
import yaml
from scipy import spatial
from sentence_transformers import SentenceTransformer

from src.data.dataloaders import (  # isort:skip
    DialogueIntentDataLoader,
    HintDataLoader,
)

random.seed(9001)


def get_sim(text1_emb, text2_emb):
    """
    Computes cosine similarity between two embeddings.Thresholds it to 0.1 to remove negatives.
    Arguments
    ----------
    text1_emb: Embedding vector one
    text2_emb: Embedding vector two

    Returns
    --------
    Similarity score
    """
    sim_result = 1 - spatial.distance.cosine(text1_emb, text2_emb)
    sim_result = max(sim_result, 0.1)
    return sim_result


def to_question_triplets_pretraing(
    dataloaders,
    dataset_names,
    model,
    data_path="data/pretrain",
    sample_size=30000,
    hard_sample=False,
    val_split=0,
):
    """
    For the list of dataloaders, creates triplets within each dataset for the required sample_size.
    Arguments
    ----------
    dataloaders: List of dataloaders
    dataset_names: List of dataset names corresponding to the dataloaders
    model: Model to be used for creating similarity scores between question pairs, which is then used for hard sampling
    data_path: Path where the training data samples will be stored
    sample_size: Required sample size per dataset
    hard_sample: Whether to hard sample or not
    val_split: If greater than 0, creates separate validation samples as well
    """
    # Will cause memory overflow for large dataset
    for i, dataloader in enumerate(dataloaders):
        data = dataloader.dataset[:]
        texts = [each["Text"].lower() for each in data]
        if hard_sample:
            embeddings = model.encode(texts)
            emb_dict = {i: j for i, j in zip(texts, embeddings)}
            with open(
                f"{data_path}/{dataset_names[i]}_question_pairs_temp.csv",
                "w",
                newline="",
            ) as f_output:
                csv_output = csv.DictWriter(
                    f_output,
                    fieldnames=["question1", "question2", "label", "weights"],
                    delimiter=",",
                )
                csv_output.writeheader()

                for q1, q2 in itertools.combinations(data, 2):
                    if q1["Label"] == q2["Label"]:
                        csv_output.writerow(
                            {
                                "question1": q1["Text"],
                                "question2": q2["Text"],
                                "label": 1,
                                "weights": 1
                                - get_sim(
                                    emb_dict[q1["Text"].lower()],
                                    emb_dict[q2["Text"].lower()],
                                ),
                            }
                        )
                    else:
                        csv_output.writerow(
                            {
                                "question1": q1["Text"],
                                "question2": q2["Text"],
                                "label": 0,
                                "weights": get_sim(
                                    emb_dict[q1["Text"].lower()],
                                    emb_dict[q2["Text"].lower()],
                                ),
                            }
                        )
        else:
            with open(
                f"{data_path}/{dataset_names[i]}_question_pairs_temp.csv",
                "w",
                newline="",
            ) as f_output:
                csv_output = csv.DictWriter(
                    f_output,
                    fieldnames=["question1", "question2", "label"],
                    delimiter=",",
                )
                csv_output.writeheader()

                for q1, q2 in itertools.combinations(data, 2):
                    if q1["Label"] == q2["Label"]:
                        csv_output.writerow(
                            {
                                "question1": q1["Text"],
                                "question2": q2["Text"],
                                "label": 1,
                            }
                        )
                    else:
                        csv_output.writerow(
                            {
                                "question1": q1["Text"],
                                "question2": q2["Text"],
                                "label": 0,
                            }
                        )
        df = pd.read_csv(f"{data_path}/{dataset_names[i]}_question_pairs_temp.csv")
        print(f"Reading from {data_path}/{dataset_names[i]}_question_pairs_temp.csv")
        pos_data = df[df.label == 1]
        neg_data = df[df.label == 0]
        triplet_anchor, triplet_pos, triplet_neg = [], [], []
        n = max(math.ceil(sample_size / len(data)), 2)
        for row in data:
            row_pos_1 = pos_data[pos_data.question1 == row["Text"]]
            row_pos_1 = row_pos_1.rename(columns={"question2": "question"})
            row_pos_1 = row_pos_1.drop(columns=["question1"])
            row_pos_2 = pos_data[pos_data.question2 == row["Text"]]
            row_pos_2 = row_pos_2.rename(columns={"question1": "question"})
            row_pos_2 = row_pos_2.drop(columns=["question2"])
            row_pos = pd.concat([row_pos_1, row_pos_2])
            row_negs_1 = neg_data[neg_data.question1 == row["Text"]]
            row_negs_1 = row_negs_1.rename(columns={"question2": "question"})
            row_negs_1 = row_negs_1.drop(columns=["question1"])
            row_negs_2 = neg_data[neg_data.question2 == row["Text"]]
            row_negs_2 = row_negs_2.rename(columns={"question1": "question"})
            row_negs_2 = row_negs_2.drop(columns=["question2"])
            row_negs = pd.concat([row_negs_1, row_negs_2])
            if hard_sample:
                if len(row_pos) != 0:
                    row_pos = row_pos.sample(
                        n=n, replace=True, weights=row_pos["weights"], random_state=1
                    )
                else:
                    row_pos = pd.DataFrame({"question": [row["Text"]] * 10})
                row_negs = row_negs.sample(
                    n=n, replace=True, weights=row_negs["weights"], random_state=1
                )
            else:
                if len(row_pos) != 0:
                    row_pos = row_pos.sample(n=n, replace=True, random_state=1)
                else:
                    row_pos = pd.DataFrame({"question": [row["Text"]] * 10})
                row_negs = row_negs.sample(n=n, random_state=1)
            positives = list(row_pos["question"])
            negatives = list(row_negs["question"])
            for pos, neg in zip(positives, negatives):
                triplet_anchor.append(row["Text"])
                triplet_pos.append(pos)
                triplet_neg.append(neg)

        sampled_data = pd.DataFrame(
            {"anchor": triplet_anchor, "positive": triplet_pos, "negative": triplet_neg}
        )
        n = min(sample_size, len(sampled_data))
        sampled_data = sampled_data.sample(n=n, random_state=1)
        if val_split > 0:
            train_samples = int(len(sampled_data) * (1 - val_split))
            sampled_data_val = sampled_data.iloc[train_samples:, :]
            sampled_data = sampled_data.iloc[:train_samples, :]
            sampled_data_val.to_csv(
                f"{data_path}/{dataset_names[i]}_val_question_triplets.csv",
                index=False,
            )

        sampled_data.to_csv(
            f"{data_path}/{dataset_names[i]}_train_question_triplets.csv",
            index=False,
        )
    write_pretraining_triplets(data_path, sample_size, data_set="train")
    if val_split > 0:
        write_pretraining_triplets(data_path, sample_size, data_set="val")


def write_pretraining_triplets(data_path, sample_size, data_set):
    """
    Helper method to read triplet data for each dataset, shuffle it and write it back
    """
    with open(
        f"data/pretraining_{data_set}_question_triplets.csv", "w", newline=""
    ) as f_output:
        csv_output = csv.DictWriter(
            f_output,
            fieldnames=["anchor", "positive", "negative"],
            delimiter=",",
        )
        csv_output.writeheader()

        chunk_size = 10000
        skip_rows = 0
        total_read = 0
        while total_read < sample_size:
            data_full = pd.DataFrame()
            for datasets in glob.glob(
                f"{data_path}/*_{data_set}_question_triplets.csv"
            ):
                # try:
                data = pd.read_csv(
                    datasets, nrows=chunk_size, skiprows=skip_rows, header=None
                )
                # except:
                #     continue
                if data.empty:
                    continue
                data.columns = ["anchor", "positive", "negative"]
                if len(data_full) == 0:
                    data_full = data
                else:
                    data_full = pd.concat([data_full, data])
            total_read = total_read + chunk_size
            skip_rows = total_read
            if len(data_full) == 0:
                continue
            data_full = data_full.sample(frac=1)
            data_full = data_full[["anchor", "positive", "negative"]]
            for row in data_full.itertuples(index=False):
                csv_output.writerow(
                    {
                        "anchor": row.anchor,
                        "positive": row.positive,
                        "negative": row.negative,
                    }
                )
    print(
        f"Pretraining data created in data/pretraining_{data_set}_question_triplets.csv"
    )


def to_question_pairs_pretraining(
    dataloaders,
    dataset_names,
    model,
    data_path="data/pretrain",
    sample_size=30000,
    hard_sample=False,
    val_split=0,
):
    """
    For the list of dataloaders, creates Question pairs within each dataset for the required sample_size.
    Arguments
    ----------
    dataloaders: List of dataloaders
    dataset_names: List of dataset names corresponding to the dataloaders
    model: Model to be used for creating similarity scores between question pairs, which is then used for hard sampling
    data_path: Path where the training data samples will be stored
    sample_size: Required sample size per dataset
    hard_sample: Whether to hard sample or not
    val_split: If greater than 0, creates separate validation samples as well
    """
    # Will cause memory overflow for large dataset
    for i, dataloader in enumerate(dataloaders):
        data = dataloader.dataset[:]
        texts = [each["Text"].lower() for each in data]
        if hard_sample:
            embeddings = model.encode(texts)
            emb_dict = {i: j for i, j in zip(texts, embeddings)}
            with open(
                f"{data_path}/{dataset_names[i]}_question_pairs.csv", "w", newline=""
            ) as f_output:
                csv_output = csv.DictWriter(
                    f_output,
                    fieldnames=["question1", "question2", "label", "weights"],
                    delimiter=",",
                )
                csv_output.writeheader()

                for q1, q2 in itertools.combinations(data, 2):
                    if q1["Label"] == q2["Label"]:
                        csv_output.writerow(
                            {
                                "question1": q1["Text"],
                                "question2": q2["Text"],
                                "label": 1,
                                "weights": 1
                                - get_sim(
                                    emb_dict[q1["Text"].lower()],
                                    emb_dict[q2["Text"].lower()],
                                ),
                            }
                        )
                    else:
                        csv_output.writerow(
                            {
                                "question1": q1["Text"],
                                "question2": q2["Text"],
                                "label": 0,
                                "weights": get_sim(
                                    emb_dict[q1["Text"].lower()],
                                    emb_dict[q2["Text"].lower()],
                                ),
                            }
                        )
        else:
            with open(
                f"{data_path}/{dataset_names[i]}_question_pairs.csv", "w", newline=""
            ) as f_output:
                csv_output = csv.DictWriter(
                    f_output,
                    fieldnames=["question1", "question2", "label"],
                    delimiter=",",
                )
                csv_output.writeheader()

                for q1, q2 in itertools.combinations(data, 2):
                    if q1["Label"] == q2["Label"]:
                        csv_output.writerow(
                            {
                                "question1": q1["Text"],
                                "question2": q2["Text"],
                                "label": 1,
                            }
                        )
                    else:
                        csv_output.writerow(
                            {
                                "question1": q1["Text"],
                                "question2": q2["Text"],
                                "label": 0,
                            }
                        )
        data = pd.read_csv(f"{data_path}/{dataset_names[i]}_question_pairs.csv")
        pos_data = data[data.label == 1]
        neg_data = data[data.label == 0]
        if hard_sample:
            sampled_pos = pos_data.sample(
                n=int(sample_size / 2),
                replace=True,
                weights=pos_data["weights"],
                random_state=1,
            )
            sampled_neg = neg_data.sample(
                n=int(sample_size / 2),
                replace=True,
                weights=neg_data["weights"],
                random_state=1,
            )
        else:
            sampled_pos = pos_data.sample(
                n=int(sample_size / 2),
                replace=True,
                random_state=1,
            )
            sampled_neg = neg_data.sample(
                n=int(sample_size / 2),
                replace=True,
                random_state=1,
            )
        sampled_data = pd.concat([sampled_pos, sampled_neg])
        sampled_data = sampled_data.sample(frac=1, random_state=1)
        if hard_sample:
            sampled_data = sampled_data.drop(columns=["weights"])
        if val_split > 0:
            train_samples = int(len(sampled_data) * (1 - val_split))
            sampled_data_val = sampled_data.iloc[train_samples:, :]
            sampled_data = sampled_data.iloc[:train_samples, :]
            sampled_data_val.to_csv(
                f"{data_path}/{dataset_names[i]}_val_question_pairs.csv",
                index=False,
                header=False,
            )
        sampled_data.to_csv(
            f"{data_path}/{dataset_names[i]}_train_question_pairs.csv",
            index=False,
            header=False,
        )
    write_pretraining_question_pairs(data_path, sample_size, data_set="train")
    if val_split > 0:
        write_pretraining_question_pairs(data_path, sample_size, data_set="val")


def write_pretraining_question_pairs(data_path, sample_size, data_set):
    """
    Helper method to read question pair data for each dataset, shuffle it and write it back
    """
    with open(
        f"data/pretraining_{data_set}_question_pairs.csv", "w", newline=""
    ) as f_output:
        csv_output = csv.DictWriter(
            f_output,
            fieldnames=["question1", "question2", "label"],
            delimiter=",",
        )
        csv_output.writeheader()

        chunk_size = 10000
        skip_rows = 0
        total_read = 0
        while total_read < sample_size:
            data_full = pd.DataFrame()
            for datasets in glob.glob(f"{data_path}/*{data_set}_question_pairs.csv"):
                # try:
                data = pd.read_csv(
                    datasets, nrows=chunk_size, skiprows=skip_rows, header=None
                )
                # except:
                #     continue
                data.columns = ["question1", "question2", "label"]
                if len(data_full) == 0:
                    data_full = data
                else:
                    data_full = pd.concat([data_full, data])
            total_read = total_read + chunk_size
            skip_rows = total_read
            data_full = data_full.sample(frac=1)
            data_full = data_full[["question1", "question2", "label"]]
            for row in data_full.itertuples(index=False):
                csv_output.writerow(
                    {
                        "question1": row.question1,
                        "question2": row.question2,
                        "label": row.label,
                    }
                )
    print(f"Pretraining data created in data/pretraining_{data_set}_question_pairs.csv")


def generate_data_pretraining(
    model,
    gen_triplets=True,
    gen_pairs=False,
    hard_sample=True,
    sample_size=100000,
    val_split=0.1,
    data_path="data/pretrain",
):
    """
    Wrapper function which creates Question pairs / Triplets for pre-training for the required config parameters.
    Arguments
    ----------
    model: Model to be used for creating similarity scores between question pairs, which is then used for hard sampling
    data_path: Path where the intermediate training data samples will be stored. The final pretraining data is stored under "data" folder.
    sample_size: Required sample size per dataset
    hard_sample: Whether to hard sample or not
    val_split: If greater than 0, creates separate validation samples as well
    """
    hint_dataset_names = ["curekart", "powerplay11", "sofmattress"]
    dialoglue_dataset_names = ["banking", "clinc", "hwu"]
    dataloaders = []
    for dataset_name in hint_dataset_names:
        dl_train = HintDataLoader(
            dataset_name=dataset_name, data_type="train", data_subset="train"
        )
        train_dataloader, _ = dl_train.get_dataloader()
        dataloaders.append(train_dataloader)
    for dataset_name in dialoglue_dataset_names:
        dl_train = DialogueIntentDataLoader(
            dataset_name=dataset_name, data_type="train", data_subset="train_10"
        )
        train_dataloader, _ = dl_train.get_dataloader()
        dataloaders.append(train_dataloader)
    if gen_pairs:
        to_question_pairs_pretraining(
            dataloaders,
            hint_dataset_names + dialoglue_dataset_names,
            model,
            sample_size=int(sample_size / (1 - val_split)),
            hard_sample=hard_sample,
            val_split=val_split,
            data_path=data_path,
        )
    if gen_triplets:
        to_question_triplets_pretraing(
            dataloaders,
            hint_dataset_names + dialoglue_dataset_names,
            model,
            sample_size=int(sample_size / (1 - val_split)),
            hard_sample=hard_sample,
            val_split=val_split,
            data_path=data_path,
        )


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    # change the model depending on the training requirements
    model = SentenceTransformer(model_name_or_path=config["PRETRAINING"]["MODEL_NAME"])
    sample_size = config["PRETRAINING"]["SAMPLE_SIZE_PER_DATASET"]
    val_split = config["PRETRAINING"]["VAL_SPLIT"]
    hard_sample = config["PRETRAINING"]["HARD_SAMPLE"]
    gen_triplets = config["PRETRAINING"]["GENERATE_TRIPLETS"]
    gen_pairs = config["PRETRAINING"]["GENERATE_PAIRS"]
    data_path = config["PRETRAINING"]["PRETRAIN_DATA_PATH"]
    generate_data_pretraining(
        model,
        gen_triplets=True,
        gen_pairs=False,
        sample_size=sample_size,
        val_split=val_split,
        data_path=data_path,
        hard_sample=hard_sample,
    )
