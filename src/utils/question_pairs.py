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


def test_question_pairs(train_dataloader, test_dataloader, data_path, data_subset):
    train_data = train_dataloader.dataset[:]
    test_data = test_dataloader.dataset[:]
    _ = [test_data[i].update({"idx": i}) for i in range(len(test_data))]
    partial_filename = data_path.split(".")[0]
    with open(
        f"{partial_filename}_{data_subset}_question_pairs.csv", "w", newline=""
    ) as f_output:
        csv_output = csv.DictWriter(
            f_output,
            fieldnames=["question1", "question2", "idx", "label"],
            delimiter=",",
        )
        csv_output.writeheader()
        for q_test, q_train in list(itertools.product(test_data, train_data)):
            if q_test["Label"] == q_train["Label"]:
                csv_output.writerow(
                    {
                        "question1": q_test["Text"],
                        "question2": q_train["Text"],
                        "idx": q_test["idx"],
                        "label": 1,
                    }
                )
            else:
                csv_output.writerow(
                    {
                        "question1": q_test["Text"],
                        "question2": q_train["Text"],
                        "idx": q_test["idx"],
                        "label": 0,
                    }
                )

        f_output.close()


def to_question_pairs(dataloader, data_path):
    """
    Creates question pairs and saves it as a csv file in the same folder as data path.
    Questions for same label are paired and given a label of 1
    Questions for different labels are paired and given a label of 0
    """
    # Will cause memory overflow for large dataset
    data = dataloader.dataset[:]
    partial_filename = data_path.split(".")[0]
    with open(f"{partial_filename}_question_pairs.csv", "w", newline="") as f_output:
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
        f_output.close()


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


def to_question_pairs_sample(
    model, dataloader, data_path, hard_sample=True, sample_size=100000
):
    """
    Creates question pairs by sampling based on sample size. If hard_sample is True, uses the model to generate hard samples.
    Arguments
    -----------
    model: Model to be used for hard sampling
    dataloader: dataloader for fetching data
    data_path: Path to the train dataset file. Same path is used for saving the Question pairs
    hard_sample: Whether or not hard sampling should be done
    sample_size: sample size needed

    """
    # Will cause memory overflow for large dataset
    data = dataloader.dataset[:]
    texts = [each["Text"].lower() for each in data]
    embeddings = model.encode(texts)
    emb_dict = {i: j for i, j in zip(texts, embeddings)}
    partial_filename = data_path.split(".")[0]
    if hard_sample:
        with open(
            f"{partial_filename}_question_pairs.csv", "w", newline=""
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
            f"{partial_filename}_question_pairs.csv", "w", newline=""
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
                        {"question1": q1["Text"], "question2": q2["Text"], "label": 1}
                    )
                else:
                    csv_output.writerow(
                        {"question1": q1["Text"], "question2": q2["Text"], "label": 0}
                    )
    data = pd.read_csv(f"{partial_filename}_question_pairs.csv")
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
    sampled_data.to_csv(f"{partial_filename}_question_pairs.csv", index=False)


def to_question_triplets(
    model, dataloader, data_path, sample_size=50000, hard_sample=False, val_split=0
):
    """
    Creates question triplets by sampling based on sample size. If hard_sample is True, uses the model to generate hard samples.

    Arguments
    -----------
    model: Model to be used for hard sampling
    dataloader: dataloader for fetching data
    data_path: Path to the train dataset file. Same path is used for saving the Question pairs
    hard_sample: Whether or not hard sampling should be done
    sample_size: sample size needed
    val_split: validation split if needed to create separate validation files. Used for Sentence Bert triplet evaluation
    """
    # Will cause memory overflow for large dataset
    data = dataloader.dataset[:]
    partial_filename = data_path.split(".")[0]
    texts = [each["Text"].lower() for each in data]
    if hard_sample:
        embeddings = model.encode(texts)
        emb_dict = {i: j for i, j in zip(texts, embeddings)}
        with open(
            f"{partial_filename}_question_pairs_temp.csv", "w", newline=""
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
            f"{partial_filename}_question_pairs_temp.csv", "w", newline=""
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
    df = pd.read_csv(f"{partial_filename}_question_pairs_temp.csv")
    print(f"Reading from {partial_filename}_question_pairs_temp.csv")
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
            f"{partial_filename}_val_question_triplets.csv", index=False
        )
    sampled_data.to_csv(
        f"{partial_filename}_train_question_triplets.csv",
        index=False,
    )


def generate_data_finetuning(
    model,
    gen_triplets=True,
    sample_size=100000,
    hard_sample=True,
    val_split=0.2,
    sub_sample_qq=False,
):
    """
    Wrapper to generate data for finetuning - question pairs/triplets. The training data is stored in the same folder as source data

    Arguments
    ----------
    model: Model to be used for hard sampling
    gen_triplets: Whether to generate triplets or not
    hard_sample: Whether or not hard sampling should be done
    sample_size: sample size needed
    val_split: validation split if needed to create separate validation files. Used for Sentence Bert triplet evaluation
    sub_sample_qq: Whether to sample True/False. If this is False, none of the other sampling parameters are used

    """
    dataset_names = ["curekart", "powerplay11", "sofmattress"]
    for train_subset in ["train", "subset_train"]:
        for dataset_name in dataset_names:
            # Hint3 train
            dl_train = HintDataLoader(
                dataset_name=dataset_name, data_type="train", data_subset=train_subset
            )
            train_dataloader, _ = dl_train.get_dataloader()
            if sub_sample_qq:
                to_question_pairs_sample(
                    model,
                    train_dataloader,
                    data_path=dl_train.data_path,
                    hard_sample=hard_sample,
                    sample_size=sample_size,
                )
                if gen_triplets:
                    to_question_triplets(
                        model,
                        train_dataloader,
                        data_path=dl_train.data_path,
                        sample_size=int(sample_size / (1 - val_split)),
                        hard_sample=hard_sample,
                        val_split=val_split,
                    )
            else:
                to_question_pairs(train_dataloader, data_path=dl_train.data_path)
            # Hint3 test
            dl_test = HintDataLoader(
                dataset_name=dataset_name,
                data_type="test",
                data_subset="test",
                intent_label_to_idx=train_dataloader.dataset.intent_label_to_idx,
            )
            test_dataloader, _ = dl_test.get_dataloader()
            test_question_pairs(
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                data_path=dl_test.data_path,
                data_subset=train_subset,
            )

    # dialogue intent
    dataset_names = ["banking", "clinc", "hwu"]
    for train_subset in ["train_5", "train_10"]:
        for dataset_name in dataset_names:
            # dialogue intent train
            dl_train = DialogueIntentDataLoader(
                dataset_name=dataset_name, data_type="train", data_subset=train_subset
            )
            train_dataloader, _ = dl_train.get_dataloader()
            if sub_sample_qq:
                to_question_pairs_sample(
                    model,
                    train_dataloader,
                    data_path=dl_train.data_path,
                    hard_sample=hard_sample,
                    sample_size=sample_size,
                )
                if gen_triplets:
                    to_question_triplets(
                        model,
                        train_dataloader,
                        data_path=dl_train.data_path,
                        sample_size=int(sample_size / (1 - val_split)),
                        hard_sample=hard_sample,
                        val_split=val_split,
                    )
            else:
                to_question_pairs(train_dataloader, data_path=dl_train.data_path)
            # dialogue intent test
            dl_test = DialogueIntentDataLoader(
                dataset_name=dataset_name, data_type="test", data_subset="test"
            )
            test_dataloader, _ = dl_test.get_dataloader()
            test_question_pairs(
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                data_path=dl_test.data_path,
                data_subset=train_subset,
            )
            # dialogue intent val
            dl_val = DialogueIntentDataLoader(
                dataset_name=dataset_name, data_type="val", data_subset="val"
            )
            val_dataloader, _ = dl_val.get_dataloader()
            test_question_pairs(
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                data_path=dl_val.data_path,
                data_subset=train_subset,
            )


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    model = SentenceTransformer(model_name_or_path=config["TRAINING"]["MODEL_NAME"])
    sample_size = config["TRAINING"]["SAMPLE_SIZE_PER_DATASET"]
    val_split = config["TRAINING"]["VAL_SPLIT"]
    hard_sample = config["TRAINING"]["HARD_SAMPLE"]
    gen_triplets = config["TRAINING"]["GENERATE_TRIPLETS"]
    sub_sample_qq = config["TRAINING"]["SUB_SAMPLE_QQ"]
    generate_data_finetuning(
        model,
        sample_size=sample_size,
        gen_triplets=gen_triplets,
        val_split=val_split,
        hard_sample=hard_sample,
        sub_sample_qq=sub_sample_qq,
    )
