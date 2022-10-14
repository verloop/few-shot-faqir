import csv
import glob
import itertools
import random

import pandas as pd
from scipy import spatial
from sentence_transformers import SentenceTransformer
from torch import embedding

from src.data.dataloaders import (  # isort:skip
    DialogueIntentDataLoader,
    HaptikDataLoader,
)

random.seed(9001)
SAMPLE_SIZE = 50000
SUB_SAMPLE_QQ = True

# change the model depending on the training requirements
model = SentenceTransformer(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")


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


def to_question_pairs(dataloader, data_path):  # no balancing
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
    sim_result = 1 - spatial.distance.cosine(text1_emb, text2_emb)
    if sim_result <= 0:
        sim_result = 0.1
    return sim_result


def to_question_pairs_sample(dataloader, data_path, sample_size=100000):
    # Will cause memory overflow for large dataset
    data = dataloader.dataset[:]
    texts = [each["Text"].lower() for each in data]
    embeddings = model.encode(texts)
    emb_dict = {i: j for i, j in zip(texts, embeddings)}
    partial_filename = data_path.split(".")[0]
    with open(f"{partial_filename}_question_pairs.csv", "w", newline="") as f_output:
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
                            emb_dict[q1["Text"].lower()], emb_dict[q2["Text"].lower()]
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
                            emb_dict[q1["Text"].lower()], emb_dict[q2["Text"].lower()]
                        ),
                    }
                )
    data = pd.read_csv(f"{partial_filename}_question_pairs.csv")
    pos_data = data[data.label == 1]
    neg_data = data[data.label == 0]
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
    sampled_data = pd.concat([sampled_pos, sampled_neg])
    sampled_data = sampled_data.sample(frac=1, random_state=1)
    sampled_data = sampled_data.drop(columns=["weights"])
    sampled_data.to_csv(f"{partial_filename}_question_pairs.csv", index=False)


def to_question_triplets_pretraing(
    dataloaders,
    dataset_names,
    data_path="data/pretrain",
    sample_size=30000,
    hard_sample=False,
):
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
        df = pd.read_csv(f"{data_path}/{dataset_names[i]}_question_pairs.csv")
        print(f"Reading from {data_path}/{dataset_names[i]}_question_pairs.csv")
        pos_data = df[df.label == 1]
        neg_data = df[df.label == 0]
        neg_data.to_csv("temp.csv", index=False)
        triplet_anchor, triplet_pos, triplet_neg = [], [], []
        n = max(int(sample_size / len(data)), 2)
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
        sampled_data.to_csv(
            f"{data_path}/{dataset_names[i]}_question_triplets.csv",
            index=False,
            header=False,
        )

    with open(f"data/pretraining_question_triplets.csv", "w", newline="") as f_output:
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
            for datasets in glob.glob(f"{data_path}/*_question_triplets.csv"):
                print(datasets)
                try:
                    data = pd.read_csv(
                        datasets, nrows=chunk_size, skiprows=skip_rows, header=None
                    )
                except:
                    continue
                data.columns = ["anchor", "positive", "negative"]
                if len(data_full) == 0:
                    data_full = data
                else:
                    data_full = pd.concat([data_full, data])
            total_read = total_read + chunk_size
            skip_rows = total_read
            data_full = data_full.sample(frac=1)
            data_full = data_full[["anchor", "positive", "negative"]]
            for row in data_full.itertuples(index=False):
                try:
                    csv_output.writerow(
                        {
                            "anchor": row.anchor,
                            "positive": row.positive,
                            "negative": row.negative,
                        }
                    )
                except:
                    continue


def to_question_pairs_pretraing(
    dataloaders,
    dataset_names,
    data_path="data/pretrain",
    sample_size=30000,
    hard_sample=False,
):
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
        sampled_data.to_csv(
            f"{data_path}/{dataset_names[i]}_question_pairs.csv",
            index=False,
            header=False,
        )

    with open(f"data/pretraining_question_pairs.csv", "w", newline="") as f_output:
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
            for datasets in glob.glob(f"{data_path}/*_question_pairs.csv"):
                print(datasets)
                try:
                    data = pd.read_csv(
                        datasets, nrows=chunk_size, skiprows=skip_rows, header=None
                    )
                except:
                    continue
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
                try:
                    csv_output.writerow(
                        {
                            "question1": row.question1,
                            "question2": row.question2,
                            "label": row.label,
                        }
                    )
                except:
                    continue


def generate_question_pairs_finetuning():
    # haptik
    dataset_names = ["curekart", "powerplay11", "sofmattress"]
    for train_subset in ["train", "subset_train"]:
        for dataset_name in dataset_names:
            # haptik train
            dl_train = HaptikDataLoader(
                dataset_name=dataset_name, data_type="train", data_subset=train_subset
            )
            train_dataloader, _ = dl_train.get_dataloader()
            if SUB_SAMPLE_QQ:
                to_question_pairs_sample(
                    train_dataloader,
                    data_path=dl_train.data_path,
                    sample_size=SAMPLE_SIZE,
                )
            else:
                to_question_pairs(train_dataloader, data_path=dl_train.data_path)
            # haptik test
            dl_test = HaptikDataLoader(
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
            if SUB_SAMPLE_QQ:
                to_question_pairs_sample(
                    train_dataloader,
                    data_path=dl_train.data_path,
                    sample_size=SAMPLE_SIZE,
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


def generate_data_pretraining():
    haptik_dataset_names = ["curekart", "powerplay11", "sofmattress"]
    dialoglue_dataset_names = ["banking", "clinc", "hwu"]
    dataloaders = []
    for dataset_name in haptik_dataset_names:
        dl_train = HaptikDataLoader(
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
    # to_question_pairs_pretraing(
    #     dataloaders,
    #     haptik_dataset_names + dialoglue_dataset_names,
    #     sample_size=100000,
    #     hard_sample=True,
    # )
    to_question_triplets_pretraing(
        dataloaders,
        haptik_dataset_names + dialoglue_dataset_names,
        sample_size=100000,
        hard_sample=True,
    )


if __name__ == "__main__":
    # generate_question_pairs_finetuning()
    generate_data_pretraining()
