import csv
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
SIMILARITY_THRESHOLD = 0.5
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


def to_question_pairs_sample(dataloader, data_path, sample_size=50000):
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
        f_output.close()
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
            n=min(int(sample_size / 2), len(neg_data)),
            weights=neg_data["weights"],
            random_state=1,
        )
        sampled_data = pd.concat([sampled_pos, sampled_neg])
        sampled_data = sampled_data.sample(frac=1, random_state=1)
        sampled_data = sampled_data.drop(columns=["weights"])
        sampled_data.to_csv(f"{partial_filename}_question_pairs.csv", index=False)


# def to_question_pairs_hard_sample(dataloader, data_path,pos_samples_per_label): # balanced
#     # Balanced approach deteriorated results because of having too few samples of a sentence, if the number of samples per intent was low
#     # Will cause memory overflow for large dataset
#     data = dataloader.dataset[:]
#     texts = [each["Text"] for each in data]
#     labels = [each["Label"] for each in data]
#     df_data = pd.DataFrame({"Text": texts, "Label": labels})
#     partial_filename = data_path.split(".")[0]
#     with open(f"{partial_filename}_question_pairs.csv", "w", newline="") as f_output:
#         csv_output = csv.DictWriter(
#             f_output,
#             fieldnames=["question1", "question2", "label","weight"],
#             delimiter=",",
#         )
#         csv_output.writeheader()
#         labels = df_data["Label"].unique()
#         for label in labels:
#             print(label)
#             df_data_label = df_data[df_data["Label"] == label]
#             data_label = [
#                 {"Text": x[1][0], "Label": x[1][1]} for x in df_data_label.iterrows()
#             ]
#             pos = 0
#             q1_subset,q2_subset,label_subset,weights = [],[],[],[]
#             for q1, q2 in itertools.combinations(data_label, 2):
#                 if q1["Label"] == q2["Label"]:
#                     q1_subset.append(q1["Text"])
#                     q2_subset.append(q2["Text"])
#                     label_subset.append(1)
#                     weights.append(1-get_sim(q1["Text"],q2["Text"]))
#                     pos = pos + 1
#             neg = 0
#             for q1, q2 in itertools.product(data_label, random.sample(data,len(data_label))):
#                 if q1["Label"] != q2["Label"]:
#                     q1_subset.append(q1["Text"])
#                     q2_subset.append(q2["Text"])
#                     label_subset.append(0)
#                     weights.append(get_sim(q1["Text"],q2["Text"]))
#                     neg += 1
#             label_data = pd.DataFrame({"q1_subset":q1_subset,"q2_subset":q2_subset,"label_subset":label_subset,"weights":weights})
#             label_data.to_csv(f"{partial_filename}_temp_question_pairs.csv")
#             pos_data = label_data[label_data.label_subset==1]
#             neg_data = label_data[label_data.label_subset==0]
#             sampled_pos = pos_data.sample(n=min(pos_samples_per_label,len(pos_data)),weights=pos_data["weights"])
#             sampled_neg = neg_data.sample(n=min(pos_samples_per_label,len(neg_data)),weights=neg_data["weights"])
#             sampled_data = pd.concat([sampled_pos,sampled_neg])
#             sampled_data = sampled_data.sample(frac=1)
#             for q1,q2,label,weight in zip(sampled_data['q1_subset'],sampled_data['q2_subset'],sampled_data['label_subset'],sampled_data['weights']):
#                 csv_output.writerow(
#                     {
#                         "question1": q1,
#                         "question2": q2,
#                         "label": label,
#                         "weight":weight
#                     }
#                 )
#         f_output.close()


# def to_question_pairs_hard(dataloader, data_path): # balanced
#     # Balanced approach deteriorated results because of having too few samples of a sentence, if the number of samples per intent was low
#     # Will cause memory overflow for large dataset
#     data = dataloader.dataset[:]
#     texts = [each["Text"] for each in data]
#     labels = [each["Label"] for each in data]
#     df_data = pd.DataFrame({"Text": texts, "Label": labels})
#     partial_filename = data_path.split(".")[0]
#     with open(f"{partial_filename}_question_pairs.csv", "w", newline="") as f_output:
#         csv_output = csv.DictWriter(
#             f_output,
#             fieldnames=["question1", "question2", "label"],
#             delimiter=",",
#         )
#         csv_output.writeheader()
#         labels = df_data["Label"].unique()
#         for label in labels:
#             print(label)
#             df_data_label = df_data[df_data["Label"] == label]
#             data_label = [
#                 {"Text": x[1][0], "Label": x[1][1]} for x in df_data_label.iterrows()
#             ]
#             pos = 0
#             q1_subset,q2_subset,label_subset = [],[],[]
#             for q1, q2 in itertools.combinations(data_label, 2):
#                 if q1["Label"] == q2["Label"] and get_sim(q1["Text"],q2["Text"]) < SIMILARITY_THRESHOLD:
#                     q1_subset.append(q1["Text"])
#                     q2_subset.append(q2["Text"])
#                     label_subset.append(1)
#                     pos = pos + 1
#             neg = 0
#             for q1, q2 in itertools.product(data_label, random.sample(data,len(data_label))):
#                 if q1["Label"] != q2["Label"] and get_sim(q1["Text"],q2["Text"]) > SIMILARITY_THRESHOLD:
#                     q1_subset.append(q1["Text"])
#                     q2_subset.append(q2["Text"])
#                     label_subset.append(0)
#                     neg += 1
#             # Balancing
#             diff_sample = []
#             if neg > pos:
#                 diff = neg - pos
#                 q1_diff_subset,q2_diff_subset,label_diff_subset = [],[],[]
#                 for q1, q2 in itertools.combinations(data_label, 2):
#                     if q1["Label"] == q2["Label"] and ((q1["Text"],q2["Text"]) not in list(zip(q1_subset,q2_subset))) :
#                         q1_diff_subset.append(q1["Text"])
#                         q2_diff_subset.append(q2["Text"])
#                         label_diff_subset.append(1)
#                 diff_sample = random.sample(list(zip(q1_diff_subset,q2_diff_subset,label_diff_subset)),min(diff,len(label_diff_subset)))
#                 print("neg > pos")
#                 print(diff)
#                 print(len(label_diff_subset))
#                 print(len(diff_sample))
#             if pos > neg:
#                 diff = pos - neg
#                 q1_diff_subset,q2_diff_subset,label_diff_subset = [],[],[]
#                 for q1, q2 in itertools.product(data_label, random.sample(data,len(data_label))):
#                     if q1["Label"] != q2["Label"] and ((q1["Text"],q2["Text"]) not in list(zip(q1_subset,q2_subset))) :
#                         q1_diff_subset.append(q1["Text"])
#                         q2_diff_subset.append(q2["Text"])
#                         label_diff_subset.append(0)
#                 diff_sample = random.sample(list(zip(q1_diff_subset,q2_diff_subset,label_diff_subset)),min(diff,len(label_diff_subset)))
#                 print("pos > neg")
#                 print(diff)
#                 print(len(label_diff_subset))
#                 print(len(diff_sample))
#             shuffled_list = list(zip(q1_subset,q2_subset,label_subset)) + diff_sample
#             random.shuffle(shuffled_list)
#             for q1,q2,label in shuffled_list:
#                 csv_output.writerow(
#                     {
#                         "question1": q1,
#                         "question2": q2,
#                         "label": label,
#                     }
#                 )
#         f_output.close()


if __name__ == "__main__":

    # haptik
    dataset_names = ["curekart", "powerplay11", "sofmattress"]
    for train_subset in ["train", "subset_train"]:
        for dataset_name in dataset_names:
            # haptik train
            dl_train = HaptikDataLoader(
                dataset_name=dataset_name, data_type="train", data_subset=train_subset
            )
            train_dataloader, _ = dl_train.get_dataloader()
            to_question_pairs_sample(train_dataloader, data_path=dl_train.data_path)
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
            to_question_pairs_sample(train_dataloader, data_path=dl_train.data_path)
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
