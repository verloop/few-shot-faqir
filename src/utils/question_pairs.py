import csv
import itertools

from src.data.dataloaders import (  # isort:skip
    DialogueIntentDataLoader,
    DialogueTopDataLoader,
    HaptikDataLoader,
)


def test_question_pairs(train_dataloader, test_dataloader, data_path):
    train_data = train_dataloader.dataset[:]
    test_data = test_dataloader.dataset[:]
    _ = [test_data[i].update({"idx": i}) for i in range(len(test_data))]
    partial_filename = data_path.split(".")[0]
    with open(f"{partial_filename}_question_pairs.csv", "w", newline="") as f_output:
        csv_output = csv.DictWriter(
            f_output,
            fieldnames=["question1", "question2", "idx", "label"],
            delimiter=",",
        )
        csv_output.writeheader()
        for q_test, q_train in list(itertools.product(test_data, train_data))[:10000]:
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


if __name__ == "__main__":

    # haptik
    dataset_names = ["curekart", "powerplay11", "sofmattress"]
    for dataset_name in dataset_names:
        # haptik train
        dl_instance = HaptikDataLoader(dataset_name=dataset_name, data_type="train")
        train_dataloader = dl_instance.get_dataloader()
        # to_question_pairs(train_dataloader, data_path=dl_instance.data_path)
        # haptik test
        dl_instance = HaptikDataLoader(
            dataset_name=dataset_name,
            data_type="test",
            intent_label_to_idx=train_dataloader.dataset.intent_label_to_idx,
        )
        test_dataloader = dl_instance.get_dataloader()
        test_question_pairs(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            data_path=dl_instance.data_path,
        )

    # # dialogue intent
    # dataset_names = ["banking", "clinc", "hwu"]
    # for dataset_name in dataset_names:
    #     # dialogue intent train
    #     dl_instance = DialogueIntentDataLoader(
    #         dataset_name=dataset_name, data_type="train"
    #     )
    #     train_dataloader = dl_instance.get_dataloader()
    #     to_question_pairs(train_dataloader, data_path=dl_instance.data_path)
    #     # dialogue intent test
    #     dl_instance = DialogueIntentDataLoader(
    #         dataset_name=dataset_name, data_type="test"
    #     )
    #     test_dataloader = dl_instance.get_dataloader()
    #     test_question_pairs(train_dataloader=train_dataloader,test_dataloader=test_dataloader,data_path=dl_instance.data_path)
    #     # dialogue intent val
    #     dl_instance = DialogueIntentDataLoader(
    #         dataset_name=dataset_name, data_type="val"
    #     )
    #     val_dataloader = dl_instance.get_dataloader()
    #     test_question_pairs(train_dataloader=train_dataloader,test_dataloader=val_dataloader,data_path=dl_instance.data_path)

    # dialogue top
    # dl_instance = DialogueTopDataLoader(dataset_name="top", data_type="train")
    # dataloader = dl_instance.get_dataloader()
    # to_question_pairs(dataloader, data_path=dl_instance.data_path)
    # dl_instance = DialogueTopDataLoader(dataset_name="top", data_type="test")
    # dataloader = dl_instance.get_dataloader()
    # to_question_pairs(dataloader, data_path=dl_instance.data_path)
    # dl_instance = DialogueTopDataLoader(dataset_name="top", data_type="eval")
    # dataloader = dl_instance.get_dataloader()
    # to_question_pairs(dataloader, data_path=dl_instance.data_path)
