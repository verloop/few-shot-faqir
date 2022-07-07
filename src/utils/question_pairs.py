import csv
import itertools

from src.data.dataloaders import (  # isort:skip
    DialogueIntentDataLoader,
    DialogueTopDataLoader,
    HaptikDataLoader,
)


def to_question_pairs(dataloader, data_path):
    # Will cause memory overflow for large dataset
    data = dataloader.dataset[:]
    partial_filename = data_path.split(".")[0]
    with open(f"{partial_filename}_question_pairs.csv", "w", newline="") as f_output:
        csv_output = csv.DictWriter(
            f_output,
            fieldnames=["question1", "question2", "is_duplicate"],
            delimiter=",",
        )
        csv_output.writeheader()
        for q1, q2 in itertools.combinations(data, 2):
            if q1["Label"] == q2["Label"]:
                csv_output.writerow(
                    {
                        "question1": q1["Text"],
                        "question2": q2["Text"],
                        "is_duplicate": 1,
                    }
                )
            else:
                csv_output.writerow(
                    {
                        "question1": q1["Text"],
                        "question2": q2["Text"],
                        "is_duplicate": 0,
                    }
                )

        f_output.close()


if __name__ == "__main__":

    # haptik
    dataset_names = ["curekart", "powerplay11", "sofmattress"]
    for dataset_name in dataset_names:
        # haptik train
        dl_instance = HaptikDataLoader(dataset_name=dataset_name, data_type="train")
        dataloader = dl_instance.get_dataloader()
        to_question_pairs(dataloader, data_path=dl_instance.data_path)
        # haptik test
        dl_instance = HaptikDataLoader(dataset_name=dataset_name, data_type="test")
        dataloader = dl_instance.get_dataloader()
        to_question_pairs(dataloader, data_path=dl_instance.data_path)

    # dialogue intent
    dataset_names = ["banking", "clinc", "hwu"]
    for dataset_name in dataset_names:
        # dialogue intent train
        dl_instance = DialogueIntentDataLoader(
            dataset_name=dataset_name, data_type="train"
        )
        dataloader = dl_instance.get_dataloader()
        to_question_pairs(dataloader, data_path=dl_instance.data_path)
        # dialogue intent test
        dl_instance = DialogueIntentDataLoader(
            dataset_name=dataset_name, data_type="test"
        )
        dataloader = dl_instance.get_dataloader()
        to_question_pairs(dataloader, data_path=dl_instance.data_path)
        # dialogue intent val
        dl_instance = DialogueIntentDataLoader(
            dataset_name=dataset_name, data_type="val"
        )
        dataloader = dl_instance.get_dataloader()
        to_question_pairs(dataloader, data_path=dl_instance.data_path)

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
