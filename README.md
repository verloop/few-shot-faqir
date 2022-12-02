
# Multi-Tenant Optimization For Few-Shot Task-Oriented FAQ Retrieval

## <u>Introduction</u>

This repository contains the code to our paper (Multi-Tenant Optimization For Few-Shot Task-Oriented FAQ Retrieval) accepted at EMNLP 2022.
In this work, we evaluate performance for Business FAQ retrieval both with standard FAQ retrieval techniques using query-Question (q-Q) similarity and few-shot intent detection techniques.We propose a novel approach to scale multi-tenant FAQ applications in real-world context by contrastive fine-tuning of the last layer in sentence Bi-Encoders along with tenant-specific weight switching.


### Setting up the Repo

1. Create a virtual environment and setup the requirements: ```pip install -r requirements.txt```

2. Download data from here -> [FAQ Data](https://drive.google.com/file/d/1QybdQ6VRHvsXfiPlE4DTZWxKQ0sn4xid/view?usp=sharing)

3. Extract under: `src/data`

```
data
 ├── dialoglue
 │     ├── banking
 │     ├── clinc
 │	   ..
 └──  hint3
  	   └── v1
  	      ├── test
  	      └── train

```

4. Download relevant models as needed (Fasttext/Glove etc)

## <u>Example Usage</u>

1. Change the parameters in `config.yaml` under the `src/config` folder and run the scripts. Check config file parameters for more details

2. Evaluate base embeddings/models: ```python -m src.evaluate```

    For example:

	* For evaluating the Dialoglue BANKING77 dataset with default Bert Embeddings, these are the configuration changes required in `config.yaml`:

    ```yaml
    DATASETS:
        DATASET_SOURCE: "dialoglue"
        DATASET_NAME: "banking"
        OOS_CLASS_NAME: "NO_NODES_DETECTED"
        DATA_SUBSET: "train_5"
        N_LABELS: 77

    EVALUATION:
        EVALUATION_METHOD : "BERT_EMBEDDINGS"
        MODEL_NAME : "bert-base-uncased"
        TOKENIZER_NAME : "bert-base-uncased"
    ```

	* Then run: ```python -m src.evaluate```

3. Fine-tuning the bi-encoders/cross-encoders with question pairs/triplets

    - Create Question Pair Training data if needed: ```python -m src.utils.question_pairs```

	- Train a bi-encoder / cross-encoder with Question Pairs ```python -m src.train```

	- Change the model to the trained nodel folder and then run ```python -m src.evaluate```

	For example:

	- For finetuning the Sentence Bert model with the Dialoglue BANKING77 dataset, these are the configuration changes required in `config.yaml`

    ```yaml
    DATASETS:
        DATASET_SOURCE: "dialoglue"
        DATASET_NAME: "banking"
        OOS_CLASS_NAME: "NO_NODES_DETECTED"
        DATA_SUBSET: "train_5"
        N_LABELS: 77

    TRAINING:
        MODEL_TYPE : "BI_ENCODER"
        MODEL_NAME : "sentence-transformers/all-mpnet-base-v2"
        TOKENIZER_NAME  : "sentence-transformers/all-mpnet-base-v2"
        LAYERS_TO_UNFREEZE : [11]
        NUM_ITERATIONS : 10000
        SCHEDULER : "WarmupLinear"
        VALIDATION_SPLIT : 0.2
    ```

    - Then run `python -m src.train`

    - Update the evaluation parameters in `config.yaml`

    ```yaml
	EVALUATION:
        EVALUATION_METHOD : "BERT_EMBEDDINGS"
        MODEL_NAME : "<model_folder>"
        TOKENIZER_NAME : "sentence-transformers/all-mpnet-base-v2"
    ```

	- Then run ```python -m src.evaluate```

4. Pretraining, followed by fine-tuning

	- Under the `"data"` folder create a folder called `"pretrain"`: ```mkdir pretrain```

	- Set the required parameters in the `config.yaml` file.

	- Generate Offline Triplets for Pre-training: ```python -m src.utils.gen_pretraining_data```

	- Pretrain the bi-encoder with offline Triplets: ```python -m src.pretrain```

	- Fine-tuning the pre-trained model: Follow same steps as listed in 3.

5. Running inference with trained models with client weight switching

	- Train the model for 2 datasets(tenants) separately and store the last layer weights: ```python -m src.train```

	- The tenant specific weights will get stored under `MODEL_DIR/clients` as specified under INFERENCE in `config.yaml`

	- Change parameters in `config.yaml` under `INFERENCE`

	- Specify the tenant names under `CLIENT_NAMES`, `INFERENCE_FILE_PATH`, `MODEL_NAME` and the `LAYERS_TO_LOAD` (the last layer of the model) and then run: ```python -m src.predict```

	- A sample `inference.txt` is present under the data folder. This contains mixed utterances from 2 sample tenants

## <u>Config file Parameters</u>

### DATASETS

| Parameter | Usage |
|---|---|
| DATASET_SOURCE | Specify dataset for running evaluation / training |
| DATASET_NAME | Specify the dataset name of the corresponding DATASET_SOURCE that should be used for evalaution/training |
| OOS_CLASS_NAME | Set "NO_NODES_DETECTED" for HINT3 datasets / "oos" for CLINC150 |
| DATA_SUBSET | Training dataset to use. "train" / "subset_train" for HINT3 datasets, "train_5"/"train_10" for Dialoglue |
| N_LABELS | Number of labels present in training data.Used by the data loaders |                                  |


### TRAINING

| Parameter | Usage |
|---|---|
| SUB_SAMPLE_QQ | Set True if subsampling is needed for Question Pair generation else False |
| SAMPLE_SIZE_PER_DATASET | Required sample size if sub-sampling |
| DATA_VAL_SPLIT | Used to create a validation dataset after question pair generation eg. 0.2. Only required if using Triplets for validation|
| GENERATE_TRIPLETS | Set True if triplet generation is required for finetuning |
| HARD_SAMPLE | Set to True if needed. Used with Sub-sampling |
| MODEL_TYPE | Set one of these "BI_ENCODER"/"BERT_CLASSIFIER"/"SBERT_CROSS_ENCODER" |
| NUM_ITERATIONS | Number of training iterations |
| TRAIN_OUTPUT_DIR | Output directory for storing the models. Default is "./models/" |
| MODEL_NAME | Specify the model like "sentence-transformers/all-MiniLM-L6-v2"/"bert-base-uncased" /"cross-encoder/stsb-distilroberta-base" |
| TOKENIZER_NAME | Specify the corresponding tokenizer name eg. "sentence-transformers/all-MiniLM-L6-v2" |
| LAYERS_TO_UNFREEZE | Specify model layer to unfreeze - 11,5 eg. [5] |
| LOSS_METRIC | Specify a loss metric for Sentence Bert Bi-Encoder models - "ContrastiveLoss" / "BatchHardTripletLoss" |
| LEARNING_RATE | Default is 2e-5 |
| SCHEDULER | Specify scheduler - "WarmupLinear" for SentenceBert / "linear" for Bert |
| VALIDATION_SPLIT | Validation split for evaluating models during training if separate validation dataset doesnt exist |


### EVALUATION


| Parameter | Usage |
|---|---|
| EVALUATION_METHOD | "BERT_EMBEDDINGS" / "BERT_CLASSIFIER" / "SBERT_CROSS_ENCODER" / "BM25" / "GLOVE" / "FASTTEXT" / "TFIDF_WORD_EMBEDDINGS" / "TFIDF_CHAR_EMBEDDINGS" / "CV_EMBEDDINGS" |
| MODEL_NAME | Specify a model name which is implemented via Huggingface - "bert-base-uncased" / "models/convbert" (DialoGLUE convbert model) / "sentence-transformers/all-MiniLM-L6-v2" |
| TOKENIZER_NAME | Corresponding tokenizer for the model |
| BATCH_SIZE | Batch size for data loaders for evaluation |
| FASTTEXT_MODEL_PATH | Location of the fasttext model "models/fasttext_ecom_model_2.bin" |
| GLOVE_MODEL_PATH | Location of glove model "models/glove.6B/glove.6B.300d.txt" |
| CHECK_SUCCESS_RATE | Set True / False for checking during evaluation |
| CHECK_PRECISION | Set True / False for checking during evaluation |
| CHECK_MAP | Set True / False for checking during evaluation |
| CHECK_NDCG | Set True / False for checking during evaluation |
| CHECK_MRR | Set True / False for checking during evaluation |
| CHECK_F1_MACRO | Set True / False for checking during evaluation |
| CHECK_F1_MICRO | Set True / False for checking during evaluation |
| CHECK_F1_WEIGHTED | Set True / False for checking during evaluation |
| CHECK_OOS_ACCURACY | Set True / False for checking during evaluation |
| OOS_THRESHOLD | Thresholds at which OOS accuracy should be checked. eg.[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] |
| K_VAL | List of k values for top-k evaluation for all metrics. Specify less than 10. Eg. [1,2,3,5] |


### PRETRAINING

| Parameter | Usage |
|---|---|
| SAMPLE_SIZE_PER_DATASET | Sample size to be taken for each dataset(Curekart,SOFMattress,PowerPlay11,BANKING77,CLINC150 and HWU64) |
| VAL_SPLIT | Validation split for pretraining. Eg. 0.1 |
| GENERATE_TRIPLETS | Set True/False depending on pretraining requirements |
| GENERATE_PAIRS | Set True/False depending on pretraining requirements |
| HARD_SAMPLE | Number of steps per epoch |
| PRETRAIN_DATA_PATH | Path where intermediate data for pretraining should be saved |
| STEPS_PER_EPOCH | Set True if Hard sampling is needed |
| NUM_TRAIN_EPOCHS | Number of train epochs |
| TRAIN_OUTPUT_DIR | Output directory for the trained model and checkpoints |
| MODEL_NAME | "sentence-transformers/all-MiniLM-L6-v2" / "sentence-transformers/all-mpnet-base-v2" |
| LOSS_METRIC | Loss metric required "ContrastiveLoss" / TripletLoss" |
| BATCH_SIZE | Batch size for pretraining. Default is 32 |
| LEARNING_RATE | Default is 2e-5 |
| SCHEDULER | 'WarmupLinear' |


### INFERENCE

Parameters here is used to test the final inference with tenant weight switching.

| Parameter | Usage |
|---|---|
| TENANT_NAMES | Set of tenant names for whom the inference engine is setup. For example ["curekart", "powerplay11"] |
| INFERENCE_FILE_PATH | Path to the file where the inference data is present. "data/inference.txt" |
| MODEL_DIR | Path to model directory where tenant weights should be stored . Default is "models" |
| MODEL_NAME | Base model to be loaded. Eg "sentence-transformers/all-MiniLM-L6-v2" |
| EMBEDDING_FILE_NAME | The file name of the embeddings of the train set which is stored in numpy format. "embeddings.npy" |
| TEXTS_FILE_NAME | The file name for texts from the train set which is stored in numpy format."texts.npy" |
| LABELS_FILE_NAME | The file name containing the labels from the train set stored in numpy format. "labels.npy" |
| LAYERS_TO_LOAD | Specify the model layer to swap for each tenant. This should match with how it was trained. Eg. [5] |


## <u>Configuration notes For different usecases</u>

**Evaluation**

- For evaluation with `BM25` / `GLOVE` / `FASTTEXT` / `TFIDF_WORD_EMBEDDINGS` / `TFIDF_CHAR_EMBEDDINGS` / `CV_EMBEDDINGS` - Set ```EVALUATION_METHOD``` to `BM25` / `GLOVE` / `FASTTEXT` / `TFIDF_WORD_EMBEDDINGS` / `TFIDF_CHAR_EMBEDDINGS` / `CV_EMBEDDINGS`. For Glove, Fasttext etc, the model path (```GLOVE_MODEL_PATH```, ```FASTTEXT_MODEL_PATH```) should be specified correctly.

- For evaluation with `BERT` / `Sentence BERT embeddings`, set ```EVALUATION_METHOD``` to `"BERT_EMBEDDINGS"`. Set ```MODEL_NAME``` and ```TOKENIZER_NAME``` with any BERT model which can be loaded with Huggingface. Eg. `"bert-base-uncased"` / `"sentence-transformers/all-MiniLM-L6-v2"`

- For evaluation using `BERT` in a classifier approach, set ```EVALUATION_METHOD``` to `"BERT_CLASSIFIER"`. Set ```MODEL_NAME``` & ```TOKENIZER_NAME``` to any of the Huggingface BERT classifier models.

- For evaluation using `SBERT cross encoders`, set ```EVALUATION_METHOD``` to `"SBERT_CROSS_ENCODER"`. Set ```MODEL_NAME``` & ```TOKENIZER_NAME``` to an SBERT cross encoder model like `"cross-encoder/stsb-distilroberta-base"`.

**Fine-tuning**

- For fine-tuning bi-encoder `SBERT` models, under ```TRAINING```, set MODEL_TYPE as `"BI_ENCODER"`, Set ```MODEL_NAME``` & ```TOKENIZER_NAME``` to Sentence BERT model like `"sentence-transformers/all-MiniLM-L6-v2"`. Set ```LAYERS_TO_UNFREEZE``` as `[5]` which will depend on the model chosen.

- For fine-tuning `cross encoder SBERT` models, under ```TRAINING```, set ```MODEL_TYPE``` as `"SBERT_CROSS_ENCODER"`, Set ```MODEL_NAME``` & ```TOKENIZER_NAME``` to Sentence Bert model like `"cross-encoder/stsb-distilroberta-base"`.Set ```LAYERS_TO_UNFREEZE``` to the last layer which will depend on the model chosen.

- For fine-tuning `BERT` based models as a classifier,under ```TRAINING``` set ```MODEL_TYPE``` as `"BERT_CLASSIFIER"` and Set ```MODEL_NAME``` & ```TOKENIZER_NAME``` to any of the Huggingface Bert classifier models. Set ```LAYERS_TO_UNFREEZE``` to the last layer which will depend on the model chosen.

**Pre-training**

- For pre-training, set a ```MODEL_NAME``` to a Sentence Bert Bi-encoder model. Set the ```LOSS_METRIC```

**Training data**

- For Question pair / Triplet generation, under ```TRAINING```, set ```SUB_SAMPLE_QQ : True``` if sampling needs to be done. In such a case, set ```SAMPLE_SIZE_PER_DATASET```and ```HARD_SAMPLE``` flag as well. Set the ```MODEL_NAME``` which will be used for generating the hard samples. If training with Triplets, set ```GENERATE_TRIPLETS : True``` to enable triplet generation.

- For Pretraining data, create the folder `"data/pretrain"`. Set ```GENERATE_TRIPLETS : True``` for Triplets and ```GENERATE_PAIRS : True``` for Question Pairs. Set ```SAMPLE_SIZE_PER_DATASET``` and ```HARD_SAMPLE``` flag as well. Set the ```MODEL_NAME``` which will be used for generating the hard samples.


## <u>Citation</u>


Coming soon
