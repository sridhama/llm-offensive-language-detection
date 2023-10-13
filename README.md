# Multi-Lingual Hate Speech Detection

We leveraged the OLID multi-lingual hate speech dataset to explore different transformer models with the aim of determining the most effective method for hate speech detection across multiple languages. In our approach, we use a multi-lingual transformer model known as XLM-RoBERTa, as well as specialized BERT models pretrained for individual languages.

[Full paper](https://drive.google.com/file/d/1YmO5iou8OHDZ_6O0cciS_lIA8spmkg94/view?usp=sharing)

## Per-language pre-trained BERT models for hate speech detection

## **Requirements:**

```
torch==1.10.2
transformers=4.18.0
pandas
scikit-learn
```

## **Training:**

****For subtask A:****
Navigate to the `/notebooks` folder and run `run_train.ipynb`. Make sure the data path and python file path are correctly specified.

In general, to run the training script from a terminal window, the following syntax can be used:

    python bert_finetune.py \
    --language 'english' \
    --logs_dir 'PATH_TO_LOGS' \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --weight_decay 1e-2 \
    --epochs 50

****For subtasks B & C:****
Navigate to the `/notebooks` folder and run `run_train_olid_subtask_b_c.ipynb`. Make sure the data path and python file path are correctly specified.

In general, to run the training script from a terminal window, the following syntax can be used:
NOTE: For training on subtask C, simply replace `subtask_b` below with `subtask_c`.

    python bert_finetune_OLID.py \
    --language 'english' \
    --logs_dir 'PATH_TO_LOGS' \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --weight_decay 1e-2 \
    --epochs 50 \
    --subtask 'subtask_b'

## **Evaluation:**

Navigate to the `/notebooks` folder and run `error_analysis.ipynb`. Make sure the data path and python file path are correctly specified.

In general, to run the evaluation script from a terminal window, the following syntax can be used:

    python bert_finetune_OLID_error_analysis.py \
    --language 'english' \
    --ckpt_loc 'PATH_TO_SUBTASK_MODEL_CHECKPOINT'
    --subtask 'subtask_a'
## **Access to our fine-tuned model checkpoints:**

[Link](https://drive.google.com/drive/folders/1-FckcqYSeOeLGeYfvdCyyPNNe6AYzFyZ?usp=sharing) to fine-tuned models for the five different languages in the dataset for Task A.

[Link](https://drive.google.com/drive/folders/1qYjEy3I4Ve8ZJ8nJS_xTb9EhIz3riJcU?usp=sharing) to fine-tuned model checkpoint for Task B.

[Link](https://drive.google.com/drive/folders/1fD8O8eNxxaS2efnX0SSL65UnoQaRtoXL?usp=sharing) to fine-tuned model checkpoint for Task C.
