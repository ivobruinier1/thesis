import pandas as pd
import evaluate
from sklearn.metrics import classification_report

from datasets import Dataset
from datasets import ClassLabel

from transformers import AutoModelForTokenClassification, Trainer, AutoTokenizer, DataCollatorForTokenClassification



import pandas as pd

# Step 1: Read the CSV file from the specified path
train_path = 'data/train.csv'  # Make sure the path is correct
test_path = 'data/test.csv'
def convert_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['tokens', 'tags'])

    # Step 2: Initialize lists to hold the transformed sentences
    sentences = []
    current_tokens = []
    current_labels = []

    end_token =["?", "!","."]
    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    # Step 3: Process the dataframe row by row
    for index, row in df.iterrows():
        token = row['tokens']
        label = row['tags']

        # If we encounter punctuation (.), assume sentence ends and save the current sentence
        if token in end_token:
            current_tokens.append(token)
            current_labels.append(label)
            sentences.append({"text": " ".join(current_tokens), "labels": current_labels})
            # Reset for next sentence
            current_tokens = []
            current_labels = []
        elif label not in label_list:
            current_tokens.append(str(token))
            current_labels.append("O")
        else:
            current_tokens.append(str(token))
            current_labels.append(str(label))

    # Step 4: Handle case when there's no punctuation at the end of a sentence
    if current_tokens:
        sentences.append({"text": " ".join(current_tokens), "labels": current_labels})

    # Now `sentences` is in the desired format
    return sentences


train_sentences = convert_data(train_path)
eval_sentences = convert_data(test_path)

# Define a ClassLabel object to use to map string labels to integers.
classmap = ClassLabel(num_classes=9, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'])

# Create the label_list variable
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


# Convert to Hugging Face Datasets
ds_train = Dataset.from_pandas(pd.DataFrame(data=train_sentences))
ds_eval = Dataset.from_pandas(pd.DataFrame(data=eval_sentences))

print(ds_eval)

# Initialize model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("pdelobelle/robbert-v2-dutch-ner",
                                                        id2label={i: classmap.int2str(i) for i in
                                                                  range(classmap.num_classes)},
                                                        label2id={c: classmap.str2int(c) for c in classmap.names},
                                                        finetuning_task="ner")
tokenizer = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-ner")
data_collator = DataCollatorForTokenClassification(tokenizer)

# Tokenize the dataset
ds_train = ds_train.map(lambda x: tokenizer(x["text"], truncation=True))
ds_eval = ds_eval.map(lambda x: tokenizer(x["text"], truncation=True))

# Convert labels to integers using ClassLabel mappings
ds_train = ds_train.map(lambda y: {"labels": [classmap.str2int(label) for label in y["labels"]]})
ds_eval = ds_eval.map(lambda y: {"labels": [classmap.str2int(label) for label in y["labels"]]})

# Load the evaluation metric
metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute metrics
    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Initialize the Trainer
trainer = Trainer(
    model=model,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Get predictions from the model
predictions, labels, _ = trainer.predict(ds_eval)

# Process the predictions
predictions = predictions.argmax(axis=2)

# Remove special tokens and prepare data for classification report
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

# Flatten the lists of labels and predictions to pass to classification_report
flat_true_predictions = [item for sublist in true_predictions for item in sublist]
flat_true_labels = [item for sublist in true_labels for item in sublist]

# Print classification report
print(classification_report(flat_true_labels, flat_true_predictions))
