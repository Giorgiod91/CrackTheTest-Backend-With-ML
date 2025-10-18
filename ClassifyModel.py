
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments, Trainer


data = [
    # Leicht
    {"question": "Was ist 2 + 2?", "label": "Leicht"},
    {"question": "Wie heißt die Hauptstadt von Deutschland?", "label": "Leicht"},
    {"question": "Welche Farbe hat der Himmel?", "label": "Leicht"},

    # Mittel
    {"question": "Erkläre den Unterschied zwischen RAM und ROM.", "label": "Mittel"},
    {"question": "Was macht eine for-Schleife in Python?", "label": "Mittel"},
    {"question": "Warum hat der Mond verschiedene Phasen?", "label": "Mittel"},

    # Schwer
    {"question": "Erkläre das Konzept der Big-O-Notation.", "label": "Schwer"},
    {"question": "Wie funktioniert Backpropagation in neuronalen Netzen?", "label": "Schwer"},
    {"question": "Beschreibe die Photosynthese auf molekularer Ebene.", "label": "Schwer"},
]
# prepare the data for the model make it numeric
convert_labe_to_numbers = {"Leicht":0, "Mittel": 1, "Schwer": 2}
# 
texts = [item["question"] for item in data]
labels = [convert_labe_to_numbers[item["label"]] for item in data]

# BERT German model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")

# test to see if the tokenizer works
print(tokenizer("i build my own model"))

# truncation cuts out the text if its too long
# padding so all the text are the same length
#  return_tensors="pt" return tensores instead of lists

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    return_tensors="pt"
)


# setup model for fine tuning with classification
model = AutoModelForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-german-cased",
    num_labels=3  # 3 classes = Leicht, Mittel, Schwer
)

# i checke out those parameter and what value and defaults to use here https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
# and here https://www.learnhuggingface.com/notebooks/hugging_face_text_classification_tutorial
training_args = TrainingArguments(
    output_dir="./results", # save the results there
    learning_rate=5e-5,  # picked the default value
    per_device_train_batch_size=8,  # found this as a default value aswell
    num_train_epochs=3, # in the documention was this also flagged with 3 as default
    eval_strategy="epos",  # evaluates end the end of each epos 
    save_strategy="no",  # no auto safe here i just manually safe it at the end
    logging_strategy="epoch"  # since i have a small dataset i can log and track loss after each epoch



)

# this step here i had create a dataset myown sonce i dont use created one  for the train_dataset= argument




# Setup Trainer
# for reference and learning i used this and the secdion is 6.4 Setting up and instance of Trainer
# https://www.learnhuggingface.com/notebooks/hugging_face_text_classification_tutorial?utm_source=chatgpt.com#setting-up-a-model-for-training

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=


)

