
import numpy as np
from datasets import Dataset
import torch
from torch.nn.functional import softmax

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments, Trainer

#:TODO: add more DATA 
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
    eval_strategy="no",  # evaluates end the end of each epoch
    save_strategy="no",  # no auto safe here i just manually safe it at the end
    logging_strategy="epoch"  # since i have a small dataset i can log and track loss after each epoch
    



)
#::TODO:: create a working train dataset
# this step here i had create a dataset myown sonce i dont use created one  for the train_dataset= argument
# source here i used is this https://huggingface.co/docs/datasets/about_dataset_features
# and this https://huggingface.co/docs/datasets/create_dataset#from-python-dictionaries


# softmax since its 3 labels for 2 i would use sigmoid


# making dataset out of the lists
dataset = Dataset.from_dict({
    "text": texts,
    "labels": labels
})

# Tokenizer amnwenden

tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)



# Setup Trainer
# for reference and learning i used this and the secdion is 6.4 Setting up and instance of Trainer
# https://www.learnhuggingface.com/notebooks/hugging_face_text_classification_tutorial

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset


)

# training starten
trainer.train()

# input to test the model

input_from_user = input("Frage stellen")
# tokenizen
tokenized_input_from_user = tokenizer(input_from_user, return_tensors="pt")


# code snippet i found online that i will learn about this is why i comment those lines now

# switch Model mode to predict and not training
model.eval()



#  we need id first before the label so we switch the dict v comes k
convert_labe_to_numbers_id_first ={}
for k,v in convert_labe_to_numbers.items():
    convert_labe_to_numbers_id_first[v] = k



with torch.no_grad():
    # use the model on the tokenized input  it gives back raw logits before the softmax
    output = model(**tokenized_input_from_user)
    # now applying softmax that converst logits into probabilites 
    probabilities = softmax(output.logits, dim=-1)
    # finds the index from the biggest value 
    predicted_class_id = torch.argmax(probabilities)


# converts back to Leicht , Mittel , Schwer
print("Predicted class:", convert_labe_to_numbers_id_first[int(predicted_class_id)])

# converts tensor into lists
print("Probabilities:", probabilities.tolist())




