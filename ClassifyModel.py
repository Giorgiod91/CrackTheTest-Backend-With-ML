
import numpy as np
from transformers import AutoModel, AutoTokenizer


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