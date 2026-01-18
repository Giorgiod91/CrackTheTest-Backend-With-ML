# Since im taking the Andrew Ng Deepl Learning course i try to implement what i learn into this model
import copy
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional external dataset path
DATA_PATH = os.path.join(os.path.dirname(__file__), "difficulty_data.json")
# loat the json data w
def load_data_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            items = payload.get("data", payload)
            data = [
                {"question": str(d.get("question", "")).strip(), "label": str(d.get("label", "")).strip()}
                for d in items
                if isinstance(d, dict) and d.get("question") and d.get("label")
            ]
            if len(data) >= 10:
                return data
    except Exception:
        pass
    return None

data = [
    # ---------------- Leicht (30) ----------------
    {"question": "Was ist 12 + 7?", "label": "Leicht"},
    {"question": "Wie viele Tage hat ein Jahr?", "label": "Leicht"},
    {"question": "Welche Farbe ergibt Rot + Gelb?", "label": "Leicht"},
    {"question": "Wie heißt die Hauptstadt von Deutschland?", "label": "Leicht"},
    {"question": "Setze fort: 3, 6, 9, 12, ...", "label": "Leicht"},
    {"question": "Welcher Monat kommt nach April?", "label": "Leicht"},
    {"question": "Was ist das Gegenteil von kalt?", "label": "Leicht"},
    {"question": "Wie viele Kontinente gibt es?", "label": "Leicht"},
    {"question": "Wie viele Minuten hat eine Stunde?", "label": "Leicht"},
    {"question": "Welches Tier bellt: Hund oder Katze?", "label": "Leicht"},
    {"question": "Wie viele Seiten hat ein Quadrat?", "label": "Leicht"},
    {"question": "Welche Zahl ist größer: 19 oder 21?", "label": "Leicht"},
    {"question": "Wie heißt die Hauptstadt von Österreich?", "label": "Leicht"},
    {"question": "Welche Farbe hat der Himmel an einem sonnigen Tag?", "label": "Leicht"},
    {"question": "Was ist 45 − 18?", "label": "Leicht"},
    {"question": "Wie viele Wochen hat ein Jahr ungefähr?", "label": "Leicht"},
    {"question": "Welche Jahreszeit folgt auf den Sommer?", "label": "Leicht"},
    {"question": "Welche Zahl ist ungerade: 14 oder 15?", "label": "Leicht"},
    {"question": "Wie viele Zähne hat ein Erwachsener ungefähr?", "label": "Leicht"},
    {"question": "Wie viele Planeten hat unser Sonnensystem?", "label": "Leicht"},
    {"question": "Was ist 7 × 6?", "label": "Leicht"},
    {"question": "Welche Einheit misst die Länge: Meter oder Liter?", "label": "Leicht"},
    {"question": "Welcher Wochentag kommt nach Freitag?", "label": "Leicht"},
    {"question": "Welches Wort passt nicht: Apfel, Banane, Auto, Birne?", "label": "Leicht"},
    {"question": "Wie viele Ecken hat ein Dreieck?", "label": "Leicht"},
    {"question": "Wie heißt der größte Ozean der Erde?", "label": "Leicht"},
    {"question": "Wie viele Stunden hat ein Tag?", "label": "Leicht"},
    {"question": "Welches ist das größte Landtier?", "label": "Leicht"},
    {"question": "Was ist 100 ÷ 4?", "label": "Leicht"},
    {"question": "Welche Sprache wird in Spanien gesprochen?", "label": "Leicht"},

    # ---------------- Schwer (30) ----------------
    {"question": "Erkläre die Big-O-Notation und ihren Zweck in der Algorithmusanalyse.", "label": "Schwer"},
    {"question": "Wie funktioniert Gradientenabstieg in der linearen Regression?", "label": "Schwer"},
    {"question": "Beschreibe das Backpropagation-Verfahren in neuronalen Netzen.", "label": "Schwer"},
    {"question": "Was ist der Unterschied zwischen Overfitting und Underfitting?", "label": "Schwer"},
    {"question": "Erkläre L2-Regularisierung und ihren Einfluss auf das Training.", "label": "Schwer"},
    {"question": "Wie berechnet man die Standardabweichung einer Stichprobe?", "label": "Schwer"},
    {"question": "Was beschreibt das Ohmsche Gesetz und wie lautet die Formel?", "label": "Schwer"},
    {"question": "Erkläre den Unterschied zwischen RAM und ROM.", "label": "Schwer"},
    {"question": "Was versteht man unter relationaler Normalisierung (1NF, 2NF, 3NF)?", "label": "Schwer"},
    {"question": "Was ist der Unterschied zwischen INNER JOIN und LEFT JOIN in SQL?", "label": "Schwer"},
    {"question": "Erkläre den Zweck von Hashfunktionen in der Kryptographie.", "label": "Schwer"},
    {"question": "Was ist symmetrische vs. asymmetrische Verschlüsselung?", "label": "Schwer"},
    {"question": "Beschreibe die Hauptunterschiede zwischen Prozess und Thread.", "label": "Schwer"},
    {"question": "Was bedeutet Referenzzählung in der Speicherverwaltung?", "label": "Schwer"},
    {"question": "Erkläre das Konzept der Zeitkomplexität am Beispiel von Mergesort.", "label": "Schwer"},
    {"question": "Was ist Regularisierung und warum verhindert sie Overfitting?", "label": "Schwer"},
    {"question": "Wie interpretiert man Präzision, Recall und F1-Score?", "label": "Schwer"},
    {"question": "Erkläre das Prinzip der Gradientenexplosion und -vanishing.", "label": "Schwer"},
    {"question": "Was ist der Unterschied zwischen REST und RPC in Web-APIs?", "label": "Schwer"},
    {"question": "Erkläre ACID-Eigenschaften in Datenbanksystemen.", "label": "Schwer"},
    {"question": "Wie funktioniert eine Garbage Collection grob?", "label": "Schwer"},
    {"question": "Was ist Bias-Varianz-Trade-off im maschinellen Lernen?", "label": "Schwer"},
    {"question": "Erkläre Kreuzvalidierung und ihren Nutzen bei kleinen Datensätzen.", "label": "Schwer"},
    {"question": "Beschreibe die Hauptideen der Normalverteilung und des Z-Scores.", "label": "Schwer"},
    {"question": "Was ist der Unterschied zwischen deterministischen und probabilistischen Algorithmen?", "label": "Schwer"},
    {"question": "Erkläre die Funktionsweise eines Verbrennungsmotors auf hoher Ebene.", "label": "Schwer"},
    {"question": "Was ist der Unterschied zwischen TCP und UDP?", "label": "Schwer"},
    {"question": "Wie funktioniert ein Cache und was sind Cache-Hits/Misses?", "label": "Schwer"},
    {"question": "Erkläre das Konzept der Entropie in der Informationstheorie.", "label": "Schwer"},
    {"question": "Was versteht man unter Feature-Engineering und warum ist es wichtig?", "label": "Schwer"},
]

# If a JSON dataset exists, use it instead of the hardcoded list
_loaded = load_data_json(DATA_PATH)
if _loaded:
    data = _loaded




texts = [d["question"] for d in data]  # list of strings
# switching text for 0 or 1
labels = [0 if d["label"] == "Leicht" else 1 for d in data]  

texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# will convert each question into a vector of numbers
# Minimal German stop words list (lowercase) for vectorization the default only knows English
GERMAN_STOP_WORDS = [
    "der", "die", "das", "ein", "eine", "einer", "eines", "einem", "einen",
    "und", "oder", "aber", "denn", "weil", "dass", "als", "wie",
    "auf", "für", "mit", "von", "zu", "im", "in", "am", "an", "aus", "bei", "nach", "über", "unter", "zwischen", "vor", "hinter", "ohne", "durch", "gegen", "seit", "bis",
    "ist", "sind", "war", "waren", "sein", "bin", "bist", "wird", "werden", "wurden",
    "nicht", "kein", "keine", "nur", "auch", "noch", "schon", "sehr", "mehr", "weniger",
    "man", "wir", "ihr", "sie", "er", "es", "ich", "du",
    "mein", "meine", "dein", "deine", "sein", "seine", "ihr", "ihre",
    "dies", "diese", "dieser", "dieses", "jener", "jene", "jenes",
    "zum", "zur", "vom", "beim", "ins", "dem", "den", "des"
]

vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words=GERMAN_STOP_WORDS, max_features=800)

# and here X_mat has the shape of (n,m)  = in this case we habe (6 samples, N features (words))
X_train_mat = vectorizer.fit_transform(texts_train).toarray()  # (m_train, n)
X_test_mat  = vectorizer.transform(texts_test).toarray() 


# transpose or switch cuase we want x to be the features cause in Math this will alling with formulas likes Z + wt*X+b

# also the reshape here to match the shapes we want them to be the same shape
X_train = X_train_mat.T
X_test  = X_test_mat.T
Y_train = np.array(labels_train).reshape(1, -1)
Y_test  = np.array(labels_test).reshape(1, -1)



# create w and b
# here we start with all weights at 0
def initalize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0.0 # scalar

    return w,b


# sigmoid function will turn any number outcome into a probability between 0 and 1
# in our case here it z is big === 1 ==> "Schwer"

def sigmoid(z):
    s = 1/(1+np.exp(-z))

    return s


# cost function

def propagate(w,b,X,Y):
    m = X.shape[1]

    #compute activation
    # forward propagation
    A= sigmoid(np.dot(w.T,X)+b)

    # compute the cost or the loss

    cost =  -(1.0/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

    # Backward propagation

    dw = (1/m) * np.dot(X,(A-Y).T)
    db = (1/m) * np.sum(A-Y)

    cost = np.squeeze(np.array(cost))
    grads = {"dw":dw, "db": db}

    return cost,grads


# function that optimizes w and b by running gradient descent algo
def optimize(w,b,X,Y, num_iterations=100, learning_rate=0.009, print_cost=False):

    # first create copies so we dont modify the original ones
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []
    # here the training loop
    for i in range(num_iterations):
        # we cann the propagate function that uses forward and backward propagation
        cost , grads = propagate(w,b,X,Y)
        # extract the value dw and db becaue we need them for the gradient descent 
        dw = grads["dw"]
        db = grads["db"]

        # and this here is the catual update steop with gradient descent
        w = w-learning_rate * dw
        b = b-learning_rate *db

        #record cost
        if i % 100 ==0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after {i}: {cost}")


        params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w,b,X):
    A = sigmoid(np.dot(w.T,X) +b)
    return (A >= 0.5).astype(int),A

# train
w0,b0 = initalize_with_zeros(X_train.shape[0])

params, grad, costs = optimize(w0, b0,X_train,Y_train, num_iterations=5000, learning_rate=0.005, print_cost=True)
# desctructe the values i need
w,b = params["w"], params["b"]

train_preds, _ = predict(w, b, X_train)
test_preds,  _ = predict(w, b, X_test)


train_acc = (train_preds.flatten() == Y_train.flatten()).mean()
test_acc  = (test_preds.flatten()  == Y_test.flatten()).mean()


# now a sample to predict on
questions_to_predict =[
    "was ist 5 + 12 ?",
    "Wie Lange braucht das Licht von der Sonne zur Erde ?",
    "Erkläre das Prinzip der Gravitation"
    

]

# Vectorize the quesiotns
vectorized_questions = vectorizer.transform(questions_to_predict).toarray().T

predicts, probab = predict(w,b, vectorized_questions)

# print and check if the model is predicting right
# flatten into 1D
for q, pred in zip(questions_to_predict, predicts.flatten()):
    if pred == 0:
        print(q, "Leicht")
    else:
        print(q, "Schwer")


print(f"Train Accuracy: {train_acc*100:.1f}%")
print(f"Test Accuracy: {test_acc*100:.1f}%")

plt.plot(costs)
plt.title("Training cost")
plt.xlabel("Iterations (x100)")
plt.ylabel("Cost")
plt.show()

