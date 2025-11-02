# Since im taking the Andrew Ng Deepl Learning course i try to implement what i learn into this model
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
data = [
    # Leicht
    {"question": "Was ist 2 + 2?", "label": "Leicht"},
    {"question": "Wie heißt die Hauptstadt von Deutschland?", "label": "Leicht"},
    {"question": "Welche Farbe hat der Himmel?", "label": "Leicht"},

       # Schwer
    {"question": "Erkläre das Konzept der Big-O-Notation.", "label": "Schwer"},
    {"question": "Wie funktioniert Backpropagation in neuronalen Netzen?", "label": "Schwer"},
    {"question": "Beschreibe die Photosynthese auf molekularer Ebene.", "label": "Schwer"},
]



texts = [d["question"] for d in data]  # list of strings
# switching text for 0 or 1
labels = [0 if d["label"] == "Leicht" else 1 for d in data]  

vectorizer = TfidfVectorizer()
X_mat = vectorizer.fit_transform(texts).toarray()
Y = np.array(labels)[None, :]  


X = np.array(X_mat).T
Y = np.array(labels)

# binary sigmoid

def sigmoid(z):
    s = 1/(1+np.exp(-z))

    return s


# create w and b

def initalize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0.0 # scalar

    return w,b


# cost function

def propagate(w,b,X,Y):
    m = X.shape[1]

    #compute activation

    A= sigmoid(np.dot(w.t,X)+b)

    # compute the cost

    cost = -1(1.0/m) * np.sum(Y.np.log(A)+ (1-Y)*np.log(1-A))

    # Backward propagation

    dw = (1/m) * np.dot(X,(A-Y).T)
    db = (1/m) * np.sum(A-Y)

    cost = np.squeeze(np.array(cost))
    grads = {"dw":dw, "db": db}

    return cost,grads


# function that optimizes w and b by running gradient descent algo
def optimize(w,b,X,Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    cost = []

    for i in range(num_iterations):
        grads , cost = propagate(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]


        w = w-learning_rate * dw
        b = b-learning_rate *db

        #record cost
        if i % 100 ==0:
            cost.append(cost)
            if print_cost:
                print("Cost after "(i,cost))

        params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, cost