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

# will convert each question into a vector of numbers
vectorizer = TfidfVectorizer()
# and here X_mat has the shape of (n,m)  = in this case we habe (6 samples, N features (words))
X_mat = vectorizer.fit_transform(texts).toarray()


# transpose or switch cuase we want x to be the features cause in Math this will alling with formulas likes Z + wt*X+b
X = X_mat.T
# also the reshape here to match the shapes we want them to be the same shape
Y = labels.reshape(1,-1)



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
    A= sigmoid(np.dot(w.t,X)+b)

    # compute the cost or the loss

    cost = -1(1.0/m) * np.sum(Y.np.log(A)+ (1-Y)*np.log(1-A))

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

    cost = []
    # here the training loop
    for i in range(num_iterations):
        # we cann the propagate function that uses forward and backward propagation
        grads , cost = propagate(w,b,X,Y)
        # extract the value dw and db becaue we need them for the gradient descent 
        dw = grads["dw"]
        db = grads["db"]

        # and this here is the catual update steop with gradient descent
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