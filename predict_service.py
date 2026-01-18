import os
import pickle
import numpy as np

class DifficultyPredictor:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "model_artifacts", "difficulty_model.pkl")
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)
        
        self.w = artifacts["w"]
        self.b = artifacts["b"]
        self.vectorizer = artifacts["vectorizer"]
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, question: str):
        # Vectorize
        X = self.vectorizer.transform([question]).toarray().T
        
        # Predict
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        prediction = (A >= 0.5).astype(int)[0][0]
        probability = float(A[0][0])
        
        label = "Schwer" if prediction == 1 else "Leicht"
        return {
            "question": question,
            "difficulty": label,
            "confidence": probability if prediction == 1 else (1 - probability)
        }