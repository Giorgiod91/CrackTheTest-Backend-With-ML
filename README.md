
# 🚀 CrackTheTest-Backend-With-ML

**Backend für CrackTheTest.ai – FastAPI + Supabase + Custom ML-Modell**

🧩 **Status:** In Entwicklung (API-Routen, DB-Integration & ML-Prediction)  
🔗 **Frontend Repo & Live Demo:** [https://github.com/Giorgiod91/CrackTheTest] [https://crack-the-test.vercel.app/](https://crack-the-test.vercel.app/)  

Dieses Repo enthält das **Python-Backend** für den AI-Test-Generator: API-Routen mit FastAPI, Supabase-Integration (PostgreSQL) und ein **custom Logistic Regression Modell** (from scratch mit NumPy – inspiriert von Andrew Ng’s Deep Learning Kurs).

---

## 🎯 Features (aktuell & geplant)

- ⚡ FastAPI-Routen für User-Management & Premium-Content
- 💾 Supabase Client für DB-Operationen (User anlegen, Content fetchen)
- 🤖 `/predict-difficulty` Endpoint: Schwierigkeit von Fragen vorhersagen (Leicht/Schwer)
- 🔒 CORS Middleware für sichere Frontend-Verbindung
- 🧠 Custom ML-Modell: Binary Logistic Regression (NumPy only) mit TF-IDF
- 🚀 Geplante Erweiterungen: Supabase Auth + JWT, Stripe Billing, Modell persistieren

---

## ⚙️ Tech Stack

- **Python + FastAPI** (API & Routing)
- **Supabase** (PostgreSQL DB + Client)
- **Pydantic** (Request Validation)
- **NumPy** (ML from scratch: Sigmoid, Gradient Descent, Cross-Entropy)
- **werkzeug** (Password Hashing)

## 🧠 ML-Modell (lehrreicher Teil)

- Binary Klassifikation: `0 = Leicht`, `1 = Schwer`
- Vollständig manuell implementiert (Forward/Backward Propagation, Gradient Descent)
- Training auf handgelabelten deutschen Fragen
- Ziel: Tieferes Verständnis der ML-Grundlagen (Andrew Ng Style)


