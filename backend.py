import os
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from werkzeug.security import generate_password_hash ,check_password_hash
from flask_sqlalchemy import SQLAlchemy


load_dotenv()
app = FastAPI()

#creating db intance
db = SQLAlchemy

db.init_app(app)

Class SupaUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    joined_at = db.Column(db.DateTime(), index=True, default=datetime.utcnow)
    

https://www.youtube.com/watch?v=fsNeGqxC4PM   reference



class User(BaseModel):
    email: str

class input(BaseModel):
    text: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend-URL
    allow_credentials=True,
    allow_methods=["*"],   # erlaubt GET, POST, PUT, DELETE usw.
    allow_headers=["*"],   # erlaubt alle Header
)
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
service_role_key: str = os.environ.get("SERVICE")
supabase: Client = create_client(url,service_role_key)
# Route
@app.post("/create-new-user/")
# Function here to add a user to the DB with supabase and return data
def add_user(user: User):

    existing = supabase.table("Users").select("*").eq("email_adress", user.email).execute()
    if existing.data and len(existing.data) > 0:
        return {"data": None, "error": "Email already exists"}


  
    result = supabase.table("Users").insert({"email_adress":user.email}).execute()

    return {"data": result.data}

# another route here for my ML model
@app.post("/predict-difficulty")
def predict(data: input):
    user_input = data.text
    predicted_difficulty = My_Model.predict()

    return {"predicted difficulty with machnine learning Model": predicted_difficulty}


# hash user password with data (user_input_pw) from the frontend validation check will be in the frontend
def create_hash_pw(user_input_pw):
    hashed_pw = generate_password_hash(user_input_pw)

    return hashed_pw

# hased_pw with the right user push into the DB

def push_hashed_pw(hashed_pw):
 #::TODO:: add the logic here and connect supabase db    

    


