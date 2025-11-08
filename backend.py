import os
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


load_dotenv()
app = FastAPI()

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
    def predicted_difficulty(data: input):
    user_input = data.text
    predicted_difficulty = My_Model.predict()

    return {"predicted difficulty with machnine learning Model": predicted_difficulty}
    


