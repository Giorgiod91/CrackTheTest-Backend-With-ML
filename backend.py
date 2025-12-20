import os
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from werkzeug.security import generate_password_hash ,check_password_hash


load_dotenv()
app = FastAPI()

#creating db intance

#https://www.youtube.com/watch?v=fsNeGqxC4PM   reference

# i allways forget how to start the server   ::TODO:  uvicorn backend:app --reload



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

#fetch the premium users data

@app.get("/display_data/{user_id}")
def get_data(user_id: int):

    # check if users exists
    user = supabase.table("Users").select("id").eq("authorid", user_id).execute()
    if not user.data:
        return {"error": "User not found"}

    # fetch premium content for dashboard
    content = supabase.table("Test").select("title, content").eq(
        "user_id", user_id
    ).execute()

    return content.data


# want a route now for the premium users

@app.get('/premium')
def premium_users():
    myChannel = "" ## my youtube channel later
    return f'Welcome to the Premium Section! <a href="https://youtube.com{myChannel}"> Show Video Guide </a>'
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

#def push_hashed_pw(hashed_pw):
 #::TODO:: add the logic here and connect supabase db    

    


