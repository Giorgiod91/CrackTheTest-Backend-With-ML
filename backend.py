import os
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from werkzeug.security import generate_password_hash ,check_password_hash

from openai import OpenAI



load_dotenv()
app = FastAPI()






client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
print(os.environ.get("OPENAI_API_KEY"))

#creating db intance

#https://www.youtube.com/watch?v=fsNeGqxC4PM   reference

# i allways forget how to start the server   ::TODO:  uvicorn backend:app --reload




class User(BaseModel):
    email: str
    password: str

class input(BaseModel):
    text: str
    title: str
    subject: str
    content: str

class Tests(BaseModel):
    user_id: int 
    title: str
    subject: str
    content: str
    

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

    email_address = "first_user@lol.com"
    username = "admin1"


  
    result = supabase.table("users").insert({"email":email_address, "username":username}).execute()

    return {"data": result.data}

#fetch the premium users data
@app.get("/display_data/{user_id}")
def get_data(user_id: int):
    # check if user exists
    user_response = supabase.table("users").select("id").eq("id", user_id).execute()
    print("User query:", user_response.data)

    if not user_response.data:
        return {"error": "User not found"}

    # fetch premium content for dashboard
    content_response = supabase.table("tests").select("title, content,id, created_at").eq("authorid", user_id).execute()

    return content_response.data

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




# Routing to create tests with the open ai API
# i make sure to check if the user has premium status
@app.post('/create_test')
def create_test(data:Tests):
    def is_user_premium():
        premium = supabase.table("users").select("premium").eq("id", data.user_id).execute()
        if not premium.data or not premium.data[0]["premium"]:
            return {"error": "User is not Premium"}     
        else:
              
             response = client.responses.create(
             model="gpt-3.5-turbo",
             input=f"Write a authentic test for this topic={data.title} and the test should include={data.content} withing this subject={data.subject}")
             supabase.table("tests").insert({"title": data.title, "subject": data.subject,"content":response.output_text}).execute()
       

        return response.output_text
            
    return is_user_premium() 
        
    
# create a route to upadate a test
@app.put('/update_test/{test_id}')
def update_test(test_id: int, data: Tests):
    response = supabase.table("tests").update({
        "title": data.title,
        "subject": data.subject,
        "content": data.content
    }).eq("id", test_id).execute()

    return {"data": response.data}


# and a route to delete a test
@app.delete('/delete_test/{test_id}')
def delete_test(test_id: int):
    response = supabase.table("tests").delete().eq("id", test_id).execute()

    return {"data": response.data}

# auth route for user login
@app.post('/login')
def auth_login(user: User):
    response = supabase.auth.sign_in_with_password(
    {
        "email": user.email,
        "password": user.password,
    }
    )
    return {"data": response}




    