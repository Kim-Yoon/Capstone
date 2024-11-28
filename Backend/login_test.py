from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Any
from datetime import datetime, timedelta
import jwt


from mongo import MongoConnection

import jwt
import yaml

env_path = './.env.yml'

with open(env_path) as f:
    env = yaml.load(f, Loader=yaml.FullLoader)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

mongo_client = AsyncIOMotorClient(env['mongo']['uri'])
db = mongo_client['example']
collection = db['users']
test_collection = db['landmarks']


# JWT 설정
SECRET_KEY = env['secret_key']
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[EmailStr] = None

class User(BaseModel):
    name : str
    email: EmailStr
    hashed_password: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    name : str
    email: EmailStr
    password: str
    

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_password_hash(password: str):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_user(email: EmailStr):
    user_data = await collection.find_one({"email": email})
    print(user_data)
    if user_data:
        return UserInDB(**user_data)

async def check_user(email: EmailStr, password: str):
    user = await get_user(email)
    if not user:
        return False
    if not pwd_context.verify(password, user.hashed_password):
        return False
    return user

async def authenticate_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    return token_data

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await check_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    print(access_token)
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/signup")
async def signup(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    user = UserCreate(name= name, email=email, password=password)
    print(user.name)
    # if await get_user(user.email):
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Email already registered",
    #     )
    
    hashed_password = get_password_hash(user.password)
    try:
        user_data = {
            "name" : user.name,
            "email": user.email,
            "hashed_password": hashed_password,
            "disabled": False,
            "complete_list" : [],
            "wish_list" : [],
        }
        await collection.insert_one(user_data)
    except Exception as e:
        print("Email already registered")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    return {"email": user.email}

@app.get("/signup", response_class=HTMLResponse)
async def get_signup_page():
    with open('static/signup.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/login", response_class=HTMLResponse)
async def get_login_page():
    with open('static/login.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)