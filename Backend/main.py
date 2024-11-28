from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import traceback
from datetime import datetime
from fastapi.responses import FileResponse

import io
import shutil
import os
import yaml
from bson.son import SON
import glob
import base64
from datetime import datetime, timedelta


from finder import Finder
# from openai_model import GPT

from fastapi import Depends,  status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Any
from datetime import datetime, timedelta
import jwt


# from bard import Bard, prefix, suffix
from utils.inference import LandmarkClassifier
from mongo import MongoConnection



env_path = './.env.yml'

with open(env_path) as f:
    env = yaml.load(f, Loader=yaml.FullLoader)

# # Get OPENAI Model
# gpt = GPT(openai_key=env['openai'], openai_org=env['openai_org'], model='gpt-3.5-turbo')

# JWT 설정
SECRET_KEY = env['secret_key']
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


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

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

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

async def authenticate_user(email: EmailStr, password: str):
    user = await get_user(email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# 이미지를 Base64 인코딩된 문자열로 변환하는 함수
def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

async def fetch_initialfItems():
    festivals = db['festivals']
    f_item={}
    # 현재 날짜와 2주 후 날짜 계산
    now = datetime.now()
    two_weeks_later = now + timedelta(weeks=2)
    now_str = now.strftime('%Y-%m-%d')
    two_weeks_later_str = two_weeks_later.strftime('%Y-%m-%d')

    festivals_pipeline = [
        {
            # 축제 시작 날짜가 현재 날짜와 2주 후 날짜 사이인 것만 필터링
            "$match": {
                "date_start": {"$gte": now_str, "$lte": two_weeks_later_str}
            }
        },
        {
            # 필요한 필드만 선택하여 가져오기
            "$project": {
                "_id": 1,
                "name": 1
            }
        }
    ]
    async for doc in festivals.aggregate(festivals_pipeline):
        f_item[doc['_id']] = doc['name']
    return f_item



async def authenticate_token(token: str = Depends(oauth2_scheme)):
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
        email_data = TokenData(email=email)
    except InvalidTokenError:
        raise credentials_exception
    return email_data


app = FastAPI()
lc = LandmarkClassifier()
mongo_client = AsyncIOMotorClient(env['mongo']['uri'])
db = mongo_client['example']

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root(request: Request):
    client_host = request.client.host
    x_real_ip = request.headers.get("X-Real-IP")
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    return {
        "client_host": client_host,
        "X-Real-IP": x_real_ip,
        "X-Forwarded-For": x_forwarded_for,
    }

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    
    try:
        image_content = await file.read()
        image = Image.open(io.BytesIO(image_content))
        results = lc.inference(image)
        print(results)
        #image.save(f"{file.filename}")  # 이미지를 서버에 저장
        #bard_answer = bard.get_answer(f'{prefix} 대한민국의 문화재인 {results[0]}에 대한 설명을 200자 이상으로 요약해 줘. "물론입니다"라는말 하지마. {suffix}')

        return JSONResponse(content={"filename": file.filename, 
                                     "detail": "Image uploaded", 
                                     "prediction": results})
                                    #  "bard": bard_answer['content']}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    #with open(os.path.join("uploads", file.filename), "wb") as buffer:
    with open(file.filename, "wb") as buffer:
        contents = await file.read()  # async read
        buffer.write(contents)
    return {"filename": file.filename}

@app.post("/default_searchlist/")
async def create_upload_file(file: UploadFile = File(...)):
    #with open(os.path.join("uploads", file.filename), "wb") as buffer:
    with open(file.filename, "wb") as buffer:
        contents = await file.read()  # async read
        buffer.write(contents)
    return {"filename": file.filename}


@app.get("/upload_page/", response_class=HTMLResponse)
async def read_item():
    with open('static/upload.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/upload_festival/{f_date}")
async def read_festival_item(f_date : str):
    results = []
    collection = db['festivals']
    
    # check f_date if validate for the date type '%Y-%m-%d'
    if datetime.strptime(f_date, '%Y-%m-%d'):
        _filter = {'date_start': {'$lte': f'{f_date}'}, 'date_end': {'$gte': f'{f_date}'}}
        cursor = collection.find(_filter)
        async for document in cursor:
            item = {}
            f_id, name, start_time, end_time = document['_id'], document['name'], document['start_time'], document['end_time']
            item['id'], item['name'] = f_id, name
            if start_time is None or end_time is None :
                item['op_time'] = "운영시간정보 없음"
            else:
                item['op_time'] = f'{start_time} ~ {end_time}'
            results.append(item)
    return results

@app.get("/upload_fdetail/{f_id}")
async def read_fdetail(f_id: str):
    print(f_id)
    collection = db['festivals']
    if type(f_id) is not int:
        try:
            f_id = int(f_id)
        except Exception as e :
            return JSONResponse(content={"error": str(e)}, status_code=400)
    document = await collection.find_one({'_id' : f_id})
    document['_id'] = str(document['_id'])
    print(document)
    return JSONResponse(content=document)

@app.get("/upload_lmdetail/{class_number}")
async def read_lmdetail(class_number : int):

    collection = db['landmarks']
    if type(class_number) is not int:
        try:
            class_number = int(class_number)
        except Exception as e :
            return JSONResponse(content={"error": str(e)}, status_code=400)

    document = await collection.find_one({'_id': class_number})

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # 이미지 경로 패턴 설정 (예: class_number_1.jpg, class_number_2.jpg 등)
    image_pattern = f"/home/yongjang/datasets/landmark/{document['_id']}_*.jpg"
    
    images_list = []
    for image_path in glob.glob(image_pattern):
        images_list.append(image_path)

    if images_list:
        document['image'] = images_list
    # print(document['image'])

    document['_id'] = str(document['_id'])
    
    return JSONResponse(content=document)
    
# searching 
@app.get("/upload_item/{keyword}")
async def read_item(keyword: str):
    try:
        lm_finder = Finder('landmarks')
        f_finder = Finder('festivals')
        lm_item = await lm_finder.send_result(keyword)
        f_item = await f_finder.send_result(keyword)
        print(f"lm : {lm_item}")
        print(f"lm : {f_item}")
        return JSONResponse(content={"lm_item": lm_item, 
                                     "f_item": f_item})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/fetch_initialItems")
async def initial_item():
    lm_item = {}
    f_items = {}
    landmarks = db['landmarks'] 
    pipeline = [
        {"$unwind": "$wish_list"},
        {"$group": {"_id": "$_id", "name": {"$first": "$name"}, "count": {"$sum": 1}}},
        {"$sort": SON([("count", -1)])},
        {"$limit": 15}
    ]
    async for doc in landmarks.aggregate(pipeline):
        lm_item[doc['_id']] = doc['name']
    f_items = await fetch_initialfItems()
    print(f_items)
    return JSONResponse(content={"lm_item": lm_item, 
                                 "f_item": f_items})


@app.get("/add_wishlist/{class_number}")
async def add_wishlist(class_number : int, email_data: TokenData = Depends(authenticate_token)):
    users = db['users']
    landmarks = db['landmarks']
    email_address = email_data.email
     # 사용자의 위시리스트에서 class_number가 존재하는지 확인
    item = await users.find_one(
        {"email": email_address, "wish_list": {"$elemMatch": {"landmark": class_number}}}
    )
    print(item)
    
    # 만약 class_number가 이미 존재한다면, 이미 추가되었다는 메시지를 반환
    if item:
        print("exist")
        return {"message": "Already add to wish_list"}
        
    else:
        await landmarks.update_one(
            {"_id": class_number},
            {"$push": {"wish_list": email_address}}
        )
        
        await users.update_one(
            {"email": email_address},
            {"$push": {"wish_list.landmark": class_number}}
        )

        return {"message": "wishlist has been updated"}

@app.get("/add_complete/{class_number}")
async def add_complete(class_number : int, email_data: TokenData = Depends(authenticate_token)):
    users = db['users']
    landmarks = db['landmarks']
    email_address = email_data.email
     # 사용자의 위시리스트에서 class_number가 존재하는지 확인
    item = await users.find_one(
        {"email": email_address, "complete_list": {"$elemMatch": {"landmark": class_number}}}
    )
    print(item)
    
    # 만약 class_number가 이미 존재한다면, 이미 추가되었다는 메시지를 반환
    if item:
        print("exist")
        return {"message": "Already add to complete_list"}
        
    else:
        await landmarks.update_one(
            {"_id": class_number},
            {"$push": {"complete_list": email_address}}
        )
        
        await users.update_one(
            {"email": email_address},
            {"$push": {"complete_list.landmark": class_number}}
        )

        return {"message": "complete_list has been updated"}


@app.get("/add_fwishlist/{class_number}")
async def add_wishlist(class_number : int, email_data: TokenData = Depends(authenticate_token)):
    
    users = db['users']
    festivals = db['festivals']
    email_address = email_data.email
     # 사용자의 위시리스트에서 class_number가 존재하는지 확인
    item = await users.find_one(
        {"email": email_address, "wish_list": {"$elemMatch": {"festival": class_number}}}
    )
    print(item)
    
    # 만약 class_number가 이미 존재한다면, 이미 추가되었다는 메시지를 반환
    if item:
        print("exist")
        return {"message": "Already add to wish_list"}
        
    else:
        await festivals.update_one(
            {"_id": class_number},
            {"$push": {"wish_list": email_address}}
        )
        
        await users.update_one(
            {"email": email_address},
            {"$push": {"wish_list.festival": class_number}}
        )

        return {"message": "wishlist has been updated"}



@app.get("/myWishList")
async def show_myWishList(email_data: TokenData = Depends(authenticate_token)):
    lm_wish = {}
    f_wish = {}
    users = db['users']
    landmarks = db['landmarks']
    festivals = db['festivals']
    email_address = email_data.email
    user = await db.users.find_one({"email": email_address})
    wish_lists = user['wish_list']
    for wish_list in wish_lists["landmark"]:
        print(wish_list)
        landmark = await landmarks.find_one({"_id": wish_list})
        class_name = landmark['name']
        lm_wish[wish_list] = class_name
    for wish_list in wish_lists["festival"]:
        festival = await festivals.find_one({"_id": wish_list})
        class_name = festival['name']
        f_wish[wish_list] = class_name
    print(lm_wish)
    print(f_wish)
    return JSONResponse(content={"lm_item": lm_wish, 
                                     "f_item": f_wish})
                                    
@app.get("/myCompleteList")
async def show_myWishList(email_data: TokenData = Depends(authenticate_token)):
    lm_complete = {}
    f_complete = {}
    users = db['users']
    landmarks = db['landmarks']
    festivals = db['festivals']
    email_address = email_data.email
    user = await db.users.find_one({"email": email_address})
    complete_lists = user['complete_list']
    for complete_list in complete_lists["landmark"]:
        landmark = await landmarks.find_one({"_id":complete_list})
        class_name = landmark['name']
        lm_complete[complete_list] = class_name
    for complete_list in complete_lists["festival"]:
        festival = await festivals.find_one({"_id": complete_list})
        class_name = festival['name']
        f_complete[complete_list] = class_name
    print(lm_complete) 
    print(f_complete)
    return JSONResponse(content={"lm_item": lm_complete, 
                                     "f_item": f_complete})

@app.post("/removeMyWishLM/{id}")
async def show_myWishList(id : int, email_data: TokenData = Depends(authenticate_token)):
    users = db['users']
    landmarks = db['landmarks']
    email_address = email_data.email
    await landmarks.update_one(
        {"_id": id},
        {"$pull": {"wish_list": email_address }}
    )
    await users.update_one(
        {"email": email_address},
        {"$pull": {"wish_list.landmark": id }}
    )
    return "remove it"

@app.post("/removeMyCompleteLM/{id}")
async def show_myCompleteList(id : int, email_data: TokenData = Depends(authenticate_token)):
    users = db['users']
    landmarks = db['landmarks']
    email_address = email_data.email
    await landmarks.update_one(
        {"_id": id},
        {"$pull": {"complete_list": email_address }}
    )
    await users.update_one(
        {"email": email_address},
        {"$pull": {"complete_list.landmark": id }}
    )
    return "remove it"

@app.post("/removeMyWishFestival/{id}")
async def show_myWishList(id : int, email_data: TokenData = Depends(authenticate_token)):
    users = db['users']
    email_address = email_data.email
    await users.update_one(
        {"email": email_address},
        {"$pull": {"wishl_ist.festival": id }}
    )
    return "remove it"


###### 로그인 기능 구현 부분 ######

with open(env_path) as f:
    env = yaml.load(f, Loader=yaml.FullLoader)

## bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

mongo_client = AsyncIOMotorClient(env['mongo']['uri'])
db = mongo_client['example']
collection = db['users']
test_collection = db['landmarks']



# JWT 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
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
            "complete_list" : { "landmark": [],
                                "festival" : []},
            "wish_list" : { "landmark": [],
                                "festival" : []},
        }
        await collection.insert_one(user_data)
    except Exception as e:
        print("Email already registered")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    return {"email": user.email}

# @app.get("/signup", response_class=HTMLResponse)
# async def get_signup_page():
#     with open('static/signup.html', 'r') as f:
#         html_content = f.read()
#     return HTMLResponse(content=html_content, status_code=200)

# @app.get("/login", response_class=HTMLResponse)
# async def get_login_page():
#     with open('static/login.html', 'r') as f:
#         html_content = f.read()
#     return HTMLResponse(content=html_content, status_code=200)

@app.get("/home/yongjang/datasets/landmark/{filename}")
async def get_image(filename: str):
    file_path = f"/home/yongjang/datasets/landmark/{filename}" # 실제 이미지 파일 경로로 변경해주세요.
    replace_path = f"/home/yongjang/datasets/landmark/replace.jpg"
    
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    else:
        return FileResponse(replace_path, media_type="image/jpeg")

@app.get("/home/yongjang/datasets/festival/{filename}")
async def get_image(filename: str):
    file_path = f"/home/yongjang/datasets/festival/{filename}" # 실제 이미지 파일 경로로 변경해주세요.
    replace_path = f"/home/yongjang/datasets/landmark/replace.jpg"
    
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    else:
        return FileResponse(replace_path, media_type="image/jpeg")

@app.post("/update_profile")
async def update_profile(old_password: str = Form(...), new_password: str = Form(...), email_data: TokenData = Depends(authenticate_token)):
    users = db['users']
    # 데이터베이스에서 사용자의 해싱된 현재 비밀번호를 가져옴
    user_data = await users.find_one({"email": email_data.email})
    if not user_data:
        print("User not found")
        return "User not found"
    
    current_hashed_pwd = user_data["hashed_password"]
    print(current_hashed_pwd)
    try:
        # 사용자로부터 입력받은 현재 비밀번호와 데이터베이스의 해싱된 비밀번호를 비교
        if verify_password(old_password, current_hashed_pwd):
            # 새 비밀번호를 해싱하고 데이터베이스에 업데이트
            new_pwd_hashed = get_password_hash(new_password)
            print(new_pwd_hashed)
            users.update_one({"email": email_data.email}, {"$set": {"hashed_password": new_pwd_hashed}})
            print("Password updated successfully")
            return "Password updated successfully"
    except Exception as e:
        print(e)
        print("Error occurred during password update")
        return "Error occurred during password update"

    else:
        print("Incorrect current password")
        return "Incorrect current password"

# # https://python.langchain.com/docs/get_started/quickstart
# @app.get("/ask/{q}", response_class=HTMLResponse)
# async def ask_gpt(q: str):
#     try:
#         print(f'Question : {q}')
#         gpt_answer = gpt.get_answer(prompt=q)
        
#         return JSONResponse(content={"response": gpt_answer})
#     except Exception as e:
#         print(e)
#         return JSONResponse(content={"error": str(e)}, status_code=400)
