# FastAPI从零基础到精通完整教程

## 目录
1. [环境准备](#环境准备)
2. [FastAPI基础](#fastapi基础)
3. [路由与请求处理](#路由与请求处理)
4. [数据验证与Pydantic](#数据验证与pydantic)
5. [SQLAlchemy ORM详解](#sqlalchemy-orm详解)
6. [数据库集成](#数据库集成)
7. [依赖注入系统](#依赖注入系统)
8. [身份验证与安全](#身份验证与安全)
9. [中间件与CORS](#中间件与cors)
10. [文件上传与处理](#文件上传与处理)
11. [后台任务](#后台任务)
12. [WebSocket实时通信](#websocket实时通信)
13. [测试](#测试)
14. [部署](#部署)
15. [最佳实践](#最佳实践)

---

## 环境准备

### 安装Python
确保安装Python 3.8或更高版本：
```bash
python --version
```

### 创建虚拟环境
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 安装依赖
```bash
pip install fastapi
pip install "uvicorn[standard]"
pip install sqlalchemy
pip install python-multipart  # 文件上传
pip install python-jose[cryptography]  # JWT
pip install passlib[bcrypt]  # 密码加密
pip install alembic  # 数据库迁移
pip install pytest  # 测试
pip install httpx  # 测试客户端
```

---

## FastAPI基础

### 第一个应用

创建 `main.py`：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

运行应用：
```bash
uvicorn main:app --reload
```

访问：
- API: http://127.0.0.1:8000
- 交互式文档: http://127.0.0.1:8000/docs
- 备用文档: http://127.0.0.1:8000/redoc

### FastAPI核心概念

**同步vs异步**：
```python
# 同步函数 - 用于CPU密集型操作
@app.get("/sync")
def sync_endpoint():
    return {"type": "sync"}

# 异步函数 - 用于I/O操作（数据库、API调用）
@app.get("/async")
async def async_endpoint():
    return {"type": "async"}
```

---

## 路由与请求处理

### 路径参数

```python
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return {"model_name": model_name, "message": "Deep Learning model"}

# 路径参数验证
from fastapi import Path

@app.get("/items/{item_id}")
async def read_item(
    item_id: int = Path(..., title="Item ID", ge=1, le=1000)
):
    return {"item_id": item_id}
```

### 查询参数

```python
from typing import Optional

@app.get("/items/")
async def read_items(
    skip: int = 0,
    limit: int = 10,
    q: Optional[str] = None
):
    results = {"skip": skip, "limit": limit}
    if q:
        results["q"] = q
    return results

# 必需查询参数
@app.get("/items/required")
async def read_items_required(q: str):
    return {"q": q}
```

### 请求体

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

### 多个参数混合

```python
@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    item: Item,
    q: Optional[str] = None
):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result
```

---

## 数据验证与Pydantic

### Pydantic模型详解

```python
from pydantic import BaseModel, Field, validator, EmailStr
from typing import List, Optional
from datetime import datetime

class User(BaseModel):
    id: Optional[int] = None
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    age: int = Field(..., ge=0, le=150)
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = []
    
    # 自定义验证器
    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'must be alphanumeric'
        return v
    
    # 配置
    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "age": 30,
                "tags": ["developer", "python"]
            }
        }

@app.post("/users/", response_model=User)
async def create_user(user: User):
    return user
```

### 响应模型

```python
class UserIn(BaseModel):
    username: str
    password: str
    email: EmailStr

class UserOut(BaseModel):
    username: str
    email: EmailStr

@app.post("/users/", response_model=UserOut)
async def create_user(user: UserIn):
    # 密码不会被返回
    return user
```

### 嵌套模型

```python
class Image(BaseModel):
    url: str
    name: str

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    images: Optional[List[Image]] = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

---

## SQLAlchemy ORM详解

### SQLAlchemy基础架构

SQLAlchemy分为两层：
1. **Core**: 底层SQL抽象
2. **ORM**: 对象关系映射

### 数据库配置

创建 `database.py`：

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLite数据库
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
# PostgreSQL示例
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/dbname"
# MySQL示例
# SQLALCHEMY_DATABASE_URL = "mysql+pymysql://user:password@localhost/dbname"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}  # 仅SQLite需要
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
```

### 定义模型

创建 `models.py`：

```python
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime, Table
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

# 多对多关系中间表
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('role_id', Integer, ForeignKey('roles.id'))
)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 一对多关系
    items = relationship("Item", back_populates="owner", cascade="all, delete-orphan")
    
    # 多对多关系
    roles = relationship("Role", secondary=user_roles, back_populates="users")

class Role(Base):
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(200))
    
    users = relationship("User", secondary=user_roles, back_populates="roles")

class Item(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), index=True)
    description = Column(String(500))
    price = Column(Float, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 多对一关系
    owner = relationship("User", back_populates="items")
```

### 创建表

```python
from database import engine
import models

# 创建所有表
models.Base.metadata.create_all(bind=engine)
```

---

## 数据库集成

### CRUD操作

创建 `crud.py`：

```python
from sqlalchemy.orm import Session
from typing import List, Optional
import models, schemas
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 用户CRUD
def get_user(db: Session, user_id: int) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[models.User]:
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    hashed_password = pwd_context.hash(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user: schemas.UserUpdate) -> Optional[models.User]:
    db_user = get_user(db, user_id)
    if db_user:
        for key, value in user.dict(exclude_unset=True).items():
            setattr(db_user, key, value)
        db.commit()
        db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int) -> bool:
    db_user = get_user(db, user_id)
    if db_user:
        db.delete(db_user)
        db.commit()
        return True
    return False

# 商品CRUD
def get_items(db: Session, skip: int = 0, limit: int = 100) -> List[models.Item]:
    return db.query(models.Item).offset(skip).limit(limit).all()

def create_user_item(db: Session, item: schemas.ItemCreate, user_id: int) -> models.Item:
    db_item = models.Item(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

# 复杂查询示例
def search_items(
    db: Session,
    keyword: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    skip: int = 0,
    limit: int = 100
) -> List[models.Item]:
    query = db.query(models.Item)
    
    if keyword:
        query = query.filter(
            models.Item.title.contains(keyword) | 
            models.Item.description.contains(keyword)
        )
    if min_price is not None:
        query = query.filter(models.Item.price >= min_price)
    if max_price is not None:
        query = query.filter(models.Item.price <= max_price)
    
    return query.offset(skip).limit(limit).all()
```

### Schemas定义

创建 `schemas.py`：

```python
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

# Item schemas
class ItemBase(BaseModel):
    title: str
    description: Optional[str] = None
    price: float

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int
    owner_id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

# User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    items: List[Item] = []
    
    class Config:
        orm_mode = True
```

---

## 依赖注入系统

### 数据库会话依赖

```python
from fastapi import Depends
from sqlalchemy.orm import Session
from database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users
```

### 通用参数依赖

```python
from typing import Optional

class CommonQueryParams:
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        q: Optional[str] = None
    ):
        self.skip = skip
        self.limit = limit
        self.q = q

@app.get("/items/")
async def read_items(commons: CommonQueryParams = Depends()):
    return {
        "skip": commons.skip,
        "limit": commons.limit,
        "q": commons.q
    }
```

### 子依赖

```python
def verify_token(token: str = Header(...)):
    if token != "secret-token":
        raise HTTPException(status_code=400, detail="Invalid token")
    return token

def verify_key(key: str = Header(...)):
    if key != "secret-key":
        raise HTTPException(status_code=400, detail="Invalid key")
    return key

def get_current_user(
    token: str = Depends(verify_token),
    key: str = Depends(verify_key)
):
    return {"token": token, "key": key}

@app.get("/protected/")
async def protected_route(user: dict = Depends(get_current_user)):
    return user
```

---

## 身份验证与安全

### JWT令牌认证

创建 `auth.py`：

```python
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

SECRET_KEY = "your-secret-key-keep-it-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(db: Session, username: str, password: str):
    user = crud.get_user_by_email(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = crud.get_user_by_email(db, email=username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: models.User = Depends(get_current_user)
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
```

### 登录端点

```python
from fastapi.security import OAuth2PasswordRequestForm

@app.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
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
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(get_current_active_user)):
    return current_user
```

---

## 中间件与CORS

### CORS配置

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 自定义中间件

```python
from fastapi import Request
import time

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

---

## 文件上传与处理

```python
from fastapi import File, UploadFile
from typing import List
import shutil
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "size": file_path.stat().st_size}

@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    filenames = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        filenames.append(file.filename)
    return {"filenames": filenames}

# 图片处理示例
from PIL import Image
import io

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # 创建缩略图
    image.thumbnail((200, 200))
    
    # 保存
    thumbnail_path = UPLOAD_DIR / f"thumb_{file.filename}"
    image.save(thumbnail_path)
    
    return {"original": file.filename, "thumbnail": f"thumb_{file.filename}"}
```

---

## 后台任务

```python
from fastapi import BackgroundTasks
import time

def write_log(message: str):
    time.sleep(5)  # 模拟耗时操作
    with open("log.txt", "a") as log:
        log.write(f"{message}\n")

@app.post("/send-notification/{email}")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(write_log, f"Notification sent to {email}")
    return {"message": "Notification sent in the background"}

# 多个后台任务
@app.post("/process/")
async def process_item(
    item: schemas.ItemCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    background_tasks.add_task(write_log, f"Processing item: {item.title}")
    background_tasks.add_task(write_log, f"Sending email notification")
    
    db_item = crud.create_item(db, item)
    return db_item
```

---

## WebSocket实时通信

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
            await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")
```

---

## 测试

创建 `test_main.py`：

```python
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base
from main import app, get_db

# 测试数据库
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_user():
    response = client.post(
        "/users/",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass123"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data

def test_read_users():
    response = client.get("/users/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_login():
    response = client.post(
        "/token",
        data={"username": "test@example.com", "password": "testpass123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

# 运行测试
# pytest test_main.py -v
```

---

## 部署

### 使用Docker

创建 `Dockerfile`：

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/dbname
    depends_on:
      - db
    volumes:
      - .:/app
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=dbname
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

运行：
```bash
docker-compose up --build
```

### 使用Gunicorn

```bash
pip install gunicorn

gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 最佳实践

### 项目结构

```
myapp/
├── alembic/              # 数据库迁移
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI应用入口
│   ├── config.py         # 配置
│   ├── database.py       # 数据库连接
│   ├── models/           # SQLAlchemy模型
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   ├── schemas/          # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   ├── crud/             # CRUD操作
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   ├── api/              # 路由
│   │   ├── __init__.py
│   │   ├── deps.py       # 依赖
│   │   └── endpoints/
│   │       ├── users.py
│   │       └── items.py
│   └── core/             # 核心功能
│       ├── __init__.py
│       ├── security.py
│       └── config.py
├── tests/
├── .env
├── requirements.txt
└── README.md
```

### 配置管理

创建 `config.py`：

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My API"
    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()
```

创建 `.env`：

```
DATABASE_URL=postgresql://user:password@localhost/dbname
SECRET_KEY=your-secret-key-here
```

### 错误处理

```python
from fastapi import HTTPException
from fastapi.responses import JSONResponse

class CustomException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(CustomException)
async def custom_exception_handler(request, exc: CustomException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something wrong"}
    )

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )
```

### 日志配置

```python
import logging
from logging.handlers import RotatingFileHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("app.log", maxBytes=10000000, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.get("/items/{item_id}")
async def read_item(item_id: int, db: Session = Depends(get_db)):
    logger.info(f"Fetching item with id: {item_id}")
    item = crud.get_item(db, item_id)
    if not item:
        logger.warning(f"Item {item_id} not found")
        raise HTTPException(status_code=404, detail="Item not found")
    return item
```

### 数据库迁移 (Alembic)

初始化Alembic：

```bash
alembic init alembic
```

配置 `alembic.ini`：

```ini
sqlalchemy.url = postgresql://user:password@localhost/dbname
```

修改 `alembic/env.py`：

```python
from app.database import Base
from app.models import user, item  # 导入所有模型

target_metadata = Base.metadata
```

创建迁移：

```bash
# 自动生成迁移
alembic revision --autogenerate -m "Create users and items tables"

# 应用迁移
alembic upgrade head

# 回滚迁移
alembic downgrade -1
```

### 分页工具

创建 `pagination.py`：

```python
from typing import Generic, TypeVar, List
from pydantic import BaseModel
from math import ceil

T = TypeVar('T')

class Page(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    size: int
    pages: int

def paginate(items: List[T], page: int = 1, size: int = 10) -> Page[T]:
    total = len(items)
    pages = ceil(total / size)
    start = (page - 1) * size
    end = start + size
    
    return Page(
        items=items[start:end],
        total=total,
        page=page,
        size=size,
        pages=pages
    )

# 使用示例
@app.get("/items/", response_model=Page[schemas.Item])
def read_items(
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db)
):
    items = crud.get_items(db, skip=0, limit=1000)
    return paginate(items, page, size)
```

### 缓存策略

```python
from functools import lru_cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis

# 启动时初始化缓存
@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

# 使用缓存装饰器
@app.get("/cached-items/")
@cache(expire=60)  # 缓存60秒
async def get_cached_items(db: Session = Depends(get_db)):
    return crud.get_items(db)

# 简单内存缓存
@lru_cache(maxsize=128)
def get_config_value(key: str):
    # 读取配置的耗时操作
    return settings.dict()[key]
```

### API版本控制

```python
from fastapi import APIRouter

# V1 API
v1_router = APIRouter(prefix="/api/v1")

@v1_router.get("/users/")
def get_users_v1():
    return {"version": "v1", "users": []}

# V2 API
v2_router = APIRouter(prefix="/api/v2")

@v2_router.get("/users/")
def get_users_v2():
    return {"version": "v2", "users": [], "metadata": {}}

# 注册路由
app.include_router(v1_router)
app.include_router(v2_router)
```

### 限流（Rate Limiting）

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/limited/")
@limiter.limit("5/minute")
async def limited_route(request: Request):
    return {"message": "This endpoint is rate limited"}
```

### 健康检查端点

```python
from sqlalchemy import text

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        # 检查数据库连接
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ok" if db_status == "healthy" else "error",
        "database": db_status,
        "version": "1.0.0"
    }
```

---

## 高级特性

### 事件处理器

```python
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    # 初始化数据库连接池
    # 加载机器学习模型
    # 预热缓存

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    # 关闭数据库连接
    # 保存状态
```

### 自定义响应类

```python
from fastapi.responses import HTMLResponse, StreamingResponse
import io

@app.get("/html", response_class=HTMLResponse)
async def get_html():
    return """
    <html>
        <head><title>FastAPI</title></head>
        <body><h1>Hello from FastAPI!</h1></body>
    </html>
    """

@app.get("/download")
async def download_file():
    def generate():
        for i in range(100):
            yield f"Line {i}\n".encode()
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=data.txt"}
    )
```

### GraphQL集成

```python
from strawberry.fastapi import GraphQLRouter
import strawberry

@strawberry.type
class User:
    id: int
    name: str
    email: str

@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: int) -> User:
        # 从数据库获取用户
        return User(id=id, name="John", email="john@example.com")

schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)

app.include_router(graphql_app, prefix="/graphql")
```

### 任务队列（Celery）

```python
from celery import Celery

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery_app.task
def send_email_task(email: str, message: str):
    # 发送邮件的耗时操作
    time.sleep(5)
    return f"Email sent to {email}"

@app.post("/send-email/")
async def send_email(email: str, message: str):
    task = send_email_task.delay(email, message)
    return {"task_id": task.id, "status": "processing"}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": task.status, "result": task.result}
```

### 数据导出

```python
import csv
from io import StringIO

@app.get("/export/users")
async def export_users(db: Session = Depends(get_db)):
    users = crud.get_users(db)
    
    # 创建CSV
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Username", "Email", "Created At"])
    
    for user in users:
        writer.writerow([user.id, user.username, user.email, user.created_at])
    
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=users.csv"}
    )
```

### 定时任务

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

def cleanup_old_data():
    logger.info("Running cleanup task...")
    # 清理逻辑

@app.on_event("startup")
async def start_scheduler():
    scheduler.add_job(cleanup_old_data, "cron", hour=2, minute=0)  # 每天凌晨2点
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_scheduler():
    scheduler.shutdown()
```

---

## 性能优化

### 数据库查询优化

```python
from sqlalchemy.orm import joinedload, selectinload

# 预加载关联对象，避免N+1查询问题
@app.get("/users-with-items/")
def get_users_with_items(db: Session = Depends(get_db)):
    users = db.query(models.User).options(
        selectinload(models.User.items)
    ).all()
    return users

# 使用索引
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)  # 索引
    username = Column(String, index=True)  # 索引
    
    # 复合索引
    __table_args__ = (
        Index('idx_username_email', 'username', 'email'),
    )
```

### 异步数据库操作

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql+asyncpg://user:password@localhost/dbname"

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_async_db():
    async with async_session() as session:
        yield session

@app.get("/async-users/")
async def get_async_users(db: AsyncSession = Depends(get_async_db)):
    from sqlalchemy import select
    result = await db.execute(select(models.User))
    users = result.scalars().all()
    return users
```

### 响应压缩

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

---

## 监控与日志

### Prometheus集成

```python
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# 访问 /metrics 查看指标
```

### Sentry错误追踪

```python
import sentry_sdk

sentry_sdk.init(
    dsn="your-sentry-dsn",
    traces_sample_rate=1.0,
)
```

### 结构化日志

```python
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    logger.info("item_requested", item_id=item_id)
    return {"item_id": item_id}
```

---

## 安全最佳实践

### 输入验证和清理

```python
from pydantic import validator, constr

class UserCreate(BaseModel):
    username: constr(min_length=3, max_length=50, regex="^[a-zA-Z0-9_]+$")
    email: EmailStr
    password: constr(min_length=8)
    
    @validator('password')
    def password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v
```

### SQL注入防护

```python
# ✅ 正确：使用参数化查询
def get_user_safe(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

# ❌ 错误：拼接SQL字符串
def get_user_unsafe(db: Session, username: str):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return db.execute(query)  # 容易受到SQL注入攻击
```

### HTTPS和安全头

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

# 只允许特定主机
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"]
)

# 强制HTTPS
app.add_middleware(HTTPSRedirectMiddleware)

# 添加安全头
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

---

## 完整示例项目

以下是一个完整的博客API示例，整合了所有知识点：

```python
# main.py - 完整博客API示例
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
import models, schemas, crud, auth
from database import engine, SessionLocal

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Blog API",
    description="A complete blog API with authentication",
    version="1.0.0"
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 认证端点
@app.post("/register", response_model=schemas.User)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)

@app.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# 用户端点
@app.get("/users/me", response_model=schemas.User)
async def read_users_me(
    current_user: models.User = Depends(auth.get_current_active_user)
):
    return current_user

@app.get("/users/", response_model=List[schemas.User])
def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

# 文章端点
@app.post("/posts/", response_model=schemas.Post)
def create_post(
    post: schemas.PostCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    return crud.create_post(db=db, post=post, user_id=current_user.id)

@app.get("/posts/", response_model=List[schemas.Post])
def read_posts(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    posts = crud.get_posts(db, skip=skip, limit=limit)
    return posts

@app.get("/posts/{post_id}", response_model=schemas.Post)
def read_post(post_id: int, db: Session = Depends(get_db)):
    db_post = crud.get_post(db, post_id=post_id)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return db_post

@app.put("/posts/{post_id}", response_model=schemas.Post)
def update_post(
    post_id: int,
    post: schemas.PostUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    db_post = crud.get_post(db, post_id=post_id)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    if db_post.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return crud.update_post(db=db, post_id=post_id, post=post)

@app.delete("/posts/{post_id}")
def delete_post(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    db_post = crud.get_post(db, post_id=post_id)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    if db_post.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    crud.delete_post(db=db, post_id=post_id)
    return {"message": "Post deleted successfully"}

# 评论端点
@app.post("/posts/{post_id}/comments/", response_model=schemas.Comment)
def create_comment(
    post_id: int,
    comment: schemas.CommentCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    return crud.create_comment(
        db=db,
        comment=comment,
        post_id=post_id,
        user_id=current_user.id
    )

@app.get("/posts/{post_id}/comments/", response_model=List[schemas.Comment])
def read_comments(post_id: int, db: Session = Depends(get_db)):
    return crud.get_comments(db, post_id=post_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 学习路径建议

### 初级阶段（1-2周）
1. 掌握FastAPI基本路由和请求处理
2. 理解Pydantic数据验证
3. 学习基本的CRUD操作
4. 熟悉依赖注入系统

### 中级阶段（2-4周）
1. 深入学习SQLAlchemy ORM
2. 实现完整的身份验证系统
3. 掌握数据库关系和查询优化
4. 学习中间件和错误处理

### 高级阶段（4-8周）
1. 异步编程和性能优化
2. WebSocket实时通信
3. 微服务架构设计
4. 部署和运维实践

### 实战项目建议
1. **博客系统**：用户、文章、评论、标签
2. **电商API**：商品、订单、购物车、支付
3. **社交媒体**：用户关系、动态、消息、通知
4. **任务管理**：项目、任务、团队协作

---

## 常见问题与解决方案

### 1. 数据库连接池耗尽
```python
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True
)
```

### 2. CORS跨域问题
确保正确配置CORS中间件并包含所需的headers。

### 3. 异步与同步混用
避免在async函数中使用同步数据库操作，使用异步驱动或run_in_executor。

### 4. 内存泄漏
确保正确关闭数据库会话，使用依赖注入管理资源生命周期。

---

## 总结

这份教程涵盖了FastAPI和SQLAlchemy的核心概念和实践技巧。持续学习和实践是掌握这些技术的关键。建议：

1. 阅读官方文档：[FastAPI文档](https://fastapi.tiangolo.com/) 和 [SQLAlchemy文档](https://docs.sqlalchemy.org/)
2. 通过实际项目练习
3. 关注最佳实践和安全问题
4. 参与开源项目贡献

祝你学习顺利！