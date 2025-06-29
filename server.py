from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timedelta
import jwt
import hashlib
from enum import Enum
import ssl
from certifi import where


 # Load environment variables first
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Initialize FastAPI app
app = FastAPI()

# Root route
@app.get("/")
def root():
    return {"message": "Backend is working fine ✅"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# MongoDB connection
mongo_url = os.environ.get('MONGODB_URI')
if not mongo_url:
    logging.error("MONGODB_URI environment variable is not set")
    raise RuntimeError("MongoDB connection string is missing")

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
ssl_context.verify_mode = ssl.CERT_REQUIRED
ssl_context.load_verify_locations(where())

# Create client with SSL context
client = AsyncIOMotorClient(
    mongo_url,
    tls=True,
    tlsAllowInvalidCertificates=False,
    ssl=ssl_context,
    tlsInsecure=False,
    connectTimeoutMS=30000,
    socketTimeoutMS=30000
)

client = AsyncIOMotorClient(mongo_url)
db_name = os.environ.get('DB_NAME', 'telecom-prod')
db = client[db_name]

# Create a router with /api prefix
api_router = APIRouter(prefix="/api")

# JWT Configuration - Use environment variable with fallback
SECRET_KEY = os.environ.get('SECRET_KEY', 'fallback_secret_key_please_change')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security
security = HTTPBearer()


# Enums
class UserRole(str, Enum):
    DISTRIBUTOR = "distributor"
    AGENT = "agent"
    RETAILER = "retailer"

class ProductType(str, Enum):
    SIM = "SIM"
    MOBILE = "Mobile"
    FIBER = "Fiber"

class ProductStatus(str, Enum):
    CREATED = "Created"
    REQUESTED = "Requested"
    APPROVED = "Approved"
    ASSIGNED = "Assigned"

class RequestStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FULFILLED = "fulfilled"

class TransactionType(str, Enum):
    DISTRIBUTOR_TO_AGENT = "distributor_to_agent"
    AGENT_TO_RETAILER = "agent_to_retailer"
    RETAILER_REQUEST = "retailer_request"
    AGENT_REQUEST = "agent_request"

# Models
class UserBase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    phone: str
    role: UserRole
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class UserCreate(BaseModel):
    email: str
    name: str
    phone: str
    password: str
    role: UserRole

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    phone: str
    role: UserRole
    created_at: datetime
    is_active: bool

class StockItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    product_type: ProductType
    quantity: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    last_reset: datetime = Field(default_factory=datetime.utcnow)

class StockUpdate(BaseModel):
    agent_id: str
    product_type: ProductType
    quantity_change: int
    transaction_type: TransactionType
    order_id: str
    notes: Optional[str] = None

class ProductBase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ProductType
    code: str
    serial_number: str
    price: float
    status: ProductStatus = ProductStatus.CREATED
    created_by: str  # distributor id
    created_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_to: Optional[str] = None  # agent or retailer id
    assigned_at: Optional[datetime] = None
    order_id: Optional[str] = None

class ProductCreate(BaseModel):
    type: ProductType
    code: str
    serial_number: str
    price: float

class ProductRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    requester_id: str  # agent or retailer id
    requester_role: UserRole
    target_id: str  # distributor or agent id (who should fulfill the request)
    product_type: ProductType
    quantity: int
    reason: str
    status: RequestStatus = RequestStatus.PENDING
    order_id: str = Field(default_factory=lambda: f"ORD-{uuid.uuid4().hex[:8].upper()}")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    fulfilled_at: Optional[datetime] = None

class ProductRequestCreate(BaseModel):
    product_type: ProductType
    quantity: int
    reason: str
    target_id: Optional[str] = None  # For retailer requests to specific agents

class StockTransaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str
    transaction_type: TransactionType
    from_user_id: str
    to_user_id: str
    product_type: ProductType
    quantity: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# Utility Functions
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed_password: str) -> bool:
    return hash_password(password) == hashed_password

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def generate_order_id() -> str:
    return f"ORD-{uuid.uuid4().hex[:8].upper()}"

async def get_or_create_stock(agent_id: str, product_type: ProductType):
    """Get or create stock entry for an agent"""
    stock = await db.stock.find_one({"agent_id": agent_id, "product_type": product_type})
    if not stock:
        stock_item = StockItem(agent_id=agent_id, product_type=product_type)
        await db.stock.insert_one(stock_item.dict())
        return stock_item
    return StockItem(**stock)

async def update_stock(agent_id: str, product_type: ProductType, quantity_change: int, transaction_type: TransactionType, order_id: str, notes: str = None):
    """Update agent stock and record transaction"""
    # Get current stock
    stock = await get_or_create_stock(agent_id, product_type)
    
    # Update quantity
    new_quantity = max(0, stock.quantity + quantity_change)  # Don't allow negative stock
    
    # Update stock in database
    await db.stock.update_one(
        {"agent_id": agent_id, "product_type": product_type},
        {
            "$set": {
                "quantity": new_quantity,
                "last_updated": datetime.utcnow()
            }
        }
    )
    
    # Record transaction
    transaction = StockTransaction(
        order_id=order_id,
        transaction_type=transaction_type,
        from_user_id=agent_id if quantity_change < 0 else "system",
        to_user_id=agent_id if quantity_change > 0 else "system",
        product_type=product_type,
        quantity=abs(quantity_change),
        notes=notes
    )
    await db.stock_transactions.insert_one(transaction.dict())
    
    return new_quantity

async def reset_all_stock():
    """Reset all agent stock to 0"""
    await db.stock.update_many(
        {},
        {
            "$set": {
                "quantity": 0,
                "last_reset": datetime.utcnow(),
                "last_updated": datetime.utcnow()
            }
        }
    )

async def check_monthly_reset():
    """Check if monthly reset is needed"""
    now = datetime.utcnow()
    if now.day == 1:
        # Check if we've already reset this month
        last_reset = await db.system_settings.find_one({"key": "last_monthly_reset"})
        if not last_reset or last_reset["value"].month != now.month:
            await reset_all_stock()
            await db.system_settings.update_one(
                {"key": "last_monthly_reset"},
                {"$set": {"value": now}},
                upsert=True
            )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = await db.users.find_one({"id": user_id})
    if user is None:
        raise credentials_exception
    return UserResponse(**user)

def require_role(allowed_roles: List[UserRole]):
    def role_checker(current_user: UserResponse = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user
    return role_checker

# Authentication Routes
@api_router.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if this is the first user (make them distributor)
    user_count = await db.users.count_documents({})
    if user_count == 0:
        # First user becomes distributor automatically
        user_data.role = UserRole.DISTRIBUTOR
    else:
        # Only allow agent/retailer registration for public
        if user_data.role == UserRole.DISTRIBUTOR:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Distributor accounts can only be created by existing distributors"
            )
    
    # Create user
    user_dict = user_data.dict()
    user_dict["password"] = hash_password(user_data.password)
    user_obj = UserBase(**{k: v for k, v in user_dict.items() if k != "password"})
    
    # Save to database
    await db.users.insert_one({**user_obj.dict(), "password": user_dict["password"]})
    
    # Create token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_obj.id}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(**user_obj.dict())
    )

# Add user creation endpoint for distributors
@api_router.post("/users/create", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: UserResponse = Depends(require_role([UserRole.DISTRIBUTOR]))
):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user_dict = user_data.dict()
    user_dict["password"] = hash_password(user_data.password)
    user_obj = UserBase(**{k: v for k, v in user_dict.items() if k != "password"})
    
    # Save to database
    await db.users.insert_one({**user_obj.dict(), "password": user_dict["password"]})
    
    return UserResponse(**user_obj.dict())

@api_router.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    user = await db.users.find_one({"email": user_credentials.email})
    if not user or not verify_password(user_credentials.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["id"]}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(**user)
    )

# Product Routes
@api_router.post("/products", response_model=ProductBase)
async def create_product(
    product_data: ProductCreate,
    current_user: UserResponse = Depends(require_role([UserRole.DISTRIBUTOR]))
):
    product_dict = product_data.dict()
    product_dict["created_by"] = current_user.id
    product_obj = ProductBase(**product_dict)
    
    await db.products.insert_one(product_obj.dict())
    return product_obj

@api_router.get("/products", response_model=List[ProductBase])
async def get_products(current_user: UserResponse = Depends(get_current_user)):
    if current_user.role == UserRole.DISTRIBUTOR:
        products = await db.products.find({"created_by": current_user.id}).to_list(1000)
    elif current_user.role == UserRole.AGENT:
        products = await db.products.find({"assigned_to": current_user.id}).to_list(1000)
    else:  # RETAILER
        products = await db.products.find({"assigned_to": current_user.id}).to_list(1000)
    
    return [ProductBase(**product) for product in products]

@api_router.get("/products/available", response_model=List[ProductBase])
async def get_available_products(current_user: UserResponse = Depends(require_role([UserRole.DISTRIBUTOR]))):
    products = await db.products.find({
        "created_by": current_user.id,
        "status": ProductStatus.CREATED
    }).to_list(1000)
    return [ProductBase(**product) for product in products]

# Monthly Tracking Models
class MonthlyReport(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    month: int
    year: int
    distributor_id: str
    product_type: ProductType
    total_approved: int = 0
    total_fulfilled: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)

# Product Request Routes
@api_router.post("/product-requests", response_model=ProductRequest)
async def create_product_request(
    request_data: ProductRequestCreate,
    current_user: UserResponse = Depends(get_current_user)
):
    # Check if the requested product type is available from distributor
    if current_user.role == UserRole.AGENT:
        # Agent requesting from distributor - check if distributor has created this product type
        target_user = await db.users.find_one({"role": UserRole.DISTRIBUTOR})
        if not target_user:
            raise HTTPException(status_code=404, detail="No distributor found")
        target_id = target_user["id"]
        
        # Check if distributor has created products of this type
        available_products = await db.products.count_documents({
            "created_by": target_id,
            "type": request_data.product_type,
            "status": ProductStatus.CREATED
        })
        if available_products == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"No {request_data.product_type} products available from distributor"
            )
    
    elif current_user.role == UserRole.RETAILER:
        # Retailer requesting from agent
        if not request_data.target_id:
            raise HTTPException(status_code=400, detail="Retailer must specify target agent")
        target_user = await db.users.find_one({"id": request_data.target_id, "role": UserRole.AGENT})
        if not target_user:
            raise HTTPException(status_code=404, detail="Target agent not found")
        target_id = request_data.target_id
        
        # Check if agent has stock of this product type
        agent_stock = await db.stock.find_one({
            "agent_id": target_id,
            "product_type": request_data.product_type
        })
        if not agent_stock or agent_stock["quantity"] < request_data.quantity:
            raise HTTPException(
                status_code=400, 
                detail=f"Agent doesn't have enough {request_data.product_type} stock. Available: {agent_stock['quantity'] if agent_stock else 0}"
            )
    
    else:
        raise HTTPException(status_code=403, detail="Only agents and retailers can create requests")
    
    # Check monthly reset
    await check_monthly_reset()
    
    request_dict = request_data.dict()
    request_dict["requester_id"] = current_user.id
    request_dict["requester_role"] = current_user.role
    request_dict["target_id"] = target_id
    request_obj = ProductRequest(**request_dict)
    
    await db.product_requests.insert_one(request_obj.dict())
    return request_obj

@api_router.get("/product-requests", response_model=List[ProductRequest])
async def get_product_requests(current_user: UserResponse = Depends(get_current_user)):
    if current_user.role == UserRole.DISTRIBUTOR:
        # Distributors see all agent requests to them
        requests = await db.product_requests.find({
            "target_id": current_user.id,
            "requester_role": UserRole.AGENT
        }).to_list(1000)
    elif current_user.role == UserRole.AGENT:
        # Agents see their own requests and retailer requests to them
        requests = await db.product_requests.find({
            "$or": [
                {"requester_id": current_user.id},
                {"target_id": current_user.id, "requester_role": UserRole.RETAILER}
            ]
        }).to_list(1000)
    else:  # RETAILER
        # Retailers see only their own requests
        requests = await db.product_requests.find({"requester_id": current_user.id}).to_list(1000)
    
    return [ProductRequest(**request) for request in requests]

@api_router.put("/product-requests/{request_id}/approve")
async def approve_product_request(
    request_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    # Find the request
    request = await db.product_requests.find_one({"id": request_id})
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Check permissions
    if request["target_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to approve this request")
    
    # Update request status
    await db.product_requests.update_one(
        {"id": request_id},
        {
            "$set": {
                "status": RequestStatus.APPROVED,
                "approved_by": current_user.id,
                "approved_at": datetime.utcnow()
            }
        }
    )
    
    return {"message": "Request approved successfully"}

@api_router.put("/product-requests/{request_id}/fulfill")
async def fulfill_product_request(
    request_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    # Find the request
    request = await db.product_requests.find_one({"id": request_id})
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Check permissions
    if request["target_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to fulfill this request")
    
    if request["status"] != RequestStatus.APPROVED:
        raise HTTPException(status_code=400, detail="Request must be approved first")
    
    # For agent fulfilling retailer request, check and update stock
    if current_user.role == UserRole.AGENT and request["requester_role"] == UserRole.RETAILER:
        current_stock = await get_or_create_stock(current_user.id, ProductType(request["product_type"]))
        if current_stock.quantity < request["quantity"]:
            raise HTTPException(status_code=400, detail="Insufficient stock")
        
        # Decrease agent stock
        await update_stock(
            agent_id=current_user.id,
            product_type=ProductType(request["product_type"]),
            quantity_change=-request["quantity"],
            transaction_type=TransactionType.AGENT_TO_RETAILER,
            order_id=request["order_id"],
            notes=f"Fulfilled request for retailer {request['requester_id']}"
        )
    
    # For distributor fulfilling agent request, increase agent stock
    elif current_user.role == UserRole.DISTRIBUTOR and request["requester_role"] == UserRole.AGENT:
        await update_stock(
            agent_id=request["requester_id"],
            product_type=ProductType(request["product_type"]),
            quantity_change=request["quantity"],
            transaction_type=TransactionType.DISTRIBUTOR_TO_AGENT,
            order_id=request["order_id"],
            notes=f"Stock allocation from distributor"
        )
    
    # Update request status
    await db.product_requests.update_one(
        {"id": request_id},
        {
            "$set": {
                "status": RequestStatus.FULFILLED,
                "fulfilled_at": datetime.utcnow()
            }
        }
    )
    
    return {"message": "Request fulfilled successfully"}

@api_router.put("/products/{product_id}/assign")
async def assign_product(
    product_id: str,
    assignee_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    # Find the product
    product = await db.products.find_one({"id": product_id})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Check permissions
    if current_user.role == UserRole.DISTRIBUTOR and product["created_by"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    elif current_user.role == UserRole.AGENT and product["assigned_to"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Update product
    new_status = ProductStatus.ASSIGNED
    await db.products.update_one(
        {"id": product_id},
        {
            "$set": {
                "assigned_to": assignee_id,
                "assigned_at": datetime.utcnow(),
                "status": new_status
            }
        }
    )
    
    return {"message": "Product assigned successfully"}

# Stock Management Routes
@api_router.get("/stock", response_model=List[StockItem])
async def get_stock(current_user: UserResponse = Depends(get_current_user)):
    # Check monthly reset
    await check_monthly_reset()
    
    if current_user.role == UserRole.DISTRIBUTOR:
        # Distributors can see all agent stock
        stock_items = await db.stock.find().to_list(1000)
    elif current_user.role == UserRole.AGENT:
        # Agents can see only their own stock
        stock_items = await db.stock.find({"agent_id": current_user.id}).to_list(1000)
    else:
        # Retailers don't have stock
        stock_items = []
    
    return [StockItem(**item) for item in stock_items]

@api_router.post("/stock/reset")
async def reset_stock(current_user: UserResponse = Depends(require_role([UserRole.DISTRIBUTOR]))):
    """Manual stock reset by distributor"""
    await reset_all_stock()
    return {"message": "All stock has been reset to 0"}

@api_router.get("/stock/transactions", response_model=List[StockTransaction])
async def get_stock_transactions(current_user: UserResponse = Depends(get_current_user)):
    if current_user.role == UserRole.DISTRIBUTOR:
        # Distributors can see all transactions
        transactions = await db.stock_transactions.find().sort("created_at", -1).to_list(1000)
    elif current_user.role == UserRole.AGENT:
        # Agents can see transactions involving them
        transactions = await db.stock_transactions.find({
            "$or": [
                {"from_user_id": current_user.id},
                {"to_user_id": current_user.id}
            ]
        }).sort("created_at", -1).to_list(1000)
    else:
        # Retailers don't have access to stock transactions
        transactions = []
    
    return [StockTransaction(**transaction) for transaction in transactions]

# Dashboard Routes
@api_router.get("/dashboard/stats")
async def get_dashboard_stats(current_user: UserResponse = Depends(get_current_user)):
    # Check monthly reset
    await check_monthly_reset()
    
    if current_user.role == UserRole.DISTRIBUTOR:
        # Total products
        total_products = await db.products.count_documents({"created_by": current_user.id})
        
        # Requests count
        pending_requests = await db.product_requests.count_documents({
            "target_id": current_user.id,
            "status": RequestStatus.PENDING
        })
        approved_requests = await db.product_requests.count_documents({
            "target_id": current_user.id,
            "status": RequestStatus.APPROVED
        })
        fulfilled_requests = await db.product_requests.count_documents({
            "target_id": current_user.id,
            "status": RequestStatus.FULFILLED
        })
        
        # Total agents and retailers
        total_agents = await db.users.count_documents({"role": UserRole.AGENT})
        total_retailers = await db.users.count_documents({"role": UserRole.RETAILER})
        
        # Total stock allocated to agents
        total_stock = await db.stock.aggregate([
            {"$group": {"_id": None, "total": {"$sum": "$quantity"}}}
        ]).to_list(1)
        total_stock = total_stock[0]["total"] if total_stock else 0
        
        return {
            "total_products": total_products,
            "pending_requests": pending_requests,
            "approved_requests": approved_requests,
            "fulfilled_requests": fulfilled_requests,
            "total_agents": total_agents,
            "total_retailers": total_retailers,
            "total_stock_allocated": total_stock
        }
    
    elif current_user.role == UserRole.AGENT:
        # Agent's current stock
        stock_items = await db.stock.find({"agent_id": current_user.id}).to_list(1000)
        stock_summary = {item["product_type"]: item["quantity"] for item in stock_items}
        
        # Agent's requests
        my_requests = await db.product_requests.count_documents({"requester_id": current_user.id})
        retailer_requests = await db.product_requests.count_documents({
            "target_id": current_user.id,
            "requester_role": UserRole.RETAILER
        })
        
        return {
            "current_stock": stock_summary,
            "my_requests": my_requests,
            "retailer_requests": retailer_requests
        }
    
    else:  # RETAILER
        # Retailer's requests
        my_requests = await db.product_requests.count_documents({"requester_id": current_user.id})
        pending_requests = await db.product_requests.count_documents({
            "requester_id": current_user.id,
            "status": RequestStatus.PENDING
        })
        fulfilled_requests = await db.product_requests.count_documents({
            "requester_id": current_user.id,
            "status": RequestStatus.FULFILLED
        })
        
        return {
            "total_requests": my_requests,
            "pending_requests": pending_requests,
            "fulfilled_requests": fulfilled_requests
        }

@api_router.get("/users/agents", response_model=List[UserResponse])
async def get_agents(current_user: UserResponse = Depends(get_current_user)):
    # Both distributors and retailers can see agents
    if current_user.role not in [UserRole.DISTRIBUTOR, UserRole.RETAILER]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    agents = await db.users.find({"role": UserRole.AGENT}).to_list(1000)
    return [UserResponse(**agent) for agent in agents]

@api_router.get("/users/retailers", response_model=List[UserResponse])
async def get_retailers(current_user: UserResponse = Depends(get_current_user)):
    if current_user.role not in [UserRole.DISTRIBUTOR, UserRole.AGENT]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    retailers = await db.users.find({"role": UserRole.RETAILER}).to_list(1000)
    return [UserResponse(**retailer) for retailer in retailers]

# Include the router in the main app
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
