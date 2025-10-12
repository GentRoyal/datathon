from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api")

class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str

@router.post("/login")
async def login(request: LoginRequest):
    """Login endpoint"""
    # Your database logic here
    email = request.email
    password = request.password
    
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Login successful"
        }
    )

@router.post("/signup")
async def signup(request: SignupRequest):
    """Signup endpoint"""
    # Your database logic here
    first_name = request.first_name
    last_name = request.last_name
    email = request.email
    password = request.password
    
    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "message": "Account created successfully"
        }
    )