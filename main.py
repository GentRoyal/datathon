import uvicorn
from dotenv import load_dotenv
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from scripts.api.chat_system import router as chat_router
from scripts.api.knowledge_system import router as document_router
from scripts.api.teacher_support import router as needs_analysis_router 
from scripts.api.auth import router as auth_router
from scripts.api.profile import router as profile_router

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EduBuddy",
    description="An AI-powered Educator Assistant that Analyzes Curricula, Lesson Plans, and Education Policies to Support Smarter Teaching Through Contextual Document Understanding.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes with /api prefix
app.include_router(chat_router)

# Serve landing page at root
@app.get("/")
async def read_root():
    """Serve the landing page"""
    landing_page_path = Path(__file__).parent / "templates" / "landing.html"
    if landing_page_path.exists():
        return FileResponse(landing_page_path)
    return {"message": "Welcome to EduBuddy API", "docs": "/docs"}

# Serve chat page
@app.get("/knowledge-coach")
async def read_chat():
    """Serve the chat interface"""
    chat_page_path = Path(__file__).parent / "templates" / "knowledge-coach.html"
    if chat_page_path.exists():
        return FileResponse(chat_page_path)
    return {"message": "Chat interface not found", "redirect": "/"}

app.include_router(document_router)
@app.get("/lesson-copilot")
async def upload_document():
    """Serve the report interface"""
    report_page_path = Path(__file__).parent / "templates" / "lesson-copilot.html"
    
    if report_page_path.exists():
        return FileResponse(report_page_path)
    return {"message": "Knowledge Compressor interface not found", "redirect": "/"}


app.include_router(needs_analysis_router)
@app.get("/need-assessment")
async def teacher_needs():
    """Serve the report interface"""
    report_page_path = Path(__file__).parent / "templates" / "need-assessment.html"
    
    if report_page_path.exists():
        return FileResponse(report_page_path)
    return {"message": "Need Analysis Interface Not Found", "redirect": "/"}

app.include_router(auth_router)
@app.get("/api/login")
async def login():
    """Serve the report interface"""
    report_page_path = Path(__file__).parent / "templates" / "login.html"
    
    if report_page_path.exists():
        return FileResponse(report_page_path)
    return {"message": "Login Interface Not Found", "redirect": "/"}

@app.get("/api/signup")
async def signup():
    """Serve the report interface"""
    report_page_path = Path(__file__).parent / "templates" / "login.html"
    
    if report_page_path.exists():
        return FileResponse(report_page_path)
    return {"message": "Login Interface Not Found", "redirect": "/"}

app.include_router(profile_router)
@app.get("/api/teacher-profile")
async def read_profile():
    """Serve the teacher profile page"""
    profile_page_path = Path(__file__).parent / "templates" / "teacher-profile.html"
    if profile_page_path.exists():
        return FileResponse(profile_page_path)
    return {"message": "Profile page not found"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)