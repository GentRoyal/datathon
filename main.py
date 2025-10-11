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
from scripts.api.feature3 import router as needs_analysis_router 

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Teacher CoPilot",
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

# Mount static files directory (for CSS, JS, images if needed)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include API routes with /api prefix
app.include_router(chat_router)

# Serve landing page at root
@app.get("/")
async def read_root():
    """Serve the landing page"""
    landing_page_path = Path(__file__).parent / "templates" / "landing.html"
    if landing_page_path.exists():
        return FileResponse(landing_page_path)
    return {"message": "Welcome to AI Teacher CoPilot API", "docs": "/docs"}

# Serve chat page
@app.get("/chat_system")
async def read_chat():
    """Serve the chat interface"""
    chat_page_path = Path(__file__).parent / "templates" / "chat.html"
    if chat_page_path.exists():
        return FileResponse(chat_page_path)
    return {"message": "Chat interface not found", "redirect": "/"}

@app.get("/report")
async def read_report():
    """Serve the report interface"""
    report_page_path = Path(__file__).parent / "templates" / "report.html"
    if report_page_path.exists():
        return FileResponse(report_page_path)
    return {"message": "Report interface not found", "redirect": "/"}


app.include_router(document_router)
@app.get("/lesson_copilot")
async def upload_document():
    """Serve the report interface"""
    report_page_path = Path(__file__).parent / "templates" / "knowledge_compressor.html"
    
    if report_page_path.exists():
        return FileResponse(report_page_path)
    return {"message": "Knowledge Compressor interface not found", "redirect": "/"}


app.include_router(needs_analysis_router)
@app.get("/teacher_needs")
async def teacher_needs():
    """Serve the report interface"""
    report_page_path = Path(__file__).parent / "templates" / "dashboard.html"
    
    if report_page_path.exists():
        return FileResponse(report_page_path)
    return {"message": "Need Analysis Interface Not Found", "redirect": "/"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)