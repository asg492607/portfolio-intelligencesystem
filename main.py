import os
from dotenv import load_dotenv
load_dotenv()

import shutil
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

import models
from database import engine, get_db
from storage import storage_client
from vector_db import vector_db
from analyzer import (
    extract_text_from_pdf,
    extract_images_from_pdf,
    scrape_url_content,
    parse_figma_content,
    generate_text_embedding,
    run_ai_analysis
)

# Initialize database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Portfolio Intelligence Agent API", version="1.0.0")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("local_storage", exist_ok=True)
app.mount("/local_storage", StaticFiles(directory="local_storage"), name="local_storage")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class UrlAnalyzeRequest(BaseModel):
    url: str

async def process_portfolio_job(job_id: str, content: str, source_label: str, extracted_images: list = None, extracted_links: list = None, local_file_to_clean: str = None):
    """Orchestrates the background analysis pipeline: S3 storage -> vector embedding -> AI analysis -> DB update."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        # Step 2: Save raw contents to MinIO/S3
        try:
            s3_url = await asyncio.to_thread(
                storage_client.upload_data,
                content.encode("utf-8"),
                f"{job_id}/source_content.txt",
                "text/plain"
            )
            print(f"Stored raw content to MinIO/S3: {s3_url}")
        except Exception as e:
            print(f"Failed storing raw content in S3: {e}")

        # Step 3: Embed content for Qdrant Vector search
        try:
            embedding = await generate_text_embedding(content)
            # Add to Qdrant collection for RAG-based similarity search
            await asyncio.to_thread(
                vector_db.add_portfolio_chunk,
                job_id=job_id,
                chunk_index=0,
                text=content[:2000],
                vector=embedding,
                metadata={"source": source_label}
            )
            print("Successfully indexed chunk vector to Qdrant.")
        except Exception as e:
            print(f"Qdrant vector indexing skipped or failed: {e}")

        # Step 4: Run parallelized AI analysis
        report_data = await asyncio.to_thread(
            run_ai_analysis,
            content,
            source_label,
            extracted_images,
            extracted_links
        )
        if isinstance(report_data, dict):
            report_data["images"] = extracted_images or []
            report_data["links"] = extracted_links or []
            
            # Compute default job matches
            dummy_jobs = [
                {
                    "job_id": "job_001",
                    "job_title": "Graphic Designer",
                    "company": "Studio Fresh",
                    "industry": "Food & Beverage",
                    "location": "Remote",
                    "seniority": "mid",
                    "required_skills": ["Branding", "Visual Identity", "Packaging Design", "Typography"],
                    "tools": ["Adobe Illustrator", "Adobe Photoshop", "Adobe InDesign"],
                    "summary": "Create fresh, emotionally resonant brand identities and packaging layouts for a portfolio of food and beverage clients."
                },
                {
                    "job_id": "job_002",
                    "job_title": "Illustrator & Visual Designer",
                    "company": "Pixel & Prose Publishing",
                    "industry": "Publishing",
                    "location": "Hybrid",
                    "seniority": "mid",
                    "required_skills": ["Illustration", "Character Design", "Storyboarding", "Layout Design"],
                    "tools": ["Procreate", "Adobe Illustrator", "Figma"],
                    "summary": "Develop unique characters, illustrations, and visual storytelling assets for children's books and educational media."
                },
                {
                    "job_id": "job_003",
                    "job_title": "Brand & Marketing Designer",
                    "company": "Nexus Tech",
                    "industry": "Technology",
                    "location": "Remote",
                    "seniority": "senior",
                    "required_skills": ["Advertising Campaigns", "Social Media Design", "OOH Advertising", "Creative Thinking"],
                    "tools": ["Figma", "Adobe After Effects", "Adobe Photoshop"],
                    "summary": "Lead visual design efforts across cross-functional tech teams, driving large-scale advertising campaigns and digital product launches."
                }
            ]
            report_data["job_matches"] = [compute_job_match(report_data, job) for job in dummy_jobs]

        # Step 5 & 6: Update DB & complete
        db_job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if db_job:
            db_job.results = report_data
            db_job.status = "completed"
            db.commit()

    except Exception as e:
        print(f"Error in background processing of job {job_id}: {e}")
        db_job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if db_job:
            db_job.status = "error"
            db.commit()
    finally:
        db.close()
        # Clean up temporary uploads if any
        if local_file_to_clean and os.path.exists(local_file_to_clean):
            try:
                os.remove(local_file_to_clean)
            except Exception as ex:
                print(f"Failed deleting temporary file: {ex}")

def compute_job_match(portfolio: dict, job: dict) -> dict:
    """Helper to calculate skills match score and breakdown."""
    port_skills = set()
    if "skills" in portfolio and isinstance(portfolio["skills"], dict):
        for cat in portfolio["skills"].values():
            if isinstance(cat, list):
                for s in cat:
                    port_skills.add(s.lower())
    if "tools" in portfolio and isinstance(portfolio["tools"], list):
        for t in portfolio["tools"]:
            port_skills.add(t.lower())

    req_skills = [s.lower() for s in job["required_skills"]]
    req_tools = [t.lower() for t in job["tools"]]

    matched_skills = [s for s in job["required_skills"] if s.lower() in port_skills]
    matched_tools = [t for t in job["tools"] if t.lower() in port_skills]

    missing_skills = [s for s in job["required_skills"] if s.lower() not in port_skills]
    missing_tools = [t for t in job["tools"] if t.lower() not in port_skills]

    # Calculate score
    total_criteria = len(req_skills) + len(req_tools)
    matched_count = len(matched_skills) + len(matched_tools)

    score = int((matched_count / total_criteria) * 100) if total_criteria > 0 else 0

    # Experience heuristic adjustment
    exp_aligned = True
    port_exp = portfolio.get("years_experience")
    if port_exp is not None:
        target_exp = 5.0 if job["seniority"] == "senior" else 2.0
        if port_exp < target_exp:
            score = max(0, score - 15)
            exp_aligned = False

    fit_status = "Strong Match" if score >= 75 else ("Good Match" if score >= 45 else "Low Match")

    return {
        "job_id": job["job_id"],
        "job_title": job["job_title"],
        "company": job["company"],
        "industry": job["industry"],
        "location": job["location"],
        "seniority": job["seniority"],
        "summary": job.get("summary", ""),
        "score": score,
        "fit_status": fit_status,
        "matched_skills": matched_skills + matched_tools,
        "missing_skills": missing_skills + missing_tools,
        "exp_aligned": exp_aligned
    }

class MatchRequest(BaseModel):
    job_description: str

@app.post("/api/v1/report/{job_id}/match")
async def match_custom_job(job_id: str, payload: MatchRequest, db: Session = Depends(get_db)):
    """API endpoint to dynamically match a candidate profile against a custom job description using Ollama."""
    db_job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")

    portfolio = db_job.results
    if not portfolio:
        raise HTTPException(status_code=400, detail="Portfolio has not been successfully analyzed yet.")

    from openai import OpenAI
    import json

    prompt = f"""
    Analyze how well the candidate's portfolio matches the following job description.
    Focus strictly on objective comparison of skills, tools, and experience. Do not give generic subjective commentary.

    Candidate Portfolio Data:
    {json.dumps(portfolio, indent=2)}

    Job Description:
    {payload.job_description}

    Return a JSON response matching exactly this template:
    {{
      "match_score": 85, // integer 0 to 100
      "fit_status": "Strong Match|Good Match|Potential Match|Low Match",
      "matching_skills": ["list of overlapping skills/tools found in both"],
      "missing_skills": ["list of skills/tools requested in the job description but missing in the portfolio"],
      "fit_reason": "A brief paragraph detailing their professional alignment with the role.",
      "recommendations": ["1-2 actionable tips for aligning this portfolio to better target this specific role."]
    }}
    """

    groq_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not groq_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY environment variable is not configured.")

    try:
        # Use Groq Llama API
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key
        )
        response = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": "You are a professional recruiting assistant specialized in matching candidate portfolios to job roles."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        import re
        clean_text = response.choices[0].message.content.strip()
        match = re.search(r'\{[\s\S]*\}', clean_text)
        if match:
            clean_text = match.group(0)
        return json.loads(clean_text)
    except Exception as e:
        # Programmatic fallback
        import uuid
        dummy_job = {
            "job_id": str(uuid.uuid4()),
            "job_title": "Custom Target Role",
            "company": "Target Employer",
            "industry": "General",
            "location": "Remote",
            "seniority": "mid",
            "required_skills": ["communication", "problem solving"],
            "tools": [],
            "summary": payload.job_description
        }
        res = compute_job_match(portfolio, dummy_job)
        return {
            "match_score": res["score"],
            "fit_status": res["fit_status"] + " (Fallback)",
            "matching_skills": res["matched_skills"],
            "missing_skills": res["missing_skills"],
            "fit_reason": f"Analyzed via local keyword intersection heuristic engine. Error calling LLM: {str(e)}",
            "recommendations": ["Highlight the tools and methodology matching the job description."]
        }

@app.post("/api/v1/analyze/pdf")
async def analyze_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a PDF portfolio and start background analysis."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    job_id = str(uuid.uuid4())
    temp_file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    # Save uploaded file temporarily
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Database state creation
    db_job = models.Job(
        id=job_id,
        filename=file.filename,
        status="processing"
    )
    db.add(db_job)
    db.commit()

    # S3 backup upload
    try:
        s3_pdf_url = await asyncio.to_thread(
            storage_client.upload_file,
            temp_file_path,
            f"{job_id}/{file.filename}"
        )
        print(f"Uploaded raw PDF to MinIO/S3: {s3_pdf_url}")
    except Exception as e:
        print(f"MinIO/S3 PDF upload failed: {e}")

    # Extract text immediately for background handoff (injects image placeholders)
    text_content = extract_text_from_pdf(temp_file_path, job_id=job_id)
    
    # Extract links and images via placeholders
    import re
    extracted_images = re.findall(r'\[IMAGE_URL:\s*([^\s\]]+)', text_content)
    extracted_links = re.findall(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', text_content)[:15]

    # Queue async processing
    background_tasks.add_task(
        process_portfolio_job,
        job_id=job_id,
        content=text_content,
        source_label=file.filename,
        extracted_images=extracted_images,
        extracted_links=extracted_links,
        local_file_to_clean=temp_file_path
    )

    return {"job_id": job_id, "status": "processing"}

@app.post("/api/v1/analyze/url")
async def analyze_url(
    payload: UrlAnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Ingest, parse, and analyze Behance, Figma, or Website portfolios."""
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    job_id = str(uuid.uuid4())
    
    # Database entry
    db_job = models.Job(
        id=job_id,
        portfolio_url=url,
        status="processing"
    )
    db.add(db_job)
    db.commit()

    # Determine scraper target
    source_label = "Website Portfolio"
    is_figma = "figma.com" in url.lower()
    is_behance = "behance.net" in url.lower()

    content = ""
    extracted_images = []
    extracted_links = []

    if is_figma:
        source_label = f"Figma Design ({url[:30]}...)"
        # Extraction
        content = await parse_figma_content(url)
    elif is_behance:
        source_label = f"Behance Project ({url[:30]}...)"
        # Web scraping
        content, extracted_images, extracted_links = await scrape_url_content(url)
    else:
        source_label = f"Web Portfolio ({url[:30]}...)"
        # Web scraping
        content, extracted_images, extracted_links = await scrape_url_content(url)

    # Queue background analysis workflow
    background_tasks.add_task(
        process_portfolio_job,
        job_id=job_id,
        content=content,
        source_label=source_label,
        extracted_images=extracted_images,
        extracted_links=extracted_links
    )

    return {"job_id": job_id, "status": "processing"}

@app.get("/api/v1/report/{job_id}")
async def get_report(job_id: str, db: Session = Depends(get_db)):
    """Retrieve report status and generated intelligence package."""
    db_job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": db_job.id,
        "status": db_job.status,
        "filename": db_job.filename,
        "portfolio_url": db_job.portfolio_url,
        "results": db_job.results
    }

@app.get("/")
@app.head("/")
async def get_index():
    """Serves the frontend homepage index.html."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
