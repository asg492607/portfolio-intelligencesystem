import os
import shutil
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

import models
from database import engine, get_db
from storage import storage_client
from vector_db import vector_db
from analyzer import (
    extract_text_from_pdf,
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

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class UrlAnalyzeRequest(BaseModel):
    url: str
    role_target: str = "UX Designer"
    seniority: str = "Mid"

async def process_portfolio_job(job_id: str, content: str, source_label: str, role_target: str, seniority: str, local_file_to_clean: str = None):
    """Orchestrates the complete 6-stage architecture background processing pipeline."""
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
            # Add to Qdrant collection for RAG benchmarking
            await asyncio.to_thread(
                vector_db.add_portfolio_chunk,
                job_id=job_id,
                chunk_index=0,
                text=content[:2000],
                vector=embedding,
                metadata={"source": source_label, "role_target": role_target, "seniority": seniority}
            )
            print("Successfully indexed chunk vector to Qdrant.")
        except Exception as e:
            print(f"Qdrant vector indexing skipped or failed: {e}")

        # Step 4: Run parallelized AI analysis
        report_data = await asyncio.to_thread(
            run_ai_analysis,
            content,
            source_label,
            role_target,
            seniority
        )

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

@app.post("/api/v1/analyze/pdf")
async def analyze_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    role_target: str = Form("UX Designer"),
    seniority: str = Form("Mid"),
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
        role_target=role_target,
        seniority=seniority,
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

    # Extract text immediately for background handoff
    text_content = extract_text_from_pdf(temp_file_path)

    # Queue async processing
    background_tasks.add_task(
        process_portfolio_job,
        job_id=job_id,
        content=text_content,
        source_label=file.filename,
        role_target=role_target,
        seniority=seniority,
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
        role_target=payload.role_target,
        seniority=payload.seniority,
        status="processing"
    )
    db.add(db_job)
    db.commit()

    # Determine scraper target
    source_label = "Website Portfolio"
    is_figma = "figma.com" in url.lower()
    is_behance = "behance.net" in url.lower()

    if is_figma:
        source_label = f"Figma Design ({url[:30]}...)"
        # Extraction
        content = await parse_figma_content(url)
    elif is_behance:
        source_label = f"Behance Project ({url[:30]}...)"
        # Web scraping
        content = await scrape_url_content(url)
    else:
        source_label = f"Web Portfolio ({url[:30]}...)"
        # Web scraping
        content = await scrape_url_content(url)

    # Queue background analysis workflow
    background_tasks.add_task(
        process_portfolio_job,
        job_id=job_id,
        content=content,
        source_label=source_label,
        role_target=payload.role_target,
        seniority=payload.seniority
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
        "role_target": db_job.role_target,
        "seniority": db_job.seniority,
        "results": db_job.results
    }

@app.get("/")
async def get_index():
    """Serves the frontend homepage index.html."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
