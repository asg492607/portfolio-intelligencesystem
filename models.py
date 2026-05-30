import datetime
from sqlalchemy import Column, String, DateTime, JSON, Text
from database import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=True)
    portfolio_url = Column(String, nullable=True)  # New: tracks Behance, Figma, Website links
    role_target = Column(String, nullable=True)
    seniority = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    results = Column(JSON, nullable=True)  # Stores the final synthesized report JSON
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
