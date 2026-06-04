import datetime
from sqlalchemy import Column, String, DateTime, JSON, Text, Integer
from database import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=True)
    portfolio_url = Column(String, nullable=True)  # New: tracks Behance, Figma, Website links
    status = Column(String, default="pending")  # pending, processing, completed, error
    results = Column(JSON, nullable=True)  # Stores the final synthesized report JSON
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(String, primary_key=True, index=True)
    job_id = Column(String, index=True, nullable=False)
    match_job_id = Column(String, index=True, nullable=False)
    rating = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

