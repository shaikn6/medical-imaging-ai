"""
SQLAlchemy ORM model for storing inference results.
"""

import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from api.database import Base


class PredictionRecord(Base):
    """Persists every API prediction for audit and analytics."""

    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(256), nullable=True)
    predicted_class = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities_json = Column(Text, nullable=False)   # JSON string
    processing_ms = Column(Float, nullable=True)
    created_at = Column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
