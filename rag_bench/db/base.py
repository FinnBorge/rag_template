from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Float, Index, JSON, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()