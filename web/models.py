from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), nullable=True, unique=True, index=True)
    fingerprint = Column(String(255), nullable=True, unique=True, index=True)
    password_hash = Column(Text, nullable=True)
    provider = Column(String(32), nullable=False, default="password")
    role = Column(String(32), nullable=False, default="user", index=True)
    created_at = Column(String(64), nullable=False)


class UsageCounter(Base):
    __tablename__ = "usage_counters"

    subject_type = Column(String(32), primary_key=True)
    subject_id = Column(String(255), primary_key=True)
    period_type = Column(String(32), primary_key=True)
    period_key = Column(String(64), primary_key=True)
    count = Column(Integer, nullable=False, default=0)
    updated_at = Column(String(64), nullable=False)


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    question_index = Column(Integer, nullable=False, default=0, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(String(64), nullable=False)
    feedback = Column(String(16), nullable=True)
    feedback_text = Column(Text, nullable=True)
