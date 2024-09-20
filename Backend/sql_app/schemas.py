from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

# Modelos para Users

class UserCreate(BaseModel):
    email: EmailStr  # Asegura que el email es válido
    full_name: Optional[str] = None
    school_name: Optional[str] = None

    class Config:
        from_attributes = True  # Pydantic v2

class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str] = None
    school_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # Pydantic v2
        orm_mode = True  # Permite la compatibilidad con modelos SQLAlchemy

# Modelos para Requests

class RequestCreate(BaseModel):
    seed: Optional[int] = None
    size: Optional[int] = None
    jobs: Optional[str] = None
    machines: Optional[str] = None
    distributions: Optional[str] = None
    speed_scaling: Optional[int] = None
    release_due_date: Optional[int] = None
    unique_id: Optional[str] = None
    user_id: int  # Referencia al usuario creador de la solicitud

    class Config:
        from_attributes = True  # Pydantic v2

class RequestOut(BaseModel):
    id: int
    seed: Optional[int] = None
    size: Optional[int] = None
    jobs: Optional[str] = None
    machines: Optional[str] = None
    distributions: Optional[str] = None
    speed_scaling: Optional[int] = None
    release_due_date: Optional[int] = None
    unique_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    user_id: int

    class Config:
        from_attributes = True  # Pydantic v2
        orm_mode = True  # Permite la compatibilidad con modelos SQLAlchemy
