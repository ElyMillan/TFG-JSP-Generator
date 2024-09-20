from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound
from .models import User, Request
from .schemas import UserCreate, RequestCreate

def create_user(db: Session, user_create: UserCreate) -> User:
    # Crear una instancia del modelo User con los datos de UserCreate
    db_user = User(
        email=user_create.email,
        full_name=user_create.full_name,
        school_name=user_create.school_name
    )
    
    # Agregar el nuevo usuario a la sesión y confirmar la transacción
    db.add(db_user)
    db.commit()
    db.refresh(db_user)  # Refresca la instancia para obtener datos generados por la DB, como el ID
    
    return db_user

def get_user_by_email(db: Session, email: str) -> User:
    # Intentar encontrar un usuario con el email dado
    try:
        return db.query(User).filter(User.email == email).one()
    except NoResultFound:
        return None  # Devuelve None si no se encuentra el usuario


def create_request(db: Session, request: RequestCreate) -> Request:
    # Crear una nueva instancia del modelo Request con los datos del esquema Pydantic
    db_request = Request(
        email=request.email,
        full_name=request.full_name,
        school_name=request.school_name,
        seed=request.seed,
        size=request.size,
        jobs=request.jobs,
        machines=request.machines,
        distributions=request.distributions,
        speed_scaling=request.speed_scaling,
        release_due_date=request.release_due_date,
        unique_id=request.unique_id,
        user_id=request.user_id  # Asegúrate de que `user_id` está incluido si es necesario
    )
    
    # Agregar el nuevo objeto Request a la sesión y confirmar la transacción
    db.add(db_request)
    db.commit()
    db.refresh(db_request)  # Refresca la instancia para obtener los datos generados por la base de datos
    
    return db_request