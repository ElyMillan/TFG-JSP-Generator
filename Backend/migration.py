from sql_app.database import engine, Base
from sql_app.models import Request  # Importa tus modelos aquí

def create_tables():
    print("Intentando crear tablas en la base de datos...")
    print("Modelos en Base.metadata.tables:", Base.metadata.tables)
    Base.metadata.create_all(engine)  # Esto debería crear las tablas
    print("Tablas creadas si no hubo errores.")

if __name__ == "__main__":
    create_tables()
