# Backend-Python-Graph-Data-Request

API en Python que permite obtener datos desde un servidor remoto sobre la grafica deseada.

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalado Python (3.8+) y pip. Este proyecto fue desarrollado usando Python 3.8.

## Configuración del Entorno

Sigue estos pasos para configurar tu entorno de desarrollo local:

### 1. Clonar el Repositorio

Primero, clona este repositorio en tu máquina local:

```bash
git clone https://github.com/SoujiGitCode/Backend-Python-Graph-Data-Request
cd Backend-Python-Graph-Data-Request
```

# Instalar virtualenv si aún no está instalado
pip install virtualenv

# Crear un entorno virtual
virtualenv venv --python=python3.11

# Activar el entorno virtual
# En Windows
venv\Scripts\activate
# En MacOS/Linux
source venv/bin/activate

# Instalar todas las dependencias necesarias ejecutando:
pip install -r requirements.txt


# CREAR LA BASE DE DATOS
nombre: generator_v2
# Definir tu database_url en database.py de la siguiente manera:
DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/generator_v2"

# Ejecutar el migration.py para crear la tabla en la base de datos
python migration.py

# Ejecutar el Proyecto de Forma local , desde el directorio principal
uvicorn app.main:app --reload

Visita http://127.0.0.1:8000 en tu navegador para ver la aplicación en funcionamiento.



### Notas Adicionales

- **requirements.txt**: Este archivo debe estar correctamente poblado con todas las dependencias necesarias para ejecutar el proyecto. Puedes crearlo ejecutando `pip freeze > requirements.txt` en tu entorno después de instalar todas tus dependencias.
- **Instrucciones de Ejecución**: Verifica que el comando para ejecutar el servidor (`uvicorn app.main:app --reload`) se ajuste a cómo está estructurado tu proyecto.
