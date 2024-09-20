from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from .models import MainFunctionDataRequest
from ..core.generadorFinal import Generador, caller
from ..core.solvers import Gecode, Gurobi, Cplex, ORTOOLS
import asyncio
from itertools import product, combinations
from datetime import datetime
import zipfile
import os
import shutil
from typing import List
router = APIRouter()
#db
from sqlalchemy.orm import Session
from sql_app.database import get_db
from sql_app.models import User, Request as DatabaseRequest
from sql_app.schemas import RequestCreate, UserCreate, RequestOut # Modelo de Pydantic
from sql_app.crud import create_request, create_user, get_user_by_email

async def welcome():
    return """
    <html>
        <head>
            <title>Bienvenido a la API</title>
            <style>
                body { font-family: 'Arial', sans-serif; background-color: #f4f4f9; color: #333; padding: 40px; text-align: center; }
                h1 { color: #333366; }
                p { font-size: 18px; }
            </style>
        </head>
        <body>
            <h1>Bienvenido a la API de Generación</h1>
          <p>Utiliza los endpoints proporcionados para interactuar con la API. Consulta la <a href="http://127.0.0.1:8000/docs">documentación de la API</a> para más detalles.</p>
        </body>
    </html>
    """
@router.get("/download/{unique_id}")
async def download_zip(unique_id: str):
    zip_path = f"./ConjuntoTFG/{unique_id}.zip"
    if os.path.exists(zip_path):
        return FileResponse(path=zip_path, filename=f"{unique_id}.zip", media_type='application/zip', headers={"Content-Disposition": "attachment"})
    else:
        raise HTTPException(status_code=404, detail="ZIP file not found.")
    

async def process_parameters(parameters, db: Session):
    # Crear un nuevo diccionario sin el 'unique_id'
  # Filter parameters to exclude certain keys
    filtered_parameters = {
        key: value for key, 
        value in parameters.items() 
        if key not in 
        {'unique_id', 
         'email', 
         'full_name', 
         'school_name', 
         'pickle_file_output', 
         'json_file_output',
         'dzn_file_output', 
         'taillard_file_output', 
         'single_folder_output',
         'custom_folder_name'
         }
    }
 
    unique_id = parameters.get('unique_id')
    email = parameters.get('email')
    full_name= parameters.get('full_name')
    school_name = parameters.get('school_name')
    pickle_file_output = parameters.get('pickle_file_output')
    json_file_output = parameters.get('json_file_output')
    dzn_file_output  = parameters.get('dzn_file_output')
    taillard_file_output = parameters.get('taillard_file_output')
    single_folder_output = parameters.get('single_folder_output')
    custom_folder_name = parameters.get('custom_folder_name')

    print("Unique id:", unique_id)
    print("Email:", email)
    print("Full name:", full_name)
    print("School name:", school_name)

    print("Unique id:", unique_id)

    print("Processing parameters:", filtered_parameters)
    combinations_list = list(product(*filtered_parameters.values()))
    parameter_vectors = [
        {"size": size, 
         "jobs": jobs, 
         "machines": machines, 
         "distributions": dist,
         "SpeedScaling": speed, 
         "ReleaseDueDate": release, 
         "index": size * i, 
         "seeds": seeds,
         "unique_id": unique_id,
        "pickle_file_output": pickle_file_output,
        "json_file_output": json_file_output,
        "dzn_file_output": dzn_file_output,
        "taillard_file_output": taillard_file_output,
        "single_folder_output": single_folder_output,
        "custom_folder_name": custom_folder_name 
         }
        for i, (size, jobs, machines, dist, speed, release, seeds) in enumerate(combinations_list)
    ]

    tasks = []
    for params in parameter_vectors:
        print("Scheduling task for params:", params)
        
        # Crear el registro en la base de datos
        print("DEBUG ACTUAL HERE DINK DONK!!", email, full_name, school_name,)
        await create_and_save_request(params, email, full_name, school_name, db)
        
        # Ejecutar caller_async en segundo plano
        task = asyncio.create_task(caller_async(params))
        tasks.append(task)
    
    await asyncio.gather(*tasks)

async def create_and_save_request(params, email: str, full_name: str, school_name: str, db: Session):
    # Validar y preparar los datos de usuario usando UserCreate
    user_data = UserCreate(email=email, full_name=full_name, school_name=school_name)

    # Intentar encontrar un usuario con el email dado
    user = get_user_by_email(db, user_data.email)
    if user:
        print(f"Usuario encontrado: {user.full_name} (ID: {user.id})")
    else:
        # Crear un nuevo usuario si no se encuentra
        user = create_user(db, user_data)
        print(f"Nuevo usuario creado: {user.full_name} (ID: {user.id})")

    # Crear la request asociada al usuario
    request_data = RequestCreate(
        seed=params.get('seeds')[0] if isinstance(params.get('seeds'), list) else params.get('seeds'),
        size=params['size'],
        jobs=str(params['jobs']),
        machines=str(params['machines']),
        distributions=params['distributions'],
        speed_scaling=params['SpeedScaling'],
        release_due_date=params['ReleaseDueDate'],
        unique_id=params['unique_id'],
        user_id=user.id  # Asocia la request con el usuario
    )

    new_request = DatabaseRequest(**request_data.dict())
    db.add(new_request)
    db.commit()
    db.refresh(new_request)

    return new_request

async def caller_async(params):
    print("Processing asynchronously:", params)
    caller(params)  # Asegúrate de que esta operación es adecuada para ser llamada aquí
    await asyncio.sleep(1)  # Simula trabajo con sleep asíncrono

async def solver_function_async(solver, path, timeout=60, replace=True, guardar=True):
    print("Processing solver asynchronously:", solver, path)
    # Diccionario que mapea nombres de solvers a sus clases correspondientes
    solvers = {
        "Gecode": Gecode,
        "Gurobi": Gurobi,
        "Cplex": Cplex,
        "ORTOOLS": ORTOOLS
    }

    # Verificar si el solver está en el diccionario
    if solver in solvers:
        # Instanciar la clase del solver correspondiente
        solver_instance = solvers[solver](path)
        # Llamar al método solve de la instancia creada
        await asyncio.to_thread(solver_instance.solve, path, timeout=timeout, replace=replace, guardar=guardar)
    else:
        print(f"No valid solver selected for '{solver}'. Exiting.")

    await asyncio.sleep(1)  # Simula trabajo con sleep asíncrono

@router.post("/default_generation")
async def generate_default(request:Request,  db: Session = Depends(get_db)):
    unique_id = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    directory_path = f"./ConjuntoTFG/{unique_id}/"
    solver = "none"

    default_parameters = {
        "email": "default_email@gmail.com",
        "full_name": "John Doe",
        "school_name": "default school",
        "size": [1], 
        "jobs": [5, 10], 
        "machines": [5, 10], 
        "distributions": ["normal"],
        "SpeedScaling": [5], 
        "ReleaseDueDate": [2],
        "unique_id" : unique_id,
        "seeds": [0], 
        "pickle_file_output": "True",
        "json_file_output": "True",
        "dzn_file_output": "True",
        "taillard_file_output":"True",
        "single_folder_output":"True",
        "custom_folder_name": "Paco_Size{size}_Jobs{jobs}_Machines{machines}_RDDD{release_due_date}_Speed{speed_scaling}_Distribution{distribution}_Seed{seed}"
    }

  # Procesar parámetros y generar datos
    await process_parameters(default_parameters, db)
    
    # Comprimir el directorio generado
    zip_path = compress_directory(directory_path )
    
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=500, detail="Failed to create zip file.")
    
    # Verificar si el directorio existe antes de borrarlo
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Directorio '{directory_path}' y su contenido fueron borrados exitosamente.")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print(f"El directorio '{directory_path}' no existe.")
    
    # Construir la URL completa dinámicamente
    zip_url = str(request.url_for('static', path=f"ConjuntoTFG/{unique_id}.zip"))
    
    # Devolver una respuesta con la ruta del archivo zip
    return {
        "message": "Custom generation completed successfully",
        "zip_url": zip_url,
        "unique_id": unique_id
    }

@router.post("/custom_generation")
async def generate_custom(request: Request, data: MainFunctionDataRequest, db: Session = Depends(get_db)):
    seeds = data.seeds[0]
    print(f"DATA MACHINES '{data.machines}'")
    print(f"DATA JOBS '{data.jobs}'")
    unique_id =f"Seed_{seeds}_" + datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    directory_path = f"./ConjuntoTFG/{unique_id}/"
    solver = data.solver
    print(f"DATA SOLVER RECEIVED :'{data.solver}'")

    custom_parameters = {
        "email": data.email,
        "full_name": data.full_name,
        "school_name": data.school_name,
        "size": [data.size],
        "jobs": list(data.jobs) if isinstance(data.jobs, tuple) else data.jobs,
        "machines": list(data.machines) if isinstance(data.machines, tuple) else data.machines,
        "distributions": list(data.distributions)if isinstance (data.distributions, tuple) else data.distributions,
        "SpeedScaling": [data.speed_scaling],
        "ReleaseDueDate": [data.release_due_date],
        "unique_id": unique_id, 
        "seeds": [data.seeds],
        "pickle_file_output": data.pickle_file_output,
        "json_file_output": data.json_file_output,
        "dzn_file_output": data.dzn_file_output,
        "taillard_file_output": data.taillard_file_output,
        "single_folder_output":data.single_folder_output,
        "custom_folder_name":data.custom_folder_name
    }
    # Procesar parámetros y generar datos
    await process_parameters(custom_parameters, db)
    
    # Solver
    # Ejecutar solverFunction en segundo plano
    solver_task = asyncio.create_task(solver_function_async(solver, directory_path))
    await solver_task

    # Comprimir el directorio generado
    zip_path = compress_directory(directory_path )
    
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=500, detail="Failed to create zip file.")
    
    # Verificar si el directorio existe antes de borrarlo
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Directorio '{directory_path}' y su contenido fueron borrados exitosamente.")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print(f"El directorio '{directory_path}' no existe.")

    # Construir la URL completa dinámicamente y convertirla a una cadena
    zip_url = str(request.url_for('static', path=f"ConjuntoTFG/{unique_id}.zip"))
    
    # Devolver una respuesta con la ruta del archivo zip y el unique_id
    return {
        "message": "Custom generation completed successfully",
        "zip_url": zip_url,
        "unique_id": unique_id
    }


def compress_directory(directory_path):
    # Crear un archivo ZIP en el mismo directorio base
    zip_path = f"{directory_path.rstrip('/')}.zip"  # Quita cualquier barra al final y añade .zip
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Recorrer todos los archivos en el directorio
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # Crear el path completo de cada archivo
                file_path = os.path.join(root, file)
                # Agregar el archivo al ZIP
                zipf.write(file_path, os.path.relpath(file_path, directory_path))
    return zip_path


@router.get("/requests", response_model=List[RequestOut])
async def read_all_requests(db: Session = Depends(get_db)):
    try:
        requests = db.query(DatabaseRequest).all()
        return requests
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/requests", response_model=RequestOut)
async def create_new_request(request: RequestCreate, db: Session = Depends(get_db)):
    try:
        new_request = create_request(db, request)
        return new_request
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))