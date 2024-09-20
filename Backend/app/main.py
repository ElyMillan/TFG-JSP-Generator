from fastapi import FastAPI, Request, status, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router as api_router  # Ensure this import path is correct
from .core.generadorFinal import Generador, caller
import asyncio


app = FastAPI()

# CORS configuration
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Include the API router
app.include_router(api_router)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    fields = [".".join(str(item) for item in error['loc']) for error in exc.errors()]
    unique_fields = set(fields)
    fields_string = ", ".join(unique_fields) if unique_fields else "No se especificaron campos correctamente."
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"message": "Por favor verifica los campos: " + fields_string},
    )

# Entry point for running the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
