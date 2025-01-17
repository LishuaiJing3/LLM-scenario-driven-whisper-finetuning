from fastapi import FastAPI
from apps.data_curation.api.endpoints import router
from apps.data_curation.utils.logger import setup_logging

# Initialize the app
app = FastAPI(title="Data Curation Service", version="1.0")

# Setup logging
setup_logging()

# Include the API router
app.include_router(router, prefix="/api/data_curation", tags=["Data Curation"])

# Root endpoint for health check
@app.get("/")
async def health_check():
    return {"status": "Data Curation Service is running"}

# add text and speech alignment 
