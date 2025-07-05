from fastapi import FastAPI
from backend.api.routes import router
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI(
    title="LitLens API",
    description="An AI-powered literature review assistant.",
    version="0.1.0",
    docs_url="/docs",              # 👈 explicitly enable Swagger UI
    redoc_url=None,                # optional: disables ReDoc
    openapi_url="/openapi.json"    # 👈 explicitly expose schema
)

# ✅ Serve files like /static/reports/my_report.md
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Include your routes
app.include_router(router)