from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return JSONResponse(
        content={
            "message": "✅ LitLens v2.0 backend is running.",
            "version": "2.0.0",
            "status": "active"
        }
    )

def handler(request):
    return app(request)