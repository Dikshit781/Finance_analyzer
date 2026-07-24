from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile,File
from uuid import uuid4
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "FinPilot AI backend is running"
    }

@app.post("/upload")
async def upload_file(file: UploadFile=File(...)):
    unique_filename=f"{uuid4().hex}_{file.filename}"
    return {
        "message": "File uploaded successfully",
        "original_filename": file.filename,
        "saved_filename": unique_filename,
        "content_type": file.content_type
    }

@app.get("/Read-ulpoads")
async def read_ulpoaded_file():
    with open("ulpoad/{filename}", "wb") as f:
        content=await f.write()
        return content