from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


app = FastAPI(title="FinPilot AI Backend")


# --------------------------------------------------
# CORS Middleware
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Upload Folder
# --------------------------------------------------

UPLOAD_FOLDER = Path("data/uploads")

# Create the folder if it does not already exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Home Route
# --------------------------------------------------

@app.get("/")
def read_root():
    return {
        "message": "FinPilot AI backend is running"
    }

ALLOWED_EXTENSIONS = {
        ".pdf",
        ".csv",
        ".xlsx",
        ".xls",
        ".json",
        ".txt"
    }

# --------------------------------------------------
# Upload File
# --------------------------------------------------

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not have a filename."
        )

    safe_filename = Path(file.filename).name
    extension = Path(safe_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} files are allowed."
        )

    unique_filename = f"{uuid4().hex}_{safe_filename}"
    file_path = UPLOAD_FOLDER / unique_filename

    content = await file.read()

    with open(file_path, "wb") as saved_file:
        saved_file.write(content)

    await file.close()

    return {
        "message": "File uploaded successfully",
        "original_filename": safe_filename,
        "saved_filename": unique_filename,
        "content_type": file.content_type,
        "file_path": str(file_path)
    }


# --------------------------------------------------
# Read or Download Uploaded File
# --------------------------------------------------

@app.get("/uploads/{filename}")
def read_uploaded_file(filename: str):
    file_path = UPLOAD_FOLDER / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    return FileResponse(
        path=file_path,
        filename=filename
    )