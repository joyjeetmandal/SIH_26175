from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import tempfile
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow Python to find run_pipeline.py
sys.path.insert(0, str(PROJECT_ROOT))


from run_pipeline import run_pipeline

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "runs"

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)

@app.get("/")
def home():
    return {
        "message": "Backend is running"
    }


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ------------------------------------------------------------
# AI PREDICTION
# ------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    # Keep the original extension
    suffix = Path(file.filename).suffix

    # Create temporary file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(image_bytes)
        temp_path = Path(temp_file.name)

    try:

        print("\n==============================")
        print("Received file:", file.filename)
        print("Temporary file:", temp_path)
        print("==============================")

        result = run_pipeline(temp_path)

        print("\nAI pipeline finished!")

        run_dir = Path(result["run_dir"])
        heightmap_path = Path(result["heightmap"])
        texture_path = Path(result["texture"])
        metadata_path = Path(result["metadata"])

        run_folder = run_dir.name

        # Send results back to React
        return {
            "success": True,
            "filename": file.filename,
            "results": {
                "run_dir": run_folder,
                "heightmap": f"/outputs/{run_folder}/{heightmap_path.name}",
                "texture": f"/outputs/{run_folder}/{texture_path.name}",
                "metadata": f"/outputs/{run_folder}/{metadata_path.name}"
            }
        }

    except Exception as exc:

        print(f"\nPipeline error: {exc}")

        return {
            "success": False,
            "error": str(exc)
        }

    finally:

        # Delete temporary uploaded image
        if temp_path.exists():
            temp_path.unlink()