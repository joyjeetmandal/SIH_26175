from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import numpy as np
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
    image_bytes = await file.read()
    suffix = Path(file.filename).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(image_bytes)
        temp_path = Path(temp_file.name)

    try:
        print("\n==============================")
        print("Received file:", file.filename)
        print("Temporary file:", temp_path)
        print("==============================")

        # Run the AI terrain pipeline
        result = run_pipeline(temp_path)

        print("\nAI pipeline finished!")
        print("Pipeline result:", result)

        # --------------------------------------------------
        # Find the output directory
        # --------------------------------------------------

        if "run_dir" in result:
            run_dir = Path(result["run_dir"])

        elif (
            "pipeline" in result
            and "output_directory" in result["pipeline"]
        ):
            run_dir = Path(result["pipeline"]["output_directory"])

        else:
            raise RuntimeError(
                f"Pipeline completed but no output directory was returned. "
                f"Result keys: {list(result.keys())}"
            )

        # --------------------------------------------------
        # Expected output files
        # --------------------------------------------------

        heightmap_path = run_dir / "heightmap.png"
        texture_path = run_dir / "texture.png"
        metadata_path = run_dir / "metadata.json"
        heightmap_npy_path = run_dir / "heightmap.npy"
        
        heightmap_data = np.load(heightmap_npy_path).astype(np.float32).tolist()

        # --------------------------------------------------
        # Verify files exist
        # --------------------------------------------------

        if not heightmap_path.exists():
            raise RuntimeError(
                f"Heightmap not found: {heightmap_path}"
            )

        if not texture_path.exists():
            raise RuntimeError(
                f"Texture not found: {texture_path}"
            )

        if not metadata_path.exists():
            raise RuntimeError(
                f"Metadata not found: {metadata_path}"
            )

        run_folder = run_dir.name

        print("\nGenerated files:")
        print("  Heightmap:", heightmap_path)
        print("  Texture:", texture_path)
        print("  Metadata:", metadata_path)

        # --------------------------------------------------
        # Send URLs to React
        # --------------------------------------------------

        return {
            "success": True,
            "filename": file.filename,
            "results": {
                "run_dir": run_folder,
                "heightmap": f"/outputs/{run_folder}/{heightmap_path.name}",
                "texture": f"/outputs/{run_folder}/{texture_path.name}",
                "metadata": f"/outputs/{run_folder}/{metadata_path.name}",
                "heightmap_data": heightmap_data,
            },
        }

    except Exception as exc:
        print(f"\nPipeline error: {exc}")

        return {
            "success": False,
            "error": str(exc),
        }

    finally:
        if temp_path.exists():
            temp_path.unlink()