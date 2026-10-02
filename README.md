# VisionQC

A local visual-defect inspection demo. It builds a PatchCore-style normal memory bank from good images, then scores camera or video frames on the same machine using CPU inference.

## Requirements

- Python 3.11 (the source can be syntax-checked on Python 3.12; install dependencies into Python 3.11 for the requested runtime).
- Node.js and npm.
- A webcam is optional. An MVTec AD download is only needed for dataset fitting/evaluation; alternatively upload at least two good photos.
- The first model startup downloads torchvision's pretrained weights if they are not already cached. Thereafter the memory bank and database are local. Chat calls OpenRouter and requires an API key; all other workflows run locally.

## Install

From the repository root, create/activate a Python 3.11 virtual environment and install the CPU PyTorch build plus backend requirements:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend\requirements.txt
```

For OpenRouter chat, add your key to the root `.env` file:

```text
OPENROUTER_API_KEY=your-openrouter-api-key
```

The default model is `openai/gpt-4o-mini`. Change `OPENROUTER_MODEL` in `backend/config.py` to another model enabled for your OpenRouter account. Restart the backend after editing `.env`.

Install the frontend packages:

```powershell
cd frontend
npm install
cd ..
```

## MVTec AD

Download the MVTec Anomaly Detection dataset from [MVTec's official AD dataset page](https://www.mvtec.com/company/research/datasets/mvtec-ad). Extract the `bottle` category so the paths are:

```text
data/mvtec/bottle/train/good/*.png
data/mvtec/bottle/test/good/*.png
data/mvtec/bottle/test/<defect-type>/*.png
```

The default category is `bottle`; change `CATEGORY` in `backend/config.py` to select another category. The app uses up to the first 30 sorted images in `train/good` for fitting and calibration.

## Run

In one PowerShell terminal from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173`. Choose **Fit from dataset** or upload at least two good photos before inspecting. Browser camera access requires localhost permission. Video files are read in-browser and submitted frame-by-frame. Data is stored under `data/`; the model is not refit on startup.

To enable the data-only chat, set `OPENROUTER_API_KEY` in the root `.env` file. Questions are answered using only inspection data from SQLite.

## Evaluate

With MVTec AD at the documented path:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\evaluate.py --category bottle
```

The script fits on up to 30 good training photos and evaluates every test image, printing AUROC, detection rate at the calibration-derived threshold, and false alarm rate. Evaluation writes the same local memory bank as the app.

## Seed dashboard data

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\seed_demo_data.py
```

This inserts 60 synthetic inspections with `source='seed'`, timestamps spread across the previous day, and a repeated FAIL cluster. Run it against a fresh/empty database for a clean demo; the script appends rows each time.

## Implementation notes

- Backbone defaults to torchvision `wide_resnet50_2`; set `BACKBONE = "resnet18"` in `backend/config.py` for a smaller CPU model.
- No anomalib or FAISS is used. Nearest-neighbor queries use chunked `torch.cdist`.
- ORB alignment is off by default because the included MVTec demo frames are already framed and provide too few stable ORB matches. Set `ALIGN_ENABLED = True` in `backend/config.py` for a fixed camera/reference rig, then restart the backend.
- The default UI/API pair uses `localhost:5173` and `localhost:8000`.
