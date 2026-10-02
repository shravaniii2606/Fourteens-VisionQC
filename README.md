# VisionQC

VisionQC is a local visual quality inspection demo for small factories. It learns what a good product looks like from example images, then checks images or video frames for visual anomalies and reports a score, heatmap, and inspection verdict.

## Project Overview

The project includes a React web interface and a FastAPI backend. The backend builds a PatchCore-style memory bank of features from known-good examples and compares each inspected frame against that reference. Inspection records, settings, and analytics are stored locally in SQLite. Most inspection workflows run on the same machine; the optional Ask the Log feature uses OpenRouter.

## Setup & Installation Instructions

### Requirements

- Python 3.11
- Node.js and npm
- Windows PowerShell commands below (adapt activation commands for your shell if needed)
- Optional: webcam for live capture, and MVTec AD for dataset fitting/evaluation
- Optional: OpenRouter API key for Ask the Log

Create a Python virtual environment from the repository root and install the CPU PyTorch build and backend dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend\requirements.txt
```

Install frontend packages and configure the API origin if it differs from the default:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
```

Set `VITE_API_URL` in `frontend/.env` to the backend URL when needed. By default, the frontend uses `http://localhost:8000`.

For Ask the Log, create a root `.env` file with:

```text
OPENROUTER_API_KEY=your-openrouter-api-key
```

The default model is `openai/gpt-4o-mini`. `OPENROUTER_MODEL` can be changed in `backend/config.py`. Restart the backend after changing configuration.

### Run locally

Start the backend in one repository-root terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```powershell
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Choose **Fit from dataset** or upload at least two good product photos before inspection. Browser camera access requires permission and works on localhost. Video is read in the browser and sent frame by frame. The first model startup may download pretrained torchvision weights; later model state and database data are stored locally under `data/`.

Create a production frontend bundle with `npm run build` from `frontend/`.

## Key Features

- Teach the model from good images or the MVTec AD dataset.
- Inspect webcam frames, uploaded images, or video frames.
- Show anomaly scores, confidence, anomaly heatmaps, and pass/fail/recapture decisions.
- Check image capture quality, including brightness, blur, and framing.
- Tune an inspection threshold and review its calibration information.
- Review inspection history, dashboard metrics, alerts, repeat-defect heatmaps, and DriftGuard status.
- Provide feedback on a result to help identify false alarms.
- Ask questions about logged inspection data with the optional OpenRouter-backed chat.
- Demonstrate crack healing in the landing page as a visual explanation of anomaly scoring.

## Technology Stack

| Area | Technologies |
| --- | --- |
| Frontend | React, Vite, React Router, Recharts, Lucide icons, plain CSS |
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| Computer vision and ML | PyTorch, torchvision, OpenCV, NumPy, Pillow; PatchCore-style feature memory and nearest-neighbor scoring |
| Storage | SQLite for inspection logs and application data; local files under `data/` for model memory and frames |
| Optional language model | OpenAI-compatible client connected to OpenRouter |
| Dataset | MVTec Anomaly Detection (MVTec AD), `bottle` category by default |

## Architecture / Workflow

```text
Good product images ──> FastAPI /fit ──> feature memory bank + calibration
                                                   │
Webcam / image / video ──> React ──> /inspect ────┤
                                                   ├──> score + heatmap + verdict
                                                   └──> SQLite history and analytics
                                                            │
React dashboard <── /stats, /history, /alerts, analytics ───┘
React Ask the Log ──> /chat ──> OpenRouter (optional; grounded in local log data)
```

The default backbone is torchvision `wide_resnet50_2`; set `BACKBONE = "resnet18"` in `backend/config.py` for a smaller CPU model. The app limits fitting to 30 sorted good images by default. Nearest-neighbor queries use chunked `torch.cdist`; the project does not use anomalib or FAISS. ORB alignment is disabled by default and can be enabled for a fixed camera/reference rig through `ALIGN_ENABLED` in `backend/config.py`.

## Dataset / API Information

### MVTec AD

Download MVTec AD from the [official dataset page](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract the `bottle` category in this layout:

```text
data/mvtec/bottle/train/good/*.png
data/mvtec/bottle/test/good/*.png
data/mvtec/bottle/test/<defect-type>/*.png
```

The default category is `bottle`; change `CATEGORY` in `backend/config.py` to select another category. Dataset fitting uses up to the first 30 sorted images in `train/good` for model fitting and calibration. Alternatively, upload at least two good images through the UI.

### Backend API

The FastAPI service defaults to `http://localhost:8000`. Interactive endpoint documentation is available at `http://localhost:8000/docs` while it is running.

| Endpoint | Purpose |
| --- | --- |
| `POST /fit` | Fit from the configured dataset or uploaded good images |
| `POST /inspect` | Inspect an uploaded frame |
| `GET/POST /threshold` | Read or update the inspection threshold |
| `GET /threshold/suggestion` | Suggest a threshold based on calibration |
| `GET /health`, `GET /model_info` | Check service and model status |
| `GET /stats`, `GET /history`, `GET /alerts` | Read inspection metrics, records, and alerts |
| `GET /cumulative_heatmap`, `GET /driftguard/status` | Read repeat-defect and drift analytics |
| `POST /feedback/{inspection_id}` | Record feedback for an inspection |
| `POST /chat` | Ask a question about logged inspection data (OpenRouter key required) |

Other settings and demo endpoints are documented by FastAPI at `/docs`. SQLite is stored at `data/visionqc.db`; runtime model and frame data are also kept under `data/`.

To run the evaluation script with the dataset installed:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\evaluate.py --category bottle
```

It reports AUROC, detection rate at the calibration-derived threshold, and false alarm rate.

To fit from a known-good video and replay it through the inspection API:

```powershell
.\.venv\Scripts\python.exe scripts\fit_from_video.py path\to\good_clip.mp4 --frames 30
```

Use a clip containing only known-good product frames for fitting. The tool samples frames, fits the model, then replays the clip and prints capture thresholds and verdict percentages.

Optional synthetic dashboard data can be added with:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\seed_demo_data.py
```

This appends 60 synthetic inspections to the local database each time it runs. Use a fresh or empty database for a clean demo.

## Screenshots / Demo Information

Run the frontend and backend using the instructions above to explore the landing page, inspection workspace, dashboard, history, alerts, analytics, and Ask the Log interface. The repository does not currently include a maintained screenshot gallery or hosted demo URL. Browser login is demo-only: any valid email and non-empty password opens the local demo workspace; replace it with real authentication before production use.
<img width="1900" height="827" alt="image" src="https://github.com/user-attachments/assets/58e77846-97c1-4185-8bb7-bbcd6c810118" />
<img width="1897" height="883" alt="image" src="https://github.com/user-attachments/assets/f06687b6-7762-4518-9bed-bcba3c666f29" />
<img width="1901" height="720" alt="image" src="https://github.com/user-attachments/assets/f5d1f5ea-48cf-476a-b80d-4f1a833b971e" />
<img width="1870" height="887" alt="image" src="https://github.com/user-attachments/assets/cf2b9b88-6ff9-4179-9acc-a3bed3dd6da6" />
<img width="1877" height="876" alt="image" src="https://github.com/user-attachments/assets/99cd248f-4b83-4fb3-8ba6-3aa3a0561c6b" />
<img width="1916" height="866" alt="image" src="https://github.com/user-attachments/assets/53ef8ee2-9915-477a-b53f-52f6de217b17" />

<img width="1886" height="907" alt="image" src="https://github.com/user-attachments/assets/48aea3d0-45d8-4a16-8787-0745fcfbc5c3" />
<img width="1897" height="910" alt="image" src="https://github.com/user-attachments/assets/fabb46b5-bda9-4ab7-a9e7-dd699e4b9a77" />
<img width="1877" height="897" alt="image" src="https://github.com/user-attachments/assets/11d14a8c-1011-439d-a1ce-8677d53694ec" />
<img width="1917" height="882" alt="image" src="https://github.com/user-attachments/assets/a229aaa2-443a-4fc3-b395-3f4f68344c7e" />
<img width="1895" height="761" alt="image" src="https://github.com/user-attachments/assets/8fc0622a-6f40-423a-a2cd-546c00dd1d33" />
<img width="1886" height="898" alt="image" src="https://github.com/user-attachments/assets/5e3e4d8d-7c2c-4136-a5da-e00c87c4e679" />
<img width="1873" height="892" alt="image" src="https://github.com/user-attachments/assets/3f8dd0fc-9f35-404b-8920-0e8160dec97e" />
<img width="1872" height="901" alt="image" src="https://github.com/user-attachments/assets/66eb20e6-16f8-40b4-8687-39882b314436" />
<img width="1882" height="632" alt="image" src="https://github.com/user-attachments/assets/7e16967c-508d-4b03-84fa-080d212e9da7" />
<img width="1870" height="882" alt="image" src="https://github.com/user-attachments/assets/bdc71974-fd28-48de-8e67-5d8d682e74be" />
<img width="1902" height="790" alt="image" src="https://github.com/user-attachments/assets/9d623f76-9dd6-49f3-8520-6516b63538d3" />

## Limitations & Future Scope

### Current limitations

- This is a local demonstration, not a production quality-control system.
- Model performance depends on representative good reference images, stable framing, lighting, and camera conditions.
- The default fitting path uses the `bottle` dataset category; other products may need configuration and their own calibration data.
- CPU inference is supported, but throughput depends on the machine and model backbone.
- ORB alignment is disabled by default and requires a suitable fixed camera/reference setup.
- Authentication is demo-only, and the optional chat requires a network connection and an OpenRouter API key.
- Evaluation metrics from MVTec AD or a demo dataset do not guarantee performance on a factory production line.

### Future scope

- Add production authentication, access control, and deployment configuration.
- Support configurable product categories, camera profiles, and multi-line workspaces.
- Improve calibration and evaluation workflows with customer-specific datasets and repeatable reports.
- Add deployment monitoring, model versioning, and operational data export.
- Expand the demo documentation with current screenshots and a recorded walkthrough.

## Team Members

- Shravani Chaudhari
- Mitanshi Khanna
- Shirley Castelino
- Shravani Kolekar
