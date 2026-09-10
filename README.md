# hardcoders
DODGE is an AI-powered road hazard intelligence system that converts dashcam footage into real-time safety alerts and road-maintenance data. Using YOLOv8, depth estimation, GPS and DBSCAN clustering, it detects, measures, prioritizes and maps potholes while removing duplicate detections. 
# Real-Time Pothole Detection & GPS Impact Analysis

A complete computer-vision pipeline that watches dashcam footage, finds potholes,
estimates how badly each one will hurt *your specific vehicle*, geolocates it,
merges repeat sightings, and produces an interactive repair map.

```
input_dashcam.mp4
      │
      ▼
 [1] DETECT      YOLOv8 (or SimulatedDetector) ──► pixel bounding boxes
      ▼
 [2] MEASURE     ground-plane projection ──► area (m²), distance, depth proxy
      ▼
 [3] SCORE       depth + area + speed + vehicle ──► ISI ──► Low / Medium / High
      ▼
 [4] GEOTAG      GNSS fix + range + bearing ──► the POTHOLE's own lat/lon
      │
      ├──► [5a] RENDER   output_annotated.mp4   (boxes, depth labels, HUD)
      ├──► [5b] ALERT    beep + banner + alerts.log on High severity
      └──► [5c] LOG      pothole_log.csv        (one row per detection)
                  │
                  ▼
            [6] CLUSTER   DBSCAN @ 3 m ──► pothole_clusters.csv
                  ▼
            [7] MAP       folium ──► pothole_map.html
```

---

## Quick start (60 seconds, no downloads)

```bash
pip install numpy opencv-python pandas scikit-learn folium
python make_sample_video.py     # renders a synthetic dashcam clip + ground truth
python main.py                  # runs all 7 stages
python evaluate.py              # scores the result against ground truth
```

Open `outputs/pothole_map.html` in a browser.

Everything works with **zero downloads and no GPU**. If `ultralytics` isn't
installed, the pipeline transparently uses the deterministic `SimulatedDetector`
so all seven stages still run and can be demonstrated.

---

## Measured results on the synthetic benchmark

| Metric | Value |
|---|---|
| Precision | **100.0 %** |
| Recall | 65.2 % |
| F1 | 0.789 |
| Median localisation error | **0.59 m** |
| 95th-percentile localisation error | 3.12 m |
| Area measurement error (pixels → m²) | **4.7 %** mean relative |
| Raw detections → unique potholes | 375 → 15 (**96 % de-duplicated**) |
| Processing speed | 41.5 fps on CPU (**1.7× real time**) |

Recall is capped by a genuine monocular limitation, not a bug: a 0.3 m-long
pothole viewed from a 1.25 m camera height projects to a **2-pixel-tall** box at
10 m range. Small potholes only become measurable inside ~6 m, giving roughly a
0.3-second detection window. Missed potholes averaged 0.67 m² versus 1.02 m² for
found ones. Mitigations: higher camera mounting, higher frame rate, or a
second rear-facing camera.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | **Every tunable parameter.** Camera geometry, vehicle profiles, ISI weights, DBSCAN radius, alert settings |
| `main.py` | Orchestrates all seven stages |
| `make_sample_video.py` | Renders a synthetic dashcam clip **plus a ground-truth JSON sidecar** |
| `evaluate.py` | Precision / recall / localisation / area-error report vs ground truth |
| `prepare_dataset.py` | Build a YOLO dataset: split, VOC→YOLO, COCO→YOLO, synthetic, verify |
| `train_yolo.py` | Fine-tune YOLOv8 with road-specific augmentation; validate; export |
| `src/detector.py` | `YoloDetector` and `SimulatedDetector` behind one interface |
| `src/depth_impact.py` | Pinhole projection, depth proxy, Impact Severity Index |
| `src/gps_simulator.py` | Route interpolation, GNSS jitter, **pothole geolocation** |
| `src/clustering.py` | Haversine DBSCAN + cluster aggregation |
| `src/mapping.py` | Interactive folium map |
| `src/alerts.py` | Beep synthesis, 4-tier audio fallback, alert log |
| `src/renderer.py` | Bounding boxes, depth labels, HUD, warning banner |
| `src/video_utils.py` | Optional ffmpeg H.264 re-encode (≈10× smaller) |

---

## The two design decisions that matter most

### 1. Geolocate the pothole, not the car

The naive implementation tags every detection with the **car's** GPS fix. That
quietly destroys the clustering stage:

> A pothole is first detected ~25 m ahead and stays visible until the car is ~3 m
> away. Logging the car's position smears those ~50 detections along **22 metres**
> of road. DBSCAN at a 3 m radius then shreds one pothole into seven clusters,
> while two genuinely distinct potholes 10 m apart have their detection trails
> tangled together and merged.

`gps_simulator.project_pothole()` pushes the fix forward by the measured range
and sideways by the measured lateral offset, rotated into the vehicle's heading:

```
north = distance · cos H  −  lateral · sin H
east  = distance · sin H  +  lateral · cos H
```

Fixing this took the result from 9 wrong clusters to 15 correct ones.

### 2. DBSCAN needs radians, not degrees

You cannot run DBSCAN on raw lat/lon with Euclidean distance — one degree of
longitude is ~111 km at the equator and ~0 km at the poles. scikit-learn's
`metric='haversine'` gives true great-circle distance, but requires coordinates
in **radians** and interprets `eps` as **radians of arc**:

```python
coords_rad = np.radians(df[["latitude", "longitude"]].to_numpy())
eps_rad    = radius_m / 6_371_008.8          # metres → radians of arc
DBSCAN(eps=eps_rad, metric="haversine", algorithm="ball_tree", min_samples=1)
```

DBSCAN is the right algorithm here specifically because it does **not** need `k`
in advance — how many distinct potholes exist is the entire question being asked.

---

## The Impact Severity Index

```
ISI = 0.55 · depth_term  +  0.25 · area_term  +  0.20 · speed_term
```

**Depth term** is scored *relative to what the vehicle can absorb*, not in
absolute centimetres:

```
depth_term = depth / (ground_clearance + bridging_capacity)
bridging_capacity = 0.06 · tyre_diameter + 0.35 · suspension_travel
```

A 12 cm pothole is a suspension-breaker for a hatchback and a thud for a truck.
**Area term** saturates at 1.2 m² — a wide pothole can't be straddled or swerved.
**Speed term** is squared, because impact energy scales with v².

Change one line in `config.py` and the whole dataset is re-scored:

| Vehicle | Clearance | Bridging | High | Medium | Low |
|---|---|---|---|---|---|
| Hatchback | 16.5 cm | 7.7 cm | **2** | 4 | 9 |
| Sedan | 15.0 cm | 8.3 cm | **2** | 4 | 9 |
| SUV | 21.0 cm | 10.6 cm | **0** | 6 | 9 |
| Truck | 28.0 cm | 13.4 cm | **0** | 6 | 9 |
| Two-Wheeler | 16.0 cm | 6.2 cm | **2** | 4 | 9 |

Same road, same potholes — the SUV simply doesn't care about the two holes that
would damage a hatchback.

---

## Honest limitation: depth is a proxy, not a measurement

**A single camera cannot observe depth.** A 2 cm dish and a 20 cm crater can be
pixel-identical. Stage 2 is an *empirical proxy* calibrated on the documented
observation that pothole depth scales with the square root of plan area:

```
depth ≈ 9.0 · √(area / 1 m²) · shape_factor
```

with a shape penalty for elongated holes (long thin strips are usually shallow
surface ravelling; compact holes are deep punch-outs).

For true depth you need stereo, LiDAR, an IMU jolt sensor, or a learned monocular
depth network (MiDaS / Depth-Anything). **The code is deliberately structured so
Stage 2 can be swapped without touching Stages 1 and 3** — `estimate_depth_cm()`
is a single pure function.

State this openly in any demo. The *area* measurement, by contrast, is real
geometry and validates at 4.7 % error against ground truth.

---

## Training on a real dataset

```bash
pip install ultralytics

# Option A — smoke-test the pipeline with zero downloads
python prepare_dataset.py --synthetic --count 1000

# Option B — a real dataset
python prepare_dataset.py --from-voc RDD2022/India --keep D40 --out dataset
python prepare_dataset.py --verify dataset

python train_yolo.py --data dataset/data.yaml --model yolov8s.pt --epochs 100
python main.py     # now uses real YOLO inference automatically
```

Trained weights are installed to `models/pothole_yolov8.pt`, exactly where
`config.YOLO_WEIGHTS` points, so `main.py` switches backends with no config edit.

**Where to get data:**

- [Roboflow Universe](https://universe.roboflow.com/) — search "pothole", exports directly in YOLOv8 format
- [RDD2022](https://github.com/sekilab/RoadDamageDetector) — ~47,000 images from India, Japan, Czechia, Norway, USA. Keep class `D40`
- Kaggle — "Pothole Detection Dataset", "Annotated Potholes Dataset"

**Augmentation is tuned for roads, not COCO** (`train_yolo.py::ROAD_AUGMENTATION`):

- `flipud=0.0` — an upside-down road never occurs; vertical flips teach the model potholes appear in the sky
- `perspective=0.0004` — the augmentation that matters most; simulates different camera mounting heights
- `hsv_v=0.5` — lighting is the dominant nuisance variable on roads
- `box=9.0` (above the 7.5 default) — box *tightness* matters more than usual here, because a loose box directly becomes a wrong depth reading

Watch **mAP@50-95**, not mAP@50, for the same reason.

---

## CLI reference

```bash
python main.py --vehicle SUV          # re-score for a different vehicle
python main.py --input my_drive.mp4   # your own footage
python main.py --radius 5             # wider merge radius
python main.py --no-video             # skip rendering (much faster)
python main.py --preview              # live OpenCV window
python main.py --simulate             # force simulated detector
python main.py --max-frames 100       # debugging

python make_sample_video.py --seconds 30 --potholes 30
python prepare_dataset.py --verify dataset
python train_yolo.py --validate-only --weights models/pothole_yolov8.pt
python train_yolo.py --export onnx
```

Re-cluster an existing run without reprocessing the video:

```python
from src.clustering import cluster_csv
from src.mapping import map_from_csv
cluster_csv(radius_m=5.0)
map_from_csv()
```

---

## Graceful degradation

Nothing is a hard dependency except numpy, OpenCV and pandas. Every optional
component degrades with a printed notice rather than a crash:

| Missing | Behaviour |
|---|---|
| `ultralytics` or no weights | Falls back to `SimulatedDetector`, all stages still run |
| `scikit-learn` | Raw log still written; clustering and map skipped |
| `folium` | Cluster CSV still written; map skipped |
| `simpleaudio` | Falls back to winsound → aplay/afplay → terminal bell → silent |
| `ffmpeg` | Keeps the larger mp4v file, prints how to install |

---

## Output artefacts

| File | Contents |
|---|---|
| `outputs/output_annotated.mp4` | Boxes, depth labels, HUD, pulsing warning banner |
| `outputs/pothole_log.csv` | One row per detection per frame — 23 columns of raw evidence |
| `outputs/pothole_clusters.csv` | One row per physical pothole after DBSCAN |
| `outputs/pothole_map.html` | Interactive map: colour-coded markers, heatmap, popups, legend |
| `outputs/alerts.log` | Timestamped High-severity warnings |
| `outputs/evaluation_matches.csv` | Per-cluster ground-truth match detail |

The map encodes severity **twice** — colour *and* marker radius — for
colour-blind accessibility.
