# Changelog

All notable changes to MangoSense are documented here.

---

## [Unreleased] — 2026-04-22

### Backend

#### Added
- `mangosense/views/symptom_views.py` — new `GET /api/symptoms/` endpoint
  - Accepts query params `disease` (string) and `plant_part` (`leaf` | `fruit`)
  - Returns the symptom list for the requested disease and plant part
  - Falls back to 4 generic observation prompts for unknown diseases or missing params
  - No authentication required (mobile app calls this before any user action)
  - Covers 8 diseases: Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Healthy, Powdery Mildew, Sooty Mould
- `mangosense/urls.py` — registered `symptoms/` route pointing to the new view

### Mobile (Angular/Ionic)

#### Changed
- `verify-detection.service.ts` — `getDiseaseSymptoms()` refactored from a synchronous local dictionary lookup to an async HTTP call (`Promise<string[]>`) against the new `/api/symptoms/` endpoint
  - Injected `HttpClient`; uses `firstValueFrom()` from RxJS
  - On network error or empty response, silently returns the same 4 generic fallback symptoms — no UI regression
  - Hardcoded `symptomsMap` dictionary removed from the frontend entirely

- `verify-symptoms.service.ts` — `extractAlternativeSymptoms()` made `async`
  - Callback parameter type updated from `() => string[]` to `() => Promise<string[]>`
  - Symptoms for the 2nd and 3rd ranked diseases are now fetched in parallel via `Promise.all` instead of sequentially

- `verify.page.ts` — symptom loading made async throughout
  - `getDiseaseSymptoms()` wrapper updated to return `Promise<string[]>`
  - Primary symptom fetch now `await`s the API call
  - `extractAlternativeSymptoms()` made `private async`, returning `Promise<void>`
  - Both awaited calls sit inside the existing `try/catch` block that drives the image-analysis loading spinner — no additional spinner changes needed

---

## [0.5.0] — 2026-04-22

### Backend

#### Added
- Supabase PostgreSQL migration replacing SQLite for production database
- AWS S3 (Supabase Storage) integration for image file storage
- `training_ready` and `training_notes` fields on `MangoImage` for admin-controlled training data approval gate
- Training data editor endpoints: `GET/PATCH /api/training-data/<pk>/`, `GET /api/training-data/summary/`, `POST /api/training-data/bulk-approve/`
- Model retraining endpoints: `POST /api/retrain/`, `GET /api/retrain/status/`, `GET /api/retrain/dataset-info/`
- Background threading pattern for non-blocking model retraining (`mangosense/ML/retrain.py`)
- Disease location endpoints for map visualisation: `GET /api/disease-locations/similar/`, `GET /api/disease-locations/all/`
- Admin training-ready verification flow in the admin dashboard

#### Fixed
- 11 silent S3/PostgreSQL integration bugs (documented in `issues and fix/`)
- Mango image file saving to S3 bucket path resolution
- `.exclude(selected_symptoms=[])` PostgreSQL empty-array query fix

---

## [0.4.0] — prior

### Backend

#### Added
- Gate model validation: `gate_leaf.keras` and `gate_fruit.keras` reject non-mango images before disease classification
- `gateValidation` field in predict response (`passed`, `message`, `top_class`, `confidence`)
- Heatmap / disease location aggregation
- Model settings admin endpoints for toggling active models without server restart
- Admin dashboard APIs: disease statistics, classified images list/detail, bulk update, export dataset

### Mobile

#### Added
- Verify page with 4-step wizard: symptom selection → location → notes → confirm
- Symptom checklist showing primary disease symptoms and alternative symptoms from top-3 ranked diseases
- Leaflet map embedded via `srcdoc` iframe (self-contained, no external script dependency issues)
- Similar disease / all disease location markers on map
- GPS location detection with reverse geocoding (Nominatim primary, BigDataCloud fallback)
- Gate rejection screen when backend rejects the uploaded image as non-mango
- User analysis history page

---

## [0.3.0] — prior

### Backend

#### Added
- JWT authentication (`djangorestframework-simplejwt`) for mobile and admin
- Admin login with refresh token endpoint
- User registration and login
- Notification system
- User confirmation / feedback save endpoint

### Mobile

#### Added
- Login and registration screens
- Profile settings with photo
- History page
- Push notification handling

---

## [0.2.0] — prior

### Backend

#### Added
- Two-stage disease classification pipeline: gate model → disease model
- `disease_leaf.keras` / `disease_fruit.keras` models (224×224 input)
- `MangoImage`, `MLModel`, `PredictionLog` database models
- `/api/predict/` endpoint accepting multipart image upload
- Confidence threshold (configurable, default 50% for gate, 30% for display)

---

## [0.1.0] — prior

- Initial Django project scaffold (`mangoAPI/`, `mangosense/`)
- Basic REST framework setup
- Docker and Railway deployment configuration
