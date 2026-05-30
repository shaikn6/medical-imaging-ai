# Security Audit — medical-imaging-ai

## Version: 1.1.0 — Security Hardened

**Audit Date:** 2026-05-30
**Auditor:** Security Reviewer (Claude Sonnet 4.6)
**Scope:** Full codebase — FastAPI predict endpoint, Streamlit dashboards (app.py, app_v2.py), DICOM pipeline (src/dicom_pipeline.py), preprocessing (preprocessing/dicom_handler.py), model loading, Grad-CAM / ScoreCAM, .gitignore

---

## HIPAA Compliance Note

This project processes medical imaging data (chest X-rays, DICOM files). The following properties are required before use in any clinical or research context involving real patient data:

- DICOM files contain Protected Health Information (PHI) in tags 0010,* (PatientName, PatientID, PatientBirthDate, etc.)
- **De-identification is mandatory** before ingesting real patient data
- The `scrub_phi()` function added to `src/dicom_pipeline.py` performs basic tag removal suitable for display; a full HIPAA de-identification must apply DICOM PS3.15 Annex E attribute confidentiality profiles
- Model weights trained on PHI-containing datasets must never be committed to version control
- SQLite prediction logs (`api/predictions.db`) are excluded from git; ensure they are encrypted at rest in production
- No real patient data was found in this repository at audit time — all training and test data is synthetically generated

---

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 2     | Fixed  |
| HIGH     | 5     | Fixed  |
| MEDIUM   | 3     | Fixed  |
| LOW      | 2     | Noted  |

---

## Findings and Fixes

### CRITICAL-1: Model weights tracked in git (HIPAA / supply-chain risk)

**File:** `model/xray_model.pth`, `.gitignore`

**Description:** `model/xray_model.pth` (9.5 MB) was committed to git history and not listed in `.gitignore`. If the model were retrained on real patient data, weights can leak PHI through embedding attacks. Additionally, tracking large binary model files in git enables supply-chain attacks if the repository is forked and weights are swapped with malicious ones.

**Fix applied:**
- `git rm --cached model/xray_model.pth` removes the file from the git index
- `.gitignore` now excludes `*.pth`, `*.pt`, `*.pkl`, `*.ckpt`, `*.bin`, `*.dcm`, `*.dicom`, `*.ima`
- Weights are regenerated locally via `python -m model.trainer`

---

### CRITICAL-2: DICOM parser accepts arbitrary files with `force=True`

**File:** `src/dicom_pipeline.py` — `load_dicom_image()`

**Description:** `pydicom.dcmread(path, force=True)` bypasses DICOM magic-byte validation, meaning any file on disk could be parsed. Combined with a caller-controlled `path` argument and no file-size check, this allowed: (a) parsing non-DICOM files as DICOM (triggering pydicom parser bugs), (b) denial-of-service via gigabyte-scale files.

**Fix applied:**
- Extension allow-list check before opening (`_ALLOWED_DICOM_EXTENSIONS`)
- File-size cap enforced via `os.path.getsize()` before any parsing (default: 200 MB, configurable via `MAX_DICOM_BYTES` env var)
- `force=True` removed — pydicom now rejects non-DICOM files at header validation
- `scrub_phi()` function added to strip PHI tags before any display or logging

---

### HIGH-1: FastAPI `/predict` had no file-size limit

**File:** `api/main.py` — `predict()` endpoint

**Description:** `await file.read()` read the entire upload into memory with no size cap. A 1 GB upload would exhaust server memory. No authentication or rate limiting exists (noted in LOW-1).

**Fix applied:**
- Chunked read with an early-exit check: rejects uploads beyond `MAX_UPLOAD_BYTES` (default: 10 MB, configurable via env var)
- HTTP 413 returned with a sanitized message (no internal details exposed)

---

### HIGH-2: MIME-type spoofing accepted for file uploads

**File:** `api/main.py` — `predict()` endpoint

**Description:** The original check `file.content_type.startswith("image/")` accepted `image/svg+xml`, `image/x-windows-bmp` and similar non-photographic types. An attacker could submit an SVG with embedded JavaScript or a crafted BMP that triggers Pillow parsing bugs.

**Fix applied:**
- Strict allow-list (`_ALLOWED_CONTENT_TYPES`) instead of prefix match
- Defense-in-depth: Pillow's magic-byte validation still runs on the actual bytes, rejecting content-type spoofs

---

### HIGH-3: Wildcard CORS policy on FastAPI

**File:** `api/main.py`

**Description:** `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]` allowed any origin to make credentialed cross-origin requests to the API. This enables CSRF-style attacks from malicious web pages against any authenticated session.

**Fix applied:**
- CORS restricted to `http://localhost:8501` by default (the Streamlit dashboard origin)
- Configurable via `ALLOWED_ORIGINS` environment variable (comma-separated)
- Methods restricted to `GET, POST`; headers restricted to `Content-Type, Accept`

---

### HIGH-4: Filesystem path leaked in 503 error response

**File:** `api/main.py` — `get_pipeline()`

**Description:** `detail=f"Model checkpoint not found at {CHECKPOINT}. Run model/trainer.py first."` exposed the absolute filesystem path of the model file in the HTTP response body. This reveals container internals to unauthenticated callers.

**Fix applied:**
- Error detail is now a static message: `"Model checkpoint unavailable. Contact the administrator or run the training script."`
- The actual path is logged server-side via `logger.error()` only

---

### HIGH-5: PIL exception detail leaked in 422 error response

**File:** `api/main.py` — `predict()`

**Description:** `detail=f"Cannot decode image: {exc}"` forwarded the raw PIL exception string (which can include internal decoder paths and memory addresses) to the HTTP client.

**Fix applied:**
- Exception detail is now a static `"Cannot decode image file."` message
- `from None` suppresses exception chaining to prevent accidental leakage through ASGI middleware

---

### MEDIUM-1: Uploaded filename stored without sanitization

**File:** `api/main.py` — `predict()`

**Description:** `file.filename` was stored directly in the SQLite `filename` column and returned in the JSON response. A path like `../../../../etc/passwd` or a very long string could cause issues in log parsers or downstream systems.

**Fix applied:**
- `os.path.basename()` strips directory components from the user-supplied filename
- Truncated to 256 characters before storage
- Column definition already specifies `String(256)` which enforces a DB-level cap

---

### MEDIUM-2: PHI metadata displayed in DICOM Explorer tab

**File:** `dashboard/app_v2.py` — Tab 4 (DICOM Explorer)

**Description:** `PatientName` and `PatientID` were rendered directly in the Streamlit UI with no escaping. While the current codebase uses synthetic data, this pattern would expose real PHI if the `generate_synthetic_dicom` call were replaced with real DICOM loading. The display code also used `st.write(f"**{k}**: `{v}`")` without HTML-escaping values, creating a stored-XSS risk.

**Fix applied:**
- `PatientName` and `PatientID` removed from the displayed `meta_items` dict
- All displayed metadata values passed through `html.escape()` before rendering
- Comment added explaining why PHI fields are excluded

---

### MEDIUM-3: Streamlit dashboards accepted uploads with no size cap

**File:** `dashboard/app.py`, `dashboard/app_v2.py`

**Description:** `st.file_uploader()` had `type=["png","jpg","jpeg"]` extension filtering but no size limit. Streamlit defaults to 200 MB; a large malicious image could cause memory exhaustion during PIL decoding.

**Fix applied:**
- `uploaded.size` checked against 10 MB before `Image.open()` is called
- `PIL.Image.open()` wrapped in `try/except` with a user-friendly error and `st.stop()`

---

### MEDIUM-4: Unbounded `limit` parameter on `/predictions`

**File:** `api/main.py` — `list_predictions()`

**Description:** `GET /predictions?limit=999999` would execute `SELECT ... LIMIT 999999` against the SQLite database, potentially returning the entire table and exhausting server memory.

**Fix applied:**
- `capped_limit = max(1, min(limit, 200))` clamps the query to [1, 200] records

---

### LOW-1: No authentication or rate limiting on API endpoints

**File:** `api/main.py`

**Description:** All endpoints (`/predict`, `/predictions`, `/health`, `/classes`) are unauthenticated and have no rate limiting. In a production deployment, repeated calls to `/predict` could exhaust GPU/CPU resources.

**Recommendation (not auto-fixed — requires infrastructure decisions):**
- Add `slowapi` or `fastapi-limiter` middleware for rate limiting
- Add API key authentication via `fastapi.security.APIKeyHeader` or OAuth2
- Deploy behind a reverse proxy (nginx/Caddy) with connection-rate limiting

---

### LOW-2: `generate_v2_assets.py` logs PatientID to stdout

**File:** `generate_v2_assets.py` line 254

**Description:** `f"Patient ID: {ds_ct.PatientID}"` is embedded in a matplotlib annotation in the documentation screenshot. Currently this is synthetic (`SYN123456`), but the pattern is unsafe if the function is ever called with a real DICOM.

**Recommendation (not auto-fixed — documentation script only):**
- Replace with `"Patient ID: [DE-IDENTIFIED]"` or use `scrub_phi()` before display

---

## Status

All CRITICAL and HIGH findings have been remediated in this commit. MEDIUM findings have been remediated. LOW-1 and LOW-2 require infrastructure decisions or documentation-only changes and are tracked for future work.

### Files Modified

| File | Change |
|------|--------|
| `api/main.py` | CORS lockdown, file-size limit (chunked read), content-type allowlist, filename sanitization, error message hardening, query limit cap |
| `src/dicom_pipeline.py` | Extension allowlist, file-size cap, `force=True` removed, `scrub_phi()` added |
| `dashboard/app.py` | Upload size cap, PIL error handling, HTML-escaped prediction output |
| `dashboard/app_v2.py` | Upload size cap, PIL error handling, HTML-escaped label chips, PHI fields removed from DICOM Explorer |
| `.gitignore` | Added `*.pth`, `*.pt`, `*.pkl`, `*.ckpt`, `*.bin`, `*.dcm`, `*.dicom`, `*.ima`, `.env` |
| `model/xray_model.pth` | Removed from git index (`git rm --cached`) |
