# Document Classifier — Developer Guide

Zero-shot document classification using SigLIP 2. All category definitions
live in `categories.yaml`; no Python code changes are needed to add or tune
classes.

---

## Running locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`pdf2image` also requires `poppler` on the system path:

```bash
# macOS
brew install poppler

# Ubuntu / Debian
sudo apt-get install poppler-utils
```

### 2. Start the API server

```bash
cd app
uvicorn main:app --reload --port 8000
```

The server preloads the SigLIP 2 model on startup (~10–20 s first run while
weights download). Subsequent starts use the HuggingFace cache.

### 3. Open the frontend

Open `frontend/index.html` directly in a browser, or serve it:

```bash
python3 -m http.server 5500 --directory frontend
# then visit http://localhost:5500
```

The frontend expects the API at `http://localhost:8000`. Change the `API`
constant at the top of the `<script>` block if you move the server.

### 4. Verify the server is healthy

```
GET http://localhost:8000/health
```

Returns model load status, device (mps / cpu), category count, and the
last-modified timestamp of `categories.yaml`.

---

## How to add a new single-hypothesis class

Edit `categories.yaml` only — no Python changes required.

```yaml
# categories.yaml  (append under the "categories:" list)

- label: insurance_policy          # snake_case, unique
  display_name: Insurance Policy   # shown in UI and API responses
  prompt: "an insurance policy document showing policyholder name coverage type sum insured premium and insurer stamp"
```

Rules:
- `label` must be unique across all categories.
- `prompt` should describe the visual appearance of the document in plain
  English. More specific = better accuracy.
- Save the file. The running server hot-reloads within ~1 s (watchdog).
- No server restart needed.

---

## How to add a new hypothesis to an existing multi-hypothesis class

Locate the class in `categories.yaml` and append one line under its
`hypotheses:` list:

```yaml
- label: kyc_document
  display_name: KYC Document
  hypotheses:
    - "an Aadhaar card showing ..."
    - "an Income Tax Department PAN card ..."
    # ... existing hypotheses ...
    - "a CGHS health card issued by the Central Government Health Scheme with beneficiary name card number and photograph"  # ← new
```

Rules:
- Add as many hypotheses as needed; the classifier takes the **max sigmoid
  score** across all of them, so extra hypotheses can only raise recall, not
  hurt precision for other classes.
- Save the file. The server hot-reloads automatically.

---

## How to add a brand new multi-hypothesis class

Use `hypotheses:` instead of `prompt:` and list at least two descriptions:

```yaml
- label: court_document
  display_name: Court Document
  hypotheses:
    - "a High Court or Supreme Court order with case number petitioner respondent and bench signature"
    - "a civil or criminal court summons or notice with court seal cause title and date of hearing"
    - "a court-certified copy of a judgment with court name case number date of order and certified true copy stamp"
```

When to prefer multi-hypothesis over single:
- The document class has visually distinct sub-types (e.g. passports vs
  Aadhaar cards are both KYC but look very different).
- A single prompt leaves accuracy gaps on common variants.
- Add one hypothesis per distinct visual sub-type, not per synonym.

---

## How to adjust the gate threshold

Edit `gate.default_threshold` in `categories.yaml`:

```yaml
gate:
  type: per_class_sigmoid
  default_threshold: 0.05   # ← change this value
  secondary_multiplier: 3.0
  reject_label: out_of_category
```

Effect of changing the threshold:

| Direction | Effect |
|---|---|
| Lower (e.g. `0.02`) | More documents classified; higher false-positive rate |
| Higher (e.g. `0.10`) | Fewer documents classified; more rejections to `out_of_category` |

`secondary_multiplier` is reserved for future two-stage gating and has no
effect in the current `per_class_sigmoid` implementation.

The server hot-reloads on save. No restart required.

---

## Known limitations

**No OCR / text understanding.** SigLIP 2 is a vision-language model that
matches image patches against text descriptions. It does not read or parse
text in the document. Two documents with identical layouts but different
textual content will score the same.

**Single-page representative result for PDFs.** For multi-page PDFs, every
page is scored independently and the page with the highest winning-class
score is returned as the top-level result. Page order is not used; a
cover-page-less contract may be classified by its signature page.

**DPI sensitivity.** PDF pages are rasterised at 200 DPI. Very dense small
print may be unreadable to the model at this resolution. Increase `dpi` in
`preprocessor._images_from_pdf` if accuracy on dense documents is poor.

**CPU / MPS only.** CUDA is not wired up. Inference on CPU for a 10-page PDF
typically takes 15–40 s depending on hardware. MPS (Apple Silicon) is
substantially faster (~3–8 s).

**Gate threshold is global.** All classes share `default_threshold`. If one
class consistently over- or under-fires, the only current lever is to rewrite
its prompt(s) to be more or less specific. Per-class threshold overrides are
not yet implemented.

**Scanned document quality.** Poor scan quality (skew, shadows, low contrast,
handwritten annotations covering printed content) degrades accuracy. The
preprocessor does not apply deskew, denoising, or contrast enhancement.

**Model cold start.** First request after server start triggers model loading
if the weights are not cached locally (~800 MB download). Subsequent starts
use the local HuggingFace cache.

---

## Demo script (client walkthrough)

Use this sequence for a live demo. Prepare sample files in advance from
`sample_documents/`.

### Step 1 — Health check (30 s)

Open `http://localhost:8000/health` in a browser tab. Point out:

- `"model_loaded": true` — model is warm, no cold-start latency
- `"device": "mps"` (or `"cpu"`) — running locally, no cloud dependency
- `"categories_count": 23` — all category definitions loaded from YAML

### Step 2 — Classify a KYC document (~5 s)

Upload an Aadhaar card or PAN card image via the frontend.

Expected result:
- **Predicted class:** KYC Document
- **Confidence:** typically HIGH band (green bar, ≥ 70%)
- Open "All Scores" — show that other classes score near zero

Talking point: *"The model has never been fine-tuned on Indian documents.
It reasons purely from the visual description in categories.yaml."*

### Step 3 — Classify a bank statement (~5 s)

Upload a multi-page PDF bank statement.

Expected result:
- **Predicted class:** Bank Account Statement
- Page tabs appear — click through to show per-page results
- Inner pages (transaction tables) typically score higher than the cover page

Talking point: *"Multi-page documents are scored page by page. The page with
the strongest signal wins."*

### Step 4 — Trigger a REJECT (~5 s)

Upload a screenshot, logo image, or any non-document file.

Expected result:
- **REJECTED — Out of Category** (red status chip)
- Confidence bar is red (LOW band)
- Warning: "Low confidence — recommend manual review"

Talking point: *"The gate threshold prevents weak matches from being
auto-classified. Anything below 5% sigmoid score is rejected."*

### Step 5 — Live YAML edit (~2 min)

With the server running, open `categories.yaml` and add a new single-hypothesis
class live:

```yaml
- label: insurance_policy
  display_name: Insurance Policy
  prompt: "an insurance policy document showing policyholder name coverage type sum insured premium and insurer stamp"
```

Save the file. Immediately call `GET /categories` — the new class appears
without a restart.

Talking point: *"Adding a category is a one-line YAML edit. No retraining,
no redeployment, no code change."*

### Step 6 — Copy JSON result

Click "Copy JSON" on any result. Paste into a text editor to show the full
response shape: `predicted_label`, `confidence`, `all_scores` array, per-page
breakdown.

Talking point: *"The API response is designed for downstream integration —
pipe it into your case management system, DMS, or workflow engine."*
