# HiLabs Workshop — GenAI Evaluation Framework

An evaluation and reliability framework for a healthcare AI pipeline that processes unstructured medical documents:

```
OCR → Clinical NLP → Entity Extraction
```

Workshop workflow mapping:

```
Medical PDF → OCR → Text → NLP Entity Extraction → Structured Data (JSON)
```

---

## 🗂️ Project Structure

```
.
├── test_data/            ← Place downloaded dataset folders here
│   ├── folder_1/
│   │   ├── folder_1.json   (ground truth entities)
│   │   └── folder_1.md     (raw OCR text)
│   └── ...               (30 folders total)
├── output/               ← LLM-generated entity JSON files (auto-created)
├── src/
│   ├── evaluator.py      ← Core entity comparison & scoring engine
│   ├── llm_extractor.py  ← LLM pipeline (Gemini/OpenAI/OpenRouter/Mistral)
│   ├── heatmap.py        ← Error heat-map & reporting
│   └── utils.py          ← Shared helpers
├── test.py               ← Submission entry-point: python test.py input.json output.json
├── run_pipeline.py       ← Full pipeline: extract + evaluate + report
├── requirements.txt
├── .env.example
└── report.md             ← Auto-generated after running pipeline
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure LLM API key

```bash
cp .env.example .env
# Edit .env and add your API key
```

Supported LLMs (set `LLM_PROVIDER` in `.env`):
- `gemini` — Google Gemini 1.5 Flash (free tier) ✅ Recommended
- `openai` — OpenAI GPT-4o mini
- `openrouter` — OpenRouter (OpenAI-compatible endpoint)
- `mistral` — Mistral API (OpenAI-compatible endpoint)

### 3. Download dataset

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1Elnuj6n7QDazhmSsgCMkL1g9UOBTEdkQ) and place all 30 folders inside `test_data/`.

---

## 🚀 Running the Pipeline

### Full pipeline (extract + evaluate + report)

```bash
python run_pipeline.py
```

This will:
1. Pass each `.md` file through the LLM → saves to `output/`
2. Compare LLM output vs. ground truth `.json`
3. Generate `report.md` + `output/evaluation_report.json`

---

## 🧪 Running Evaluation Only (`test.py`)

```bash
python test.py input.json output.json
```

**`input.json` format** (single document evaluation):
```json
{
  "ground_truth_path": "test_data/folder_1/folder_1.json",
  "prediction_path": "output/folder_1.json"
}
```

Or batch evaluation across all documents:
```json
{
  "mode": "batch",
  "test_data_dir": "test_data",
  "output_dir": "output"
}
```

**`output.json`** will contain a structured evaluation report with per-entity scores, field-level accuracy, and error analysis.

---

## 📊 Output & Reports

After running the pipeline:
- `output/evaluation_report.json` — machine-readable full evaluation
- `report.md` — human-readable report with heat-maps, error analysis, and proposed guardrails

The generated report includes:
- Input vs output comparison examples (source text vs AI output)
- Error classification rows: source text, AI output, error type, explanation, corrected output
- Reliability checks: source consistency, hallucination flags, negation validation, temporal validation, subject attribution validation

---

## 🛠️ Tech Stack

| Layer | Choice |
|---|---|
| LLM | Gemini 1.5 Flash / GPT-4o mini / OpenRouter / Mistral |
| Language | Python 3.9+ |
| Matching | RapidFuzz (fuzzy entity matching) |
| Visualization | Matplotlib + Seaborn (heat-maps) |
| Reporting | Markdown auto-generation |
