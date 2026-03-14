#!/usr/bin/env python3
"""Script to create mock test data for verifying the evaluation pipeline."""
import json, os

os.makedirs("test_data/patient_001", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Mock raw OCR text
md_content = """PATIENT CHART

Name: John Doe | DOB: 1965-04-12 | MRN: 123456

CHIEF COMPLAINT
Patient presents with chest pain and shortness of breath.

MEDICATIONS
- Aspirin 81mg daily (oral)
- Lisinopril 10mg once daily for hypertension

PAST MEDICAL HISTORY
- Hypertension (diagnosed 2010)
- Type 2 Diabetes Mellitus

ASSESSMENT & PLAN
Patient has stable coronary artery disease. Continue aspirin.
Schedule stress test. Monitor HbA1c.

VITAL SIGNS
Blood Pressure: 140/90 mmHg
Heart Rate: 78 bpm
HbA1c: 7.2%
"""

# Ground truth entities
gt = [
  {"entity": "Aspirin", "entity_type": "MEDICINE", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT", "heading": "MEDICATIONS", "text": "Aspirin 81mg daily (oral)", "metadata_from_qa": {"DOSE": "81mg", "FREQUENCY": "daily", "ROUTE": "oral"}},
  {"entity": "Lisinopril", "entity_type": "MEDICINE", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT", "heading": "MEDICATIONS", "text": "Lisinopril 10mg once daily for hypertension", "metadata_from_qa": {"DOSE": "10mg", "FREQUENCY": "once daily"}},
  {"entity": "Hypertension", "entity_type": "PROBLEM", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT", "heading": "PAST MEDICAL HISTORY", "text": "Hypertension (diagnosed 2010)", "metadata_from_qa": {}},
  {"entity": "Type 2 Diabetes Mellitus", "entity_type": "PROBLEM", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT", "heading": "PAST MEDICAL HISTORY", "text": "Type 2 Diabetes Mellitus", "metadata_from_qa": {}},
  {"entity": "coronary artery disease", "entity_type": "PROBLEM", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT", "heading": "ASSESSMENT & PLAN", "text": "Patient has stable coronary artery disease", "metadata_from_qa": {}},
  {"entity": "stress test", "entity_type": "PROCEDURE", "assertion": "POSITIVE", "temporality": "UPCOMING", "subject": "PATIENT", "heading": "ASSESSMENT & PLAN", "text": "Schedule stress test", "metadata_from_qa": {}},
  {"entity": "HbA1c", "entity_type": "TEST", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT", "heading": "VITAL SIGNS", "text": "HbA1c: 7.2%", "metadata_from_qa": {"TEST_VALUE": "7.2", "TEST_UNIT": "%"}}
]

# Mock LLM output (intentionally imperfect for testing)
pred = [
  {"entity": "aspirin", "entity_type": "MEDICINE", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT", "heading": "MEDICATIONS", "text": "Aspirin 81mg daily", "metadata_from_qa": {"DOSE": "81mg", "FREQUENCY": "daily", "ROUTE": "oral"}},
  {"entity": "lisinopril", "entity_type": "MEDICINE", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT", "heading": "MEDICATIONS", "text": "Lisinopril 10mg once daily", "metadata_from_qa": {"DOSE": "10mg", "FREQUENCY": "once daily"}},
  {"entity": "hypertension", "entity_type": "PROBLEM", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT", "heading": "PAST MEDICAL HISTORY", "text": "Hypertension", "metadata_from_qa": {}},
  {"entity": "diabetes", "entity_type": "PROBLEM", "assertion": "POSITIVE", "temporality": "CLINICAL_HISTORY", "subject": "PATIENT", "heading": "PAST MEDICAL HISTORY", "text": "Type 2 Diabetes Mellitus", "metadata_from_qa": {}},
  {"entity": "hallucinated drug", "entity_type": "MEDICINE", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT", "heading": "MEDICATIONS", "text": "hallucinated drug 50mg", "metadata_from_qa": {}}
]

with open("test_data/patient_001/patient_001.md",   "w") as f: f.write(md_content)
with open("test_data/patient_001/patient_001.json", "w") as f: json.dump(gt, f, indent=2)
with open("output/patient_001.json",                "w") as f: json.dump(pred, f, indent=2)

print("Mock data created successfully!")
print(f"  test_data/patient_001/patient_001.md   ({len(md_content)} chars)")
print(f"  test_data/patient_001/patient_001.json  ({len(gt)} entities)")
print(f"  output/patient_001.json                 ({len(pred)} pred entities)")
