# Selective Demo Batch Report (Complete)
**Mode:** fast  
**Generated:** 2026-11-27T15:43:02.184468+00:00  
**Patients evaluated:** 4  

## 📊 Aggregate Performance Summary

- **Total Patients:** 4
- **Path Match Rate:** 4/4 (100%)
- **Fast Path:** 2  |  **Slow Path:** 2
- **Human Review Required:** 2
- **Total Quality Flags:** 15

- **Total Time (s):** median=385.2  min=237.9  max=559.9  mean=360.9
- **Scribe Time (s):** median=260.7  min=230.3  max=365.7
- **Coding Time (s):** median=119.5  min=0.0  max=329.6
- **Scribe % of Total:** 75.8%
- **Coding % of Total:** 31.1%

## Summary Comparison

| Patient | Name | Expected | Actual | Match | Total (s) | Scribe (s) | Coding (s) | Primary ICD-10 | Secondary | CPT | Flags | Human Review |
|---------|------|----------|--------|-------|-----------|------------|------------|----------------|-----------|-----|-------|--------------|
| P001 | John Doe | fast | fast | ✅ | 237.9 | 237.8 | 0.0 | E11.9 - Type 2 diabetes mellitus without complications | R53.83 - Other fatigue | 99214 - Office visit, established patient, moderate medical decision making | 3 | — |
| P002 | Maria Gonzalez | fast | fast | ✅ | 260.8 | 260.7 | 0.0 | I10 - Essential (primary) hypertension | E78.5 - Hyperlipidemia, unspecified; N18.31 - Chronic kidney disease, stage 3a | 99215 - Office visit, established patient, high medical decision making | 4 | ⚠️ |
| P005 | James Wilson | slow | slow | ✅ | 559.9 | 230.3 | 329.6 | R05 - Cough | J06.9 - Acute upper respiratory infection, unspecified | 99214 - Office visit, established patient, moderate medical decision making | 5 | ⚠️ |
| P007 | Ahmed Hassan | slow | slow | ✅ | 385.2 | 365.7 | 119.5 | R07.9 - Chest pain, unspecified | R06.02 - Shortness of breath; R11.0 - Nausea; R53.83 - Other fatigue | 99214 - Office visit, established patient, moderate medical decision making | 3 | — |

## Detailed Per-Patient Results

### P001 — John Doe
**Expected Path:** fast  
**Actual Path:** fast  
**Path Match:** ✅ Yes  
**Demo Notes:** Classic chronic DM+HTN follow-up with lab data. Primary ICD-10 (E11.9, I10) and CPT (99214) derivable deterministically. Benchmark case.  

#### Timing
- **Total:** 237.9s  
- **Memory Load Seconds:** 0.0s  
- **Scribe Seconds:** 237.8s  
- **Cleanup Seconds:** 0.0s  
- **Coding Seconds:** 0.0s  
- **Validation Seconds:** 0.0s  
- **Supervisor Seconds:** 0.0s  
- **Memory Save Seconds:** 0.0s  

#### Coding Results
- **Primary ICD-10:** E11.9 - Type 2 diabetes mellitus without complications  
- **Secondary ICD-10:** R53.83 - Other fatigue  
- **CPT:** 99214 - Office visit, established patient, moderate medical decision making  
- **CPT Grounding:** ✅ Present  
  - MDM Level: moderate  
  - Justification: MDM 2-of-3: problems=moderate, data=low, risk=moderate -> at least two elements meet or exceed 'moderate' -> MDM moderate -> CPT 99214.  

#### Governance & Safety
- **Retry Count:** 0  
- **Empty SOAP Retries:** 0  
- **Needs Human Review:** No  
- **Final Decision:** finalize  
- **Quality Flags (3):**
  - Scribe content incomplete: no ICD-10 suggestions (defaulted by the pipeline)  
  - Fast Path: added symptom code R53.83 - Other fatigue ('fatigue' documented in visit context)  
  - Fast Path: deterministic CPT 99214 (MDM moderate)  

#### SOAP Preview (truncated)
**Subjective:** Follow-up for diabetes and hypertension. Reports increased fatigue and occasional dizziness......  
**Objective:** Blood pressure 150/94, HbA1c 7.8%. Metformin 500mg BID, Lisinopril 10mg daily, Atorvastatin 20mg nightly......  
**Assessment:** Type 2 Diabetes Mellitus with suboptimal control (HbA1c 7.8%). Essential Hypertension with elevated BP......  
**Plan:** 1. Increase Metformin to 1000mg BID. 2. Continue Lisinopril 10mg daily. 3. Monitor HbA1c in 3 months......  

---

### P002 — Maria Gonzalez
**Expected Path:** fast  
**Actual Path:** fast  
**Path Match:** ✅ Yes  
**Demo Notes:** Multi-morbidity chronic follow-up (HTN+HLD+CKD). All conditions have clear ICD-10 mappings. Good test of secondary code resolution for CKD.  

#### Timing
- **Total:** 260.8s  
- **Memory Load Seconds:** 0.0s  
- **Scribe Seconds:** 260.7s  
- **Cleanup Seconds:** 0.0s  
- **Coding Seconds:** 0.0s  
- **Validation Seconds:** 0.0s  
- **Supervisor Seconds:** 0.0s  
- **Memory Save Seconds:** 0.0s  

#### Coding Results
- **Primary ICD-10:** I10 - Essential (primary) hypertension  
- **Secondary ICD-10:** E78.5 - Hyperlipidemia, unspecified  
- **Secondary ICD-10:** N18.31 - Chronic kidney disease, stage 3a  
- **CPT:** 99215 - Office visit, established patient, high medical decision making  
- **CPT Grounding:** ✅ Present  
  - MDM Level: high  
  - Justification: MDM 2-of-3: problems=high, data=moderate, risk=moderate -> at least two elements meet or exceed 'high' -> MDM high -> CPT 99215.  

#### Governance & Safety
- **Retry Count:** 2  
- **Empty SOAP Retries:** 0  
- **Needs Human Review:** ⚠️ Yes  
- **Final Decision:** human_review  
- **Quality Flags (4):**
  - Fast Path: deterministic CPT 99214 (MDM moderate)  
  - Fast Path: deterministic CPT 99213 (MDM low)  
  - Fast Path: deterministic CPT 99215 (MDM high)  
  - Coding: removed unsupported cardiac claim from the medical necessity note  

#### SOAP Preview (truncated)
**Subjective:** Follow-up for hypertension and hyperlipidemia. Routine medication review. Reports feeling well......  
**Objective:** Blood pressure 128/76, LDL 82, eGFR 52. Lisinopril 20mg, Amlodipine 5mg, Atorvastatin 40mg......  
**Assessment:** Essential Hypertension well-controlled. Hyperlipidemia at goal. CKD Stage 3a stable......  
**Plan:** 1. Continue current antihypertensive regimen. 2. Continue Atorvastatin 40mg. 3. Monitor renal function......  

---

### P005 — James Wilson
**Expected Path:** slow  
**Actual Path:** slow  
**Path Match:** ✅ Yes  
**Demo Notes:** New patient with undiagnosed persistent cough. No chronic conditions in context. Diagnostic uncertainty (cough variant asthma? GERD? post-infectious? URI?). Requires clinical reasoning.  

#### Timing
- **Total:** 559.9s  
- **Memory Load Seconds:** 0.0s  
- **Scribe Seconds:** 230.3s  
- **Cleanup Seconds:** 0.0s  
- **Coding Seconds:** 329.6s  
- **Validation Seconds:** 0.0s  
- **Supervisor Seconds:** 0.0s  
- **Memory Save Seconds:** 0.0s  

#### Coding Results
- **Primary ICD-10:** R05 - Cough  
- **Secondary ICD-10:** J06.9 - Acute upper respiratory infection, unspecified  
- **CPT:** 99214 - Office visit, established patient, moderate medical decision making  
- **CPT Grounding:** ✅ Present  
  - MDM Level: moderate  
  - Justification: MDM 2-of-3: problems=moderate, data=moderate, risk=low -> at least two elements meet or exceed 'moderate' -> MDM moderate -> CPT 99214.  

#### Governance & Safety
- **Retry Count:** 2  
- **Empty SOAP Retries:** 0  
- **Needs Human Review:** ⚠️ Yes  
- **Final Decision:** human_review  
- **Quality Flags (5):**
  - Scribe content incomplete: no ICD-10 suggestions (defaulted by the pipeline)  
  - Coding: added symptom code R05 - Cough ('cough' documented in chief complaint)  
  - Coding: removed unsupported cardiac claim from the medical necessity note  
  - Coding: deterministic CPT 99214 (MDM moderate)  
  - Coding: deterministic CPT 99213 (MDM low)  

#### SOAP Preview (truncated)
**Subjective:** New patient. Persistent dry cough for 6 weeks. Worse at night and with exercise......  
**Objective:** CXR normal, CBC normal, CMP normal. Spirometry FEV1/FVC 82%. No wheezes on exam......  
**Assessment:** Persistent dry cough x6 weeks. Normal CXR and spirometry. Likely post-viral cough or GERD......  
**Plan:** 1. Trial of PPI for possible silent GERD. 2. Consider inhaled corticosteroid if cough variant asthma... 3. Follow up in 4 weeks....  

---

### P007 — Ahmed Hassan
**Expected Path:** slow  
**Actual Path:** slow  
**Path Match:** ✅ Yes  
**Demo Notes:** SAFETY CASE: Acute coronary syndrome presentation. Clear diagnostic uncertainty (NSTEMI vs unstable angina). Must route to full LLM agent for proper acute management coding and safety checks.  

#### Timing
- **Total:** 385.2s  
- **Memory Load Seconds:** 0.0s  
- **Scribe Seconds:** 365.7s  
- **Cleanup Seconds:** 0.0s  
- **Coding Seconds:** 119.5s  
- **Validation Seconds:** 0.0s  
- **Supervisor Seconds:** 0.0s  
- **Memory Save Seconds:** 0.0s  

#### Coding Results
- **Primary ICD-10:** R07.9 - Chest pain, unspecified  
- **Secondary ICD-10:** R06.02 - Shortness of breath  
- **Secondary ICD-10:** R11.0 - Nausea  
- **Secondary ICD-10:** R53.83 - Other fatigue  
- **CPT:** 99214 - Office visit, established patient, moderate medical decision making  
- **CPT Grounding:** ✅ Present  
  - MDM Level: moderate  
  - Justification: MDM 2-of-3: problems=moderate, data=moderate, risk=high -> at least two elements meet or exceed 'moderate' -> MDM moderate -> CPT 99214.  

#### Governance & Safety
- **Retry Count:** 1  
- **Empty SOAP Retries:** 0  
- **Needs Human Review:** No  
- **Final Decision:** finalize  
- **Quality Flags (3):**
  - Scribe content incomplete: no ICD-10 suggestions (defaulted by the pipeline)  
  - Cardiac claim not supported by the chart: ischemic etiology, acute coronary syndrome, cardiac diagnostic claim  
  - Fast Path: deterministic CPT 99214 (MDM moderate)  

#### SOAP Preview (truncated)
**Subjective:** Acute chest pain radiating to left arm for 45 minutes. Diaphoresis, nausea, shortness of breath......  
**Objective:** ECG: ST depression V4-V6, T wave inversions. Troponin I: 0.18 ng/mL (elevated). HR 98, BP 158/96......  
**Assessment:** Acute chest pain concerning for ACS/NSTEMI. Elevated troponin, ischemic ECG changes. High cardiac risk......  
**Plan:** 1. Immediate ED referral for rule-out ACS. 2. Aspirin 325mg given. 3. Serial troponins. 4. Cardiology consult......  

---
