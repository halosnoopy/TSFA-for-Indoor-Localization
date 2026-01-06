# TSFA-for-Indoor-Localization


This repository contains the implementation and experimental workflow for **TSFA**, a Wi-Fi fingerprinting–based indoor localization algorithm designed to improve **accuracy and robustness** under RSS instability and missing access point conditions.

<img width="938" height="679" alt="image" src="https://github.com/user-attachments/assets/7fda7aef-e412-41ce-8512-c5d0fbc65133" />

---

## Overview

TSFA addresses a key limitation of conventional fingerprinting methods: raw RSS values are highly sensitive to environmental changes, even though spatial signal **patterns** remain relatively stable.

To exploit this property, TSFA integrates three mechanisms:

- **Instance-wise transformation** to normalize RSS patterns  
- **Similarity filtering** to dynamically refine the candidate fingerprint set  
- **Adaptive selection** to ensure robust neighbor selection for localization  

By combining transformed fingerprints for pattern matching and original fingerprints for distance estimation, TSFA achieves a balanced and interpretable localization framework.

---

## Method Summary

### Offline Phase

- Replace missing RSS values with a fixed placeholder (−100).
- Construct two fingerprint datasets:
  - **FP**: original RSS fingerprints preserving absolute signal strength information
  - **TP**: instance-wise transformed fingerprints emphasizing signal patterns
- Store both FP and TP for online localization.

<img width="772" height="490" alt="image" src="https://github.com/user-attachments/assets/e2f5a01c-17af-4907-8376-f72c2e1fa150" />


### Online Phase

- Compute cosine similarity between the input fingerprint and TP.
- Select candidate fingerprints within a similarity interval controlled by the threshold factor θ.
- Compute Euclidean distances using FP for refined neighbor ranking.
- Apply adaptive k selection and average the coordinates of selected neighbors to estimate the user location.

This two-stage design improves localization stability, reduces high-error cases, and lowers runtime compared to conventional KNN-based methods.

<img width="860" height="479" alt="image" src="https://github.com/user-attachments/assets/7b9106e6-a6d4-4d8a-82a5-796cac5f50b9" />

---

## Repository Files

Only the following three files are required for running TSFA experiments:

| File | Description |
|------|------------|
| `ag2025.py` | Core project functions supporting TSFA experiments, including shared utilities and data handling routines. |
| `TSFA_test.ipynb` | Main notebook for testing and evaluating the TSFA algorithm under different parameter settings. |
| `result_visualization.ipynb` | Notebook for visualizing localization errors, robustness analysis, and comparative results. |


