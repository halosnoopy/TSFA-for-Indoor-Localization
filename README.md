# TSFA-for-Indoor-Localization



\## Overview



TSFA addresses a key limitation of conventional fingerprinting methods: raw RSS values are highly sensitive to environmental changes, even though spatial signal \*\*patterns\*\* remain relatively stable.



To exploit this property, TSFA integrates three mechanisms:



\- \*\*Instance-wise transformation\*\* to normalize RSS patterns  

\- \*\*Similarity filtering\*\* to dynamically refine the candidate fingerprint set  

\- \*\*Adaptive selection\*\* to ensure robust neighbor selection for localization  



By combining transformed fingerprints for pattern matching and original fingerprints for distance estimation, TSFA achieves a balanced and interpretable localization framework.



---



\## Method Summary



\### Offline Phase



\- Replace missing RSS values with a fixed placeholder (−100).

\- Construct two fingerprint datasets:

&nbsp; - \*\*FP\*\*: original RSS fingerprints preserving absolute signal strength information

&nbsp; - \*\*TP\*\*: instance-wise transformed fingerprints emphasizing signal patterns

\- Store both FP and TP for online localization.



\### Online Phase



\- Compute cosine similarity between the input fingerprint and TP.

\- Select candidate fingerprints within a similarity interval controlled by the threshold factor θ.

\- Compute Euclidean distances using FP for refined neighbor ranking.

\- Apply adaptive k selection and average the coordinates of selected neighbors to estimate the user location.



This two-stage design improves localization stability, reduces high-error cases, and lowers runtime compared to conventional KNN-based methods.



---



\## Repository Files



Only the following three files are required for running TSFA experiments:



| File | Description |

|------|------------|

| `ag2025.py` | Core project functions supporting TSFA experiments, including shared utilities and data handling routines. |

| `TSFA\_test.ipynb` | Main notebook for testing and evaluating the TSFA algorithm under different parameter settings. |

| `result\_visualization.ipynb` | Notebook for visualizing localization errors, robustness analysis, and comparative results. |



---

