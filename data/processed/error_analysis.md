# Error Analysis — Network-Only MLP

## 1. Overall Performance

- **Total test pairs:** 12,672
- **Accuracy:** 0.8129
- **Total errors:** 2,371 (18.7%)
- **False Positives:** 812 (non-interacting predicted as interacting)
- **False Negatives:** 1,559 (interacting predicted as non-interacting)

```
              precision    recall  f1-score   support

non-interact       0.79      0.88      0.83      6582
    interact       0.85      0.74      0.79      6090

    accuracy                           0.81     12672
   macro avg       0.82      0.81      0.81     12672
weighted avg       0.82      0.81      0.81     12672

```

**Confusion Matrix:**

|  | Pred Non-Interact | Pred Interact |
|---|---|---|
| **Actual Non-Interact** | 5,770 | 812 |
| **Actual Interact** | 1,559 | 4,531 |

## 2. Error Rate by Confidence Bin

How often the model is wrong at different confidence levels. Errors near 0.5 are
expected (uncertain predictions). Errors near 0.0 or 1.0 are confident mistakes.

| Confidence Bin | Total | Errors | Error Rate |
|---|---|---|---|
| 0.0–0.1 | 3,758 | 422 | 0.112 |
| 0.1–0.2 | 1,508 | 336 | 0.223 |
| 0.2–0.3 | 885 | 277 | 0.313 |
| 0.3–0.4 | 640 | 260 | 0.406 |
| 0.4–0.5 | 538 | 264 | 0.491 |
| 0.5–0.6 | 493 | 219 | 0.444 |
| 0.6–0.7 | 509 | 181 | 0.356 |
| 0.7–0.8 | 547 | 147 | 0.269 |
| 0.8–0.9 | 725 | 143 | 0.197 |
| 0.9–1.0 | 3,069 | 122 | 0.040 |

## 3. Cosine Similarity — Correct vs Incorrect Predictions

Embedding cosine similarity between protein pairs, split by prediction outcome.
If false positives have high cosine similarity, the model is fooled by proteins
that are close in embedding space but don't actually interact.

| Category | Count | Mean Cosine Sim | Std |
|---|---|---|---|
| Correct Positives | 4,531 | 0.5597 | 0.1366 |
| Correct Negatives | 5,770 | 0.2544 | 0.1575 |
| False Positives | 812 | 0.4436 | 0.1239 |
| False Negatives | 1,559 | 0.3340 | 0.1520 |

## 4. Top 10 Most Confident False Positives

Non-interacting pairs the model is most sure are interacting:

| Protein A | Protein B | Pred Prob | Cosine Sim |
|---|---|---|---|
| H2BS1 | H2BC15 | 1.000 | 0.9480 |
| H3C6 | H3Y2 | 1.000 | 0.9051 |
| EPOR | MPL | 1.000 | 0.6374 |
| H4C7 | H3C8 | 0.999 | 0.7114 |
| SGPP1 | CERS1 | 0.999 | 0.6100 |
| H2AC1 | LOC102724334 | 0.999 | 0.7296 |
| KCNE4 | KCNJ5 | 0.998 | 0.6464 |
| ATP5MPL | NDUFC1 | 0.998 | 0.6341 |
| H2AC20 | LOC102724334 | 0.998 | 0.6819 |
| H2BC8 | H3Y2 | 0.996 | 0.6766 |

## 5. Top 10 Most Confident False Negatives

Interacting pairs the model is most sure are NOT interacting:

| Protein A | Protein B | Pred Prob | Cosine Sim |
|---|---|---|---|
| SYMPK | RPLP0 | 0.002 | -0.0102 |
| XYLB | F5H5P2_HUMAN | 0.002 | 0.5388 |
| SECISBP2 | RPL36A | 0.002 | 0.2128 |
| RPL7 | SECISBP2 | 0.002 | 0.2529 |
| RPL36AL | SECISBP2 | 0.003 | 0.2410 |
| ACSS2 | ECHDC1 | 0.003 | 0.5608 |
| SECISBP2 | RPL23 | 0.003 | 0.1912 |
| TUBB | RPL36AL | 0.003 | 0.0779 |
| TUBB | RPL7 | 0.003 | 0.1018 |
| PIF1 | PRH1 | 0.004 | 0.1443 |

## 6. Most Frequently Misclassified Proteins

Proteins that appear most often across all misclassified pairs:

| Protein | Name | Error Count |
|---|---|---|
| ENSP00000261769 | CDH1 | 28 |
| ENSP00000275493 | EGFR | 27 |
| ENSP00000216037 | XBP1 | 22 |
| ENSP00000451828 | AKT1 | 19 |
| ENSP00000360609 | HSP90AB1 | 17 |
| ENSP00000420311 | RPL23 | 16 |
| ENSP00000339001 | TUBB | 15 |
| ENSP00000364965 | SECISBP2 | 15 |
| ENSP00000347184 | HTT | 15 |
| ENSP00000291295 | CALM3 | 15 |
| ENSP00000302961 | HSPA4 | 14 |
| ENSP00000430269 | EPHX2 | 14 |
| ENSP00000392094 | EFTUD2 | 14 |
| ENSP00000366927 | ALDH1B1 | 14 |
| ENSP00000355890 | EPRS1 | 13 |

## 7. Error Rate by Protein Size

Average amino acid length of the pair vs error rate:

| Size Bin (aa) | Total | Errors | Error Rate |
|---|---|---|---|
| <200 | 1,000 | 130 | 0.130 |
| 200-400 | 3,707 | 622 | 0.168 |
| 400-600 | 3,471 | 727 | 0.209 |
| 600-800 | 1,941 | 375 | 0.193 |
| 800-1000 | 1,024 | 220 | 0.215 |
| >1000 | 1,529 | 297 | 0.194 |

## 8. Functional Keyword Enrichment in Errors

Which functional categories are over/under-represented in errors compared to
correct predictions. Enrichment > 1.0 means the keyword appears more often in
errors; < 1.0 means it appears less.

| Keyword | Rate in Errors | Rate in Correct | Enrichment |
|---|---|---|---|
| cytoplasm | 0.0974 | 0.0766 | 1.27x |
| enzyme | 0.1485 | 0.1195 | 1.24x |
| kinase | 0.1738 | 0.1412 | 1.23x |
| signaling | 0.1936 | 0.1736 | 1.12x |
| receptor | 0.2758 | 0.2483 | 1.11x |
| binding | 0.3294 | 0.3006 | 1.10x |
| membrane | 0.2488 | 0.2296 | 1.08x |
| channel | 0.0700 | 0.0658 | 1.06x |
| transcription | 0.2855 | 0.2761 | 1.03x |
| chaperone | 0.0304 | 0.0307 | 0.99x |
| transport | 0.1324 | 0.1344 | 0.99x |
| DNA | 0.1906 | 0.1952 | 0.98x |
| nuclear | 0.1012 | 0.1081 | 0.94x |
| apoptosis | 0.0751 | 0.0808 | 0.93x |
| histone | 0.0839 | 0.0912 | 0.92x |
| RNA | 0.2438 | 0.2648 | 0.92x |
| ubiquitin | 0.0911 | 0.1003 | 0.91x |
| mitochond | 0.1088 | 0.1301 | 0.84x |
| ribosom | 0.1126 | 0.1705 | 0.66x |
| proteasome | 0.0211 | 0.0319 | 0.66x |
