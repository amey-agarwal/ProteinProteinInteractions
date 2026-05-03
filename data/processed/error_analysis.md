# Error Analysis — Network-Only MLP

## 1. Overall Performance

- **Total test pairs:** 12,672
- **Accuracy:** 0.8134
- **Total errors:** 2,365 (18.7%)
- **False Positives:** 809 (non-interacting predicted as interacting)
- **False Negatives:** 1,556 (interacting predicted as non-interacting)

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
| **Actual Non-Interact** | 5,773 | 809 |
| **Actual Interact** | 1,556 | 4,534 |

## 2. Error Rate by Confidence Bin

How often the model is wrong at different confidence levels. Errors near 0.5 are
expected (uncertain predictions). Errors near 0.0 or 1.0 are confident mistakes.

| Confidence Bin | Total | Errors | Error Rate |
|---|---|---|---|
| 0.0–0.1 | 3,717 | 413 | 0.111 |
| 0.1–0.2 | 1,536 | 334 | 0.217 |
| 0.2–0.3 | 880 | 289 | 0.328 |
| 0.3–0.4 | 658 | 256 | 0.389 |
| 0.4–0.5 | 538 | 264 | 0.491 |
| 0.5–0.6 | 486 | 215 | 0.442 |
| 0.6–0.7 | 507 | 194 | 0.383 |
| 0.7–0.8 | 491 | 137 | 0.279 |
| 0.8–0.9 | 729 | 130 | 0.178 |
| 0.9–1.0 | 3,130 | 133 | 0.042 |

## 3. Cosine Similarity — Correct vs Incorrect Predictions

Embedding cosine similarity between protein pairs, split by prediction outcome.
If false positives have high cosine similarity, the model is fooled by proteins
that are close in embedding space but don't actually interact.

| Category | Count | Mean Cosine Sim | Std |
|---|---|---|---|
| Correct Positives | 4,534 | 0.5597 | 0.1360 |
| Correct Negatives | 5,773 | 0.2541 | 0.1568 |
| False Positives | 809 | 0.4466 | 0.1258 |
| False Negatives | 1,556 | 0.3338 | 0.1535 |

## 4. Top 10 Most Confident False Positives

Non-interacting pairs the model is most sure are interacting:

| Protein A | Protein B | Pred Prob | Cosine Sim |
|---|---|---|---|
| H2BS1 | H2BC15 | 1.000 | 0.9480 |
| H3C6 | H3Y2 | 1.000 | 0.9051 |
| EPOR | MPL | 0.999 | 0.6374 |
| SGPP1 | CERS1 | 0.999 | 0.6100 |
| KCNE4 | KCNJ5 | 0.999 | 0.6464 |
| H4C7 | H3C8 | 0.999 | 0.7114 |
| H2AC1 | LOC102724334 | 0.998 | 0.7296 |
| H4C15 | H3Y2 | 0.996 | 0.7427 |
| NEFH | SYN3 | 0.996 | 0.4981 |
| ATP5MPL | NDUFC1 | 0.995 | 0.6341 |

## 5. Top 10 Most Confident False Negatives

Interacting pairs the model is most sure are NOT interacting:

| Protein A | Protein B | Pred Prob | Cosine Sim |
|---|---|---|---|
| SYMPK | RPLP0 | 0.002 | -0.0102 |
| SECISBP2 | RPL36A | 0.002 | 0.2128 |
| TUBB | RPL6 | 0.003 | 0.1441 |
| RPL7 | SECISBP2 | 0.003 | 0.2529 |
| RPL36AL | SECISBP2 | 0.004 | 0.2410 |
| TUBB | RPL7 | 0.004 | 0.1018 |
| TPI1 | CWC27 | 0.004 | 0.0614 |
| PIF1 | PRH1 | 0.004 | 0.1443 |
| ACSS2 | ECHDC1 | 0.005 | 0.5608 |
| SECISBP2 | RPS24 | 0.005 | 0.1751 |

## 6. Most Frequently Misclassified Proteins

Proteins that appear most often across all misclassified pairs:

| Protein | Name | Error Count |
|---|---|---|
| ENSP00000261769 | CDH1 | 29 |
| ENSP00000275493 | EGFR | 28 |
| ENSP00000216037 | XBP1 | 20 |
| ENSP00000451828 | AKT1 | 19 |
| ENSP00000420311 | RPL23 | 16 |
| ENSP00000291295 | CALM3 | 16 |
| ENSP00000334448 | GNG2 | 16 |
| ENSP00000339001 | TUBB | 15 |
| ENSP00000364965 | SECISBP2 | 15 |
| ENSP00000347184 | HTT | 15 |
| ENSP00000302961 | HSPA4 | 14 |
| ENSP00000326031 | PPP1CA | 14 |
| ENSP00000430269 | EPHX2 | 14 |
| ENSP00000320171 | PKM | 13 |
| ENSP00000360609 | HSP90AB1 | 13 |

## 7. Error Rate by Protein Size

Average amino acid length of the pair vs error rate:

| Size Bin (aa) | Total | Errors | Error Rate |
|---|---|---|---|
| <200 | 1,000 | 121 | 0.121 |
| 200-400 | 3,707 | 635 | 0.171 |
| 400-600 | 3,471 | 732 | 0.211 |
| 600-800 | 1,941 | 362 | 0.187 |
| 800-1000 | 1,024 | 226 | 0.221 |
| >1000 | 1,529 | 289 | 0.189 |

## 8. Functional Keyword Enrichment in Errors

Which functional categories are over/under-represented in errors compared to
correct predictions. Enrichment > 1.0 means the keyword appears more often in
errors; < 1.0 means it appears less.

| Keyword | Rate in Errors | Rate in Correct | Enrichment |
|---|---|---|---|
| enzyme | 0.1535 | 0.1184 | 1.30x |
| cytoplasm | 0.0981 | 0.0765 | 1.28x |
| kinase | 0.1793 | 0.1400 | 1.28x |
| channel | 0.0761 | 0.0644 | 1.18x |
| binding | 0.3378 | 0.2986 | 1.13x |
| receptor | 0.2778 | 0.2479 | 1.12x |
| signaling | 0.1941 | 0.1735 | 1.12x |
| membrane | 0.2533 | 0.2286 | 1.11x |
| transport | 0.1370 | 0.1333 | 1.03x |
| transcription | 0.2816 | 0.2770 | 1.02x |
| chaperone | 0.0309 | 0.0306 | 1.01x |
| DNA | 0.1920 | 0.1949 | 0.98x |
| apoptosis | 0.0774 | 0.0802 | 0.96x |
| nuclear | 0.1027 | 0.1078 | 0.95x |
| ubiquitin | 0.0943 | 0.0995 | 0.95x |
| histone | 0.0846 | 0.0910 | 0.93x |
| RNA | 0.2431 | 0.2650 | 0.92x |
| mitochond | 0.1108 | 0.1296 | 0.85x |
| proteasome | 0.0220 | 0.0317 | 0.69x |
| ribosom | 0.1154 | 0.1698 | 0.68x |
