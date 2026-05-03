# Error Analysis — Network-Only MLP

## 1. Overall Performance

- **Total test pairs:** 12,672
- **Accuracy:** 0.8125
- **Total errors:** 2,376 (18.8%)
- **False Positives:** 800 (non-interacting predicted as interacting)
- **False Negatives:** 1,576 (interacting predicted as non-interacting)

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
| **Actual Non-Interact** | 5,782 | 800 |
| **Actual Interact** | 1,576 | 4,514 |

## 2. Error Rate by Confidence Bin

How often the model is wrong at different confidence levels. Errors near 0.5 are
expected (uncertain predictions). Errors near 0.0 or 1.0 are confident mistakes.

| Confidence Bin | Total | Errors | Error Rate |
|---|---|---|---|
| 0.0–0.1 | 3,792 | 422 | 0.111 |
| 0.1–0.2 | 1,476 | 338 | 0.229 |
| 0.2–0.3 | 889 | 288 | 0.324 |
| 0.3–0.4 | 655 | 261 | 0.398 |
| 0.4–0.5 | 546 | 267 | 0.489 |
| 0.5–0.6 | 486 | 224 | 0.461 |
| 0.6–0.7 | 467 | 163 | 0.349 |
| 0.7–0.8 | 527 | 138 | 0.262 |
| 0.8–0.9 | 680 | 139 | 0.204 |
| 0.9–1.0 | 3,154 | 136 | 0.043 |

## 3. Cosine Similarity — Correct vs Incorrect Predictions

Embedding cosine similarity between protein pairs, split by prediction outcome.
If false positives have high cosine similarity, the model is fooled by proteins
that are close in embedding space but don't actually interact.

| Category | Count | Mean Cosine Sim | Std |
|---|---|---|---|
| Correct Positives | 4,514 | 0.5604 | 0.1361 |
| Correct Negatives | 5,782 | 0.2539 | 0.1568 |
| False Positives | 800 | 0.4497 | 0.1225 |
| False Negatives | 1,576 | 0.3344 | 0.1517 |

## 4. Top 10 Most Confident False Positives

Non-interacting pairs the model is most sure are interacting:

| Protein A | Protein B | Pred Prob | Cosine Sim |
|---|---|---|---|
| H2BS1 | H2BC15 | 1.000 | 0.9480 |
| H3C6 | H3Y2 | 1.000 | 0.9051 |
| H4C15 | H3Y2 | 0.999 | 0.7427 |
| EPOR | MPL | 0.999 | 0.6374 |
| H4C7 | H3C8 | 0.999 | 0.7114 |
| CYBRD1 | ERFE | 0.998 | 0.7014 |
| KCNE4 | KCNJ5 | 0.998 | 0.6464 |
| SGPP1 | CERS1 | 0.997 | 0.6100 |
| NEFH | SYN3 | 0.996 | 0.4981 |
| H2BC8 | H3Y2 | 0.996 | 0.6766 |

## 5. Top 10 Most Confident False Negatives

Interacting pairs the model is most sure are NOT interacting:

| Protein A | Protein B | Pred Prob | Cosine Sim |
|---|---|---|---|
| SECISBP2 | RPL23 | 0.002 | 0.1912 |
| SYMPK | RPLP0 | 0.003 | -0.0102 |
| XYLB | F5H5P2_HUMAN | 0.003 | 0.5388 |
| DTL | PBK | 0.003 | 0.4502 |
| PIGR | RAB3B | 0.003 | 0.1473 |
| TUBB | RPL6 | 0.003 | 0.1441 |
| PIF1 | PRH1 | 0.004 | 0.1443 |
| RPL36AL | SECISBP2 | 0.004 | 0.2410 |
| RPL7 | SECISBP2 | 0.004 | 0.2529 |
| SECISBP2 | RPL36A | 0.004 | 0.2128 |

## 6. Most Frequently Misclassified Proteins

Proteins that appear most often across all misclassified pairs:

| Protein | Name | Error Count |
|---|---|---|
| ENSP00000275493 | EGFR | 30 |
| ENSP00000261769 | CDH1 | 29 |
| ENSP00000216037 | XBP1 | 21 |
| ENSP00000451828 | AKT1 | 19 |
| ENSP00000347184 | HTT | 17 |
| ENSP00000291295 | CALM3 | 17 |
| ENSP00000420311 | RPL23 | 15 |
| ENSP00000334448 | GNG2 | 15 |
| ENSP00000339001 | TUBB | 15 |
| ENSP00000364965 | SECISBP2 | 15 |
| ENSP00000360609 | HSP90AB1 | 14 |
| ENSP00000430269 | EPHX2 | 14 |
| ENSP00000366927 | ALDH1B1 | 14 |
| ENSP00000302961 | HSPA4 | 13 |
| ENSP00000052754 | DCN | 13 |

## 7. Error Rate by Protein Size

Average amino acid length of the pair vs error rate:

| Size Bin (aa) | Total | Errors | Error Rate |
|---|---|---|---|
| <200 | 1,000 | 136 | 0.136 |
| 200-400 | 3,707 | 630 | 0.170 |
| 400-600 | 3,471 | 714 | 0.206 |
| 600-800 | 1,941 | 367 | 0.189 |
| 800-1000 | 1,024 | 226 | 0.221 |
| >1000 | 1,529 | 303 | 0.198 |

## 8. Functional Keyword Enrichment in Errors

Which functional categories are over/under-represented in errors compared to
correct predictions. Enrichment > 1.0 means the keyword appears more often in
errors; < 1.0 means it appears less.

| Keyword | Rate in Errors | Rate in Correct | Enrichment |
|---|---|---|---|
| cytoplasm | 0.0985 | 0.0763 | 1.29x |
| enzyme | 0.1490 | 0.1194 | 1.25x |
| kinase | 0.1751 | 0.1409 | 1.24x |
| channel | 0.0774 | 0.0641 | 1.21x |
| receptor | 0.2820 | 0.2469 | 1.14x |
| binding | 0.3342 | 0.2994 | 1.12x |
| signaling | 0.1932 | 0.1737 | 1.11x |
| membrane | 0.2534 | 0.2285 | 1.11x |
| transport | 0.1368 | 0.1334 | 1.03x |
| transcription | 0.2753 | 0.2785 | 0.99x |
| DNA | 0.1890 | 0.1956 | 0.97x |
| apoptosis | 0.0762 | 0.0805 | 0.95x |
| nuclear | 0.1006 | 0.1083 | 0.93x |
| RNA | 0.2454 | 0.2645 | 0.93x |
| chaperone | 0.0282 | 0.0312 | 0.90x |
| histone | 0.0825 | 0.0915 | 0.90x |
| ubiquitin | 0.0896 | 0.1006 | 0.89x |
| mitochond | 0.1111 | 0.1296 | 0.86x |
| ribosom | 0.1149 | 0.1700 | 0.68x |
| proteasome | 0.0206 | 0.0321 | 0.64x |
