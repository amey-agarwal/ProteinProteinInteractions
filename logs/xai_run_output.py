Embeddings loaded
Embeddings loaded
  train=50,000  test=10,000
network embeddings
    built 50,000 pairs  (skipped 0)
    built 10,000 pairs  (skipped 0)
sequence embeddings
    built 50,000 pairs  (skipped 0)
    built 10,000 pairs  (skipped 0)
sequence + network (combined)
STRING evidence features only
    [network               ] F1=0.7706  ROC-AUC=0.8644
    [sequence              ] F1=0.6201  ROC-AUC=0.7274
    [sequence+network      ] F1=0.7764  ROC-AUC=0.8642
    [evidence scores       ] F1=1.0000  ROC-AUC=1.0000

embedding geometry
  [network] pos cos=0.5015  neg cos=0.2779  Δcos=0.2236
  [sequence] pos cos=0.5207  neg cos=0.4643  Δcos=0.0564

kNN Neighbourhood overlap
  pos overlap=0.1270  neg overlap=0.0220

[PCA dimensionality
  network cumvar@20=62.4%  sequence cumvar@20=54.7%

Prototype pairs

Component ablation (embedding dim=512) …
    [full                ] F1=0.7683  ROC-AUC=0.8659
    [absdiff_only        ] F1=0.6843  ROC-AUC=0.7851
    [elemprod_only       ] F1=0.7318  ROC-AUC=0.8391

SHAP
KernelSHAP on 300 samples

Generating HTML report

Report saved to: data/processed/xai_report.html
