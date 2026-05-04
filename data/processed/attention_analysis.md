# Attention Analysis — Two-Tower MLP with Cross-Tower Attention

## Performance Comparison

| Model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Two-Tower (no attention) | 0.805 | 0.896 | 0.902 |
| **Two-Tower + Attention** | **0.786** | **0.888** | **0.898** |


## Attention Weight Analysis

The 2x2 attention matrix shows how each tower queries the other:
- **net→net**: network tower attending to itself
- **net→seq**: network tower attending to sequence
- **seq→net**: sequence tower attending to network
- **seq→seq**: sequence tower attending to itself

| Category | Count | net→net | net→seq | seq→net | seq→seq |
|---|---|---|---|---|---|
| All predictions | 12,672 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| Correct positives | 4,299 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| Correct negatives | 6,034 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| False positives | 548 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| False negatives | 1,791 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| High confidence (>0.9) | 1,754 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| Low confidence (0.4-0.6) | 1,020 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

**Overall tower reliance:**
- Network tower receives 50.0% of attention
- Sequence tower receives 50.0% of attention

**Correct positives vs false negatives:**
- Correct positives: net→seq = 0.5000, seq→net = 0.5000
- False negatives: net→seq = 0.5000, seq→net = 0.5000
- Cross-tower attention is 0.0000 lower for correct positives

## Architecture

```
net_tower:  1024 → 128  (BN + ReLU + Dropout 0.3)
seq_tower:  2048 → 128  (BN + ReLU + Dropout 0.3)
attention:  4-head self-attention over 2 tokens of dim 128
            + residual connection + LayerNorm
head:       256 → 128 → 1  (BN + ReLU + Dropout 0.2)
```

Parameters: 493,569
