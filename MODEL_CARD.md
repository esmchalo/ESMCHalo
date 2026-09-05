# ESMCHalo-v2 master source of truth

Status: **AUTHORITATIVE FOR THE CURRENT V2 MODEL AND PRIMARY PERFORMANCE REPORTING**

The previous LR/test200 source-of-truth files are superseded for current model,
calibration, threshold and primary comparative-performance statements. They may
only support explicitly labelled legacy analyses that do not conflict with V2.

## Frozen model

- ESMC-600M layer-36 canonical mean representation, 1,152 dimensions.
- Student: 1,152 → 256 → 64 → 1.
- K2 distillation: alpha=0.5, beta=0.5, gamma=0, T=2.
- H0: unweighted training; H1–H3 rejected.
- Five frozen fold models; equal mean of raw logits.
- C1 Platt: coefficient 1.65445192210991; intercept -0.059280024051954906.
- Deployment threshold: 0.5.
- DeepSaltPro is not required at inference.

## One-time blind evaluation

- n=1,865; positives=1,107; negatives=758; 1,808 homology groups.
- Prediction file was generated and SHA256-locked before label reveal.
- ESMCHalo-v2: AP=0.991655706984, AUROC=0.987546269482, MCC=0.884238593909, accuracy=0.944235924933.
- DeepSaltPro: AP=0.989121035130, AUROC=0.983364437866, MCC=0.863136800512, accuracy=0.934048257373.
- Group-bootstrap differences: AP +0.002535 [0.000654, 0.004472]; AUROC +0.004182 [0.001308, 0.007078]; MCC +0.021102 [0.000028, 0.041949].
- Exact McNemar: 55 versus 36 discordant correct predictions; two-sided P=0.058574.

## Claim boundaries

- AP and AUROC improvements are supported by positive 95% group-bootstrap intervals.
- MCC has a positive but boundary-adjacent lower confidence limit and must be described cautiously.
- McNemar P is not below 0.05; do not claim statistically significant accuracy superiority.
- Do not subtract legacy test200 metrics from blind-set metrics and call the difference model improvement; the evaluation sets differ.
- The clean component-level predictive gain is K2 versus K0 on the same development OOF protocol.
- C1 improves calibration, not ranking.
- No isolated causal performance gain is assigned to five-model ensembling.
- No broad cross-domain or mutation-effect claim is permitted.

## Intended use
Protein-level halophilic-likeness ranking and classification; not mutation-effect prediction.
