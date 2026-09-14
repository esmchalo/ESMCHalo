# Model card: ESMCHalo frozen H0 ensemble

Version of software candidate: 2.0.2-rc1. Scientific model unchanged from v2.0.1: five fold heads with their own StandardScaler, dimensions1152–256–64–1, GELU and dropout0.2; average raw logits then fixed Platt (a=1.65445192210991,b=-0.059280024051954906), probability threshold0.5.

Historical representation: frozen ESMC-600M final residue representation; 2046-residue windows,1790 stride,end-aligned final window. Original short pooling arithmetic and BF16 weight loading are preserved through the original extractor; overlapping long-window states are resolved per residue before global pooling.

Use: halophile-associated protein prediction. Not a calibrated guarantee across organisms/domains; not salt-concentration activity or mutation-effect prediction. No consistent classification superiority was established on the existing challenge datasets.

Teacher supervision has cross-student-fold dependencies; historical evaluation-data usage limits independence claims. No “leakage-free” or project-wide first-blind claim is made here. See docs/VALIDATION.md for the exact acceptance scope and limitations.
