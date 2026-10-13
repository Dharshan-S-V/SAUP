# SAUP Limitations

## Dataset Limitations
1. **UCI Adult:** Historical 1994 US census data with documented representation biases and sensitive demographic attributes.
2. **UCI Bank Marketing:** Portuguese banking data with heavy class imbalance (~11.7% positive). While `duration` is dropped to prevent target leakage, telemarketing phone patterns may not generalize to digital channels.
3. **UCI Wine Quality:** Physicochemical measurements of Portuguese Vinho Verde wines. Quality scores reflect subjective human sensory evaluations.

## Methodological Limitations
1. Controlled distribution shifts (covariate, label, mixed, severe) are synthetically constructed to allow ground-truth benchmarking; performance on arbitrary external real-world shifts may vary.
2. Density ratio estimation for weighted conformal prediction degrades in high-dimensional or sparse feature representations.
3. BBSE assumes that conditional distribution P(X|Y) remains invariant under shift.
