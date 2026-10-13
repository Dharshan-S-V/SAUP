# SAUP References

## Primary Literature

1. **Ai, R. and Ren, Z.** (2024). Not All Distributional Shifts Are Equal: Fine-Grained Robust Conformal Inference. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR 235. [https://proceedings.mlr.press/v235/ai24a.html](https://proceedings.mlr.press/v235/ai24a.html)

2. **Liu, Y., Xu, Z., and Kolar, M.** (2024). Multi-Source Conformal Inference Under Distribution Shift. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR 235. [https://proceedings.mlr.press/v235/liu24ag.html](https://proceedings.mlr.press/v235/liu24ag.html)

3. **Jonkers, L., Vercruyssen, V., and Davis, J.** (2024). Conformal Predictive Systems Under Covariate Shift. *Proceedings of the 13th Symposium on Conformal and Probabilistic Prediction with Applications (COPA 2024)*, PMLR 230. [https://proceedings.mlr.press/v230/jonkers24a.html](https://proceedings.mlr.press/v230/jonkers24a.html)

4. **Amoukou, S.I., Bénard, C., and Sabourin, A.** (2024). Sequential Harmful Shift Detection Without Labels. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. [https://proceedings.neurips.cc/paper_files/paper/2024/hash/e9721921b799b6ea98d37f9e77f1a7fe-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e9721921b799b6ea98d37f9e77f1a7fe-Abstract-Conference.html)

5. **Popordanoska, T., Kasa, M., and Feuerriegel, S.** (2024). LaSCal: Label-Shift Calibration without Target Labels. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. [https://proceedings.neurips.cc/paper_files/paper/2024/hash/783c5986e1d6112cb4688d9b2105609a-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2024/hash/783c5986e1d6112cb4688d9b2105609a-Abstract-Conference.html)

6. **Kasa, M., Popordanoska, T., and Feuerriegel, S.** (2025). Adapting Prediction Sets to Distribution Shifts Without Labels. *Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS 2025)*, PMLR 286. [https://proceedings.mlr.press/v286/kasa25a.html](https://proceedings.mlr.press/v286/kasa25a.html)

7. **Baby, D., Wang, Y., and Yu, Y.** (2025). Adapting to Online Distribution Shifts in Deep Learning: A Black-Box Approach. *Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS 2025)*, PMLR 258. [https://proceedings.mlr.press/v258/baby25a.html](https://proceedings.mlr.press/v258/baby25a.html)

8. **Lee, J.** (2026). Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study. *PMLR 337*. [https://proceedings.mlr.press/v337/lee26g.html](https://proceedings.mlr.press/v337/lee26g.html)

9. **Jiang, Y.** (2026). Score-Based Diffusion Priors for Adaptive Conformal Inference under Distribution Shift. *PMLR 337*. [https://proceedings.mlr.press/v337/jiang26a.html](https://proceedings.mlr.press/v337/jiang26a.html)

10. **Laghuvarapu, S., Deb, N., and Sun, Y.** (2026). KMM-CP: Practical Conformal Prediction under Covariate Shift via Selective Kernel Mean Matching. *PMLR 337*. [https://proceedings.mlr.press/v337/laghuvarapu26a.html](https://proceedings.mlr.press/v337/laghuvarapu26a.html)

## Dataset

11. **Becker, B. and Kohavi, R.** (1996). Adult / Census Income Dataset. UCI Machine Learning Repository. [https://archive.ics.uci.edu/dataset/2/adult](https://archive.ics.uci.edu/dataset/2/adult)

## Foundational Methods

12. **Vovk, V., Gammerman, A., and Shafer, G.** (2005). *Algorithmic Learning in a Random World*. Springer. — Foundational work on conformal prediction.

13. **Guo, C., Pleiss, G., Sun, Y., and Weinberger, K.Q.** (2017). On Calibration of Modern Neural Networks. *ICML 2017*. — Temperature scaling for probability calibration.

14. **Lipton, Z.C., Wang, Y.-X., and Smola, A.** (2018). Detecting and Correcting for Label Shift with Black Box Predictors. *ICML 2018*. — Black Box Shift Estimation (BBSE).

15. **Tibshirani, R.J., Barber, R.F., Candès, E.J., and Ramdas, A.** (2019). Conformal Prediction Under Covariate Shift. *NeurIPS 2019*. — Importance-weighted conformal prediction.

## How References Support SAUP Components

| SAUP Component | Supported By |
|---|---|
| Conformal prediction | Vovk et al. (2005), Tibshirani et al. (2019), Refs 1-3, 6, 8-10 |
| Temperature scaling | Guo et al. (2017) |
| BBSE / Label shift | Lipton et al. (2018), Refs 5-7 |
| Shift detection | Refs 4, 8 |
| Adaptive conformal under shift | Refs 1, 3, 6, 9, 10 |
| Distribution shift robustness | Refs 1-10 |

**Note:** SAUP is described as an integration and evaluation framework. Its novelty lies in the systematic combination and evaluation of these methods, not in claiming algorithmic novelty for the individual components.
