# Raw Data Sources

This file documents the origin of every raw data file in this directory, with
full APA and BibTeX citations.

---

## Shannon_Effective_Ionic_Radii.csv

Ionic radii extracted from the supplementary data of the Bartel *et al.* (2019) tolerance-factor paper.

**APA**
> Bartel, C. J., Sutton, C., Goldsmith, B. R., Ouyang, R., Musgrave, C. B., Ghiringhelli, L. M., & Scheffler, M. (2019). New tolerance factor to predict the stability of perovskite oxides and halides. *Science Advances*, *5*(2), eaav0693. https://doi.org/10.1126/sciadv.aav0693

<details>
<summary>BibTeX</summary>

```bibtex
@article{Bartel2019,
  author  = {Bartel, Christopher J. and Sutton, Christopher and Goldsmith, Bryan R.
             and Ouyang, Runhai and Musgrave, Charles B. and Ghiringhelli, Luca M.
             and Scheffler, Matthias},
  title   = {New tolerance factor to predict the stability of perovskite oxides and halides},
  journal = {Science Advances},
  year    = {2019},
  volume  = {5},
  number  = {2},
  pages   = {eaav0693},
  doi     = {10.1126/sciadv.aav0693}
}
```
</details>

---

## Expanded_Shannon_Effective_Ionic_Radii.csv

Shannon ionic radii extended to a wider range of oxidation states and
coordination environments using machine learning.

**APA**
> Baloch, A. A. B., Alqahtani, S. M., Mumtaz, F., Muqaibel, A. H., Rashkeev, S. N., & Alharbi, F. H. (2021). Extending Shannon's ionic radii database using machine learning. *Physical Review Materials*, *5*(4), 043804. https://doi.org/10.1103/PhysRevMaterials.5.043804

<details>
<summary>BibTeX</summary>

```bibtex
@article{Baloch2021,
  author  = {Baloch, Ahmer A. B. and Alqahtani, Saad M. and Mumtaz, Faisal
             and Muqaibel, Ali H. and Rashkeev, Sergey N. and Alharbi, Fahhad H.},
  title   = {Extending {Shannon}'s ionic radii database using machine learning},
  journal = {Physical Review Materials},
  year    = {2021},
  volume  = {5},
  number  = {4},
  pages   = {043804},
  doi     = {10.1103/PhysRevMaterials.5.043804}
}
```
</details>

---

## Turnley_Ionic_Radii.xlsx

Revised ionic radii proposed specifically for chalcogenide perovskite tolerance
factor analysis.

**APA**
> Turnley, J. W., Agarwal, S., & Agrawal, R. (2024). Rethinking tolerance factor analysis for chalcogenide perovskites. *Materials Horizons*, *11*(19), 4802–4808. https://doi.org/10.1039/D4MH00689E

<details>
<summary>BibTeX</summary>

```bibtex
@article{Turnley2024,
  author  = {Turnley, Jonathan W. and Agarwal, Shubhanshu and Agrawal, Rakesh},
  title   = {Rethinking tolerance factor analysis for chalcogenide perovskites},
  journal = {Materials Horizons},
  year    = {2024},
  volume  = {11},
  number  = {19},
  pages   = {4802--4808},
  doi     = {10.1039/D4MH00689E}
}
```
</details>

---

## atomic_features.csv & electronegativities.csv

Elemental features (atomic number, electronegativity, etc.) used as ML
descriptors; sourced from the supplementary data of Bartel *et al.* (2019)
(same reference as `Shannon_Effective_Ionic_Radii.csv`).

**APA**
> Bartel, C. J., et al. (2019). *Science Advances*, *5*(2), eaav0693. https://doi.org/10.1126/sciadv.aav0693

*(See full citation above.)*

---

## pettifor_embedding.csv

Non-orthogonal Pettifor elemental embeddings derived from a similarity measure
between chemical elements, used as compositional descriptors in ML models.

**APA**
> Cerqueira, T. F. T., Wang, H., Botti, S., & Marques, M. A. L. (2024). A non-orthogonal representation of the chemical space. *arXiv preprint*, arXiv:2406.19761. https://doi.org/10.48550/arXiv.2406.19761

<details>
<summary>BibTeX</summary>

```bibtex
@misc{Cerqueira2024,
  author        = {Cerqueira, Tiago F. T. and Wang, Haichen and Botti, Silvana
                   and Marques, Miguel A. L.},
  title         = {A non-orthogonal representation of the chemical space},
  year          = {2024},
  eprint        = {2406.19761},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mtrl-sci},
  doi           = {10.48550/arXiv.2406.19761}
}
```
</details>

---

## chalcogen_semicon_bandgap.csv

Auto-generated database of semiconductor band gaps mined from the literature using NLP (ChemDataExtractor 2.0).

**APA**
> Dong, Q., & Cole, J. M. (2022). Auto-generated database of semiconductor band gaps using ChemDataExtractor. *Scientific Data*, *9*(1), 193. https://doi.org/10.1038/s41597-022-01294-6

<details>
<summary>BibTeX</summary>

```bibtex
@article{Dong2022,
  author  = {Dong, Qingyang and Cole, Jacqueline M.},
  title   = {Auto-generated database of semiconductor band gaps using {ChemDataExtractor}},
  journal = {Scientific Data},
  year    = {2022},
  volume  = {9},
  number  = {1},
  pages   = {193},
  doi     = {10.1038/s41597-022-01294-6}
}
```
</details>

---

## chalcogenides_bandgap_devices.csv

Device performance data compiled from three sources:

### [1] Sopiha *et al.* (2022) — review of chalcogenide perovskite optoelectronics

**APA**
> Sopiha, K. V., Comparotto, C., Márquez, J. A., & Scragg, J. J. S. (2022). Chalcogenide perovskites: Tantalizing prospects, challenging materials. *Advanced Optical Materials*, *10*(3), 2101704. https://doi.org/10.1002/adom.202101704

<details>
<summary>BibTeX</summary>

```bibtex
@article{Sopiha2022,
  author  = {Sopiha, Kostiantyn V. and Comparotto, Corrado and
             M{\'{a}}rquez, Jos{\'{e}} A. and Scragg, Jonathan J. S.},
  title   = {Chalcogenide Perovskites: Tantalizing Prospects, Challenging Materials},
  journal = {Advanced Optical Materials},
  year    = {2022},
  volume  = {10},
  number  = {3},
  pages   = {2101704},
  doi     = {10.1002/adom.202101704}
}
```
</details>

### [2] Zhang *et al.* (2025) — LaScS₃ transparent conducting films

**APA**
> Zhang, H., Zhu, J., Fang, J., Wu, X., Zeng, B., Han, Y., Ming, C., & Sun, Y.-Y. (2025). p-Type transparent conducting material realized by composite thin film of chalcogenide perovskite LaScS₃ and graphene. *Advanced Functional Materials*, e2524382. https://doi.org/10.1002/adfm.202524382

<details>
<summary>BibTeX</summary>

```bibtex
@article{Zhang2025,
  author  = {Zhang, Han and Zhu, Junhao and Fang, Jiao and Wu, Xiaowei and
             Zeng, Biao and Han, Yanbing and Ming, Chen and Sun, Yi-Yang},
  title   = {p-Type Transparent Conducting Material Realized by Composite Thin Film
             of Chalcogenide Perovskite {LaScS}$_3$ and Graphene},
  journal = {Advanced Functional Materials},
  year    = {2025},
  pages   = {e2524382},
  doi     = {10.1002/adfm.202524382}
}
```
</details>

### [3] Perera *et al.* (2016) — chalcogenide perovskites as emerging ionic semiconductors

**APA**
> Perera, S., Hui, H., Zhao, C., Xue, H., Sun, F., Deng, C., Gross, N., Milleville, C., Xu, X., Watson, D. F., Weinstein, B., Sun, Y.-Y., Zhang, S., & Zeng, H. (2016). Chalcogenide perovskites – an emerging class of ionic semiconductors. *Nano Energy*, *22*, 129–135. https://doi.org/10.1016/j.nanoen.2016.02.020

<details>
<summary>BibTeX</summary>

```bibtex
@article{Perera2016,
  author  = {Perera, Samanthe and Hui, Haolei and Zhao, Chuan and Xue, Hongtao and
             Sun, Fan and Deng, Chenhua and Gross, Nelson and Milleville, Chris and
             Xu, Xiaohong and Watson, David F. and Weinstein, Bernard and
             Sun, Yi-Yang and Zhang, Shengbai and Zeng, Hao},
  title   = {Chalcogenide perovskites -- an emerging class of ionic semiconductors},
  journal = {Nano Energy},
  year    = {2016},
  volume  = {22},
  pages   = {129--135},
  doi     = {10.1016/j.nanoen.2016.02.020}
}
```
</details>

---

## perovskite_bandgap_devices.csv

NOMAD dataset of perovskite device data.

**APA**
> NOMAD Repository. (2024). *Perovskite device dataset* [Data set]. NOMAD. https://doi.org/10.17172/NOMAD/2024.09.24-1

<details>
<summary>BibTeX</summary>

```bibtex
@misc{NOMAD2024,
  title     = {Perovskite device dataset},
  year      = {2024},
  publisher = {NOMAD Repository},
  doi       = {10.17172/NOMAD/2024.09.24-1},
  url       = {https://nomad-lab.eu/prod/v1/gui/dataset/doi/10.17172/NOMAD/2024.09.24-1}
}
```
</details>

---

## shuffled_dataset_chalcogenide_pvk.csv & shuffled_dataset_chalcogenide_pvk_new_r.csv

Chalcogenide perovskite experimental stability dataset compiled from
Sopiha *et al.* (2022) and Bartel *et al.* (2019) (both cited above).
The `_new_r` variant uses updated ionic radii from Turnley *et al.* (2024)
(also cited above).

*(No additional citation required beyond those already listed.)*

---

## SJ_limit.csv & DJ_limit.csv

Theoretical PCE limits from the Shockley–Queisser detailed-balance model.
Numerical data computed using the open-source implementation at
https://github.com/marcus-cmc/Shockley-Queisser-limit, based on the original paper:

**APA**
> Shockley, W., & Queisser, H. J. (1961). Detailed balance limit of efficiency of *p*-*n* junction solar cells. *Journal of Applied Physics*, *32*(3), 510–519. https://doi.org/10.1063/1.1736034

<details>
<summary>BibTeX</summary>

```bibtex
@article{Shockley1961,
  author  = {Shockley, William and Queisser, Hans J.},
  title   = {Detailed Balance Limit of Efficiency of $p$-$n$ Junction Solar Cells},
  journal = {Journal of Applied Physics},
  year    = {1961},
  volume  = {32},
  number  = {3},
  pages   = {510--519},
  doi     = {10.1063/1.1736034}
}
```
</details>

---

## cols.csv & ops.csv

Internal helper files listing column names and order-parameter descriptor labels used during feature engineering. No external citation required.

---

## Summary of DOIs

| File(s) | DOI |
|---------|-----|
| `Shannon_Effective_Ionic_Radii.csv`, `atomic_features.csv`, `electronegativities.csv`, `shuffled_dataset_chalcogenide_pvk.csv` | [10.1126/sciadv.aav0693](https://doi.org/10.1126/sciadv.aav0693) |
| `Expanded_Shannon_Effective_Ionic_Radii.csv` | [10.1103/PhysRevMaterials.5.043804](https://doi.org/10.1103/PhysRevMaterials.5.043804) |
| `Turnley_Ionic_Radii.xlsx`, `shuffled_dataset_chalcogenide_pvk_new_r.csv` | [10.1039/D4MH00689E](https://doi.org/10.1039/D4MH00689E) |
| `pettifor_embedding.csv` | [10.48550/arXiv.2406.19761](https://doi.org/10.48550/arXiv.2406.19761) |
| `chalcogen_semicon_bandgap.csv` | [10.1038/s41597-022-01294-6](https://doi.org/10.1038/s41597-022-01294-6) |
| `chalcogenides_bandgap_devices.csv` | [10.1002/adom.202101704](https://doi.org/10.1002/adom.202101704) · [10.1002/adfm.202524382](https://doi.org/10.1002/adfm.202524382) · [10.1016/j.nanoen.2016.02.020](https://doi.org/10.1016/j.nanoen.2016.02.020) |
| `perovskite_bandgap_devices.csv` | [10.17172/NOMAD/2024.09.24-1](https://doi.org/10.17172/NOMAD/2024.09.24-1) |
| `SJ_limit.csv`, `DJ_limit.csv` | [10.1063/1.1736034](https://doi.org/10.1063/1.1736034) |
