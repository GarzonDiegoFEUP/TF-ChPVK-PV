# Sustainability Data Sources

This file documents the origin of every raw data file in this directory, with
full APA and BibTeX citations.

---

## MCS2025_World_Data.csv

World mineral production, capacity, and reserves data extracted from the USGS
Mineral Commodity Summaries 2025, used to compute the
Herfindahl–Hirschman Index (HHI) supply-risk scores.

**APA**
> National Minerals Information Center. (2025). *U.S. Geological Survey Mineral Commodity Summaries 2025 Data Release* (ver. 2.0, April 2025) [Data set]. U.S. Geological Survey. https://doi.org/10.5066/P13XCP3R

<details>
<summary>BibTeX</summary>

```bibtex
@misc{USGS_MCS2025,
  author    = {{National Minerals Information Center}},
  title     = {{U.S. Geological Survey Mineral Commodity Summaries 2025 Data Release}},
  year      = {2025},
  version   = {2.0},
  publisher = {U.S. Geological Survey},
  doi       = {10.5066/P13XCP3R},
  url       = {https://www.sciencebase.gov/catalog/item/6798fd34d34ea8c18376e8ee}
}
```
</details>

---

## ESG_World_Data_2023.csv

World Bank Environment, Social and Governance (ESG) dataset covering 17
sustainability themes across 200+ countries, used to derive country-level
supply-risk governance scores.

**APA**
> World Bank Group. (2023). *Environment, Social and Governance Data* [Data set]. World Bank Data Catalog. https://datacatalog.worldbank.org/search/dataset/0037651

<details>
<summary>BibTeX</summary>

```bibtex
@misc{WorldBank_ESG2023,
  author    = {{World Bank Group}},
  title     = {Environment, Social and Governance Data},
  year      = {2023},
  publisher = {World Bank Data Catalog},
  url       = {https://datacatalog.worldbank.org/search/dataset/0037651}
}
```
</details>

## earth_abundance_data.csv

Elemental abundance data (relative to Si = 10⁶ or by mass fraction in the Earth's crust).

**APA**
> Wikipedia contributors. (2025). *Abundances of the elements (data page)*. In *Wikipedia, The Free Encyclopedia*. Retrieved February 20, 2026, from https://en.wikipedia.org/wiki/Abundances_of_the_elements_(data_page)

<details>
<summary>BibTeX</summary>

```bibtex
@misc{Wikipedia_ElemAbundance,
  author    = {{Wikipedia contributors}},
  title     = {Abundances of the elements (data page)},
  year      = {2025},
  publisher = {Wikipedia, The Free Encyclopedia},
  url       = {https://en.wikipedia.org/wiki/Abundances_of_the_elements_(data_page)},
  note      = {Retrieved February 20, 2026}
}
```
</details>

---

## results_HHI.csv

Derived output file — HHI supply-risk scores computed from `MCS2025_World_Data.csv`
and `ESG_World_Data_2023.csv`. No external citation required.

---

## mapping/

Internal mapping files used to link element names / commodity names between
datasets. No external citation required.

---

## Summary of DOIs

| File | DOI / URL |
|------|-----------|
| `MCS2025_World_Data.csv` | [10.5066/P13XCP3R](https://doi.org/10.5066/P13XCP3R) |
| `ESG_World_Data_2023.csv` | [datacatalog.worldbank.org/search/dataset/0037651](https://datacatalog.worldbank.org/search/dataset/0037651) |
| `earth_abundance_data.csv` | [en.wikipedia.org/wiki/Abundances_of_the_elements_(data_page)](https://en.wikipedia.org/wiki/Abundances_of_the_elements_(data_page)) |
