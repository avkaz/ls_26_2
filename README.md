# Projekt: Textová klasifikace (klasický Machine Learning)

## Přehled projektu
Hlavním přístupem zvoleným v tomto projektu je **klasický machine learning**.  

Existují i další možné přístupy, například využití **transformerových modelů** (BERT, LLM apod.), nicméně podle mého názoru by to pro tento konkrétní případ bylo zbytečně komplexní řešení.  
Věřím, že **klasický ML přístup je pro tento problém dostačující** a nabízí lepší interpretovatelnost i jednodušší experimentování.

---

## Struktura projektu

Projekt je strukturován s důrazem na přehlednost a znovupoužitelnost kódu:

├── experiment.ipynb
├── utility.py
├── raw_data/
├── mlruns/
└── README.md


### `experiment.ipynb`
Hlavní notebook, ve kterém probíhá:
- trénování modelů
- testování různých přístupů
- ladění hyperparametrů
- analýza výsledků

Notebook obsahuje **více modelů a různá nastavení**.  
Jeho struktura záměrně odráží tok mých myšlenek, postupné experimentování a pozorování získaná během práce na projektu.

---

### `utility.py`
Soubor obsahuje pomocné funkce, které byly vyčleněny mimo notebook:
- pro lepší **reusability**
- pro přehlednější notebook
- pro snadnější údržbu kódu

---

## Experiment tracking – MLflow
Pro sledování výsledků jednotlivých experimentů je použit **MLflow**.  
Pomocí MLflow jsou logovány:
- použité modely
- jejich parametry
- dosažené metriky
---

