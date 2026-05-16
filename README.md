# F1 Results Analysis

Projekt analityczny dotyczący wyników wyścigów Formuły 1 z wykorzystaniem eksploracyjnej analizy danych (EDA), wizualizacji oraz modeli uczenia maszynowego.

Celem projektu było zbadanie czynników wpływających na wyniki wyścigów oraz analiza ryzyka wystąpienia DNF (*Did Not Finish*) na podstawie historycznych danych Formula 1.

---

## Zakres projektu

Projekt obejmuje:

- przygotowanie i czyszczenie danych,
- eksploracyjną analizę danych (EDA),
- analizę zależności między kwalifikacjami a wynikami wyścigów,
- analizę wpływu torów, konstruktorów i kierowców,
- analizę niezawodności zespołów i kierowców,
- tworzenie wizualizacji statystycznych,
- feature engineering,
- budowę modeli klasyfikacyjnych,
- predykcję DNF.

---

## Technologie

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Struktura projektu

```bash
F1_Results_Analysis/
│
├── Analysis/
│   ├── eda.py
│   ├── ml.py
│   └── visualisations.py
│
├── data/
│   ├── circuits.csv
│   ├── constructors.csv
│   ├── drivers.csv
│   ├── qualifying.csv
│   ├── races.csv
│   ├── results.csv
│   └── status.csv
│
└── F1Analysis.py
```

---

## Opis plików

| Plik | Opis |
|---|---|
| `F1Analysis.py` | Główny pipeline projektu |
| `Analysis/eda.py` | Eksploracyjna analiza danych |
| `Analysis/ml.py` | Modele machine learning |
| `Analysis/visualisations.py` | Wizualizacje danych |
| `data/*.csv` | Historyczne dane Formula 1 |

---

## Exploratory Data Analysis (EDA)

W projekcie przeanalizowano m.in.:

- zależność pozycji startowej od wyniku wyścigu,
- wpływ kwalifikacji na końcowy rezultat,
- skuteczność kierowców na różnych torach,
- zmienność pozycji w trakcie wyścigów,
- niezawodność konstruktorów,
- ekspozycję na DNF mechaniczne i wyścigowe,
- historyczne trendy wyników.

Projekt zawiera również wizualizacje takie jak:

- heatmapy,
- wykresy korelacji,
- wykresy trendów sezonowych,
- analizy niezawodności konstruktorów,
- analizy racecraftu kierowców.

---

## Machine Learning

W projekcie wykorzystano modele klasyfikacyjne do przewidywania prawdopodobieństwa DNF.

### Wykorzystane modele

- Logistic Regression
- Random Forest
- Histogram Gradient Boosting

### Przykładowe cechy wykorzystane w modelach

#### Podstawowe cechy:
- pozycja startowa (`grid`),
- pozycja kwalifikacyjna (`qualiPosition`),
- wiek kierowcy (`driverAge`).

#### Cechy kategoryczne:
- konstruktor,
- tor wyścigowy.

#### Feature engineering:
- historyczny współczynnik mechanical DNF kierowcy,
- historyczny współczynnik mechanical DNF konstruktora,
- historyczny współczynnik incident DNF toru.

### Metryki ewaluacyjne

- Accuracy
- F1-score
- Classification Report
- ROC-AUC
- Confusion Matrix

---

## Najważniejsze wnioski

- Pozycja startowa ma bardzo silny wpływ na wynik wyścigu.
- Siła zależności między kwalifikacjami a wynikiem różni się między torami.
- Niektóre tory charakteryzują się znacznie większą zmiennością pozycji podczas wyścigu.
- Konstruktorzy różnią się poziomem niezawodności mechanicznej.
- DNF mają częściowo losowy charakter, co ogranicza maksymalną skuteczność modeli predykcyjnych.


