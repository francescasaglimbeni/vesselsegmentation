# Master Pipeline - Sistema Completo di Analisi Airways & Validazione

Sistema automatizzato per l'analisi completa delle vie aeree da scansioni TC toraciche, con validazione tecnica e correlazione clinica con FVC%.

**Autore:** Francesca Saglimbeni  
**Data:** Gennaio 2026  
**Dataset:** OSIC Pulmonary Fibrosis Progression

---

## 📋 Panoramica

Il **Master Pipeline** orchestra l'intero workflow di analisi in **3 step sequenziali**:

```
CT Scan (OSIC)
    ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: AIRWAY PIPELINE                                    │
│  • Segmentazione vie aeree (TotalSegmentator)               │
│  • Preprocessing & cleaning                                 │
│  • Analisi morfometrica (volume, tortuosity, symmetry...)   │
│  • Metriche parenchimali (entropy, density, GGO, fibrosis)  │
│  • Dual Fibrosis Scoring (AIRWAY_ONLY + COMBINED)          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: TECHNICAL VALIDATION                               │
│  • Confronto metriche vs letteratura                        │
│  • Classificazione RELIABLE / UNRELIABLE                    │
│  • Identificazione problemi tecnici                         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: FVC CORRELATION ANALYSIS                           │
│  • Correlazioni metriche vs FVC% (normalized)               │
│  • Validazione dual scoring system                          │
│  • Confronto AIRWAY_ONLY vs COMBINED                        │
│  • Visualizzazioni statistiche                              │
└─────────────────────────────────────────────────────────────┘
    ↓
Risultati completi + report + grafici
```

---

## 🚀 Uso Rapido

### Workflow completo OSIC (consigliato)
```powershell
python master_pipeline.py
```

Esegue automaticamente:
- Pipeline completa su tutti gli scan OSIC
- Validazione tecnica
- Analisi correlazione FVC%

### Modalità fast (per test)
```powershell
python master_pipeline.py --fast
```

Usa TotalSegmentator in modalità fast (più veloce ma meno accurato).

### Singolo scan
```powershell
python master_pipeline.py --single path/to/scan.mhd
```

Processa un solo scan (solo pipeline, senza validazione).

---

## 📂 Struttura Directory

```
vesselsegmentation/
├── master_pipeline.py              # ← SCRIPT PRINCIPALE
│
├── airway_segmentation/
│   ├── main_pipeline.py            # Pipeline completa (6 step)
│   ├── fibrosis_scoring.py         # Dual scoring system
│   ├── parenchymal_metrics.py      # Metriche polmonari
│   └── ...
│
├── validation_pipeline/
│   ├── air_val/
│   │   └── air_val.py              # Validazione tecnica
│   └── OSIC_metrics_validation/
│       └── analyze_osic_metrics.py # Correlazione FVC%
│
├── datasets/
│   └── OSIC_correct/               # Scansioni TC (.mhd/.raw)
│
└── ../results/
    └── results_OSIC_newMetrcis/    # Output completi
        ├── ID00xxx.../
        │   ├── step1_airway_mask/
        │   ├── step2_airway_refined/
        │   ├── step3_airway_cleaned/
        │   ├── step4_advanced_metrics/
        │   │   ├── advanced_metrics.json    # Metriche airways
        │   │   └── visualization.png
        │   ├── step5_parenchymal_metrics/
        │   │   ├── parenchymal_metrics.json # Metriche polmone
        │   │   └── visualization.png
        │   └── step6_fibrosis_score/
        │       ├── fibrosis_report.json     # Dual scores
        │       └── fibrosis_visualization.png
        └── ...
```

---

## 🔬 STEP 1: Airway Pipeline (main_pipeline.py)

### Processo completo in 6 step:

#### **Step 1: Segmentazione**
- Input: CT scan (.mhd/.raw)
- Tool: TotalSegmentator
- Output: Maschera 3D vie aeree + polmoni

#### **Step 2: Preprocessing & Refinement**
- Riempimento gap (airway_gap_filler.py)
- Smoothing morfologico
- Validazione connettività

#### **Step 3: Cleaning & Skeleton**
- Rimozione artefatti
- Scheletrizzazione 3D
- Identificazione nodi/biforcazioni

#### **Step 4: Advanced Metrics**
Calcola **metriche morfometriche avanzate**:
- `volume_ml`: Volume vie aeree totale
- `surface_area_cm2`: Superficie totale
- `mean_diameter_mm`: Diametro medio
- `mean_tortuosity`: Tortuosità media (indice di distorsione)
- `symmetry_score`: Simmetria dx-sx (0-1)
- `branch_count`: Numero biforcazioni
- `peripheral_density`: Densità periferica (arborescenza)
- `peripheral_volume_ratio`: % volume zone periferiche
- `parent_child_ratio`: Rapporto generazioni genitori/figli

Output: `advanced_metrics.json`

#### **Step 5: Parenchymal Metrics**
Calcola **metriche parenchimali** da polmoni:
- `mean_hu`: Densità media Hounsfield Units
- `std_hu`: Deviazione standard densità
- `parenchymal_entropy`: Entropia texture (eterogeneità)
- `parenchymal_density_score`: Score densità normalizzato
- `percent_ground_glass_opacity`: % Ground Glass Opacity (0-100)
- `percent_fibrotic_patterns`: % Pattern fibrotici (0-100)
- `basal_predominance_index`: Predominanza basale (0-1)

Output: `parenchymal_metrics.json`

#### **Step 6: Dual Fibrosis Scoring**
Calcola **DUE score di fibrosi** con pesi diversi:

**AIRWAY_ONLY** (Opzione 1 - focus vie aeree):
```python
weights = {
    'peripheral_density': 0.35,      # Peso maggiore su arborescenza
    'peripheral_volume': 0.25,       # Volume periferico
    'parent_child_ratio': 0.20,      # Rapporto generazioni
    'tortuosity': 0.15,              # Distorsione
    'symmetry': 0.05                 # Asimmetria
}
```
**Correlazione con FVC%:** r = +0.280*** (p<0.001) ⚠ direzione sbagliata

**COMBINED** (Opzione 2 - CONSIGLIATO - include parenchima):
```python
weights = {
    'parenchymal_entropy': 0.35,     # Eterogeneità texture
    'parenchymal_density': 0.25,     # Densità polmonare
    'peripheral_density': 0.15,      # Arborescenza
    'peripheral_volume': 0.15,       # Volume periferico
    'tortuosity': 0.05,              # Distorsione
    'symmetry': 0.05                 # Asimmetria
}
```
**Correlazione con FVC%:** r = -0.497*** (p<0.001) ✓ direzione corretta  
**Miglioramento:** +77.5% rispetto ad AIRWAY_ONLY

Output: `fibrosis_report.json` con entrambi gli score

---

## ✅ STEP 2: Technical Validation (air_val.py)

Valida risultati confrontando con **range letteratura** (studi pubblicati):

### Metriche validate:
- Volume totale vie aeree (ml)
- Numero biforcazioni
- Diametro medio (mm)
- Tortuosità
- Simmetria dx-sx

### Classificazione:
- **RELIABLE:** Tutte le metriche nei range attesi
- **UNRELIABLE:** Una o più metriche fuori range

### Output:
- `OSIC_validation_newmetrics.csv`
- Report con:
  - Totale casi
  - % RELIABLE / UNRELIABLE
  - Issue più comuni (es: "volume_too_low", "excessive_tortuosity")

**Risultati tipici OSIC:** ~89% RELIABLE (40/45)

---

## 📊 STEP 3: FVC Correlation Analysis (analyze_osic_metrics.py)

Analizza correlazioni tra metriche CT e **FVC%** (funzionalità respiratoria):

### FVC% Normalized
```
FVC% = (FVC_observed / FVC_predicted) × 100

FVC_predicted corretto per:
- Età
- Sesso
- Altezza
```

### Analisi eseguite:

#### 1. Correlazioni singole metriche
- Pearson correlation per ogni metrica vs FVC%
- Significatività statistica (p-value)
- Heatmap correlazioni

**Top correlazioni (negative = peggiora con FVC basso):**
- `parenchymal_entropy`: r = -0.69*** (texture disorganizzata)
- `parenchymal_density`: r = -0.65*** (aumento densità)
- `peripheral_density`: r = +0.47*** (perdita arborescenza)

#### 2. Validazione Dual Fibrosis Score
Confronta performance:
- **AIRWAY_ONLY score** (Opzione 1)
- **COMBINED score** (Opzione 2 - RACCOMANDATO)

#### 3. Visualizzazioni
- Scatter plot con regressione lineare
- ROC-style comparison
- Bar chart miglioramento

### Output:
```
validation_pipeline/OSIC_metrics_validation/results_analysis/
├── integrated_dataset.csv               # Dataset completo
├── correlation_results.csv              # Tutte le correlazioni
├── fibrosis_score_comparison.json       # Confronto dual scoring
├── fibrosis_score_comparison.png        # Grafico confronto
└── correlation_summary.png              # Heatmap correlazioni
```

---

## 📈 Risultati Finali

Al termine del workflow completo:

### Pipeline Results
```
X:\Francesca Saglimbeni\tesi\results\results_OSIC_newMetrcis\
├── ID00xxx.../
│   ├── step4_advanced_metrics/advanced_metrics.json
│   ├── step5_parenchymal_metrics/parenchymal_metrics.json
│   └── step6_fibrosis_score/fibrosis_report.json
```

Ogni `fibrosis_report.json` contiene:
```json
{
  "scoring_methods": {
    "airway_only": {
      "fibrosis_score": 0.45,
      "grade": "MODERATE",
      "correlation_with_fvc": 0.280
    },
    "combined": {
      "fibrosis_score": 0.72,
      "grade": "SEVERE",
      "correlation_with_fvc": -0.497
    }
  },
  "recommended_method": "combined",
  "improvement_percent": 77.5
}
```

### Validation Results
```
validation_pipeline/air_val/OSIC_validation_newmetrics.csv
```

| patient | volume_ml | branch_count | tortuosity | status | issues |
|---------|-----------|--------------|------------|--------|--------|
| ID00xxx | 145.2 | 47 | 1.23 | RELIABLE | - |
| ID00yyy | 89.3 | 28 | 1.89 | UNRELIABLE | tortuosity_high |

### FVC Analysis Results
```
validation_pipeline/OSIC_metrics_validation/results_analysis/
└── fibrosis_score_comparison.json
```

```json
{
  "airway_only": {
    "n_measurements": 351,
    "n_patients": 40,
    "correlation": 0.280,
    "p_value": 4.2e-8,
    "direction": "POSITIVE (unexpected)"
  },
  "combined": {
    "n_measurements": 351,
    "n_patients": 40,
    "correlation": -0.497,
    "p_value": 6.8e-23,
    "direction": "NEGATIVE (correct)"
  },
  "improvement": {
    "absolute": 0.217,
    "relative_percent": 77.5
  },
  "recommendation": "Use COMBINED scoring method"
}
```

---

## 🔧 Requisiti Tecnici

### Dipendenze Python
```
numpy
scipy
pandas
matplotlib
seaborn
scikit-image
SimpleITK
networkx
nibabel
```

### Tool esterni
- **TotalSegmentator** (segmentazione automatica CT)
  ```bash
  pip install TotalSegmentator
  ```

### Hardware
- RAM: ≥ 16 GB (consigliato 32 GB)
- Storage: ~500 MB per scan
- GPU: Opzionale (accelera TotalSegmentator)

---

## 📝 Note Importanti

### Dual Scoring System

Il sistema calcola **due score** con filosofie diverse:

1. **AIRWAY_ONLY** (Opzione 1)
   - Focus: Morfologia vie aeree
   - Vantaggio: Non richiede segmentazione polmonare accurata
   - Svantaggio: Correlazione debole con FVC% (r=0.28)
   - Uso: Quando segmentazione parenchimale fallisce

2. **COMBINED** (Opzione 2 - RACCOMANDATO)
   - Focus: Parenchima + airways
   - Vantaggio: Correlazione forte (r=-0.50, 77.5% meglio)
   - Svantaggio: Richiede parenchima valido
   - Uso: Default per analisi cliniche

**Raccomandazione:** Usare sempre COMBINED quando disponibile.

### Gestione Errori

Il pipeline gestisce automaticamente:
- Scansioni con segmentazione fallita (skip)
- Metriche parenchimali mancanti (fallback ad AIRWAY_ONLY)
- Validazione unreliable (marcata ma processata)

### Performance

Tempi medi per scan (CPU Intel i7, 16GB RAM):
- Step 1 (segmentazione): ~3-5 min
- Step 2-4 (preprocessing + analisi): ~2 min
- Step 5 (parenchima): ~1 min
- Step 6 (fibrosis scoring): ~10 sec

**Totale: ~6-8 minuti per scan**  
**Dataset completo (45 scans): ~5-6 ore**

Con `--fast`: ~2-3 ore (ma meno accurato)

---

## 🐛 Troubleshooting

### "Dataset directory not found"
Verifica path:
```powershell
ls "X:\Francesca Saglimbeni\tesi\vesselsegmentation\datasets\OSIC_correct"
```

### "TotalSegmentator not found"
Installa:
```powershell
pip install TotalSegmentator
```

### "No parenchymal metrics"
Alcuni scan potrebbero non avere step5 valido → usa AIRWAY_ONLY score.

### "Analysis failed"
Verifica che:
1. Pipeline sia completata (almeno 40 scans)
2. File `osic_fvc_data.csv` esista in validation_pipeline/OSIC_metrics_validation/

---

## 📚 Riferimenti

- **TotalSegmentator:** Wasserthal et al., "TotalSegmentator: robust segmentation of 104 anatomical structures in CT images", 2023
- **OSIC Dataset:** Kaggle OSIC Pulmonary Fibrosis Progression Challenge
- **FVC Prediction:** GLI-2012 equations (Quanjer et al.)

---

## 👤 Contatti

**Autore:** Francesca Saglimbeni  
**Progetto:** Tesi - Analisi Morfometrica Airways in Fibrosi Polmonare  
**Anno:** 2026

