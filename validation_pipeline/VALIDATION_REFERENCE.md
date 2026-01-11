# 📋 Airway Pipeline Validation - Literature References

Questo documento descrive i valori di letteratura utilizzati per validare la pipeline di analisi delle vie aeree.

## 🎯 Obiettivo della Validazione

Dato che non abbiamo ground truth o dataset annotati per training/testing, utilizziamo **valori normativi da letteratura** per validare:

1. **Segmentazione** - Volume e superficie delle vie aeree
2. **Grafo** - Struttura e topologia dell'albero bronchiale
3. **Weibel Generations** - Profondità e diametri lungo le generazioni
4. **Fibrosi** - Metriche morfologiche (PC ratio, tortuosity)

---

## 📚 Valori di Riferimento

### 1️⃣ SEGMENTAZIONE

#### Volume delle vie aeree (ml)
- **Media**: 180 ml
- **Std Dev**: ±50 ml
- **Range valido**: 80-350 ml
- **Fonte**: Montaudon et al. 2007
- **Note**: Volume totale dell'albero bronchiale in adulti sani visibile su CT

#### Superficie totale (cm²)
- **Media**: 200 cm²
- **Std Dev**: ±80 cm²
- **Range valido**: 80-400 cm²
- **Fonte**: Studi di morfometria CT
- **Note**: Superficie interna delle vie aeree

---

### 2️⃣ GRAFO

#### Numero di rami (branch count)
- **Media**: 1500
- **Std Dev**: ±500
- **Range valido**: 500-3000
- **Fonte**: Weibel 1963
- **Note**: Numero di segmenti bronchiali visibili in CT (dipende da risoluzione)

#### Rapporto di biforcazione (bifurcation ratio)
- **Media**: 0.15
- **Std Dev**: ±0.05
- **Range valido**: 0.05-0.30
- **Fonte**: Horsfield & Cumming 1968
- **Note**: Ratio tra nodi di biforcazione (degree ≥3) e nodi totali

#### Lunghezza media dei rami (mm)
- **Media**: 12.0 mm
- **Std Dev**: ±5.0 mm
- **Range valido**: 5.0-25.0 mm
- **Fonte**: Modelli di Weibel & Horsfield
- **Note**: Lunghezza media dei segmenti tra biforcazioni

---

### 3️⃣ WEIBEL GENERATIONS

#### Generazione massima
- **Media**: 18
- **Std Dev**: ±3
- **Range valido**: 12-23
- **Fonte**: Weibel 1963
- **Note**: Profondità massima dell'albero bronchiale (in CT tipicamente 12-20)

#### Tapering ratio (parent/child diameter)
- **Media**: 0.793
- **Std Dev**: ±0.05
- **Range valido**: 0.70-0.88
- **Fonte**: Weibel (2^(-1/3))
- **Note**: Ratio ideale per conservazione di area tra generazioni

#### Diametro trachea (gen 0) (mm)
- **Media**: 18.0 mm
- **Std Dev**: ±2.0 mm
- **Range valido**: 14.0-22.0 mm
- **Fonte**: Dati normativi CT
- **Note**: Diametro della trachea in adulti

#### Diametro generazione 5 (mm)
- **Media**: 3.5 mm
- **Std Dev**: ±1.0 mm
- **Range valido**: 2.0-6.0 mm
- **Fonte**: Tabelle di Weibel
- **Note**: Diametro approssimativo alla 5ª generazione

---

### 4️⃣ FIBROSI & MORFOLOGIA

#### PC Ratio (Peripheral/Central) - Soggetti Sani
- **Media**: 0.45
- **Std Dev**: ±0.15
- **Range valido**: 0.25-0.65
- **Fonte**: Studi quantitativi CT in IPF
- **Note**: Ratio tra volume periferico e centrale in soggetti sani

#### PC Ratio - Soggetti Fibrotici
- **Media**: 0.20
- **Std Dev**: ±0.10
- **Range valido**: 0.05-0.35
- **Fonte**: Studi CT in UIP/IPF
- **Note**: PC ratio ridotto indica perdita periferica (fibrosi)

**⚠️ Interpretazione**: 
- PC ratio **alto** (>0.40) → Normale distribuzione periferica
- PC ratio **basso** (<0.30) → Possibile fibrosi periferica o pattern centrale predominante

#### Tortuosity (path tortuosity)
- **Media**: 1.25
- **Std Dev**: ±0.15
- **Range valido**: 1.0-1.6
- **Fonte**: Studi di morfometria CT
- **Note**: Ratio tra lunghezza del path e distanza euclidea (1.0 = perfettamente dritto)

**⚠️ Interpretazione**:
- Tortuosity **~1.0-1.3** → Vie aeree relativamente dritte
- Tortuosity **>1.5** → Possibile distorsione o pattern patologico

---

## 🔍 Logica di Validazione

### Status dei Check

Ogni metrica viene valutata con uno dei seguenti status:

1. **PASS** ✓
   - Valore all'interno del `valid_range` di letteratura
   - Indica che la pipeline produce risultati plausibili

2. **WARNING** ⚠
   - Valore fuori dal `valid_range` ma entro 2σ dalla media
   - Potrebbe essere accettabile ma merita attenzione

3. **FAIL** ✗
   - Valore implausibile (fuori dal valid_range e oltre 2σ)
   - Indica problemi nella pipeline o caso patologico estremo

### Pipeline Usability

La pipeline è considerata **USABLE** se:
- Nessun check ha status `FAIL`
- I WARNING sono accettabili e documentati

---

## 📊 Output della Validazione

### File generati per ogni caso

1. **`PIPELINE_VALIDATION.json`** in ogni cartella caso
   - Report dettagliato con tutti i check
   - Status per ogni metrica
   - Valori misurati vs riferimenti

### File summary generale

2. **`PIPELINE_VALIDATION_SUMMARY.csv`**
   - Tabella riassuntiva di tutti i casi
   - Colonne: case, status, total_checks, passed, warnings, failed
   - Metriche chiave: volume, branch_count, max_generation, pc_ratio

---

## 🔬 Riferimenti Bibliografici

- **Weibel, E.R. (1963)** - "Morphometry of the Human Lung"
- **Horsfield, K. & Cumming, G. (1968)** - "Morphology of the bronchial tree in man"
- **Montaudon et al. (2007)** - "Assessment of airways with three-dimensional quantitative thin-section CT"
- **CT morphometry studies** - Vari studi di morfometria quantitativa delle vie aeree

---

## ⚙️ Utilizzo

```bash
# Esegui validazione batch
python air_val.py

# Output:
# - PIPELINE_VALIDATION.json per ogni caso
# - PIPELINE_VALIDATION_SUMMARY.csv nella cartella output
# - airway_pipeline_validation_summary.csv nella cartella validation_pipeline
```

---

## 📝 Note Importanti

1. **Variabilità fisiologica**: I range sono ampi per includere la variabilità normale
2. **Dipendenza da CT**: Molti valori dipendono da risoluzione e qualità della CT
3. **Patologie**: Casi patologici possono validamente essere fuori dai range "sani"
4. **Uso clinico**: Questa validazione è per QC della pipeline, non per diagnosi clinica

---

**Autore**: Pipeline Validation Tool  
**Data**: Gennaio 2026  
**Versione**: 1.0
