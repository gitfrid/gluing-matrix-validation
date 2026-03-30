# Gluing Matrix Validation — Eine verständliche wissenschaftlich korrekte Fassung

## Abstract

**Kurzfassung**  
Wir verbinden die philosophische Idee des Peirceanischen Kontinuums mit einer robusten numerischen Methode zur Analyse von Quanten-Tunneln in eindimensionalen Potentialen.  

Zentraler Befund:

```
T(E) ∝ σ_min²(E)
```

Die Arbeit erklärt intuitiv die Herkunft dieser Idee aus Peirces Kontinuumsvorstellung, leitet die Matrixformalismen her, beschreibt eine analytische Vorbedingung zur Beseitigung exponentieller Ill-Conditioning-Effekte und liefert ein reproduzierbares numerisches Protokoll zur Validierung und Resonanzanalyse.  

Ziel ist ein Manuskript, das Laien verstehen können, ohne die mathematische Strenge für Fachleser zu verlieren.

---

## Introduction — Peircean Origin

### Intuition

Charles S. Peirce beschreibt das Kontinuum als ein Geflecht lokaler Abschnitte, die auf Überlappungen zusammenpassen müssen, damit ein globales Ganzes entsteht.

Übertragen auf die Quantenmechanik:

- In stückweise konstanten Potentialen existieren lokale Wellenlösungen  
- Ein globaler Zustand entsteht nur durch **kompatibles Zusammenkleben**

---

### Warum Peirce relevant ist

Peirces Bild liefert die konzeptionelle Brücke:

- physikalisches Problem → **Gluing Problem**
- mathematische Sprache → **Garben / Kohomologie**
- numerische Umsetzung → **Gluing Matrix**

Der kleinste Singulärwert misst:

```
Grad der Kohäsion ≈ Wie gut lokale Lösungen zusammenpassen
```

---

## Mathematical Framework

### Aufbau der Gluing Matrix

In jedem Bereich ist die Wellenfunktion eine Linearkombination von:

- propagierenden Komponenten: `exp(± i k x)`
- evaneszenten Komponenten: `exp(± κ x)`

mit:

```
κ = sqrt(V0 - E)
```

Kontinuitätsbedingungen für `ψ` und `ψ'` führen zu:

```
A(E) x = 0
```

- `A(E)` = Gluing Matrix  
- `x` = Koeffizientenvektor  

Physikalische Lösungen entsprechen **(nahe-)Kernvektoren** von `A(E)`.

---

### Singulärwerte als Maß für Kohäsion

Singulärwerte:

```
σ1 ≤ σ2 ≤ ... ≤ σn
```

- `σ_min = σ1` misst Distanz zur Singularität  
- kleine `σ_min` ⇒ starke Kopplung ⇒ physikalischer Kanal  

Im Tunnelbereich (`E < V0`):

```
σ_min(E) ≈ pre(E) * exp(-κ a)
```

⇒ daraus folgt:

```
T(E) ≈ C(E) * σ_min²(E)
```

---

### Resonanzen und Multiplizität

Resonanzen zeigen sich durch:

```
σ1 ≈ σ2 ≈ ... ≈ σC << σC+1
```

- `C` = Multiplizität (Anzahl kohärenter Modi)  
- entspricht strukturell einer Kohomologie-Dimension  

---

## Numerical Methods and Implementation

### Problem der Ill-Conditioning

Evaneszente Terme:

```
exp(± κ x)
```

führen zu:

- extrem großen / kleinen Matrixeinträgen  
- instabiler SVD  
- Verlust physikalischer Information  

---

### Analytische Vorbedingung

Skalierung:

```
A → D A
```

mit:

```
D ≈ diag(exp(-κ x_i))
```

Eigenschaften:

- erhält Kernstruktur  
- erhält Singularwert-Verhältnisse  
- verbessert Kondition drastisch  

---

### SVD Workflow

1. Matrix aufbauen:
   ```
   A(E)
   ```

2. SVD berechnen:
   ```
   A = U Σ V*
   ```

3. Normalisieren:
   ```
   σ̃_i = σ_i / σ_max
   ```

4. Multiplizität erkennen:

   ```
   g_i = σ̃_(i+1) / (σ̃_i + ε_tiny)
   ```

5. Regression:

   ```
   log T(E) vs 2 log σ_min(E)
   ```

6. Tunnelbereich einschränken:
   ```
   κ a > 2
   ```

7. Kennzahlen:
   - Pearson
   - Spearman
   - R²

---

## Reproduzierbarkeit und technische Details

- Python: `numpy`, `scipy`, `matplotlib`, `pandas`, `sklearn`
- Notebook enthält:
  - Matrix-Builder
  - Energie-Sweeps
  - SVD-Spektren
  - Regression
  - Multiplizitätsplots
  - CSV-Export

Wichtig:

- Toleranzen dokumentieren  
- Skalierung offenlegen  
- Beispielskripte bereitstellen  

---

## Beispielcode — Stabilized Matrix Builder

```python
# siehe repository: build_A_matrix_double_stabilized
```

---

## Kernaussage

```
T(E) ∝ σ_min²(E)
```

ist **kein numerischer Zufall**, sondern Ausdruck davon,  
wie gut lokale Lösungen global zusammenpassen.

---

## Interpretation

- lokale Lösungen = „Sections“
- Matching = „Gluing“
- globale Lösung = „Global Section“

Die SVD misst:

```
Kohärenz des Systems
```

und verbindet damit:

- Quantenphysik  
- Numerik  
- Peirce’s Kontinuum  
