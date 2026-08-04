# Assessment Guide — Code by Hand

You type this code yourself, cell by cell, in `assessment.ipynb`. Each block below is
ONE notebook cell. Type it, run it, check the output, move to the next.

**How to use:** start a new code cell with **+ Code** in VS Code, type the block, press
Shift+Enter. Where a line says `# <- CHANGE`, replace it with a column from YOUR dataset
(the question PDF names the exact columns to use).

---

## The 45-minute plan

| Part | Focus | Time |
|------|-------|------|
| 1 | Data validation & visualization | 0-15 |
| 2 | Text cleaning & categorization | 15-30 |
| 3 | Comprehensive analysis | 30-45 |

Submit: this notebook (code + outputs + report cells at the bottom) + screen recording.

---

## SETUP (one cell)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("dataset/dataset.csv")   # <- your CSV filename
print("Rows:", df.shape[0], "| Columns:", df.shape[1])
df.head()
```

---

## Part 1 — Data Validation and Visualization

### Cell 1: inspect, missing, duplicates

```python
df.info()
print("Missing values:")
print(df.isna().sum()[df.isna().sum() > 0])
print("Duplicate rows:", df.duplicated().sum())
```

### Cell 2: numeric summary

```python
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", num_cols)
df[num_cols].describe()
```

### Cell 3: hidden-issue detector (text that looks numeric)

```python
for c in df.columns:
    if c in num_cols:
        continue
    x = pd.to_numeric(df[c], errors="coerce")
    if x.notna().mean() > 0.5:
        print(f"[!] {c!r} is text but {x.notna().sum()}/{len(df)} values parse as numbers")
```

### Cell 4: histograms

```python
for c in num_cols:
    df[c].dropna().hist(bins=40)
    plt.title(f"Histogram: {c}")
    plt.show()
```

**Look for:** the `[!]` flags in Cell 3, impossible values in `describe()` (Cell 2),
weird spikes/gaps in the histograms. That's your data-quality issue.

---

## Part 2 — Text Cleaning and Categorization

### Cell 5: clean the text columns

```python
text_cols = [c for c in df.columns if str(df[c].dtype) in ("object", "str", "category")]
print("Text columns:", text_cols)

import re
def clean(s):
    if pd.isna(s):
        return s
    return re.sub(r"\s+", " ", str(s).strip().lower())

for c in text_cols:
    df[c + "_clean"] = df[c].apply(clean)
    print(f"\n{c}: {df[c].nunique()} unique -> {df[c + '_clean'].nunique()} cleaned")
    print(df[c + "_clean"].value_counts().head(15))
```

### Cell 6: merge near-duplicate categories

```python
col = "gender_clean"          # <- CHANGE to the cleaned text column you are categorizing
df[col].value_counts()        # look for typos / label variants

# Add any pairs that mean the same thing, then recount:
df[col] = df[col].replace({
    "female": "f",            # <- your pairs
    # "st marys": "st mary's",
})
df[col].value_counts()
```

*(Skip ID-like columns — they have no meaningful categories.)*

---

## Part 3 — Comprehensive Analysis

### Cell 7: group and aggregate

```python
group_col = "school_name"   # <- CHANGE to a categorical column in your data
mean_col  = "exam_score"    # <- CHANGE to a numeric column
df.groupby(group_col)[mean_col].agg(["count", "mean", "median", "min", "max"]).round(2).sort_values("mean", ascending=False)
```

### Cell 8: cross-tabulate two categorical columns

```python
# group_col vs the cleaned text column with the FEWEST categories
cat = min(text_cols, key=lambda c: df[c + "_clean"].nunique())
pd.crosstab(df[group_col], df[cat + "_clean"], normalize="index").round(3)
```

### Cell 9: chart + extremes

```python
df.groupby(group_col)[mean_col].mean().sort_values().plot.bar()
plt.xticks(rotation=45)
plt.title(f"Mean {mean_col} by {group_col}")
plt.show()

df.nlargest(5, mean_col)    # top 5
df.nsmallest(5, mean_col)   # bottom 5
```

**Interpret:** which group is highest/lowest, and does it relate to the Part 1 issue?

---

## Report cells (bottom of notebook)

Paste 2-4 sentences per part into the Report section of the notebook.

**Part 1:** what you checked; the data-quality issue; how you handled it.
**Part 2:** cleaning applied; unique counts before/after; final categories.
**Part 3:** key finding; one chart; connection to the Part 1 issue.

---

## Handy pandas quick-reference

```python
df.info()                  # dtypes + non-null counts
df.isna().sum()            # missing per column
df.duplicated().sum()      # duplicate rows
df.describe()              # numeric summary
df[col].value_counts()     # category counts
df[col].nunique()          # unique count
df.groupby(a)[b].mean()    # aggregation
pd.to_numeric(s, errors="coerce")  # text -> number (NaN where invalid)
df.sort_values(col)  |  df[df[col] > threshold]
```
