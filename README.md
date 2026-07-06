<img src="https://capsule-render.vercel.app/api?type=waving&height=250&color=0:000814,20:001D3D,40:003566,60:00509D,80:3A86FF,100:ADE8F4&text=Student%20Score%20Analysis&fontSize=54&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Statistics%20•%20Probability%20•%20Linear%20Algebra&descAlignY=66&descSize=20"/>

This project presents a **Statistics + Probability + Linear Algebra** analysis on a real-world student performance dataset containing **5,000 records**. The objective is to apply descriptive statistics, probability concepts, distribution fitting, and vector operations to understand student performance patterns and the relationship between study hours and passing.

The project combines mathematical theory with practical implementation in Python (Jupyter Notebook), covering the complete analytical workflow — from data loading and central tendency measures to probability, distribution visualization, and vector-based linear algebra.

---

## 🎯 Objective

To convert raw student marks and study-hour data into evidence-based insights using statistical and mathematical techniques — covering Mean/Median/Mode, Variance/Std Dev, Probability & Conditional Probability, Distribution shape (Skewness, Kurtosis, Q-Q Plot), and Vector operations (Dot Product, Norms, Angle).

---

## 🗂️ Project Files

| File | Description |
|------|-------------|
| 📓 `Student_score.ipynb` | Complete statistics, probability & linear algebra analysis notebook |
| 📊 `students_scores.csv` | Student performance dataset — 5,000 records, 7 columns |
| 📄 `Part_A_Theory.pdf` | Theoretical foundations & formulas (Part A) |
| 📘 `README.md` | Project documentation (this file) |

---

## 🛠️ Tools & Libraries

<div>
  
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-2C2D72?style=for-the-badge&logo=plotly&logoColor=white"/>

</div>

---

# 🎬 Project Demo

[![Watch Demo](https://img.shields.io/badge/▶️%20Watch%20Demo-Google%20Drive-blue?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/15rVC3j1UWCeM1BUH4yeZeG4bfqLHGLoZ/view?usp=sharing)

📹 Click the badge above to watch the complete project demonstration.

---

## 📂 Dataset Overview

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

df = pd.read_csv("students_scores.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())
```

**Columns:** `Student_ID`, `Age`, `Math_Score`, `Science_Score`, `English_Score`, `Hours_Studied`, `Pass_Fail`

✅ **Shape:** 5,000 students × 7 columns — no missing values, clean and ready-to-use.

---

## 📊 Step 1 — Measures of Central Tendency & Dispersion

### 🧮 Mean, Median, Mode of Math_Score
```python
mean_math = df["Math_Score"].mean()
median_math = df["Math_Score"].median()
mode_math = df["Math_Score"].mode()[0]
```
**Result:** Mean = 67.13, Median = 67.0, Mode = 40

**💡 Insight:** Mean (67.13) and Median (67.0) are almost identical, so Math_Score is fairly symmetric overall. However, the Mode (40) sits well below both — a smaller cluster of students is scoring on the lower end, even though the majority of the batch performs moderately well without any extreme skew.

### 📏 Range, Variance, Std Deviation of Science_Score
```python
range_science = df["Science_Score"].max() - df["Science_Score"].min()
variance_science = df["Science_Score"].var()
std_science = df["Science_Score"].std()
```
**Result:** Range = 65, Variance ≈ 361, Std Dev ≈ 19

**💡 Insight:** A range of 65 shows a wide gap between the highest and lowest Science scores. With a standard deviation of ~19 and variance of ~361, Science performance is highly spread out — some students score very well, others quite poorly, unlike the more concentrated Math scores.

---

## 🎲 Step 2 — Probability Basics

### ✅ Probability of Passing
```python
prob_pass = (df["Pass_Fail"] == 1).mean()
```
**Result:** P(Pass) = **94.54%**

**💡 Insight:** With a 94.54% pass rate, only ~5.46% of students fail — this is a high-performing batch overall, though that small failing group likely needs targeted support.

### 📋 Contingency Table — Pass_Fail vs Hours_Studied > 5
```python
df["Hours_Studied_GT5"] = df["Hours_Studied"] > 5
table = pd.crosstab(df["Pass_Fail"], df["Hours_Studied_GT5"])
```

**💡 Insight:** Students studying more than 5 hours show a much higher pass count (1,770) versus fails (98). Students studying ≤5 hours still pass at a high rate, but proportionally show more failures — a clear pattern that more study hours reduce the chance of failing.

### 🎯 Conditional Probability — P(Pass | Hours_Studied > 5)
```python
students_gt5 = df[df["Hours_Studied"] > 5]
conditional_probability = (students_gt5["Pass_Fail"] == 1).mean()
```
**Result:** P(Pass | Hours > 5) = **94.75%**

**💡 Insight:** This is slightly higher than the overall pass probability (94.54%), confirming that studying more than 5 hours modestly improves the odds of passing — students crossing that threshold have roughly a 95% chance of passing.

---

## 📈 Step 3 — Distribution & Visualization

### 🔔 Histogram + Normal Curve — Math_Score
```python
plt.hist(df["Math_Score"], bins=15, density=True, edgecolor="black")
x = np.linspace(df["Math_Score"].min(), df["Math_Score"].max(), 100)
y = stats.norm.pdf(x, df["Math_Score"].mean(), df["Math_Score"].std())
plt.plot(x, y, linewidth=2)
```

**💡 Insight:** The histogram closely follows a bell-shaped curve, and the overlaid Normal Distribution curve matches it well — Math_Score is approximately normally distributed with no major skew.

### ⚖️ Skewness & Kurtosis — Science_Score
```python
skewness = df["Science_Score"].skew()
kurtosis = df["Science_Score"].kurt()
```
**Result:** Skewness ≈ 0 (−0.0000158), Kurtosis ≈ −1.20

**💡 Insight:** Skewness near zero means Science_Score is almost perfectly symmetric. The negative kurtosis (−1.20) indicates a platykurtic distribution — flatter than Normal with thinner tails, meaning scores are spread evenly without a heavy cluster of outliers.

### 📐 Q-Q Plot — English_Score
```python
stats.probplot(df["English_Score"], dist="norm", plot=plt)
```

**💡 Insight:** Most English_Score points align closely with the theoretical straight line, confirming scores roughly follow a Normal Distribution — only the extreme tails show minor deviation, hinting at a few outliers.

---

## ➗ Step 4 — Linear Algebra Mini Task

### 🧮 Vector Representation (first 5 students)
```python
math_vector = df["Math_Score"].head(5).values
science_vector = df["Science_Score"].head(5).values
```
**💡 Insight:** The first 5 students' Math and Science scores are represented as numeric vectors — the base for the dot product, norm, and angle calculations below.

### ✖️ Dot Product
```python
dot_product = np.dot(math_vector, science_vector)
```
**Result:** Dot Product = **22,519**

**💡 Insight:** This combined value reflects the overall magnitude and alignment between the Math and Science score vectors for these 5 students.

### 📏 Norms of the Math Vector
```python
norm1 = np.linalg.norm(math_vector, 1)
norm2 = np.linalg.norm(math_vector, 2)
```
**Result:** Norm-1 (Manhattan) = 303.0, Norm-2 (Euclidean) = 141.71

**💡 Insight:** Norm-1 is simply the sum of absolute Math scores, while Norm-2 gives the vector's true straight-line magnitude — two different mathematical views of the same vector's "size."

### 📐 Angle Between Vectors
```python
cos_theta = dot_product / (norm_math * norm_science)
angle = np.degrees(np.arccos(cos_theta))
```
**Result:** Angle ≈ **21.25°**

**💡 Insight:** A small angle of 21.25° means the Math and Science score vectors point in a similar direction — students performing well in Math also tend to perform well in Science. It isn't a perfect 0° (fully proportional), but the correlation between the two subjects is clearly strong.

---

## 🏁 Final Conclusion

- 🧠 **Overall Pass Rate:** 94.54% of students pass — a strong-performing batch overall.
- ⏰ **Study Hours Impact:** Students studying more than 5 hours have about a **95% chance of passing**, slightly better than those studying less — a positive but moderate effect.
- 📊 **Math_Score:** Mean and Median are close, so the data is symmetric, but a lower Mode reveals a small low-scoring subgroup.
- 📈 **Science_Score:** Near-zero skewness and negative kurtosis show scores are symmetric and evenly spread, without heavy outliers.
- 🔔 **English_Score:** The Q-Q plot confirms an approximately Normal Distribution, with only minor tail deviation.
- ➗ **Linear Algebra:** A 21.25° angle between the Math and Science vectors shows strong correlation between the two subjects' performance.

✅ **In short:** This dataset represents a high-performing student group where study hours positively influence the probability of passing, and Math and Science performance are strongly correlated.

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scipy matplotlib statsmodels

# 2. Launch Jupyter
jupyter notebook

# 3. Open and run
Student_score.ipynb
```

---

## ✅ Project Checklist

- [x] Data Loading & Cleaning Check
- [x] Mean, Median, Mode (Math_Score)
- [x] Range, Variance, Standard Deviation (Science_Score)
- [x] Probability of Passing
- [x] Contingency Table (Pass_Fail vs Hours_Studied)
- [x] Conditional Probability
- [x] Histogram + Normal Curve (Math_Score)
- [x] Skewness & Kurtosis (Science_Score)
- [x] Q-Q Plot (English_Score)
- [x] Vector Representation
- [x] Dot Product
- [x] Vector Norms (L1, L2)
- [x] Angle Between Vectors
- [x] Dataset Included (5,000 records)
- [x] Jupyter Notebook Included
- [x] Theory PDF Included

---

## 👩‍💻 Author

📍 Ahmedabad, Gujarat, India

*"Data-Driven Decisions · Statistical Thinking · Evidence-Based Conclusions"*

⭐ If you found this project helpful, give it a star and feel free to fork!
