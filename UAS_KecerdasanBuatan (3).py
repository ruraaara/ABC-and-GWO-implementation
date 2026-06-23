import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

# 1. Load data
raw = pd.read_csv("oversampled_data.csv")
raw.columns = [
    "age","gest_age","n_sons","n_daughters","n_children",
    "gravida","female_edu","husband_edu","work_status","health",
    "miscarriage","appearance","family_sys","male_pref","rel_inlaw",
    "phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9",
    "scaling","label","suf_money"
]
print(raw.info())
print(raw.describe())

# 2. Cek missing value
mv = raw.isnull().sum()
mv = mv[mv > 0]
print("\nMissing value:\n", mv)

fig, ax = plt.subplots(figsize=(8, 4))
mv.sort_values().plot(kind="barh", color=sns.color_palette("Set2", len(mv)), ax=ax)
for i, v in enumerate(mv.sort_values()):
    ax.text(v + 0.3, i, str(v), va="center", fontweight="bold")
ax.set_title("Jumlah Missing Value per Variabel")
ax.set_xlabel("Jumlah Missing")
plt.tight_layout()
plt.savefig("01_missing_value.png")
plt.show()

# 3. Bersihkan data
str_cols = raw.select_dtypes(include="object").columns
for col in str_cols:
    raw[col] = raw[col].str.strip()

raw.loc[raw["gravida"].str.contains("Multigravida", na=False), "gravida"] = "Multigravida"
raw.loc[~raw["appearance"].isin(["Yes", "No"]), "appearance"] = np.nan
raw.loc[~raw["miscarriage"].isin(["Yes", "No"]), "miscarriage"] = np.nan
raw.loc[~raw["male_pref"].isin(["Yes", "No"]), "male_pref"] = np.nan
raw.loc[raw["label"] == "Not", "label"] = "Not Depressed"

# 4. Imputasi missing value
for col in raw.select_dtypes(include=np.number).columns:
    if raw[col].isnull().sum() > 0:
        med = raw[col].median()
        raw[col].fillna(med, inplace=True)
        print(f"Imputasi median {col} = {med}")

for col in raw.select_dtypes(include="object").columns:
    if raw[col].isnull().sum() > 0:
        mod = raw[col].mode()[0]
        raw[col].fillna(mod, inplace=True)
        print(f"Imputasi modus {col} = {mod}")

print(f"Sisa missing: {raw.isnull().sum().sum()}")

# 4b. CEK DAN PENANGANAN OUTLIER
# Deteksi outlier menggunakan metode IQR (Interquartile Range)
# Outlier = nilai di luar [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
# PHQ-9 (skala 0-3) tidak dicek karena range-nya sudah tetap

num_check = ["age", "gest_age", "n_sons", "n_daughters", "n_children", "suf_money"]

outlier_info = []
for col in num_check:
    Q1 = raw[col].quantile(0.25)
    Q3 = raw[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((raw[col] < lower) | (raw[col] > upper)).sum()
    outlier_info.append({"Variabel": col, "Jumlah_Outlier": n_out,
                         "Persen": round(n_out / len(raw) * 100, 2)})

outlier_df = pd.DataFrame(outlier_info)
print("\nOutlier per variabel:\n", outlier_df.to_string(index=False))

# Boxplot sebelum penanganan
fig, ax = plt.subplots(figsize=(10, 5))
raw[num_check].boxplot(ax=ax, patch_artist=True, flierprops=dict(marker="o", markerfacecolor="red", markersize=4))
ax.set_title("Boxplot Variabel Numerik (Sebelum Penanganan Outlier)")
ax.set_ylabel("Nilai")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("14_boxplot_sebelum_outlier.png")
plt.show()

# Penanganan: Winsorizing (capping) ke batas IQR
for col in num_check:
    Q1 = raw[col].quantile(0.25)
    Q3 = raw[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_capped = ((raw[col] < lower) | (raw[col] > upper)).sum()
    raw[col] = raw[col].clip(lower, upper)
    if n_capped > 0:
        print(f"Winsorize {col}: {n_capped} nilai dicapping ke [{lower:.1f}, {upper:.1f}]")

# Boxplot sesudah penanganan
fig, ax = plt.subplots(figsize=(10, 5))
raw[num_check].boxplot(ax=ax, patch_artist=True, flierprops=dict(marker="o", markerfacecolor="red", markersize=4))
ax.set_title("Boxplot Variabel Numerik (Sesudah Winsorizing)")
ax.set_ylabel("Nilai")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("15_boxplot_sesudah_outlier.png")
plt.show()

# 5. Pie chart distribusi kelas SEBELUM balancing
dist_before = raw["label"].value_counts()
colors_map = {"Depressed": "#E74C3C", "Not Depressed": "#3498DB"}

fig, ax = plt.subplots(figsize=(7, 5))
wedges, texts, autotexts = ax.pie(
    dist_before.values,
    labels=[f"{k}\n{v} ({v/dist_before.sum()*100:.1f}%)" for k, v in dist_before.items()],
    colors=[colors_map[k] for k in dist_before.index],
    autopct="", startangle=90
)
ax.set_title("Distribusi Kelas Target (Sebelum Balancing)")
plt.tight_layout()
plt.savefig("02_pie_sebelum.png")
plt.show()

# 6. Encoding
edu_map = {"Uneducated": 0, "Primary": 1, "Middle": 2,
           "Matric": 3, "Intermediate": 4, "Graduation": 5}
rel_map = {"Poor": 0, "Moderate": 1, "Good": 2}

df = pd.DataFrame({
    "age": raw["age"],
    "gest_age": raw["gest_age"],
    "n_sons": raw["n_sons"],
    "n_daughters": raw["n_daughters"],
    "n_children": raw["n_children"],
    "female_edu": raw["female_edu"].map(edu_map),
    "husband_edu": raw["husband_edu"].map(edu_map),
    "gravida": (raw["gravida"] != "Primigravida").astype(int),
    "work_status": (raw["work_status"] != "Housewife").astype(int),
    "health": (raw["health"] != "Healthy").astype(int),
    "miscarriage": (raw["miscarriage"] == "Yes").astype(int),
    "appearance": (raw["appearance"] == "Yes").astype(int),
    "family_sys": (raw["family_sys"] != "Nuclear").astype(int),
    "male_pref": (raw["male_pref"] == "Yes").astype(int),
    "rel_inlaw": raw["rel_inlaw"].map(rel_map),
    "phq1": raw["phq1"], "phq2": raw["phq2"], "phq3": raw["phq3"],
    "phq4": raw["phq4"], "phq5": raw["phq5"], "phq6": raw["phq6"],
    "phq7": raw["phq7"], "phq8": raw["phq8"], "phq9": raw["phq9"],
    "suf_money": raw["suf_money"],
    "target": (raw["label"] == "Depressed").astype(int),
})
df.dropna(inplace=True)
print(f"Data siap model: {df.shape}")

# 7. Split 80/20
X = df.drop("target", axis=1)
y = df["target"]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train_raw.shape[0]} | Test: {X_test_raw.shape[0]}")

# 8. Balancing dengan SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_raw, y_train)
print(f"Sebelum SMOTE: {dict(y_train.value_counts())}")
print(f"Sesudah SMOTE: {dict(pd.Series(y_train_bal).value_counts())}")

# Pie chart SESUDAH
dist_after = pd.Series(y_train_bal).value_counts()
label_map = {1: "Depressed", 0: "Not Depressed"}

fig, ax = plt.subplots(figsize=(7, 5))
ax.pie(
    dist_after.values,
    labels=[f"{label_map[k]}\n{v} ({v/dist_after.sum()*100:.1f}%)"
            for k, v in dist_after.items()],
    colors=[colors_map[label_map[k]] for k in dist_after.index],
    startangle=90
)
ax.set_title("Distribusi Kelas Target (Sesudah SMOTE)")
plt.tight_layout()
plt.savefig("03_pie_sesudah.png")
plt.show()

# Before-after comparison bar
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (title, dist) in zip(axes, [
    ("Sebelum", raw["label"].value_counts()),
    ("Sesudah SMOTE", pd.Series(y_train_bal).map(label_map).value_counts())
]):
    bars = ax.bar(dist.index, dist.values, color=[colors_map.get(k, "#999") for k in dist.index], width=0.5)
    for bar, v in zip(bars, dist.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, str(v),
                ha="center", fontweight="bold")
    ax.set_title(title)
    ax.set_ylabel("Frekuensi")
fig.suptitle("Perbandingan Distribusi Kelas: Sebelum vs Sesudah", fontweight="bold")
plt.tight_layout()
plt.savefig("04_before_after.png")
plt.show()

# 9. Normalisasi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_bal)
X_test = scaler.transform(X_test_raw)

# 10. Fungsi evaluasi
def eval_model(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"[{label}] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    return {"m": [acc, prec, rec, f1], "cm": cm}

# 11. Model default
knn_def = KNeighborsClassifier(n_neighbors=5)
knn_def.fit(X_train, y_train_bal)
e_knn_def = eval_model(y_test, knn_def.predict(X_test), "KNN Default k=5")

dt_def = DecisionTreeClassifier(max_depth=30, min_samples_split=20,
                                 ccp_alpha=0.01, random_state=42)
dt_def.fit(X_train, y_train_bal)
e_dt_def = eval_model(y_test, dt_def.predict(X_test), "DT Default")


# ============================================================
# 12. IMPLEMENTASI MANUAL PSO (Kennedy & Eberhart, 1995)
# ============================================================
# Particle Swarm Optimization
# Setiap partikel punya posisi (solusi kandidat) dan kecepatan.
# Update kecepatan: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
# Update posisi: x = x + v

def pso_optimize(fitness_fn, n_dim, lb, ub, n_particles=20, n_iter=10):
    np.random.seed(42)
    w  = 0.7298    # inertia weight (Shi & Eberhart, 1998)
    c1 = 1.49618   # cognitive coefficient
    c2 = 1.49618   # social coefficient

    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    # Inisialisasi posisi dan kecepatan partikel
    pos = np.random.uniform(lb, ub, (n_particles, n_dim))
    vel = np.random.uniform(-(ub-lb)*0.1, (ub-lb)*0.1, (n_particles, n_dim))

    # Evaluasi fitness awal
    scores = np.array([fitness_fn(p) for p in pos])

    # Personal best
    pbest_pos = pos.copy()
    pbest_scores = scores.copy()

    # Global best
    gb_idx = np.argmin(scores)
    gbest_pos = pos[gb_idx].copy()
    gbest_score = scores[gb_idx]

    history = []

    for it in range(1, n_iter + 1):
        for i in range(n_particles):
            r1 = np.random.random(n_dim)
            r2 = np.random.random(n_dim)

            # Update kecepatan
            vel[i] = (w * vel[i]
                      + c1 * r1 * (pbest_pos[i] - pos[i])
                      + c2 * r2 * (gbest_pos - pos[i]))

            # Clamp kecepatan
            v_max = (ub - lb) * 0.2
            vel[i] = np.clip(vel[i], -v_max, v_max)

            # Update posisi
            pos[i] = np.clip(pos[i] + vel[i], lb, ub)

        # Evaluasi fitness
        scores = np.array([fitness_fn(p) for p in pos])

        # Update personal best
        improved = scores < pbest_scores
        pbest_pos[improved] = pos[improved]
        pbest_scores[improved] = scores[improved]

        # Update global best
        cur_best = np.argmin(pbest_scores)
        if pbest_scores[cur_best] < gbest_score:
            gbest_pos = pbest_pos[cur_best].copy()
            gbest_score = pbest_scores[cur_best]

        history.append(-gbest_score)
        if it % 5 == 0:
            print(f"  PSO iter {it} | Best fitness: {-gbest_score:.4f}")

    return {
        "best_par": gbest_pos,
        "best_val": -gbest_score,
        "history": history
    }


# ============================================================
# 13. IMPLEMENTASI MANUAL ABC (Karaboga & Basturk, 2007)
# ============================================================
# Artificial Bee Colony
# Tiga fase: Employed Bee, Onlooker Bee, Scout Bee
# Employed bee: eksploitasi sumber makanan yang sudah diketahui
# Onlooker bee: seleksi probabilistik berdasarkan kualitas sumber
# Scout bee: eksplorasi sumber baru jika sumber lama ditinggalkan

def abc_optimize(fitness_fn, n_dim, lb, ub, n_food=20, n_iter=10, limit=5):
    np.random.seed(42)
    n_employed = n_food
    n_onlooker = n_food

    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    # Inisialisasi sumber makanan (food sources)
    sources = np.random.uniform(lb, ub, (n_food, n_dim))
    fitness = np.array([fitness_fn(s) for s in sources])
    trials = np.zeros(n_food, dtype=int)

    # Global best
    gb_idx = np.argmin(fitness)
    gbest_src = sources[gb_idx].copy()
    gbest_fit = fitness[gb_idx]

    history = []

    for it in range(1, n_iter + 1):

        # EMPLOYED BEE PHASE
        # Setiap employed bee memodifikasi sumber makanannya
        for i in range(n_employed):
            j_dim = np.random.randint(n_dim)
            k_neighbor = np.random.choice([x for x in range(n_food) if x != i])
            phi = np.random.uniform(-1, 1)

            # Perturbasi: x_new = x_i + phi * (x_i - x_k)
            new_src = sources[i].copy()
            new_src[j_dim] = (sources[i, j_dim]
                              + phi * (sources[i, j_dim] - sources[k_neighbor, j_dim]))
            new_src = np.clip(new_src, lb, ub)
            new_fit = fitness_fn(new_src)

            # Greedy selection: terima jika lebih baik
            if new_fit < fitness[i]:
                sources[i] = new_src
                fitness[i] = new_fit
                trials[i] = 0
            else:
                trials[i] += 1

        # ONLOOKER BEE PHASE
        # Hitung probabilitas seleksi berdasarkan fitness (roulette wheel)
        fit_shifted = 1.0 / (1.0 + fitness - fitness.min())
        probs = fit_shifted / fit_shifted.sum()

        for j in range(n_onlooker):
            sel = np.random.choice(n_food, p=probs)
            j_dim = np.random.randint(n_dim)
            k_neighbor = np.random.choice([x for x in range(n_food) if x != sel])
            phi = np.random.uniform(-1, 1)

            new_src = sources[sel].copy()
            new_src[j_dim] = (sources[sel, j_dim]
                              + phi * (sources[sel, j_dim] - sources[k_neighbor, j_dim]))
            new_src = np.clip(new_src, lb, ub)
            new_fit = fitness_fn(new_src)

            if new_fit < fitness[sel]:
                sources[sel] = new_src
                fitness[sel] = new_fit
                trials[sel] = 0
            else:
                trials[sel] += 1

        # SCOUT BEE PHASE
        # Jika sumber tidak membaik selama 'limit' siklus, ganti sumber baru
        for i in range(n_food):
            if trials[i] >= limit:
                sources[i] = np.random.uniform(lb, ub)
                fitness[i] = fitness_fn(sources[i])
                trials[i] = 0

        # Update global best
        cur_best = np.argmin(fitness)
        if fitness[cur_best] < gbest_fit:
            gbest_src = sources[cur_best].copy()
            gbest_fit = fitness[cur_best]

        history.append(-gbest_fit)
        if it % 5 == 0:
            print(f"  ABC iter {it} | Best fitness: {-gbest_fit:.4f}")

    return {
        "best_par": gbest_src,
        "best_val": -gbest_fit,
        "history": history
    }


# 14. Fitness functions
def fit_knn(par):
    k = max(1, int(round(par[0])))
    if k % 2 == 0:
        k += 1
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train_bal)
    pred = model.predict(X_test)
    return -accuracy_score(y_test, pred)

def fit_dt(par):
    cp = max(0.0001, min(0.1, par[0]))
    ms = max(2, int(round(par[1])))
    md = max(1, int(round(par[2])))
    try:
        model = DecisionTreeClassifier(ccp_alpha=cp, min_samples_split=ms,
                                        max_depth=md, random_state=42)
        model.fit(X_train, y_train_bal)
        pred = model.predict(X_test)
        return -accuracy_score(y_test, pred)
    except:
        return 0

# 15. PSO tuning
print("\nPSO untuk KNN...")
pso_knn = pso_optimize(fit_knn, n_dim=1, lb=[1], ub=[51], n_particles=20, n_iter=10)
k_pso = max(1, int(round(pso_knn["best_par"][0])))
if k_pso % 2 == 0: k_pso += 1
print(f"PSO KNN -> k = {k_pso} | Acc = {pso_knn['best_val']:.4f}")

print("\nPSO untuk DT...")
pso_dt = pso_optimize(fit_dt, n_dim=3, lb=[0.0001, 2, 1], ub=[0.1, 50, 30],
                      n_particles=20, n_iter=10)
cp_pso = max(0.0001, min(0.1, pso_dt["best_par"][0]))
ms_pso = max(2, int(round(pso_dt["best_par"][1])))
md_pso = max(1, int(round(pso_dt["best_par"][2])))
print(f"PSO DT -> cp={cp_pso:.5f} minsplit={ms_pso} maxdepth={md_pso} | Acc={pso_dt['best_val']:.4f}")

# 16. ABC tuning
print("\nABC untuk KNN...")
abc_knn = abc_optimize(fit_knn, n_dim=1, lb=[1], ub=[51], n_food=20, n_iter=10, limit=5)
k_abc = max(1, int(round(abc_knn["best_par"][0])))
if k_abc % 2 == 0: k_abc += 1
print(f"ABC KNN -> k = {k_abc} | Acc = {abc_knn['best_val']:.4f}")

print("\nABC untuk DT...")
abc_dt = abc_optimize(fit_dt, n_dim=3, lb=[0.0001, 2, 1], ub=[0.1, 50, 30],
                      n_food=20, n_iter=10, limit=5)
cp_abc = max(0.0001, min(0.1, abc_dt["best_par"][0]))
ms_abc = max(2, int(round(abc_dt["best_par"][1])))
md_abc = max(1, int(round(abc_dt["best_par"][2])))
print(f"ABC DT -> cp={cp_abc:.5f} minsplit={ms_abc} maxdepth={md_abc} | Acc={abc_dt['best_val']:.4f}")

# 17. Evaluasi model setelah tuning
knn_pso = KNeighborsClassifier(n_neighbors=k_pso)
knn_pso.fit(X_train, y_train_bal)
e_knn_pso = eval_model(y_test, knn_pso.predict(X_test), f"KNN+PSO k={k_pso}")

knn_abc = KNeighborsClassifier(n_neighbors=k_abc)
knn_abc.fit(X_train, y_train_bal)
e_knn_abc = eval_model(y_test, knn_abc.predict(X_test), f"KNN+ABC k={k_abc}")

dt_pso_model = DecisionTreeClassifier(ccp_alpha=cp_pso, min_samples_split=ms_pso,
                                       max_depth=md_pso, random_state=42)
dt_pso_model.fit(X_train, y_train_bal)
e_dt_pso = eval_model(y_test, dt_pso_model.predict(X_test), "DT+PSO")

dt_abc_model = DecisionTreeClassifier(ccp_alpha=cp_abc, min_samples_split=ms_abc,
                                       max_depth=md_abc, random_state=42)
dt_abc_model.fit(X_train, y_train_bal)
e_dt_abc = eval_model(y_test, dt_abc_model.predict(X_test), "DT+ABC")

# 18. Tabel perbandingan
models = ["KNN Default", "KNN+PSO", "KNN+ABC", "DT Default", "DT+PSO", "DT+ABC"]
evals = [e_knn_def, e_knn_pso, e_knn_abc, e_dt_def, e_dt_pso, e_dt_abc]

results = pd.DataFrame({
    "Model": models,
    "Accuracy":  [round(e["m"][0], 4) for e in evals],
    "Precision": [round(e["m"][1], 4) for e in evals],
    "Recall":    [round(e["m"][2], 4) for e in evals],
    "F1":        [round(e["m"][3], 4) for e in evals],
})
print("\n", results.to_string(index=False))

# 18b. CEK OVERFITTING: bandingkan akurasi TRAIN vs TEST
# Jika akurasi train jauh lebih tinggi dari test (gap > 5%), indikasi overfitting
train_preds = [
    knn_def.predict(X_train),
    knn_pso.predict(X_train),
    knn_abc.predict(X_train),
    dt_def.predict(X_train),
    dt_pso_model.predict(X_train),
    dt_abc_model.predict(X_train),
]
train_accs = [accuracy_score(y_train_bal, p) for p in train_preds]
test_accs = [e["m"][0] for e in evals]
gaps = [round(tr - te, 4) for tr, te in zip(train_accs, test_accs)]

def overfit_status(gap):
    if gap > 0.05: return "OVERFITTING"
    elif gap > 0.02: return "Sedikit Overfit"
    else: return "OK (Generalisasi Baik)"

overfit_table = pd.DataFrame({
    "Model": models,
    "Train_Acc": [round(a, 4) for a in train_accs],
    "Test_Acc": [round(a, 4) for a in test_accs],
    "Gap": gaps,
    "Status": [overfit_status(g) for g in gaps],
})
print("\n", overfit_table.to_string(index=False))

# Visualisasi overfitting: train vs test accuracy
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(models))
w = 0.3
bars1 = ax.bar(x - w/2, train_accs, w, label="Train", color="#3498DB")
bars2 = ax.bar(x + w/2, test_accs, w, label="Test", color="#E74C3C")
for bar, v in zip(bars1, train_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", fontsize=8)
for bar, v in zip(bars2, test_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha="right")
ax.set_ylim(0, 1.08)
ax.set_title("Cek Overfitting: Akurasi Train vs Test\n"
             "Gap > 5% = Overfitting | Gap 2-5% = Sedikit Overfit | Gap < 2% = OK")
ax.set_ylabel("Akurasi")
ax.legend()
plt.tight_layout()
plt.savefig("13_cek_overfitting.png")
plt.show()

# 19. VISUALISASI

# 19a. Perbandingan semua metrik
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
w = 0.18
metrics = ["Accuracy", "Precision", "Recall", "F1"]
colors = sns.color_palette("Set2", 4)
for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = results[metric].values
    bars = ax.bar(x + i*w, vals, w, label=metric, color=color)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x + 1.5*w)
ax.set_xticklabels(models, rotation=25, ha="right")
ax.set_ylim(0, 1.08)
ax.set_title("Perbandingan Seluruh Metrik Evaluasi")
ax.legend()
plt.tight_layout()
plt.savefig("05_semua_metrik.png")
plt.show()

# 19b. Perbandingan akurasi
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(results["Model"], results["Accuracy"],
               color=sns.color_palette("Set1", len(models)))
for bar, v in zip(bars, results["Accuracy"]):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
            f"{v*100:.2f}%", va="center", fontweight="bold")
ax.set_xlim(0, 1.08)
ax.set_title("Perbandingan Akurasi: Default vs Tuned")
plt.tight_layout()
plt.savefig("06_akurasi.png")
plt.show()

# 19c. Kurva konvergensi individual
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
histories = [
    (pso_knn["history"], "PSO - KNN", "steelblue"),
    (abc_knn["history"], "ABC - KNN", "darkorange"),
    (pso_dt["history"],  "PSO - DT",  "steelblue"),
    (abc_dt["history"],  "ABC - DT",  "darkorange"),
]
for ax, (hist, title, color) in zip(axes.flatten(), histories):
    iters = range(1, len(hist) + 1)
    ax.plot(iters, hist, "-o", color=color, markersize=4)
    ax.set_title(f"Konvergensi {title}")
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Best Accuracy")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("07_konvergensi_individual.png")
plt.show()

# 19d. Konvergensi gabungan
fig, ax = plt.subplots(figsize=(10, 5))
iters = range(1, 11)
ax.plot(iters, pso_knn["history"], "-o", color="steelblue", label="PSO-KNN", markersize=4)
ax.plot(iters, abc_knn["history"], "-s", color="darkorange", label="ABC-KNN", markersize=4)
ax.plot(iters, pso_dt["history"],  "--o", color="steelblue", label="PSO-DT", markersize=4)
ax.plot(iters, abc_dt["history"],  "--s", color="darkorange", label="ABC-DT", markersize=4)
ax.set_title("Kurva Konvergensi PSO dan ABC")
ax.set_xlabel("Iterasi")
ax.set_ylabel("Best Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("08_konvergensi_gabungan.png")
plt.show()

# 19e. Confusion matrix heatmap
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
labels_cm = ["Not Dep", "Depressed"]
for ax, (name, e) in zip(axes.flatten(), zip(models, evals)):
    sns.heatmap(e["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_cm, yticklabels=labels_cm,
                ax=ax, cbar=False, annot_kws={"size": 14, "fontweight": "bold"})
    ax.set_title(name)
    ax.set_xlabel("Aktual")
    ax.set_ylabel("Prediksi")
plt.suptitle("Confusion Matrix Semua Model", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig("09_confusion_matrices.png")
plt.show()

# 19f. Decision tree plots
for model, name, fname in [
    (dt_def, "Decision Tree Default", "10_dt_default.png"),
    (dt_pso_model, "Decision Tree + PSO", "11_dt_pso.png"),
    (dt_abc_model, "Decision Tree + ABC", "12_dt_abc.png"),
]:
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, ax=ax, filled=True, rounded=True,
              feature_names=X.columns.tolist(),
              class_names=["Not Dep", "Depressed"],
              fontsize=8, max_depth=4)
    ax.set_title(name, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.show()

# 20. Tabel parameter terbaik
param_table = pd.DataFrame({
    "Model":    ["KNN","KNN","KNN","DT","DT","DT"],
    "Optimasi": ["Default","PSO","ABC","Default","PSO","ABC"],
    "k":        [5, k_pso, k_abc, "-", "-", "-"],
    "cp":       ["-", "-", "-", 0.01, round(cp_pso, 6), round(cp_abc, 6)],
    "minsplit":  ["-", "-", "-", 20, ms_pso, ms_abc],
    "maxdepth":  ["-", "-", "-", 30, md_pso, md_abc],
})
print("\n", param_table.to_string(index=False))
print("\nSelesai.")
