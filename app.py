# =============================================================================
# APP STREAMLIT — BILINÉARISATION COURBE PUSHOVER
# Eurocode 8 + ASCE 41
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

# =============================================================================
# CONFIGURATION PAGE
# =============================================================================

st.set_page_config(
    page_title="Bilinéarisation Pushover",
    layout="wide",
    page_icon="📈"
)

# =============================================================================
# STYLE CSS
# =============================================================================

st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}

.metric-box {
    padding: 15px;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #e0e0e0;
    text-align: center;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
}

h1, h2, h3 {
    color: #1f3b5c;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# TITRE
# =============================================================================

st.title("📈 Bilinéarisation de la Courbe Pushover")
st.markdown("""
Application basée sur :

- Eurocode 8 — Annexe B
- ASCE 41-23 — §7.4.3.2.5
""")

# =============================================================================
# GUIDE THÉORIQUE
# =============================================================================

with st.expander("📖 Guide méthodologique & Rappels théoriques"):

    st.markdown("""
### Principe de la bilinéarisation

La bilinéarisation d'une courbe de capacité consiste à remplacer la courbe
pushover réelle par une courbe simplifiée équivalente.

Le principe fondamental utilisé dans l’Eurocode 8 repose sur :

### Équivalence énergétique

L’aire sous la courbe réelle doit être égale à l’aire sous la courbe
bilinéaire.

\[
E_m = \int_0^{d_m} F(d)\,dd
\]

La courbe idéale conserve donc la même capacité énergétique que la structure réelle.

### Eurocode 8 — Équation B.6

\[
d_y = 2 \times \left(d_m - \frac{E_m}{V_y}\right)
\]

avec :

- \(V_y\) : force de plastification
- \(d_y\) : déplacement de plastification
- \(d_m\) : déplacement ultime
- \(E_m\) : énergie réelle

### ASCE 41

La méthode ASCE utilise une procédure itérative basée sur :

- rigidité sécante à 60% de \(V_y\)
- convergence énergétique
- courbe tri-linéaire
""")

# =============================================================================
# FONCTIONS
# =============================================================================

def charger_donnees(fichier):

    try:
        if fichier.name.endswith(".csv"):
            df = pd.read_csv(fichier)

        else:
            df = pd.read_excel(fichier)

        # suppression lignes vides
        df = df.dropna(how="all")

        # recherche colonnes
        col_d = [c for c in df.columns if str(c).lower().startswith("d")]
        col_f = [c for c in df.columns if str(c).lower().startswith("f")]

        if not col_d or not col_f:
            raise ValueError("Colonnes d/f introuvables")

        # conversion numérique forcée
        d = pd.to_numeric(df[col_d[0]], errors="coerce")
        f = pd.to_numeric(df[col_f[0]], errors="coerce")

        # suppression NaN
        temp = pd.DataFrame({"d": d, "f": f}).dropna()

        d = temp["d"].values.astype(float)
        f = temp["f"].values.astype(float)

        if len(d) < 5:
            raise ValueError("Nombre de points insuffisant")

        # tri
        idx = np.argsort(d)
        d = d[idx]
        f = f[idx]

        # suppression doublons
        d, unique_idx = np.unique(d, return_index=True)
        f = f[unique_idx]

        return d, f

    except Exception as e:
        raise ValueError(
            "Erreur dans le format des données, veuillez vérifier vos colonnes."
        ) from e


# =============================================================================

def lissage_donnees(d, f):

    n = len(f)

    wl = max(5, int(n * 0.1))

    if wl % 2 == 0:
        wl += 1

    wl = min(wl, n - 1)

    if wl % 2 == 0:
        wl -= 1

    if wl < 5:
        wl = 5

    poly = 3

    if wl <= poly:
        poly = 2

    f_lisse = savgol_filter(f, wl, poly)

    f_lisse = np.maximum(f_lisse, 0)

    return f_lisse


# =============================================================================

def bilinearisation_ec8(d, f):

    idx_max = np.argmax(f)

    Vy = f[idx_max]
    dm = d[idx_max]

    mask = d <= dm

    Em = trapezoid(f[mask], d[mask])

    dy = 2 * (dm - Em / Vy)

    if dy <= 0:
        raise ValueError("dy <= 0")

    Ke = Vy / dy

    mu = dm / dy

    d_bilin = np.array([0, dy, d.max()])
    f_bilin = np.array([0, Vy, Vy])

    return {
        "Vy": Vy,
        "dy": dy,
        "dm": dm,
        "Ke": Ke,
        "mu": mu,
        "Em": Em,
        "d_bilin": d_bilin,
        "f_bilin": f_bilin
    }


# =============================================================================

def tracer(d, f_brut, f_lisse, res):

    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(
        d,
        f_brut,
        color="gray",
        linewidth=1.5,
        alpha=0.5,
        label="Courbe brute"
    )

    ax.plot(
        d,
        f_lisse,
        color="#1565C0",
        linewidth=3,
        label="Courbe lissée"
    )

    ax.plot(
        res["d_bilin"],
        res["f_bilin"],
        "--",
        color="#C62828",
        linewidth=3,
        label=(
            f"Bilinéaire EC8 | "
            f"Ke={res['Ke']:.2f} | "
            f"μ={res['mu']:.2f}"
        )
    )

    # Point de fluage
    ax.scatter(
        res["dy"],
        res["Vy"],
        s=180,
        color="red",
        edgecolors="black",
        zorder=10,
        label="Point de fluage"
    )

    # Point ultime
    ax.scatter(
        res["dm"],
        res["Vy"],
        s=220,
        marker="*",
        color="purple",
        edgecolors="black",
        zorder=10,
        label="Point ultime"
    )

    ax.set_title(
        "Bilinéarisation de la Courbe de Capacité",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel("Déplacement", fontsize=14)
    ax.set_ylabel("Effort tranchant", fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.3)

    ax.legend(fontsize=11)

    plt.tight_layout()

    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("⚙️ Paramètres")

uploaded_file = st.sidebar.file_uploader(
    "📂 Charger un fichier Excel ou CSV",
    type=["xlsx", "xls", "csv"]
)

# =============================================================================
# PIPELINE
# =============================================================================

if uploaded_file is not None:

    try:

        d, f = charger_donnees(uploaded_file)

        f_lisse = lissage_donnees(d, f)

        res = bilinearisation_ec8(d, f_lisse)

        # =========================================================================
        # INDICATEURS
        # =========================================================================

        st.subheader("📊 Indicateurs principaux")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("⚡ Effort max Vy", f"{res['Vy']:.2f}")
        c2.metric("📐 Déplacement dy", f"{res['dy']:.2f}")
        c3.metric("🧱 Raideur Ke", f"{res['Ke']:.2f}")
        c4.metric("🔁 Ductilité μ", f"{res['mu']:.2f}")

        # =========================================================================
        # GRAPHIQUE
        # =========================================================================

        st.subheader("📈 Graphique de Bilinéarisation")

        fig = tracer(d, f, f_lisse, res)

        st.pyplot(fig)

        # =========================================================================
        # TABLEAU
        # =========================================================================

        st.subheader("📋 Résultats numériques")

        df_res = pd.DataFrame({
            "Paramètre": [
                "Vy",
                "dy",
                "dm",
                "Ke",
                "μ"
            ],
            "Valeur": [
                res["Vy"],
                res["dy"],
                res["dm"],
                res["Ke"],
                res["mu"]
            ]
        })

        st.dataframe(df_res, use_container_width=True)

    except Exception:

        st.error(
            "Erreur dans le format des données, veuillez vérifier vos colonnes."
        )

else:

    st.info("Chargez un fichier pour lancer la bilinéarisation.")
