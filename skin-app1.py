"""
GlowMatch - Beauty & Skin Care Product Recommender (Streamlit)

Content-based recommender built from the Beauty_Skin_Care notebook
(dataset: "Beauty and Cosmetics.csv"). No regression / rating prediction:
products are ranked by how closely their attributes match the shopper's
profile (cosine similarity on one-hot features), blended lightly with the
product's observed rating and popularity.

Run:   streamlit run skin_recommender_app.py
Needs: streamlit, pandas, numpy, scikit-learn
Place "Beauty and Cosmetics.csv" in the same folder as this file.
"""

import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------------
# Page config & styling (dark-green sidebar, serif type, soft green gradient)
# ----------------------------------------------------------------------------
st.set_page_config(page_title="GlowMatch", page_icon="🧴", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif:wght@400;600;700&display=swap');

:root {
    --tone-1: #f9e3d3; --tone-2: #f1c9a5; --tone-3: #d9a47a;
    --tone-4: #b07a54; --tone-5: #8a5a3c; --tone-6: #5e3b27;
    --mocha: #4a3228; --cream: #fff7f0; --blush: #e8b9a6; --gold: #e0a96d;
}
html, body, [class*="css"], .stMarkdown, h1, h2, h3, h4, label, p, span, div {
    font-family: 'Noto Serif', Georgia, serif;
}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] {
    background: linear-gradient(160deg, #fdf3ea 0%, #f6dfcc 100%);
}
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: var(--mocha); }
.block-container { padding-top: 3rem; max-width: 980px; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--mocha); }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown { color: #fbeee3; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: var(--cream); border-radius: 10px; border: none;
}
[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #3a2a22 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { border-color: var(--gold) !important; }
.brand { font-size: 1.6rem; font-weight: 700; color: #fff; margin: .2rem 0 .4rem 0; }
.brand small { display:block; font-size:.8rem; font-weight:400; opacity:.75; }
.swatches { display:flex; gap:6px; margin: 4px 0 14px 0; }
.swatches span { width:22px; height:22px; border-radius:50%; border:2px solid rgba(255,255,255,.55); }

/* Buttons */
div.stButton > button {
    background: var(--gold); color: var(--mocha); border: none; border-radius: 10px;
    font-weight: 700; padding: .6rem 1rem; width: 100%;
}
div.stButton > button:hover { background: #f0be85; color: #2b1d16; }

/* Product cards */
.card {
    display: flex; gap: 18px; align-items: center;
    background: #fffaf5; border-radius: 16px; padding: 16px 20px; margin: 12px 0;
    box-shadow: 0 2px 12px rgba(94,59,39,.12); border-left: 6px solid var(--tone-4);
}
.card .body { flex: 1; min-width: 0; }
.thumb {
    flex: 0 0 92px; height: 92px; border-radius: 14px; overflow: hidden;
    background: linear-gradient(145deg, var(--tone-1), var(--tone-2));
    display:flex; align-items:center; justify-content:center;
}
.thumb svg, .thumb img { width: 76px; height: 76px; object-fit: cover; }
.card h4 { margin: 0 0 2px 0; color: var(--mocha); font-size: 1.05rem; }
.card .sub { color: #8b6f60; font-size: .85rem; margin-bottom: 8px; }
.chip {
    display: inline-block; background: var(--tone-1); color: var(--mocha); border-radius: 999px;
    padding: 2px 10px; margin: 2px 4px 2px 0; font-size: .78rem;
}
.chip.hit { background: var(--gold); font-weight: 700; }
.metrics { display:flex; flex-wrap: wrap; gap: 22px; margin-top: 8px; font-size: .9rem; color: var(--mocha); }
.bar { height: 7px; background: #f0dccb; border-radius: 99px; margin-top: 10px; }
.bar > div { height: 7px; background: linear-gradient(90deg, var(--tone-3), var(--tone-5)); border-radius: 99px; }

/* Category tiles */
.tiles { display:flex; flex-wrap:wrap; gap:14px; margin: 6px 0 4px 0; }
.tile {
    width: 104px; text-align:center; background:#fffaf5; border-radius:16px; padding:10px 6px;
    box-shadow: 0 2px 10px rgba(94,59,39,.10);
}
.tile .thumb { margin: 0 auto 6px auto; flex: none; width: 76px; height: 76px; }
.tile .thumb svg, .tile .thumb img { width: 60px; height: 60px; }
.tile b { font-size: .8rem; color: var(--mocha); display:block; }
.tile small { color:#8b6f60; font-size:.72rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DATA_FILE = "Beauty and Cosmetics.csv"
SEARCH_DIRS = [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent / "data"]

REQUIRED = [
    "Product_Name", "Brand", "Category", "Usage_Frequency", "Price_USD", "Rating",
    "Number_of_Reviews", "Product_Size", "Skin_Type", "Gender_Target",
    "Packaging_Type", "Main_Ingredient", "Cruelty_Free", "Country_of_Origin",
]

# Attributes compared between the shopper's profile and each product,
# with how much each one counts.
MATCH_WEIGHTS = {
    "Skin_Type": 3.0,
    "Category": 2.5,
    "Main_Ingredient": 2.0,
    "Usage_Frequency": 1.0,
    "Gender_Target": 1.0,
    "Brand": 1.0,
}
MATCH_COLS = list(MATCH_WEIGHTS)
PRETTY = {
    "Skin_Type": "Skin type", "Category": "Category", "Main_Ingredient": "Ingredient",
    "Usage_Frequency": "Usage", "Gender_Target": "Target", "Brand": "Brand",
}

# Final ranking = profile match + a light nudge for rating and popularity
W_MATCH, W_RATING, W_POP = 0.70, 0.20, 0.10

SKINCARE_HINTS = ["cleanser", "toner", "serum", "moisturi", "sunscreen", "spf", "mask",
                  "cream", "face wash", "essence", "eye", "oil", "lotion"]

SKIN_TIPS = {
    "oily": "Lightweight, oil-free or gel textures; look for ingredients that balance shine without stripping skin.",
    "dry": "Rich, hydrating textures; layer a humectant serum under a cream that locks moisture in.",
    "combination": "Balance is key: lighter products on the T-zone, richer ones on dry cheeks.",
    "sensitive": "Keep routines short, fragrance-free and gentle; patch-test anything new.",
    "normal": "You have flexibility. Focus on sun protection, hydration and consistency.",
}

NAV = ["🏠 Home", "🧴 Routine Builder", "🔍 Similar Products", "📊 Insights", "📖 Skin Guide", "❓ Help"]


# ----------------------------------------------------------------------------
# Product pictures: built-in skin-tone SVG illustrations per product type.
# To use real photos instead, drop images named after the category into an
# "images" folder next to this file, e.g. images/lip_gloss.png, images/toner.jpg
# ----------------------------------------------------------------------------
_C1, _C2, _C3, _C4, _C5 = "#f6d5bd", "#e8b48f", "#c98a5e", "#8a5a3c", "#4a3228"
_CREAM, _GOLD, _ROSE = "#fff7f0", "#d9a441", "#c9706b"


def _svg(body):
    return f'<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


ICONS = {
    "lipgloss": _svg(
        f'<rect x="25" y="28" width="14" height="28" rx="4" fill="{_ROSE}"/>'
        f'<rect x="27" y="6" width="10" height="22" rx="3" fill="{_C5}"/>'
        f'<rect x="28" y="32" width="3" height="18" rx="1.5" fill="#fff" opacity=".45"/>'
        f'<path d="M45 14c3 4 3 7 0 9-3-2-3-5 0-9z" fill="{_ROSE}" opacity=".8"/>'),
    "lipstick": _svg(
        f'<rect x="21" y="38" width="22" height="18" rx="2" fill="{_C5}"/>'
        f'<rect x="21" y="31" width="22" height="8" rx="1" fill="{_GOLD}"/>'
        f'<polygon points="24,31 24,18 40,9 40,31" fill="{_ROSE}"/>'
        f'<polygon points="24,18 40,9 40,13 24,22" fill="#fff" opacity=".25"/>'),
    "toner": _svg(
        f'<rect x="26" y="5" width="12" height="9" rx="2" fill="{_C5}"/>'
        f'<rect x="28" y="14" width="8" height="6" fill="{_C4}"/>'
        f'<rect x="17" y="20" width="30" height="39" rx="9" fill="{_C2}"/>'
        f'<rect x="22" y="32" width="20" height="16" rx="3" fill="{_CREAM}"/>'
        f'<rect x="25" y="37" width="14" height="2.5" rx="1" fill="{_C3}"/>'
        f'<rect x="25" y="42" width="9" height="2.5" rx="1" fill="{_C3}"/>'),
    "serum": _svg(
        f'<rect x="27" y="3" width="10" height="14" rx="5" fill="{_C5}"/>'
        f'<rect x="23" y="16" width="18" height="6" rx="1" fill="{_GOLD}"/>'
        f'<rect x="19" y="22" width="26" height="37" rx="7" fill="{_C3}"/>'
        f'<rect x="19" y="40" width="26" height="19" rx="7" fill="{_C4}" opacity=".35"/>'
        f'<rect x="23" y="29" width="18" height="12" rx="3" fill="{_CREAM}"/>'
        f'<rect x="25" y="33" width="14" height="2.5" rx="1" fill="{_C3}"/>'),
    "jar": _svg(
        f'<rect x="11" y="21" width="42" height="13" rx="3" fill="{_C5}"/>'
        f'<rect x="13" y="34" width="38" height="23" rx="7" fill="{_C2}"/>'
        f'<rect x="20" y="40" width="24" height="11" rx="3" fill="{_CREAM}"/>'
        f'<rect x="23" y="44" width="18" height="2.5" rx="1" fill="{_C3}"/>'),
    "pump": _svg(
        f'<rect x="31" y="3" width="17" height="6" rx="3" fill="{_C5}"/>'
        f'<rect x="28" y="8" width="7" height="11" fill="{_C5}"/>'
        f'<rect x="23" y="18" width="17" height="5" fill="{_GOLD}"/>'
        f'<rect x="19" y="23" width="25" height="35" rx="7" fill="{_C2}"/>'
        f'<rect x="23" y="32" width="17" height="14" rx="3" fill="{_CREAM}"/>'
        f'<circle cx="31.500" cy="39" r="3.500" fill="{_C3}"/>'),
    "sun": _svg(
        f'<rect x="21" y="7" width="22" height="14" rx="3" fill="{_C5}"/>'
        f'<path d="M17 21h30l-2 36H19z" fill="{_C3}"/>'
        f'<circle cx="32" cy="38" r="6" fill="#f2c230"/>'
        f'<g stroke="#f2c230" stroke-width="2" stroke-linecap="round">'
        f'<path d="M32 28v-2M32 50v-2M22 38h-2M44 38h-2M25 31l-1.500-1.500M40 45l1.500 1.500M25 45l-1.500 1.500M40 31l1.500-1.500"/></g>'),
    "foundation": _svg(
        f'<rect x="23" y="5" width="18" height="20" rx="3" fill="{_C5}"/>'
        f'<rect x="19" y="25" width="26" height="32" rx="6" fill="{_C2}"/>'
        f'<rect x="19" y="41" width="26" height="16" rx="6" fill="{_C3}"/>'
        f'<rect x="23" y="30" width="18" height="8" rx="2" fill="{_CREAM}"/>'),
    "mask": _svg(
        f'<ellipse cx="32" cy="32" rx="19" ry="25" fill="{_CREAM}" stroke="{_C3}" stroke-width="2"/>'
        f'<ellipse cx="24" cy="26" rx="5" ry="3" fill="{_C2}"/>'
        f'<ellipse cx="40" cy="26" rx="5" ry="3" fill="{_C2}"/>'
        f'<rect x="26" y="42" width="12" height="4" rx="2" fill="{_C2}"/>'
        f'<path d="M32 28v9" stroke="{_C3}" stroke-width="1.500" stroke-linecap="round"/>'),
    "compact": _svg(
        f'<circle cx="32" cy="34" r="25" fill="{_C5}"/>'
        f'<circle cx="32" cy="34" r="19" fill="{_C2}"/>'
        f'<circle cx="32" cy="34" r="13" fill="{_ROSE}"/>'
        f'<circle cx="27" cy="29" r="4" fill="#fff" opacity=".3"/>'
        f'<rect x="28" y="6" width="8" height="5" rx="2" fill="{_GOLD}"/>'),
    "mascara": _svg(
        f'<rect x="26" y="28" width="12" height="29" rx="3" fill="{_C5}"/>'
        f'<rect x="29" y="4" width="6" height="26" rx="2" fill="{_C3}"/>'
        f'<g stroke="{_C5}" stroke-width="2" stroke-linecap="round"><path d="M27 8h-4M27 13h-4M27 18h-4M37 8h4M37 13h4M37 18h4"/></g>'
        f'<rect x="24" y="28" width="16" height="4" rx="1" fill="{_GOLD}"/>'),
    "perfume": _svg(
        f'<rect x="26" y="8" width="12" height="10" rx="2" fill="{_GOLD}"/>'
        f'<rect x="29" y="18" width="6" height="7" fill="{_C5}"/>'
        f'<rect x="15" y="25" width="34" height="32" rx="6" fill="{_C2}" opacity=".9"/>'
        f'<rect x="22" y="33" width="20" height="14" rx="3" fill="{_CREAM}"/>'
        f'<circle cx="32" cy="40" r="3" fill="{_C3}"/>'),
}

ICON_KEYWORDS = [
    ("lip gloss", "lipgloss"), ("gloss", "lipgloss"), ("lipstick", "lipstick"), ("lip", "lipgloss"),
    ("toner", "toner"), ("essence", "toner"), ("mist", "toner"),
    ("serum", "serum"), ("oil", "serum"), ("ampoule", "serum"),
    ("moistur", "jar"), ("cream", "jar"), ("balm", "jar"), ("butter", "jar"),
    ("cleans", "pump"), ("wash", "pump"), ("lotion", "pump"), ("scrub", "pump"),
    ("sunscreen", "sun"), ("spf", "sun"), ("sun", "sun"),
    ("foundation", "foundation"), ("concealer", "foundation"), ("primer", "foundation"), ("bb", "foundation"),
    ("mask", "mask"),
    ("blush", "compact"), ("powder", "compact"), ("highlight", "compact"), ("bronzer", "compact"),
    ("shadow", "compact"), ("palette", "compact"),
    ("mascara", "mascara"), ("liner", "mascara"), ("brow", "mascara"), ("lash", "mascara"),
    ("perfume", "perfume"), ("fragrance", "perfume"), ("cologne", "perfume"),
]


@st.cache_data(show_spinner=False)
def thumb_html(category: str) -> str:
    """Photo from ./images if available, otherwise the built-in illustration."""
    slug = category.lower().strip().replace(" ", "_")
    for d in SEARCH_DIRS:
        for ext in ("png", "jpg", "jpeg", "webp"):
            p = d / "images" / f"{slug}.{ext}"
            if p.exists():
                mime = "jpeg" if ext == "jpg" else ext
                b64 = base64.b64encode(p.read_bytes()).decode()
                return f'<img src="data:image/{mime};base64,{b64}" alt="{html.escape(category)}"/>'
    key = category.lower()
    icon = next((v for k, v in ICON_KEYWORDS if k in key), "toner")
    return ICONS[icon]


# ----------------------------------------------------------------------------
# Data + recommender engine
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading products…")
def load_data():
    path = next((d / DATA_FILE for d in SEARCH_DIRS if (d / DATA_FILE).exists()), None)
    if path is None:
        return None, "missing"
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, "Missing columns: " + ", ".join(missing)

    df = df.drop_duplicates().dropna(subset=["Product_Name", "Brand", "Category", "Skin_Type"]).copy()
    for c in ["Product_Name", "Brand", "Category", "Usage_Frequency", "Skin_Type",
              "Gender_Target", "Main_Ingredient", "Packaging_Type", "Country_of_Origin", "Product_Size"]:
        df[c] = df[c].astype(str).str.strip()
    for c in ["Price_USD", "Rating", "Number_of_Reviews"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Price_USD", "Rating"]).copy()
    df["Number_of_Reviews"] = df["Number_of_Reviews"].fillna(0)
    df["Cruelty_Free"] = (
        df["Cruelty_Free"].astype(str).str.strip().str.lower().isin(["true", "yes", "1", "y", "t"])
    )
    return df.reset_index(drop=True), None


class Recommender:
    """Content-based matcher: one-hot product attributes, weighted cosine similarity."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.X = self.enc.fit_transform(df[MATCH_COLS])
        self.col_group = np.concatenate(
            [[col] * len(cats) for col, cats in zip(MATCH_COLS, self.enc.categories_)]
        )
        self.w = np.array([MATCH_WEIGHTS[g] for g in self.col_group])
        self.rating_n = ((df["Rating"] - 1) / 4).clip(0, 1).to_numpy()
        logrev = np.log1p(df["Number_of_Reviews"].to_numpy())
        self.pop_n = logrev / logrev.max() if logrev.max() > 0 else logrev
        p = df["Price_USD"].to_numpy()
        self.price_n = (p - p.min()) / (p.max() - p.min() + 1e-9)

    def profile_match(self, prefs: dict, idx: np.ndarray):
        """Cosine similarity between the shopper profile and candidate products,
        computed only over the attributes the shopper actually chose."""
        chosen = [c for c in MATCH_COLS if prefs.get(c, "Any") != "Any"]
        if not chosen:
            return np.zeros(len(idx)), chosen
        row = {c: self.enc.categories_[i][0] for i, c in enumerate(MATCH_COLS)}
        row.update({c: prefs[c] for c in chosen})
        u = self.enc.transform(pd.DataFrame([row]))[0]
        mask = np.isin(self.col_group, chosen).astype(float)
        u = u * self.w * mask
        Xc = self.X[idx] * self.w * mask
        denom = np.linalg.norm(Xc, axis=1) * np.linalg.norm(u)
        sims = np.divide(Xc @ u, denom, out=np.zeros(len(idx)), where=denom > 0)
        return sims, chosen

    def recommend(self, prefs: dict, top_n: int = 6, categories=None) -> pd.DataFrame:
        df = self.df
        mask = df["Price_USD"].between(*prefs["price"]) & (df["Rating"] >= prefs["min_rating"])
        if prefs["cruelty_free"]:
            mask &= df["Cruelty_Free"]
        if prefs.get("Skin_Type", "Any") != "Any":
            # keep products made for this skin type, plus "all skin types" products
            mask &= (df["Skin_Type"] == prefs["Skin_Type"]) | df["Skin_Type"].str.lower().str.contains("all")
        if categories is not None:
            mask &= df["Category"].isin(categories)
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) == 0:
            return df.iloc[0:0]

        match, _ = self.profile_match(prefs, idx)
        score = W_MATCH * match + W_RATING * self.rating_n[idx] + W_POP * self.pop_n[idx]
        out = df.iloc[idx].copy()
        out["Match"] = match
        out["Score"] = score
        out = out.sort_values(["Score", "Rating"], ascending=False)
        out = out.drop_duplicates(["Product_Name", "Brand"])
        return out.head(top_n)

    def similar_to(self, i: int, top_n: int = 6) -> pd.DataFrame:
        base = np.hstack([self.X * self.w, self.price_n[:, None] * 1.5, self.rating_n[:, None] * 1.0])
        v = base[i]
        sims = base @ v / (np.linalg.norm(base, axis=1) * np.linalg.norm(v) + 1e-9)
        out = self.df.copy()
        out["Match"] = sims
        out["Score"] = sims
        ref = self.df.iloc[i]
        out = out[~((out["Product_Name"] == ref["Product_Name"]) & (out["Brand"] == ref["Brand"]))]
        out = out.sort_values(["Match", "Rating"], ascending=False)
        return out.drop_duplicates(["Product_Name", "Brand"]).head(top_n)


@st.cache_resource(show_spinner="Building recommender…")
def get_recommender(_df):
    return Recommender(_df)


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def esc(x):
    return html.escape(str(x))


def render_card(row, prefs=None):
    hits = set()
    if prefs:
        for col in MATCH_COLS:
            if prefs.get(col, "Any") != "Any" and row[col] == prefs[col]:
                hits.add(col)
    chips = "".join(
        f'<span class="chip {"hit" if c in hits else ""}">{PRETTY[c]}: {esc(row[c])}</span>'
        for c in ["Skin_Type", "Category", "Main_Ingredient", "Usage_Frequency", "Gender_Target"]
    )
    if row["Cruelty_Free"]:
        chips += '<span class="chip">🐰 Cruelty-free</span>'
    pct = int(round(row["Match"] * 100))
    st.markdown(
        f"""
<div class="card">
  <div class="thumb">{thumb_html(row['Category'])}</div>
  <div class="body">
  <h4>{esc(row['Product_Name'])}</h4>
  <div class="sub">{esc(row['Brand'])} · {esc(row['Product_Size'])} · {esc(row['Packaging_Type'])} · {esc(row['Country_of_Origin'])}</div>
  {chips}
  <div class="metrics">
    <span>💲 <b>{row['Price_USD']:.2f}</b></span>
    <span>⭐ <b>{row['Rating']:.1f}</b></span>
    <span>💬 <b>{int(row['Number_of_Reviews']):,}</b> reviews</span>
    <span>🎯 <b>{pct}%</b> match</span>
  </div>
  <div class="bar"><div style="width:{pct}%"></div></div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def results_table(df):
    cols = ["Product_Name", "Brand", "Category", "Skin_Type", "Main_Ingredient",
            "Price_USD", "Rating", "Number_of_Reviews", "Match"]
    show = df[cols].copy()
    show["Match"] = (show["Match"] * 100).round(0).astype(int).astype(str) + "%"
    with st.expander("📄 View as table / download"):
        st.dataframe(show, hide_index=True)
        st.download_button("⬇️ Download CSV", show.to_csv(index=False).encode(),
                           "recommendations.csv", "text/csv")


def sidebar_preferences(df, show_category=True):
    st.sidebar.markdown("---")
    prefs = {}
    if show_category:
        prefs["Category"] = st.sidebar.selectbox(
            "🎯 What are you shopping for?", ["Any"] + sorted(df["Category"].unique()))
    prefs["Skin_Type"] = st.sidebar.selectbox(
        "🧴 Skin type", ["Any"] + sorted(df["Skin_Type"].unique()))
    prefs["Gender_Target"] = st.sidebar.selectbox(
        "🙋 Product target", ["Any"] + sorted(df["Gender_Target"].unique()))
    prefs["Usage_Frequency"] = st.sidebar.selectbox(
        "🔁 Usage frequency", ["Any"] + sorted(df["Usage_Frequency"].unique()))
    prefs["Main_Ingredient"] = st.sidebar.selectbox(
        "🧪 Preferred ingredient", ["Any"] + sorted(df["Main_Ingredient"].unique()))
    prefs["Brand"] = st.sidebar.selectbox(
        "🏷️ Favourite brand", ["Any"] + sorted(df["Brand"].unique()))

    lo, hi = int(np.floor(df["Price_USD"].min())), int(np.ceil(df["Price_USD"].max()))
    prefs["price"] = st.sidebar.slider("💲 Budget (USD)", lo, hi, (lo, hi))
    prefs["min_rating"] = st.sidebar.slider("⭐ Minimum rating", 1.0, 5.0, 3.0, 0.1)
    prefs["cruelty_free"] = st.sidebar.toggle("🐰 Cruelty-free only", value=False)
    return prefs


def summary_line(prefs):
    bits = []
    if prefs.get("Category", "Any") != "Any":
        bits.append(f"**{prefs['Category']}**")
    else:
        bits.append("**any product**")
    if prefs.get("Skin_Type", "Any") != "Any":
        bits.append(f"for **{prefs['Skin_Type']}** skin")
    return " ".join(bits)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home(df, rec):
    st.markdown("# ✨ GlowMatch")
    st.markdown("## Beauty & Skin Care Recommendation System")
    prefs = sidebar_preferences(df, show_category=True)
    top_n = st.sidebar.slider("📦 Number of results", 3, 20, 6)
    st.sidebar.markdown("")
    if st.sidebar.button("✨ Recommend Products"):
        st.session_state["ran"] = True

    st.markdown(
        f"👉 You're shopping for {summary_line(prefs)}. Choose your preferences in the "
        "sidebar, then hit **Recommend Products**."
    )
    st.divider()

    st.markdown("### 📋 Your Profile")
    c1, c2, c3 = st.columns(3)
    c1.metric("Products in catalogue", f"{len(df):,}")
    c2.metric("Brands", df["Brand"].nunique())
    c3.metric("Categories", df["Category"].nunique())

    st.markdown("### 🛍️ Shop by Category")
    counts = df["Category"].value_counts()
    tiles = "".join(
        f'<div class="tile"><div class="thumb">{thumb_html(c)}</div>'
        f'<b>{esc(c)}</b><small>{n:,} products</small></div>'
        for c, n in counts.items()
    )
    st.markdown(f'<div class="tiles">{tiles}</div>', unsafe_allow_html=True)

    if not st.session_state.get("ran"):
        return

    st.divider()
    st.markdown("### 🌟 Your Recommendations")
    res = rec.recommend(prefs, top_n=top_n)
    if res.empty:
        st.warning("No products match those filters. Try widening your budget, lowering the "
                   "minimum rating, or switching off cruelty-free only.")
        return
    st.caption("Ranked by how closely each product matches your profile, with a small boost "
               "for higher ratings and popularity. Highlighted chips show what matched.")
    for _, row in res.iterrows():
        render_card(row, prefs)
    results_table(res)


def page_routine(df, rec):
    st.markdown("# 🧴 Routine Builder")
    st.markdown("Get the best-matching product for each step of your routine.")
    prefs = sidebar_preferences(df, show_category=False)

    cats = sorted(df["Category"].unique())
    default = [c for c in cats if any(h in c.lower() for h in SKINCARE_HINTS)][:5] or cats[:4]
    chosen = st.multiselect("📋 Steps in your routine", cats, default=default)
    st.sidebar.markdown("")
    if st.sidebar.button("✨ Build My Routine"):
        st.session_state["routine_ran"] = True
    if not st.session_state.get("routine_ran"):
        st.info("Pick your steps above, set your profile in the sidebar, then hit **Build My Routine**.")
        return

    st.divider()
    total = 0.0
    for step, cat in enumerate(chosen, 1):
        p = dict(prefs, Category=cat)
        res = rec.recommend(p, top_n=1, categories=[cat])
        st.markdown(f"#### Step {step} · {cat}")
        if res.empty:
            st.caption("No product in this category matches your filters.")
            continue
        render_card(res.iloc[0], p)
        total += float(res.iloc[0]["Price_USD"])
    if total:
        st.success(f"💰 Estimated routine cost: **${total:,.2f}**")


def page_similar(df, rec):
    st.markdown("# 🔍 Similar Products")
    st.markdown("Found something you like? Find products with a similar profile.")
    st.sidebar.markdown("---")
    cat = st.sidebar.selectbox("🎯 Category", ["Any"] + sorted(df["Category"].unique()))
    sub = df if cat == "Any" else df[df["Category"] == cat]
    brand = st.sidebar.selectbox("🏷️ Brand", sorted(sub["Brand"].unique()))
    sub = sub[sub["Brand"] == brand].drop_duplicates(["Product_Name"])
    labels = {i: f"{r.Product_Name} · {r.Category} · ${r.Price_USD:.0f}" for i, r in sub.iterrows()}
    pick = st.sidebar.selectbox("🧴 Product", list(labels), format_func=labels.get)
    top_n = st.sidebar.slider("📦 Number of results", 3, 20, 6)

    st.markdown("### Your product")
    ref = df.loc[pick].copy()
    ref["Match"] = 1.0
    render_card(ref)

    st.markdown("### 🌟 You might also like")
    for _, row in rec.similar_to(pick, top_n).iterrows():
        render_card(row)


def page_insights(df, rec):
    st.markdown("# 📊 Insights")
    st.caption("What the catalogue looks like - useful context behind the recommendations.")
    st.sidebar.markdown("---")
    min_n = st.sidebar.slider("Minimum products per brand", 1, 50, 10)

    a, b = st.columns(2)
    with a:
        st.markdown("#### Most common categories")
        st.bar_chart(df["Category"].value_counts())
    with b:
        st.markdown("#### Most common ingredients")
        st.bar_chart(df["Main_Ingredient"].value_counts().head(12))

    st.markdown("#### Highest-rated brands")
    g = df.groupby("Brand").agg(Avg_Rating=("Rating", "mean"), Avg_Price=("Price_USD", "mean"),
                                Products=("Product_Name", "count"), Reviews=("Number_of_Reviews", "sum"))
    g = g[g["Products"] >= min_n].sort_values("Avg_Rating", ascending=False).round(2)
    st.dataframe(g)

    st.markdown("#### Ingredients in highly rated products (rating ≥ 4)")
    hi = df[df["Rating"] >= 4]["Main_Ingredient"].value_counts().head(12)
    st.bar_chart(hi if len(hi) else df["Main_Ingredient"].value_counts().head(12))

    st.markdown("#### Products per skin type and category")
    st.dataframe(pd.crosstab(df["Skin_Type"], df["Category"]))


def page_guide(df, rec):
    st.markdown("# 📖 Skin Guide")
    st.sidebar.markdown("---")
    skin = st.sidebar.selectbox("🧴 Skin type", sorted(df["Skin_Type"].unique()))
    tip = SKIN_TIPS.get(skin.lower())
    if tip:
        st.info(f"**{skin} skin** - {tip}")
    sub = df[df["Skin_Type"] == skin]
    st.markdown(f"### What the catalogue offers for {skin.lower()} skin")
    a, b, c = st.columns(3)
    a.metric("Products", f"{len(sub):,}")
    b.metric("Avg rating", f"{sub['Rating'].mean():.2f}")
    c.metric("Avg price", f"${sub['Price_USD'].mean():.2f}")
    l, r = st.columns(2)
    with l:
        st.markdown("#### Top categories")
        st.bar_chart(sub["Category"].value_counts().head(8))
    with r:
        st.markdown("#### Top ingredients")
        st.bar_chart(sub["Main_Ingredient"].value_counts().head(8))


def page_help(df, rec):
    st.markdown("# ❓ Help")
    st.markdown(
        """
**How recommendations work**

1. You describe yourself in the sidebar: skin type, what you're shopping for, preferred
   ingredient, usage, brand, budget and minimum rating.
2. Budget, minimum rating, cruelty-free and skin type act as **filters** - products that fail
   them never appear. (Products labelled for *all* skin types are kept.)
3. Remaining products are scored with a **content-based match**: each product's attributes are
   encoded and compared with your profile using cosine similarity, weighted so skin type,
   category and ingredient count most.
4. The final order adds a small boost for higher ratings and more reviews.

**What the match % means** - how closely a product's attributes line up with what you chose.
It is *not* a predicted rating.

**Tips** - leave a field on *Any* if you don't mind; use **Routine Builder** for a full set of
steps and **Similar Products** to find alternatives to something you already use.

*This tool gives product suggestions, not medical advice. If you have a skin condition,
consult a dermatologist.*
"""
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    df, err = load_data()
    if df is None:
        st.markdown("# ✨ GlowMatch")
        if err == "missing":
            st.error(f"Couldn't find **{DATA_FILE}**. Put it in the same folder as this script "
                     "and rerun the app.")
        else:
            st.error(err)
        st.stop()

    rec = get_recommender(df)

    st.sidebar.markdown('<div class="brand">🧴 GlowMatch<small>Skin care recommender</small></div>'
        '<div class="swatches">'
        + "".join(f'<span style="background:{c}"></span>' for c in
                  ["#f9e3d3", "#f1c9a5", "#d9a47a", "#b07a54", "#8a5a3c", "#5e3b27"])
        + "</div>",
                        unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", NAV, label_visibility="collapsed")

    {
        NAV[0]: page_home,
        NAV[1]: page_routine,
        NAV[2]: page_similar,
        NAV[3]: page_insights,
        NAV[4]: page_guide,
        NAV[5]: page_help,
    }[page](df, rec)


main()
