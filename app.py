from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Beauty & Cosmetics Explorer",
    page_icon="💄",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "Beauty_Cosmetics_Cleaned.csv"
COLOR = "#B5446E"
SEQ = px.colors.sequential.Magenta


@st.cache_data(show_spinner=False)
def load_data(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    df["Cruelty_Free"] = df["Cruelty_Free"].astype(bool)
    df["Size_ml"] = df["Product_Size"].str.extract(r"(\d+)").astype(float)
    return df


# ---------- Data loading ----------
if DATA_FILE.exists():
    df = load_data(DATA_FILE)
else:
    up = st.file_uploader("Upload Beauty_Cosmetics_Cleaned.csv", type="csv")
    if up is None:
        st.info("Place the CSV next to app.py or upload it here to get started.")
        st.stop()
    df = load_data(up)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")


def multi(label, col):
    opts = sorted(df[col].unique())
    return st.sidebar.multiselect(label, opts, default=[], placeholder="All")


sel_cat = multi("Category", "Category")
sel_brand = multi("Brand", "Brand")
sel_skin = multi("Skin type", "Skin_Type")
sel_gender = multi("Gender target", "Gender_Target")
sel_pack = multi("Packaging", "Packaging_Type")
sel_ing = multi("Main ingredient", "Main_Ingredient")
sel_country = multi("Country of origin", "Country_of_Origin")
sel_freq = multi("Usage frequency", "Usage_Frequency")

pmin, pmax = float(df.Price_USD.min()), float(df.Price_USD.max())
price_rng = st.sidebar.slider("Price (USD)", pmin, pmax, (pmin, pmax))
rating_rng = st.sidebar.slider("Rating", 1.0, 5.0, (1.0, 5.0), step=0.1)
cf_choice = st.sidebar.radio("Cruelty free", ["All", "Yes", "No"], horizontal=True)
search = st.sidebar.text_input("Search product name")

mask = df.Price_USD.between(*price_rng) & df.Rating.between(*rating_rng)
for col, sel in [
    ("Category", sel_cat),
    ("Brand", sel_brand),
    ("Skin_Type", sel_skin),
    ("Gender_Target", sel_gender),
    ("Packaging_Type", sel_pack),
    ("Main_Ingredient", sel_ing),
    ("Country_of_Origin", sel_country),
    ("Usage_Frequency", sel_freq),
]:
    if sel:
        mask &= df[col].isin(sel)
if cf_choice != "All":
    mask &= df.Cruelty_Free == (cf_choice == "Yes")
if search:
    mask &= df.Product_Name.str.contains(search, case=False, na=False)

f = df[mask]

# ---------- Header & KPIs ----------
st.title("💄 Beauty & Cosmetics Explorer")
st.caption(f"Showing {len(f):,} of {len(df):,} products")

if f.empty:
    st.warning("No products match the current filters.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Products", f"{len(f):,}")
k2.metric("Brands", f"{f.Brand.nunique()}")
k3.metric("Avg price", f"${f.Price_USD.mean():,.2f}")
k4.metric("Avg rating", f"{f.Rating.mean():.2f} ★")
k5.metric("Cruelty free", f"{f.Cruelty_Free.mean():.0%}")

tab_over, tab_price, tab_brand, tab_ing, tab_data = st.tabs(
    ["Overview", "Price & Rating", "Brands", "Ingredients & Skin", "Data"]
)

# ---------- Overview ----------
with tab_over:
    c1, c2 = st.columns(2)
    cat = f.Category.value_counts().reset_index()
    cat.columns = ["Category", "Products"]
    fig = px.bar(
        cat.sort_values("Products"),
        x="Products",
        y="Category",
        orientation="h",
        title="Products by category",
        color_discrete_sequence=[COLOR],
    )
    c1.plotly_chart(fig, use_container_width=True)

    ctry = f.Country_of_Origin.value_counts().reset_index()
    ctry.columns = ["Country", "Products"]
    fig = px.pie(
        ctry,
        names="Country",
        values="Products",
        hole=0.45,
        title="Country of origin",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    c2.plotly_chart(fig, use_container_width=True)

    c3, c4, c5 = st.columns(3)
    for col, title, widget in [
        ("Packaging_Type", "Packaging type", c3),
        ("Gender_Target", "Gender target", c4),
        ("Usage_Frequency", "Usage frequency", c5),
    ]:
        d = f[col].value_counts().reset_index()
        d.columns = [col, "Products"]
        fig = px.bar(
            d, x=col, y="Products", title=title, color_discrete_sequence=[COLOR]
        )
        fig.update_layout(xaxis_title=None)
        widget.plotly_chart(fig, use_container_width=True)

# ---------- Price & rating ----------
with tab_price:
    c1, c2 = st.columns(2)
    fig = px.histogram(
        f,
        x="Price_USD",
        nbins=40,
        title="Price distribution",
        color_discrete_sequence=[COLOR],
    )
    c1.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
        f,
        x="Rating",
        nbins=20,
        title="Rating distribution",
        color_discrete_sequence=[COLOR],
    )
    c2.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    fig = px.box(
        f,
        x="Category",
        y="Price_USD",
        title="Price by category",
        color_discrete_sequence=[COLOR],
    )
    c3.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        f,
        x="Category",
        y="Rating",
        title="Rating by category",
        color_discrete_sequence=[COLOR],
    )
    c4.plotly_chart(fig, use_container_width=True)

    st.subheader("Price vs rating")
    color_by = st.selectbox(
        "Colour points by",
        ["Category", "Skin_Type", "Gender_Target", "Packaging_Type", "Cruelty_Free"],
    )
    sample = f.sample(min(len(f), 3000), random_state=1)
    fig = px.scatter(
        sample,
        x="Price_USD",
        y="Rating",
        color=color_by,
        size="Number_of_Reviews",
        hover_data=["Product_Name", "Brand"],
        opacity=0.6,
    )
    st.plotly_chart(fig, use_container_width=True)
    if len(f) > 3000:
        st.caption("Scatter shows a random sample of 3,000 products for speed.")

# ---------- Brands ----------
with tab_brand:
    brand = (
        f.groupby("Brand")
        .agg(
            Products=("Product_Name", "count"),
            Avg_Price=("Price_USD", "mean"),
            Avg_Rating=("Rating", "mean"),
            Total_Reviews=("Number_of_Reviews", "sum"),
            Cruelty_Free=("Cruelty_Free", "mean"),
        )
        .reset_index()
    )
    top_n = st.slider("Brands to show", 5, min(40, len(brand)), min(15, len(brand)))
    metric = st.selectbox(
        "Rank brands by",
        ["Products", "Avg_Price", "Avg_Rating", "Total_Reviews", "Cruelty_Free"],
    )
    top = brand.sort_values(metric, ascending=False).head(top_n)
    fig = px.bar(
        top.sort_values(metric),
        x=metric,
        y="Brand",
        orientation="h",
        title=f"Top {top_n} brands by {metric.replace('_', ' ').lower()}",
        color_discrete_sequence=[COLOR],
    )
    fig.update_layout(height=max(400, top_n * 28))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        brand,
        x="Avg_Price",
        y="Avg_Rating",
        size="Products",
        hover_name="Brand",
        title="Brand positioning: average price vs average rating",
        color_discrete_sequence=[COLOR],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        brand.sort_values("Products", ascending=False).style.format(
            {
                "Avg_Price": "${:,.2f}",
                "Avg_Rating": "{:.2f}",
                "Total_Reviews": "{:,}",
                "Cruelty_Free": "{:.0%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------- Ingredients & skin ----------
with tab_ing:
    c1, c2 = st.columns(2)
    ing = (
        f.groupby("Main_Ingredient")
        .agg(Avg_Rating=("Rating", "mean"), Products=("Rating", "count"))
        .reset_index()
        .sort_values("Avg_Rating")
    )
    fig = px.bar(
        ing,
        x="Avg_Rating",
        y="Main_Ingredient",
        orientation="h",
        title="Average rating by main ingredient",
        color_discrete_sequence=[COLOR],
        hover_data=["Products"],
    )
    c1.plotly_chart(fig, use_container_width=True)

    skin = (
        f.groupby("Skin_Type")
        .agg(Avg_Rating=("Rating", "mean"), Avg_Price=("Price_USD", "mean"))
        .reset_index()
    )
    fig = px.bar(
        skin,
        x="Skin_Type",
        y="Avg_Rating",
        title="Average rating by skin type",
        color_discrete_sequence=[COLOR],
        hover_data=["Avg_Price"],
    )
    c2.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmaps")
    h1, h2 = st.columns(2)
    heat = f.pivot_table(
        index="Main_Ingredient", columns="Skin_Type", values="Rating", aggfunc="mean"
    )
    fig = px.imshow(
        heat,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=SEQ,
        title="Avg rating: ingredient × skin type",
    )
    h1.plotly_chart(fig, use_container_width=True)

    heat2 = f.pivot_table(
        index="Category", columns="Packaging_Type", values="Price_USD", aggfunc="mean"
    )
    fig = px.imshow(
        heat2,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale=SEQ,
        title="Avg price ($): category × packaging",
    )
    h2.plotly_chart(fig, use_container_width=True)

    cf = (
        f.groupby("Country_of_Origin")
        .Cruelty_Free.mean()
        .reset_index()
        .sort_values("Cruelty_Free")
    )
    fig = px.bar(
        cf,
        x="Cruelty_Free",
        y="Country_of_Origin",
        orientation="h",
        title="Share of cruelty-free products by country",
        color_discrete_sequence=[COLOR],
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Data ----------
with tab_data:
    sort_col = st.selectbox("Sort by", list(f.columns.drop("Size_ml")), index=5)
    asc = st.checkbox("Ascending", value=False)
    view = f.drop(columns="Size_ml").sort_values(sort_col, ascending=asc)
    st.dataframe(view, use_container_width=True, hide_index=True, height=520)
    st.download_button(
        "Download filtered data (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        file_name="beauty_cosmetics_filtered.csv",
        mime="text/csv",
    )
