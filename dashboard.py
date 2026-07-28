import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Customer Segments", layout="centered")

COLORS = {
    "High Value":   "#1D9E75",
    "Medium Value": "#2E75B6",
    "Low Value":    "#BA7517",
    "Dormant":      "#E24B4A",
}

# ── How each segment is defined (real numbers from the model) ─────────
SEGMENT_INFO = {
    "High Value": {
        "customers": 251,
        "recency":   "Transacted within the last 1 day",
        "frequency": "1,439 to 3,036 transactions",
        "monetary":  "$70,238 to $168,831 total spend",
        "avg_txn":   "$53 per transaction",
        "fraud":     "0.09% fraud rate — very safe",
        "why":       "These customers use their card almost every day and spend the most. "
                     "The model put them here because their Frequency is above 1,439 txns "
                     "AND their total spend is above $70,000. They are the bank's most valuable customers.",
        "action":    "Reward with cashback, loyalty points, and premium card upgrades.",
    },
    "Medium Value": {
        "customers": 309,
        "recency":   "Transacted within the last 1–3 days",
        "frequency": "861 to 1,539 transactions",
        "monetary":  "$38,204 to $84,593 total spend",
        "avg_txn":   "$50 per transaction",
        "fraud":     "0.16% fraud rate — safe",
        "why":       "Good customers but not as active as High Value. "
                     "Frequency between 861 and 1,539 txns and spend between $38K and $84K. "
                     "They are close to High Value — the right offer could push them up.",
        "action":    "Send upgrade offers and spend-more rewards to push them into High Value.",
    },
    "Low Value": {
        "customers": 174,
        "recency":   "Transacted within the last 1–5 days",
        "frequency": "435 to 530 transactions",
        "monetary":  "$15,702 to $40,779 total spend",
        "avg_txn":   "$53 per transaction",
        "fraud":     "0.37% fraud rate — slightly elevated",
        "why":       "Active customers but with fewer transactions (435–530) and lower total spend ($15K–$40K). "
                     "Their spend per transaction is similar to High Value ($53) — they just shop less often.",
        "action":    "Encourage more frequent use with promotions and category-specific offers.",
    },
    "Dormant": {
        "customers": 47,
        "recency":   "Last transacted 6 to 513 days ago (median: 247 days)",
        "frequency": "1 to 18 transactions only",
        "monetary":  "$6 to $930 total spend",
        "avg_txn":   "$17 per transaction",
        "fraud":     "97.87% fraud rate — VERY HIGH",
        "why":       "These customers have barely used their card. Low frequency (1–18 txns), "
                     "very low spend (under $930 total), and most importantly — not seen for months. "
                     "The extremely high fraud rate (97.87%) suggests most of their few transactions "
                     "were fraudulent. The model isolated them because they look completely different "
                     "from all other customers.",
        "action":    "Investigate fraud activity. Send re-engagement offers to the non-fraud ones.",
    },
}

@st.cache_resource
def load_model():
    with open("models/kmeans_model.pkl",  "rb") as f: model  = pickle.load(f)
    with open("models/scaler.pkl",        "rb") as f: scaler = pickle.load(f)
    with open("models/cluster_names.pkl", "rb") as f: names  = pickle.load(f)
    return model, scaler, names

@st.cache_data
def load_data():
    df = pd.read_csv("credit_card_transactions.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    for col in ["amt", "age", "city_pop"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        df     = df[(df[col] >= Q1-1.5*IQR) & (df[col] <= Q3+1.5*IQR)]
    ref = df["trans_date_trans_time"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("cc_num").agg(
        Recency   = ("trans_date_trans_time", lambda x: (ref - x.max()).days),
        Frequency = ("trans_num",  "count"),
        Monetary  = ("amt",        "sum"),
        Avg_Amt   = ("amt",        "mean"),
        Fraud_Rate= ("is_fraud",   "mean"),
    ).reset_index()
    rfm["Recency_log"]  = np.log1p(rfm["Recency"])
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])
    return rfm

def predict(recency, frequency, monetary, scaler, model, names):
    X = pd.DataFrame(
        [[np.log1p(recency), frequency, np.log1p(monetary)]],
        columns=["Recency_log", "Frequency", "Monetary_log"]
    )
    num = model.predict(scaler.transform(X))[0]
    return names[num]

model, scaler, names = load_model()
rfm = load_data()
rfm["Segment"] = rfm.apply(
    lambda r: predict(r["Recency"], r["Frequency"], r["Monetary"], scaler, model, names), axis=1
)

# ── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.title("Pages")
page = st.sidebar.radio("", ["Home", "Segments", "Predict", "Evaluation"])

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════
if page == "Home":
    st.title("Customer Segmentation Dashboard")
    st.write("781 customers grouped into 4 segments based on their credit card spending behaviour.")
    st.markdown("---")

    # 4 metric cards
    col1, col2, col3, col4 = st.columns(4)
    counts = rfm["Segment"].value_counts()
    col1.metric("🟢 High Value",   counts.get("High Value",   0))
    col2.metric("🔵 Medium Value", counts.get("Medium Value", 0))
    col3.metric("🟠 Low Value",    counts.get("Low Value",    0))
    col4.metric("🔴 Dormant",      counts.get("Dormant",      0))

    st.markdown("---")

    # Pie chart
    st.subheader("Customers per segment")
    seg_counts = rfm["Segment"].value_counts()
    colors     = [COLORS.get(s, "#888") for s in seg_counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(seg_counts.values, labels=seg_counts.index,
           colors=colors, autopct="%1.0f%%", startangle=90)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Summary table with real numbers per customer
    st.subheader("What each segment looks like — real numbers")
    st.caption("These are the actual ranges from each customer in that group, not just averages.")

    order = ["High Value","Medium Value","Low Value","Dormant"]
    for seg in order:
        sub   = rfm[rfm["Segment"] == seg]
        color = COLORS[seg]
        info  = SEGMENT_INFO[seg]

        st.markdown(f"<span style='font-size:15px;font-weight:500;color:{color}'>● {seg}</span>",
                    unsafe_allow_html=True)

        st.write(f"👥 **Customers:** {info['customers']}")
        st.write(f"📅 **Days inactive (median):** {int(sub['Recency'].median())} days")
        st.write(f"🔁 **Transactions (median):** {int(sub['Frequency'].median()):,} transactions")
        st.write(f"💰 **Total spend per customer (median):** ${sub['Monetary'].median():,.0f}")
        st.write(f"💳 **Avg spend per single transaction:** ${sub['Avg_Amt'].mean():,.0f}")

        with st.expander(f"See full details for {seg}"):
            st.write(f"**Recency (how recently they transacted):** {info['recency']}")
            st.write(f"**Frequency (how many transactions):** {info['frequency']}")
            st.write(f"**Monetary (total spend per customer):** {info['monetary']}")
            st.write(f"**Average spend per single transaction:** {info['avg_txn']}")
            st.write(f"**Fraud rate:** {info['fraud']}")
            st.markdown("---")
            st.write(f"**Why are they in this segment?**")
            st.info(info["why"])
            st.write(f"**What should the bank do?**")
            st.success(info["action"])

        st.write("")

    st.markdown("---")

    # How the model classifies
    st.subheader("How does the model decide which segment a customer goes into?")
    st.write("The model uses 3 numbers for every customer — Recency, Frequency, and Monetary. "
             "It draws invisible boundaries in 3D space and puts each customer in the nearest group.")
    st.write("Here are the rough cutoffs based on the data:")

    st.markdown("""
| Segment | Recency | Frequency | Monetary |
|---|---|---|---|
| 🟢 **High Value** | 1 day inactive | 1,439 – 3,036 txns | $70,238 – $168,831 |
| 🔵 **Medium Value** | 1–3 days inactive | 861 – 1,539 txns | $38,204 – $84,593 |
| 🟠 **Low Value** | 1–5 days inactive | 435 – 530 txns | $15,702 – $40,779 |
| 🔴 **Dormant** | 6 – 513 days inactive | 1 – 18 txns | $6 – $930 |
""")
    st.caption("Note: these are not hard cutoffs. K-Means uses distance in 3D space, "
               "not simple if/else rules. But these ranges give you a good idea of where each group sits.")

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENTS
# ══════════════════════════════════════════════════════════════════════
elif page == "Segments":
    st.title("Segment Details")
    st.write("Pick a segment to see its customers in detail.")
    st.markdown("---")

    chosen = st.selectbox("Choose a segment", ["High Value","Medium Value","Low Value","Dormant"])
    subset = rfm[rfm["Segment"] == chosen]
    color  = COLORS[chosen]
    info   = SEGMENT_INFO[chosen]

    # Header numbers — real ranges not just averages
    st.subheader(f"{chosen} — {len(subset)} customers")

    st.write(f"📅 **Days inactive:** min = {subset['Recency'].min()} day,  max = {subset['Recency'].max()} days,  median = {int(subset['Recency'].median())} days")
    st.write(f"🔁 **Transactions:** min = {subset['Frequency'].min():,},  max = {subset['Frequency'].max():,},  median = {int(subset['Frequency'].median()):,}")
    st.write(f"💰 **Total spend per customer:** min = ${subset['Monetary'].min():,.0f},  max = ${subset['Monetary'].max():,.0f},  median = ${subset['Monetary'].median():,.0f}")
    st.write(f"💳 **Avg spend per single transaction:** ${subset['Avg_Amt'].mean():,.0f}  (lowest: ${subset['Avg_Amt'].min():,.0f},  highest: ${subset['Avg_Amt'].max():,.0f})")

    st.caption("'Avg spend per transaction' = total spend ÷ number of transactions for each customer")
    st.markdown("---")

    # Why explanation with numbers
    st.subheader("Why are these customers classified as " + chosen + "?")
    st.info(info["why"])

    st.subheader("What should the bank do with them?")
    st.success(info["action"])
    st.markdown("---")

    # Histograms
    st.subheader("Distribution charts")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3))

    axes[0].hist(subset["Recency"],   bins=20, color=color, edgecolor="white")
    axes[0].set_title("Recency (days since last txn)")
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Customers")
    axes[0].axvline(subset["Recency"].median(), color="black", ls="--", lw=1.5,
                    label=f"Median = {subset['Recency'].median():.0f}d")
    axes[0].legend(fontsize=8)

    axes[1].hist(subset["Frequency"], bins=20, color=color, edgecolor="white")
    axes[1].set_title("Frequency (transactions)")
    axes[1].set_xlabel("Number of transactions")
    axes[1].axvline(subset["Frequency"].median(), color="black", ls="--", lw=1.5,
                    label=f"Median = {subset['Frequency'].median():.0f}")
    axes[1].legend(fontsize=8)

    axes[2].hist(subset["Monetary"],  bins=20, color=color, edgecolor="white")
    axes[2].set_title("Monetary (total spend $)")
    axes[2].set_xlabel("Total spend ($)")
    axes[2].axvline(subset["Monetary"].median(), color="black", ls="--", lw=1.5,
                    label=f"Median = ${subset['Monetary'].median():,.0f}")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Spend per transaction histogram
    st.subheader("How much does each customer spend per single transaction?")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(subset["Avg_Amt"], bins=20, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(subset["Avg_Amt"].mean(), color="black", ls="--", lw=1.5,
               label=f"Average = ${subset['Avg_Amt'].mean():.0f}/txn")
    ax.set_xlabel("Avg spend per transaction ($)")
    ax.set_ylabel("Customers")
    ax.set_title("Spend per transaction — " + chosen)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Compare to others
    st.subheader("How does this segment compare to others?")
    order      = ["High Value","Medium Value","Low Value","Dormant"]
    profile    = rfm.groupby("Segment")[["Recency","Frequency","Monetary","Avg_Amt"]].median().round(0).reindex(order)
    bar_colors = [COLORS[s] for s in order]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for ax, col, title in zip(axes,
        ["Recency","Frequency","Monetary","Avg_Amt"],
        ["Median Days Inactive","Median Transactions","Median Total Spend ($)","Median Spend per Txn ($)"]):
        bars = ax.bar(order, profile[col], color=bar_colors, edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        for i, seg in enumerate(order):
            if seg == chosen:
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(2.5)
        for bar, val in zip(bars, profile[col]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+profile[col].max()*0.01,
                    f"{val:,.0f}", ha="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption(f"The {chosen} bar has a black outline. Numbers shown on top of each bar.")

    st.markdown("---")

    # Customer table
    st.subheader("Full customer list")
    st.dataframe(
        subset[["cc_num","Recency","Frequency","Monetary","Avg_Amt","Fraud_Rate"]]
        .rename(columns={
            "cc_num":     "Customer ID",
            "Recency":    "Days Inactive",
            "Frequency":  "Total Transactions",
            "Monetary":   "Total Spend ($)",
            "Avg_Amt":    "Avg per Transaction ($)",
            "Fraud_Rate": "Fraud Rate",
        })
        .sort_values("Total Spend ($)", ascending=False)
        .round(2).reset_index(drop=True),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT
# ══════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("Predict a Customer's Segment")
    st.write("Enter a customer's 3 numbers and the model tells you which segment they belong to.")
    st.markdown("---")

    # Reference guide
    with st.expander("What numbers should I enter? (click to see guide)"):
        st.markdown("""
| Segment | Recency | Frequency | Monetary |
|---|---|---|---|
| 🟢 High Value | 1 day | 1,439–3,036 txns | $70K–$168K |
| 🔵 Medium Value | 1–3 days | 861–1,539 txns | $38K–$84K |
| 🟠 Low Value | 1–5 days | 435–530 txns | $15K–$40K |
| 🔴 Dormant | 6–513 days | 1–18 txns | under $930 |
""")
        st.caption("Use this as a reference. The model uses distance, not hard cutoffs — so a customer near the boundary could go either way.")

    st.markdown("---")

    recency   = st.number_input("Recency — days since last transaction",
                                 min_value=0, max_value=600, value=2, step=1)
    frequency = st.number_input("Frequency — total number of transactions",
                                 min_value=1, max_value=5000, value=1200, step=10)
    monetary  = st.number_input("Monetary — total amount spent in dollars",
                                 min_value=0.0, max_value=200000.0, value=60000.0, step=500.0)

    st.markdown("---")

    if st.button("Predict →", type="primary"):
        result = predict(recency, frequency, monetary, scaler, model, names)
        color  = COLORS.get(result, "#888")
        info   = SEGMENT_INFO[result]

        st.markdown(
            f"<div style='background:{color}18;border-left:4px solid {color};"
            f"border-radius:8px;padding:1rem 1.25rem;'>"
            f"<div style='font-size:12px;color:{color};font-weight:500;text-transform:uppercase;"
            f"letter-spacing:.06em;margin-bottom:4px;'>Predicted Segment</div>"
            f"<div style='font-size:28px;font-weight:500;color:{color};'>{result}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("Why did the model pick this segment?")
        st.info(info["why"])

        st.subheader("What should the bank do?")
        st.success(info["action"])

        st.markdown("---")
        st.subheader("How does this customer compare to the segment?")

        seg_data = rfm[rfm["Segment"] == result]
        seg_med_r = int(seg_data['Recency'].median())
        seg_med_f = int(seg_data['Frequency'].median())
        seg_med_m = seg_data['Monetary'].median()

        st.write(f"📅 **Your customer:** {recency} days inactive  |  Segment median: {seg_med_r} days  |  Difference: {recency - seg_med_r:+d} days")
        st.write(f"🔁 **Your customer:** {frequency:,} transactions  |  Segment median: {seg_med_f:,}  |  Difference: {frequency - seg_med_f:+,}")
        st.write(f"💰 **Your customer:** ${monetary:,.0f} total spend  |  Segment median: ${seg_med_m:,.0f}  |  Difference: ${monetary - seg_med_m:+,.0f}")

        st.markdown("---")
        st.subheader("Where does this customer sit in the segment?")

        fig, axes = plt.subplots(1, 3, figsize=(13, 3))
        for ax, col, user_val, label in zip(axes,
            ["Recency","Frequency","Monetary"],
            [recency, frequency, monetary],
            ["Days Inactive","Transactions","Total Spend ($)"]):
            ax.hist(seg_data[col], bins=20, color=COLORS[result], edgecolor="white", alpha=0.6,
                    label=f"{result} customers")
            ax.axvline(user_val, color="black", lw=2.5, ls="--", label="This customer")
            ax.set_xlabel(label)
            ax.set_ylabel("Customers")
            ax.legend(fontsize=8)
        plt.suptitle(f"This customer vs the {result} segment", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Try these examples")
    examples = [
        ("High spender, very active",      1,   2000, 110000),
        ("Regular active customer",        2,   1200,  60000),
        ("Active but low spend",           1,    490,  24000),
        ("Has not used card in months", 300,      3,     60),
    ]
    for label, r, f, m in examples:
        pred  = predict(r, f, m, scaler, model, names)
        color = COLORS.get(pred, "#888")
        st.markdown(
            f"<span style='color:{color}'>●</span> **{label}** "
            f"(R={r}d, F={f:,} txns, M=${m:,}) → **{pred}**",
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════════
elif page == "Evaluation":
    st.title("Model Evaluation")
    st.write("3 checks to see how good the clustering is.")
    st.markdown("---")

    from sklearn.metrics import silhouette_score, silhouette_samples

    X_eval   = scaler.transform(rfm[["Recency_log","Frequency","Monetary_log"]])
    labels   = model.predict(X_eval)
    sil      = silhouette_score(X_eval, labels)
    sil_vals = silhouette_samples(X_eval, labels)
    wcss     = model.inertia_

    # Check 1
    st.subheader("Check 1 — Silhouette Score")
    st.write("Measures how well separated the 4 clusters are. Ranges -1 to +1. Higher is better.")
    st.metric("Overall Score", f"{sil:.3f}",
              delta="Good" if sil>=0.5 else "Acceptable" if sil>=0.3 else "Weak")
    if sil >= 0.5:
        st.success(f"Score of {sil:.3f} means the 4 segments are clearly separated from each other.")
    elif sil >= 0.3:
        st.warning("Some overlap between segments.")
    else:
        st.error("Weak separation — try a different K.")

    # Per cluster silhouette with numbers on bars
    st.write("**Score per segment** (higher = that segment is more distinct):")
    order  = ["High Value","Medium Value","Low Value","Dormant"]
    colors = [COLORS[n] for n in order]
    cluster_sil = {names[c]: sil_vals[labels==c].mean() for c in set(labels)}
    vals = [cluster_sil.get(n, 0) for n in order]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(order, vals, color=colors, edgecolor="white")
    ax.axhline(sil, color="black", ls="--", lw=1.5, label=f"Overall avg = {sil:.3f}")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score per segment")
    ax.legend()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Check 2 — cluster sizes with numbers
    st.subheader("Check 2 — Cluster Sizes")
    st.write("No cluster should be extremely tiny. Here are the exact counts:")

    sizes = rfm["Segment"].value_counts().reindex(order)
    pcts  = (sizes / len(rfm) * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(order, sizes.values, color=colors, edgecolor="white")
    ax.set_ylabel("Number of customers")
    ax.set_title("How many customers in each segment?")
    for bar, count, pct in zip(bars, sizes.values, pcts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{count} customers\n({pct}%)", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    for seg, count, pct in zip(order, sizes.values, pcts.values):
        st.write(f"**{seg}:** {count} customers ({pct}%)")

    if pcts.min() < 5:
        st.warning("Dormant is the smallest group (6.0%) — expected, as truly inactive customers are rare.")
    else:
        st.success("All segments have a reasonable number of customers.")

    st.markdown("---")

    # Check 3 — business sense with actual numbers
    st.subheader("Check 3 — Does it make business sense?")
    st.write("We verify that High Value actually has the best numbers and Dormant has the worst.")

    profile = rfm.groupby("Segment")[["Recency","Frequency","Monetary","Avg_Amt"]].median().round(1).reindex(order)
    profile.columns = ["Median Days Inactive","Median Transactions","Median Total Spend ($)","Median Spend/Txn ($)"]
    st.dataframe(profile, use_container_width=True)

    checks = [
        ("High Value has the fewest inactive days (1 day median)",
         profile.loc["High Value","Median Days Inactive"] == profile["Median Days Inactive"].min()),
        ("High Value has the most transactions (1,992 median)",
         profile.loc["High Value","Median Transactions"] == profile["Median Transactions"].max()),
        ("High Value has the highest total spend ($103,502 median)",
         profile.loc["High Value","Median Total Spend ($)"] == profile["Median Total Spend ($)"].max()),
        ("Dormant has the most inactive days (247 day median)",
         profile.loc["Dormant","Median Days Inactive"] == profile["Median Days Inactive"].max()),
        ("Dormant has the fewest transactions (2 median)",
         profile.loc["Dormant","Median Transactions"] == profile["Median Transactions"].min()),
        ("Dormant has the lowest total spend ($18 median)",
         profile.loc["Dormant","Median Total Spend ($)"] == profile["Median Total Spend ($)"].min()),
    ]

    all_passed = True
    for label, passed in checks:
        st.write(f"{'✅' if passed else '❌'} {label}")
        if not passed: all_passed = False

    st.markdown("---")
    if all_passed:
        st.success("All 6 checks passed — the model is working correctly and the segments make business sense.")
    else:
        st.warning("Some checks failed. The cluster numbering may have shifted — update cluster_names in your notebook.")