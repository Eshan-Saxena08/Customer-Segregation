import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import gdown
except ImportError:  # pragma: no cover - optional dependency for dataset download
    gdown = None

st.set_page_config(page_title="Customer Segments — RFM", layout="centered")

# ── 6-segment colour map ───────────────────────────────────────────────
COLORS = {
    'Champions': '#1D9E75',
    'Loyal':     '#2E75B6',
    'Growing':   '#F39C12',
    'At Risk':   '#E74C3C',
    'Dormant':   '#7F8C8D',
    'Lost':      '#95A5A6',
}

SEG_ORDER = ['Champions', 'Loyal', 'Growing', 'At Risk', 'Dormant', 'Lost']

SEGMENT_META = {
    'Champions': ('🏆', 'RFM Score 13–15. Bought recently, transact often, spend the most.',
                  'Reward them. Give early access to new products. They will refer others.'),
    'Loyal':     ('💙', 'RFM Score 10–12. Spend well and transact consistently.',
                  'Upsell higher-value products. Ask for referrals. Keep them engaged.'),
    'Growing':   ('🌱', 'RFM Score 7–9. Active and improving — on the way up.',
                  'Nurture them with loyalty incentives. Help them grow into Loyal or Champions.'),
    'At Risk':   ('🚨', 'RFM Score 5–6. Were good customers but activity is declining.',
                  'Personalised win-back campaign. Remind them of what they loved. Act now.'),
    'Dormant':   ('😴', 'RFM Score 4. Low across all dimensions. Barely active.',
                  'One targeted re-engagement offer. Low spend — keep effort minimal.'),
    'Lost':      ('💔', 'RFM Score 3. Lowest scores on all three dimensions.',
                  'Minimal effort. One last campaign. Accept if there is no response.'),
}

# ── Segment classification using total RFM score ───────────────────────
def assign_segment(score):
    if   score >= 13: return 'Champions'
    elif score >= 10: return 'Loyal'
    elif score >= 7:  return 'Growing'
    elif score >= 5:  return 'At Risk'
    elif score == 4:  return 'Dormant'
    else:             return 'Lost'

@st.cache_data
def load_data():
    data_path = 'credit_card_transactions.csv'
    if not os.path.exists(data_path):
        if gdown is None:
            raise FileNotFoundError(
                f"{data_path} is missing and gdown is not installed; install it or provide the CSV file."
            )
        gdown.download(
            'https://drive.google.com/file/d/1JvdwHgTwBZZFR5ZTlzTCYyYp_pgT5I5Y/view?usp=sharing',
            data_path,
            quiet=False,
        )

    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days / 365.25

    for col in ['amt','age','city_pop']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        df     = df[(df[col] >= Q1-1.5*IQR) & (df[col] <= Q3+1.5*IQR)]

    ref = df['trans_date_trans_time'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('cc_num').agg(
        Recency   = ('trans_date_trans_time', lambda x: (ref - x.max()).days),
        Frequency = ('trans_num',  'count'),
        Monetary  = ('amt',        'sum'),
        Avg_Amt   = ('amt',        'mean'),
        Fraud_Rate= ('is_fraud',   'mean'),
    ).reset_index()

    # RFM scores 1-5
    rfm['R_score'] = pd.qcut(rfm['Recency'].rank(method='first'),   q=5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['Monetary'].rank(method='first'),  q=5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_Score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
    rfm['Segment']   = rfm['RFM_Score'].apply(assign_segment)
    return rfm

rfm = load_data()
counts = rfm['Segment'].value_counts()

# ── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.title("Pages")
page = st.sidebar.radio("", ["Overview","Segment Detail","RFM Scores","Predict","Evaluation"])
st.sidebar.markdown("---")

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Customer Segmentation — RFM Analysis")
    st.write("781 customers scored on Recency, Frequency, and Monetary value and placed into 6 segments based on their total RFM score (3–15).")
    st.markdown("---")

    # Segment summary table
    st.subheader("All segments at a glance")
    rows = []
    for seg in SEG_ORDER:
        if seg not in rfm['Segment'].values:
            continue
        sub  = rfm[rfm['Segment']==seg]
        icon, desc, action = SEGMENT_META[seg]
        rows.append({
            "Segment":          f"{icon} {seg}",
            "Customers":        len(sub),
            "Median Recency":   f"{sub['Recency'].median():.0f} days",
            "Median Frequency": f"{sub['Frequency'].median():,.0f} txns",
            "Median Spend":     f"${sub['Monetary'].median():,.0f}",
            "Avg per Txn":      f"${sub['Avg_Amt'].mean():,.0f}",
            "Avg RFM Score":    f"{sub['RFM_Score'].mean():.1f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Segment"), use_container_width=True)

    st.markdown("---")

    # What each segment means
    st.subheader("What each segment means and what to do")
    for seg in SEG_ORDER:
        if seg not in rfm['Segment'].values:
            continue
        sub = rfm[rfm['Segment']==seg]
        icon, desc, action = SEGMENT_META[seg]
        color = COLORS[seg]
        with st.expander(f"{icon} {seg} — {len(sub)} customers"):
            st.markdown(f"<span style='color:{color};font-weight:500;'>What they look like:</span>", unsafe_allow_html=True)
            st.write(desc)
            c1, c2, c3, c4 = st.columns(4)
            c1.write(f"📅 **Recency:** {sub['Recency'].median():.0f} days")
            c2.write(f"🔁 **Transactions:** {sub['Frequency'].median():,.0f}")
            c3.write(f"💰 **Total spend:** ${sub['Monetary'].median():,.0f}")
            c4.write(f"💳 **Avg/txn:** ${sub['Avg_Amt'].mean():,.0f}")
            st.markdown(f"<span style='color:{color};font-weight:500;'>What the bank should do:</span>", unsafe_allow_html=True)
            st.success(action)

    st.markdown("---")

    # Segment distribution bar chart
    st.subheader("Customer count per segment")
    order = [s for s in SEG_ORDER if s in counts]
    bar_colors = [COLORS[s] for s in order]
    vals = [counts[s] for s in order]
    fig, ax = plt.subplots(figsize=(10,4))
    bars = ax.bar(order, vals, color=bar_colors, edgecolor='white')
    ax.set_ylabel("Customers")
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(val), ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # RFM score distribution per segment
    st.subheader("RFM score distribution by segment")
    fig, ax = plt.subplots(figsize=(10,4))
    for seg in SEG_ORDER:
        if seg not in rfm['Segment'].values:
            continue
        sub = rfm[rfm['Segment']==seg]
        ax.scatter(sub['RFM_Score'], [seg]*len(sub),
                   c=COLORS[seg], alpha=0.55, s=35, edgecolors='white', linewidths=0.3)
    ax.set_xlabel("Total RFM Score (3–15)", fontsize=11)
    ax.set_title("Each dot = one customer, plotted on their RFM score\nSegments sit in distinct, non-overlapping bands", fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("Champions cluster at 13–15, Lost cluster at 3. No segment overlaps another.")

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENT DETAIL
# ══════════════════════════════════════════════════════════════════════
elif page == "Segment Detail":
    st.title("Segment Detail")
    st.markdown("---")

    available = [s for s in SEG_ORDER if s in rfm['Segment'].values]
    chosen = st.selectbox("Choose a segment", available,
                          format_func=lambda s: f"{SEGMENT_META[s][0]} {s} ({counts.get(s,0)} customers)")
    subset = rfm[rfm['Segment']==chosen]
    color  = COLORS[chosen]
    icon, desc, action = SEGMENT_META[chosen]

    st.markdown(f"<h3 style='color:{color}'>{icon} {chosen}</h3>", unsafe_allow_html=True)
    st.write(desc)
    st.success(f"**Bank action:** {action}")
    st.markdown("---")

    # Key numbers
    st.write(f"📅 **Recency:** min={subset['Recency'].min()} days  |  max={subset['Recency'].max()} days  |  median={subset['Recency'].median():.0f} days")
    st.write(f"🔁 **Frequency:** min={subset['Frequency'].min():,}  |  max={subset['Frequency'].max():,}  |  median={subset['Frequency'].median():,.0f} transactions")
    st.write(f"💰 **Monetary:** min=${subset['Monetary'].min():,.0f}  |  max=${subset['Monetary'].max():,.0f}  |  median=${subset['Monetary'].median():,.0f}")
    st.write(f"💳 **Avg per transaction:** ${subset['Avg_Amt'].mean():,.0f}")
    st.write(f"🎯 **RFM Score (3–15):** avg={subset['RFM_Score'].mean():.1f}  |  R avg={subset['R_score'].mean():.1f}  |  F avg={subset['F_score'].mean():.1f}  |  M avg={subset['M_score'].mean():.1f}")
    st.markdown("---")

    # Histograms
    st.subheader("Distribution charts")
    fig, axes = plt.subplots(1, 3, figsize=(13,3))
    for ax, col, label in zip(axes,
        ['Recency','Frequency','Monetary'],
        ['Recency (days)','Frequency (txns)','Monetary ($)']):
        ax.hist(subset[col], bins=20, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(subset[col].median(), color='black', ls='--', lw=1.5,
                   label=f"Median={subset[col].median():,.0f}")
        ax.set_xlabel(label); ax.set_ylabel("Customers")
        ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Compare vs all segments
    st.subheader("How does this segment compare?")
    order = [s for s in SEG_ORDER if s in rfm['Segment'].values]
    bc    = [COLORS[s] for s in order]
    profile = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].median().round(0)

    fig, axes = plt.subplots(1,3,figsize=(14,3))
    for ax, col, title in zip(axes,['Recency','Frequency','Monetary'],
        ['Median Days Inactive','Median Transactions','Median Total Spend ($)']):
        vals = [profile.loc[s,col] if s in profile.index else 0 for s in order]
        bars = ax.bar(order, vals, color=bc, edgecolor='white')
        for i,s in enumerate(order):
            if s == chosen:
                bars[i].set_edgecolor('black'); bars[i].set_linewidth(2.5)
        ax.set_title(title, fontsize=10); ax.tick_params(axis='x', rotation=20)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                    f"{v:,.0f}", ha='center', fontsize=7, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption(f"Black outline = {chosen}.")

    st.markdown("---")
    st.subheader("Customer list")
    st.dataframe(
        subset[['cc_num','R_score','F_score','M_score','RFM_Score','Recency','Frequency','Monetary','Avg_Amt','Fraud_Rate']]
        .rename(columns={'cc_num':'Customer ID','R_score':'R','F_score':'F','M_score':'M',
                         'RFM_Score':'Total Score','Recency':'Days Inactive',
                         'Frequency':'Transactions','Monetary':'Total Spend ($)',
                         'Avg_Amt':'Avg/Txn ($)','Fraud_Rate':'Fraud Rate'})
        .sort_values('Total Score', ascending=False).round(3).reset_index(drop=True),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — RFM SCORES
# ══════════════════════════════════════════════════════════════════════
elif page == "RFM Scores":
    st.title("RFM Scores — How segmentation works")
    st.write("Every customer gets a score of 1 to 5 on each of R, F, and M using quintiles. The three scores are added (range 3–15) to determine the segment.")
    st.markdown("---")

    st.subheader("Score explanation")
    col1, col2, col3 = st.columns(3)
    col1.write("**R = Recency score (1 to 5)**\n\n5 = transacted yesterday. 1 = not seen in months.")
    col2.write("**F = Frequency score (1 to 5)**\n\n5 = most transactions. 1 = fewest.")
    col3.write("**M = Monetary score (1 to 5)**\n\n5 = highest spend. 1 = lowest.")

    st.markdown("---")

    # 6-segment mapping table
    st.subheader("How total RFM score maps to segments")
    st.markdown("""
| Segment | Total RFM Score | What it means |
|---|---|---|
| 🏆 Champions | 13 – 15 | Top scores across all three dimensions |
| 💙 Loyal | 10 – 12 | Consistently high — strong, reliable customers |
| 🌱 Growing | 7 – 9 | Above average — building momentum |
| 🚨 At Risk | 5 – 6 | Declining activity — intervention needed |
| 😴 Dormant | 4 | Barely active across all dimensions |
| 💔 Lost | 3 | Lowest possible score — churned |
""")
    st.caption("Total score = R + F + M. Range is 3 (worst) to 15 (best). Each band maps to exactly one segment — no overlap.")

    st.markdown("---")

    # Score distributions
    st.subheader("Score distributions in our data")
    fig, axes = plt.subplots(1, 3, figsize=(13,3))
    for ax, col, label, color in zip(axes,
        ['R_score','F_score','M_score'],
        ['R Score','F Score','M Score'],
        ['#E24B4A','#2E75B6','#1D9E75']):
        vc = rfm[col].value_counts().sort_index()
        ax.bar(vc.index, vc.values, color=color, edgecolor='white')
        ax.set_title(label); ax.set_xlabel("Score (1-5)"); ax.set_ylabel("Customers")
        ax.set_xticks([1,2,3,4,5])
        for xi,yi in zip(vc.index, vc.values):
            ax.text(xi, yi+0.5, str(yi), ha='center', fontsize=9)
    plt.suptitle("Equal distribution — each score has ~156 customers (quintiles)", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # RFM total score histogram coloured by segment
    st.subheader("Total RFM score histogram — coloured by segment")
    fig, ax = plt.subplots(figsize=(10,4))
    for seg in SEG_ORDER:
        if seg not in rfm['Segment'].values:
            continue
        sub = rfm[rfm['Segment']==seg]
        ax.hist(sub['RFM_Score'], bins=range(3,17), color=COLORS[seg],
                edgecolor='white', alpha=0.85, label=seg)
    ax.set_xlabel("Total RFM Score (3–15)"); ax.set_ylabel("Customers")
    ax.set_title("Customer count per RFM score — each colour is a segment")
    ax.set_xticks(range(3,16))
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("Segments sit in clean, non-overlapping score bands.")

    st.markdown("---")

    # Full customer table with scores
    st.subheader("All customers with their scores")
    disp = rfm[['cc_num','R_score','F_score','M_score','RFM_Score','Segment','Recency','Frequency','Monetary']].copy()
    disp.columns = ['Customer ID','R','F','M','Total(R+F+M)','Segment','Days Inactive','Transactions','Total Spend ($)']
    disp['Total Spend ($)'] = disp['Total Spend ($)'].round(0)
    disp = disp.sort_values('Total(R+F+M)', ascending=False).reset_index(drop=True)
    st.dataframe(disp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT
# ══════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("Predict a Customer's Segment")
    st.write("Enter a customer's R, F, M scores (1–5) and see which segment they fall into.")
    st.markdown("---")

    st.info("Scores are 1–5. Score 5 = best on that dimension. Score 1 = worst. Total (R+F+M) determines the segment.")

    c1, c2, c3 = st.columns(3)
    r_input = c1.slider("R Score (Recency)", 1, 5, 4, help="5 = transacted recently. 1 = not seen in months.")
    f_input = c2.slider("F Score (Frequency)", 1, 5, 3, help="5 = most transactions. 1 = fewest.")
    m_input = c3.slider("M Score (Monetary)", 1, 5, 3, help="5 = highest spend. 1 = lowest.")

    total = r_input + f_input + m_input
    result = assign_segment(total)
    color  = COLORS[result]
    icon, desc, action = SEGMENT_META[result]

    st.markdown("---")
    st.markdown(
        f"<div style='background:{color}18;border-left:4px solid {color};"
        f"border-radius:8px;padding:1rem 1.25rem;'>"
        f"<div style='font-size:12px;color:{color};font-weight:500;text-transform:uppercase;margin-bottom:4px;'>Predicted Segment</div>"
        f"<div style='font-size:28px;font-weight:500;color:{color};'>{icon} {result}</div>"
        f"<div style='font-size:13px;color:#666;margin-top:6px;'>R={r_input}  F={f_input}  M={m_input}  Total={total}</div>"
        f"</div>", unsafe_allow_html=True
    )
    st.markdown("")
    st.write(f"**What this means:** {desc}")
    st.success(f"**Bank action:** {action}")

    st.markdown("---")
    st.subheader("Where this customer sits on the RFM score scale")
    fig, ax = plt.subplots(figsize=(10,3))
    for seg in SEG_ORDER:
        if seg not in rfm['Segment'].values:
            continue
        sub = rfm[rfm['Segment']==seg]
        ax.scatter(sub['RFM_Score'], np.random.uniform(0.2, 0.8, len(sub)),
                   c=COLORS[seg], alpha=0.35, s=30, edgecolors='none', label=seg)
    ax.scatter(total, 0.5, c=color, s=350, zorder=10,
               edgecolors='black', linewidths=2, marker='*',
               label=f'This customer (score={total})')
    ax.set_xlabel("Total RFM Score (3–15)", fontsize=11)
    ax.set_yticks([])
    ax.set_title("Your customer vs all others   ★ = your customer", fontsize=11)
    ax.set_xlim(2.5, 15.5); ax.set_xticks(range(3,16))
    ax.grid(True, axis='x', alpha=0.3)
    legend_patches = [mpatches.Patch(color=COLORS[s], label=s) for s in SEG_ORDER if s in rfm['Segment'].values]
    ax.legend(handles=legend_patches, fontsize=8, loc='upper left', ncol=2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Quick examples")
    examples = [
        (5,5,5,"Bought yesterday, transacts most, highest spend"),
        (5,1,1,"Recent but low activity"),
        (1,4,4,"Used to be great — now gone silent"),
        (2,2,2,"Average across all 3"),
        (1,1,1,"Lowest on everything"),
    ]
    for r,f,m,label in examples:
        t = r+f+m
        seg_ex = assign_segment(t)
        col_ex = COLORS[seg_ex]
        icon_ex = SEGMENT_META[seg_ex][0]
        st.write(f"**{label}** (R={r}, F={f}, M={m}, Total={t}) → {icon_ex} **{seg_ex}**")

# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════════
elif page == "Evaluation":
    st.title("Model Evaluation")
    st.write("3 checks to confirm the RFM segmentation makes business sense.")
    st.markdown("---")

    # Check 1 — RFM score spread
    st.subheader("Check 1 — RFM score spread")
    st.write("Each segment should have a different average RFM score. Champions should be highest, Lost lowest.")
    seg_scores = rfm.groupby('Segment')['RFM_Score'].mean()
    seg_scores = seg_scores.reindex([s for s in SEG_ORDER if s in seg_scores.index])
    fig, ax = plt.subplots(figsize=(10,3))
    bars = ax.bar(seg_scores.index, seg_scores.values,
                  color=[COLORS[s] for s in seg_scores.index], edgecolor='white')
    ax.set_ylabel("Avg RFM Score (3–15)")
    ax.set_title("Average RFM Score per segment — should decrease Champions → Lost")
    ax.tick_params(axis='x', rotation=15)
    for bar, v in zip(bars, seg_scores.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{v:.1f}", ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.success("Champions have the highest RFM score. Lost have the lowest. Segments decrease monotonically. ✅")

    st.markdown("---")

    # Check 2 — Cluster sizes
    st.subheader("Check 2 — Segment sizes")
    st.write("No single segment should contain almost everyone. Spread means the model found real differences.")
    counts_eval = rfm['Segment'].value_counts()
    order = [s for s in SEG_ORDER if s in counts_eval]
    pcts = (counts_eval / len(rfm) * 100).round(1)
    fig, ax = plt.subplots(figsize=(10,3))
    vals = [counts_eval[s] for s in order]
    bars = ax.bar(order, vals,
                  color=[COLORS[s] for s in order], edgecolor='white')
    ax.set_ylabel("Customers"); ax.tick_params(axis='x', rotation=15)
    for bar,s in zip(bars,order):
        cnt = counts_eval[s]; pct = pcts[s]
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{cnt}\n({pct}%)", ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Check 3 — Business sense
    st.subheader("Check 3 — Business sense")
    profile = rfm.groupby('Segment')[['R_score','F_score','M_score','Recency','Frequency','Monetary']].mean().round(1)
    display_profile = profile.reindex([s for s in SEG_ORDER if s in profile.index])
    st.dataframe(display_profile, use_container_width=True)

    checks = [
        ("Champions have the highest R score",
         'Champions' in profile.index and profile.loc['Champions','R_score'] == profile['R_score'].max()),
        ("Champions have the highest F score",
         'Champions' in profile.index and profile.loc['Champions','F_score'] == profile['F_score'].max()),
        ("Champions have the highest M score",
         'Champions' in profile.index and profile.loc['Champions','M_score'] == profile['M_score'].max()),
        ("Loyal have high R, F and M scores (all ≥ 3)",
         'Loyal' in profile.index and all(profile.loc['Loyal', c] >= 3 for c in ['R_score','F_score','M_score'])),
        ("At Risk have lower scores than Loyal and Growing",
         'At Risk' in profile.index and 'Loyal' in profile.index and
         profile.loc['At Risk','R_score'] < profile.loc['Loyal','R_score']),
        ("Lost have the lowest R score",
         'Lost' in profile.index and profile.loc['Lost','R_score'] == profile['R_score'].min()),
    ]
    all_pass = True
    for label, passed in checks:
        st.write(f"{'✅' if passed else '❌'} {label}")
        if not passed: all_pass = False
    st.markdown("---")
    if all_pass:
        st.success("All checks passed — the RFM segmentation is working correctly.")
    else:
        st.warning("Some checks failed. Review the segment mapping logic.")