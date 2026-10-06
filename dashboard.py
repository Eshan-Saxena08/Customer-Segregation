import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

st.set_page_config(page_title="Customer Segments — RFM", layout="centered")

# ── 11-segment colour map (Putler methodology) ────────────────────────
COLORS = {
    'Champions':          '#1D9E75',
    'Loyal Customers':    '#2E75B6',
    'Potential Loyalist': '#27AE60',
    'New Customers':      '#3498DB',
    'Promising':          '#F39C12',
    'Needs Attention':    '#E67E22',
    'About To Sleep':     '#E74C3C',
    'At Risk':            '#C0392B',
    'Cannot Lose Them':   '#8E44AD',
    'Hibernating':        '#7F8C8D',
    'Lost':               '#95A5A6',
}

SEGMENT_META = {
    'Champions':          ('🏆', 'Bought recently, transact often, spend the most.',          'Reward them. Give early access to new products. They will refer others.'),
    'Loyal Customers':    ('💙', 'Spend well and transact often. Responsive to offers.',       'Upsell higher-value products. Ask for referrals. Keep them engaged.'),
    'Potential Loyalist': ('🌱', 'Active recently but frequency and spend are still growing.', 'Offer a loyalty program. Recommend products they have not tried yet.'),
    'New Customers':      ('🆕', 'Bought most recently but not yet frequent.',                  'Onboard well. Give them early wins. Build the relationship.'),
    'Promising':          ('⭐', 'Recent activity but spend is still low.',                    'Show them what they are missing. Build brand awareness.'),
    'Needs Attention':    ('⚠️',  'Average scores across all three dimensions.',              'Limited-time offers. Recommend based on past behaviour. Reactivate.'),
    'About To Sleep':     ('😴', 'Below average recency, frequency, and spend. Fading.',       'Send a win-back campaign. Offer a discount or bonus.'),
    'At Risk':            ('🚨', 'Used to transact a lot but have gone quiet recently.',       'Personalised outreach. Remind them of what they loved. Act now.'),
    'Cannot Lose Them':   ('🔒', 'Made big purchases often but disappeared completely.',       'Immediate personal contact. Top priority. Do not lose to a competitor.'),
    'Hibernating':        ('❄️',  'Last purchase was long ago. Low frequency and spend.',     'Try a relevant discount. Low investment — last attempt.'),
    'Lost':               ('💔', 'Lowest scores on all three dimensions.',                     'Minimal effort. One last campaign. Accept if no response.'),
}

# ── Segment classification using Putler's R + FM methodology ─────────
def assign_segment(r, fm):
    if   r >= 4 and fm >= 4:                     return 'Champions'
    elif r <= 1 and fm >= 4:                     return 'Cannot Lose Them'
    elif r <= 2 and fm >= 3:                     return 'At Risk'
    elif r >= 3 and fm >= 3:                     return 'Loyal Customers'
    elif r >= 4 and fm < 2:                      return 'New Customers'
    elif r == 3 and fm < 2:                      return 'Promising'
    elif 2 <= r <= 3 and 2 <= fm <= 3:           return 'Needs Attention'
    elif 2 <= r <= 3 and fm < 2:                 return 'About To Sleep'
    elif 1 <= r <= 2 and 1 <= fm <= 2:           return 'Hibernating'
    elif r <= 2 and fm < 1:                      return 'Lost'
    else:                                        return 'Hibernating'

@st.cache_data
def load_data():
    df = pd.read_csv('credit_card_transactions.csv')
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

    # FM combined (average, keeps 1-5 scale)
    rfm['FM_score'] = (rfm['F_score'] + rfm['M_score']) / 2
    rfm['Segment']  = rfm.apply(lambda r: assign_segment(r['R_score'], r['FM_score']), axis=1)
    return rfm

rfm = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.title("Pages")
page = st.sidebar.radio("", ["Overview","Segment Detail","RFM Scores","Predict","Evaluation"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Methodology**")
st.sidebar.caption("Based on Putler's 11-segment RFM framework.\nR score + FM combined score → segment.")
st.sidebar.markdown("**Segments found**")
counts = rfm['Segment'].value_counts()
for seg, cnt in counts.items():
    color = COLORS.get(seg,'#888')
    icon  = SEGMENT_META[seg][0]
    st.sidebar.markdown(f"<span style='color:{color}'>{icon} **{seg}**: {cnt}</span>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Customer Segmentation — RFM Analysis")
    st.write("781 customers scored on Recency, Frequency, and Monetary value and placed into segments using the Putler 11-segment methodology.")
    st.markdown("---")

    # Segment summary table
    st.subheader("All segments at a glance")
    rows = []
    for seg in sorted(rfm['Segment'].unique(), key=lambda s: -counts.get(s,0)):
        sub  = rfm[rfm['Segment']==seg]
        icon, desc, action = SEGMENT_META[seg]
        rows.append({
            "Segment":        f"{icon} {seg}",
            "Customers":      len(sub),
            "Median Recency": f"{sub['Recency'].median():.0f} days",
            "Median Frequency": f"{sub['Frequency'].median():,.0f} txns",
            "Median Spend":   f"${sub['Monetary'].median():,.0f}",
            "Avg per Txn":    f"${sub['Avg_Amt'].mean():,.0f}",
            "R Score":        f"{sub['R_score'].mean():.1f}",
            "FM Score":       f"{sub['FM_score'].mean():.1f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Segment"), use_container_width=True)

    st.markdown("---")

    # What each segment means
    st.subheader("What each segment means and what to do")
    for seg in sorted(rfm['Segment'].unique(), key=lambda s: -counts.get(s,0)):
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
    seg_order = counts.index.tolist()
    bar_colors = [COLORS.get(s,'#888') for s in seg_order]
    fig, ax = plt.subplots(figsize=(12,4))
    bars = ax.bar(seg_order, counts.values, color=bar_colors, edgecolor='white')
    ax.set_ylabel("Customers")
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(val), ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # RFM score vs segment scatter
    st.subheader("How segments sit on R vs FM space")
    fig, ax = plt.subplots(figsize=(10,6))
    for seg in rfm['Segment'].unique():
        sub = rfm[rfm['Segment']==seg]
        ax.scatter(sub['R_score'], sub['FM_score'],
                   c=COLORS.get(seg,'#888'), label=seg, alpha=0.65, s=40,
                   edgecolors='white', linewidths=0.3)
    ax.set_xlabel("R Score (1=least recent, 5=most recent)", fontsize=11)
    ax.set_ylabel("FM Score (avg of F+M, 1=lowest, 5=highest)", fontsize=11)
    ax.set_title("Customers plotted on R × FM space\nEach dot = one customer", fontsize=11)
    ax.set_xticks([1,2,3,4,5]); ax.set_yticks([1,2,3,4,5])
    ax.grid(True, alpha=0.3)
    legend_patches = [mpatches.Patch(color=COLORS.get(s,'#888'), label=s) for s in rfm['Segment'].unique()]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("Champions (top-right) have high R and high FM. Hibernating (bottom-left) have low scores on both.")

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENT DETAIL
# ══════════════════════════════════════════════════════════════════════
elif page == "Segment Detail":
    st.title("Segment Detail")
    st.markdown("---")

    available = sorted(rfm['Segment'].unique(), key=lambda s: -counts.get(s,0))
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
    st.write(f"🎯 **RFM Score (1-15):** avg={subset['RFM_Score'].mean():.1f}  |  R_score avg={subset['R_score'].mean():.1f}  |  FM_score avg={subset['FM_score'].mean():.1f}")
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
    order = sorted(rfm['Segment'].unique(), key=lambda s: -rfm[rfm['Segment']==s]['Monetary'].median())
    bc    = [COLORS.get(s,'#888') for s in order]
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
    st.caption(f"Black outline = {chosen}. Sorted by median spend (highest first).")

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
    st.write("Every customer gets a score of 1 to 5 on each of R, F, and M. The segment is determined by combining R score and FM score (average of F and M).")
    st.markdown("---")

    st.subheader("Score explanation")
    col1, col2, col3 = st.columns(3)
    col1.write("**R = Recency score (1 to 5)**\n\n5 = transacted yesterday. 1 = not seen in months.")
    col2.write("**F = Frequency score (1 to 5)**\n\n5 = most transactions. 1 = fewest.")
    col3.write("**M = Monetary score (1 to 5)**\n\n5 = highest spend. 1 = lowest.")

    st.markdown("---")

    # Putler segment mapping table
    st.subheader("How R and FM scores map to segments")
    st.markdown("""
| Segment | R Score | FM Score (avg of F+M) |
|---|---|---|
| 🏆 Champions | 4 – 5 | 4 – 5 |
| 💙 Loyal Customers | 3 – 5 | 3 – 5 |
| 🌱 Potential Loyalist | 4 – 5 | 1 – 2 |
| ⭐ Promising | 3 | 1 – 2 |
| ⚠️ Needs Attention | 2 – 3 | 2 – 3 |
| 😴 About To Sleep | 2 – 3 | 1 – 2 |
| 🚨 At Risk | 1 – 2 | 3 – 5 |
| 🔒 Cannot Lose Them | 1 | 4 – 5 |
| ❄️ Hibernating | 1 – 2 | 1 – 2 |
| 💔 Lost | 1 – 2 | 1 |
""")
    st.caption("FM score = average of F score and M score. Keeps scale 1–5.")

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

    # Full customer table with scores
    st.subheader("All customers with their scores")
    disp = rfm[['cc_num','R_score','F_score','M_score','FM_score','RFM_Score','Segment','Recency','Frequency','Monetary']].copy()
    disp.columns = ['Customer ID','R','F','M','FM','Total(R+F+M)','Segment','Days Inactive','Transactions','Total Spend ($)']
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

    st.info("Scores are 1–5. Score 5 = best on that dimension. Score 1 = worst.")

    c1, c2, c3 = st.columns(3)
    r_input  = c1.slider("R Score (Recency)", 1, 5, 4, help="5 = transacted recently. 1 = not seen in months.")
    f_input  = c2.slider("F Score (Frequency)", 1, 5, 3, help="5 = most transactions. 1 = fewest.")
    m_input  = c3.slider("M Score (Monetary)", 1, 5, 3, help="5 = highest spend. 1 = lowest.")

    fm = (f_input + m_input) / 2
    result = assign_segment(r_input, fm)
    color  = COLORS[result]
    icon, desc, action = SEGMENT_META[result]

    st.markdown("---")
    st.markdown(
        f"<div style='background:{color}18;border-left:4px solid {color};"
        f"border-radius:8px;padding:1rem 1.25rem;'>"
        f"<div style='font-size:12px;color:{color};font-weight:500;text-transform:uppercase;margin-bottom:4px;'>Predicted Segment</div>"
        f"<div style='font-size:28px;font-weight:500;color:{color};'>{icon} {result}</div>"
        f"<div style='font-size:13px;color:#666;margin-top:6px;'>R={r_input}  F={f_input}  M={m_input}  FM={fm:.1f}</div>"
        f"</div>", unsafe_allow_html=True
    )
    st.markdown("")
    st.write(f"**What this means:** {desc}")
    st.success(f"**Bank action:** {action}")

    st.markdown("---")
    st.subheader("Where this customer sits on the R × FM chart")
    fig, ax = plt.subplots(figsize=(8,6))
    for seg in rfm['Segment'].unique():
        sub = rfm[rfm['Segment']==seg]
        ax.scatter(sub['R_score'], sub['FM_score'],
                   c=COLORS.get(seg,'#888'), alpha=0.3, s=25, edgecolors='none')
    ax.scatter(r_input, fm, c=color, s=300, zorder=10,
               edgecolors='black', linewidths=2, marker='*',
               label=f'This customer (R={r_input}, FM={fm:.1f})')
    ax.set_xlabel("R Score"); ax.set_ylabel("FM Score")
    ax.set_xticks([1,2,3,4,5]); ax.set_yticks([1,2,3,4,5])
    ax.set_title("Your customer vs all others\n★ = your customer", fontsize=11)
    ax.grid(True, alpha=0.3)
    legend_patches = [mpatches.Patch(color=COLORS.get(s,'#888'), label=s) for s in rfm['Segment'].unique()]
    ax.legend(handles=legend_patches, fontsize=7, loc='upper left', ncol=2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Quick examples")
    examples = [
        (5,5,5,"Bought yesterday, transacts most, highest spend"),
        (5,1,1,"New or promising — recent but low activity"),
        (1,4,4,"Used to be great — now gone silent"),
        (2,2,2,"Average across all 3 — needs attention"),
        (1,1,1,"Lowest on everything — lost or hibernating"),
    ]
    for r,f,m,label in examples:
        fm_ex = (f+m)/2
        seg_ex = assign_segment(r, fm_ex)
        col_ex = COLORS[seg_ex]
        icon_ex = SEGMENT_META[seg_ex][0]
        st.write(f"**{label}** (R={r}, F={f}, M={m}) → {icon_ex} **{seg_ex}**")

# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════════
elif page == "Evaluation":
    st.title("Model Evaluation")
    st.write("3 checks to see if the RFM segmentation makes sense.")
    st.markdown("---")

    # Check 1 — RFM score spread
    st.subheader("Check 1 — RFM score spread")
    st.write("Each segment should have a different average RFM score. Champions should be highest, Lost/Hibernating lowest.")
    seg_scores = rfm.groupby('Segment')['RFM_Score'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,3))
    bars = ax.bar(seg_scores.index, seg_scores.values,
                  color=[COLORS.get(s,'#888') for s in seg_scores.index], edgecolor='white')
    ax.set_ylabel("Avg RFM Score (3–15)")
    ax.set_title("Average RFM Score per segment")
    ax.tick_params(axis='x', rotation=25)
    for bar, v in zip(bars, seg_scores.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{v:.1f}", ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.success("Champions have the highest RFM score. Hibernating/At Risk have the lowest. ✅")

    st.markdown("---")

    # Check 2 — Cluster sizes
    st.subheader("Check 2 — Segment sizes")
    st.write("No single segment should contain almost everyone. Spread means the model found real differences.")
    counts_eval = rfm['Segment'].value_counts()
    pcts = (counts_eval / len(rfm) * 100).round(1)
    fig, ax = plt.subplots(figsize=(10,3))
    bars = ax.bar(counts_eval.index, counts_eval.values,
                  color=[COLORS.get(s,'#888') for s in counts_eval.index], edgecolor='white')
    ax.set_ylabel("Customers"); ax.tick_params(axis='x', rotation=25)
    for bar,cnt,pct in zip(bars, counts_eval.values, pcts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{cnt}\n({pct}%)", ha='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Check 3 — Business sense
    st.subheader("Check 3 — Business sense")
    profile = rfm.groupby('Segment')[['R_score','F_score','M_score','Recency','Frequency','Monetary']].mean().round(1)
    st.dataframe(profile.sort_values('R_score', ascending=False), use_container_width=True)

    checks = [
        ("Champions have the highest R score",
         'Champions' in profile.index and profile.loc['Champions','R_score'] == profile['R_score'].max()),
        ("Champions have the highest F score",
         'Champions' in profile.index and profile.loc['Champions','F_score'] == profile['F_score'].max()),
        ("Champions have the highest M score",
         'Champions' in profile.index and profile.loc['Champions','M_score'] == profile['M_score'].max()),
        ("Loyal Customers have high FM but not necessarily high R",
         'Loyal Customers' in profile.index and profile.loc['Loyal Customers','FM_score' if 'FM_score' in profile.columns else 'F_score'] >= 3),
        ("At Risk have low R but decent FM (were good, now gone quiet)",
         'At Risk' in profile.index and profile.loc['At Risk','R_score'] <= 2),
        ("Hibernating have low scores across all dimensions",
         'Hibernating' in profile.index and profile.loc['Hibernating','R_score'] <= 2),
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