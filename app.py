"""
NeuroScope — AI-Powered Customer Intelligence Platform
Entry point: injects global CSS and configures multi-page navigation.
"""

import streamlit as st

st.set_page_config(
    page_title="NeuroScope | AI Customer Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg:       #050b18;
        --bg2:      #0d1626;
        --card:     rgba(13,22,38,0.60);
        --border:   rgba(255,255,255,0.06);
        --b-cyan:   rgba(0,212,255,0.22);
        --cyan:     #00d4ff;
        --purple:   #7c3aed;
        --coral:    #ff6b6b;
        --green:    #10b981;
        --amber:    #f59e0b;
        --text:     #e2e8f0;
        --muted:    #64748b;
        --r:        14px;
    }

    /* App shell */
    .stApp { background: var(--bg) !important; font-family: 'Inter', sans-serif !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg2) !important;
        border-right: 1px solid var(--border) !important;
    }

    /* Block container */
    .block-container { padding-top: 2rem !important; padding-bottom: 4rem !important; max-width: 1380px !important; }

    /* Headings */
    h1,h2,h3,h4,h5,h6 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; }

    /* ── Hero ── */
    .hero-wrap {
        position: relative; overflow: hidden;
        padding: 3rem 2.5rem; border-radius: 20px;
        background: linear-gradient(135deg,#050b18 0%,#0c1a2e 55%,#070d1a 100%);
        border: 1px solid rgba(0,212,255,0.14);
        margin-bottom: 2rem;
    }
    .hero-wrap::before {
        content:''; position:absolute; inset:-60%; width:220%; height:220%;
        background:
            radial-gradient(circle at 28% 55%,rgba(0,212,255,0.07) 0%,transparent 48%),
            radial-gradient(circle at 72% 22%,rgba(124,58,237,0.07) 0%,transparent 48%);
        animation: aurora 9s ease-in-out infinite alternate;
        pointer-events:none;
    }
    @keyframes aurora { 0%{transform:scale(1) rotate(0deg);} 100%{transform:scale(1.08) rotate(4deg);} }

    .hero-badge {
        display:inline-flex; align-items:center; gap:.4rem;
        background:rgba(0,212,255,0.09); border:1px solid rgba(0,212,255,0.28);
        border-radius:100px; padding:.3rem .85rem;
        font-size:.73rem; color:var(--cyan); font-weight:600;
        letter-spacing:.06em; margin-bottom:.9rem; width:fit-content;
    }
    .hero-title {
        font-family:'Space Grotesk',sans-serif;
        font-size:clamp(2.4rem,5vw,3.7rem); font-weight:700; line-height:1.08;
        background:linear-gradient(135deg,#fff 0%,#00d4ff 42%,#7c3aed 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
        margin:0 0 .7rem 0;
    }
    .hero-sub { font-size:1.05rem; color:var(--muted); max-width:580px; line-height:1.65; margin:0 0 1.4rem 0; }

    /* ── Stat cards ── */
    .stat-card {
        background:rgba(0,212,255,0.04); border:1px solid rgba(0,212,255,0.14);
        border-radius:var(--r); padding:1.2rem 1.5rem; text-align:center;
        transition:all .28s ease;
    }
    .stat-card:hover { border-color:rgba(0,212,255,0.38); background:rgba(0,212,255,0.08); transform:translateY(-2px); }
    .stat-num { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; color:var(--cyan); display:block; }
    .stat-lbl { font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-top:.2rem; }

    /* ── Feature cards ── */
    .feat-card {
        background:var(--card); border:1px solid var(--border); border-radius:var(--r);
        padding:1.4rem; transition:all .28s ease; height:100%;
    }
    .feat-card:hover { border-color:rgba(0,212,255,0.28); transform:translateY(-2px); box-shadow:0 8px 32px rgba(0,212,255,0.07); }
    .feat-icon { font-size:2rem; margin-bottom:.7rem; }
    .feat-title { font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:600; color:var(--text); margin-bottom:.35rem; }
    .feat-desc { font-size:.83rem; color:var(--muted); line-height:1.55; }

    /* ── Step cards ── */
    .step-wrap { display:flex; align-items:flex-start; gap:1rem; padding:1rem; border-radius:var(--r); background:var(--card); border:1px solid var(--border); margin-bottom:.7rem; }
    .step-num {
        width:32px; height:32px; border-radius:50%; flex-shrink:0;
        background:linear-gradient(135deg,var(--cyan),var(--purple));
        display:flex; align-items:center; justify-content:center;
        font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:.82rem; color:#fff;
    }

    /* ── Tech badges ── */
    .tech-badge {
        display:inline-flex; align-items:center; gap:.3rem;
        padding:.28rem .75rem; background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.09); border-radius:100px;
        font-size:.78rem; color:var(--text); font-weight:500;
    }

    /* ── Persona cards ── */
    .persona-card {
        background:linear-gradient(135deg,rgba(124,58,237,0.08),rgba(0,212,255,0.04));
        border:1px solid rgba(124,58,237,0.18); border-radius:16px;
        padding:1.5rem; text-align:center; transition:all .28s ease;
    }
    .persona-card:hover { border-color:rgba(124,58,237,0.45); transform:translateY(-3px); box-shadow:0 12px 40px rgba(124,58,237,0.12); }
    .persona-emoji { font-size:2.4rem; margin-bottom:.45rem; display:block; }
    .persona-name { font-family:'Space Grotesk',sans-serif; font-size:.98rem; font-weight:600; color:var(--text); }
    .persona-count { font-size:.78rem; color:var(--muted); margin-top:.18rem; }

    /* ── Winner banner ── */
    .winner-banner {
        background:linear-gradient(135deg,rgba(16,185,129,0.09),rgba(0,212,255,0.05));
        border:1px solid rgba(16,185,129,0.28); border-radius:12px;
        padding:.9rem 1.4rem; margin:1rem 0;
        display:flex; align-items:center; gap:1rem;
        animation:bannerIn .45s ease both;
    }
    @keyframes bannerIn { from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:none;} }

    /* Respect reduced-motion preferences */
    @media (prefers-reduced-motion: reduce) {
        .hero-wrap::before, .winner-banner, .stat-card, .feat-card, .persona-card,
        [data-testid="stMetric"] { animation:none!important; transition:none!important; }
    }

    /* ── Section titles with accent bar ── */
    .section-title {
        display:flex; align-items:center; gap:.6rem;
        font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:600; color:var(--text);
        margin:1.4rem 0 .2rem 0;
    }
    .section-title::before {
        content:''; width:4px; height:1.15rem; border-radius:2px;
        background:linear-gradient(180deg,var(--cyan),var(--purple)); flex-shrink:0;
    }

    /* ── Page sub-header (shared by all pages) ── */
    .page-head { margin-bottom:1.6rem; }
    .page-head h1 {
        font-family:'Space Grotesk',sans-serif; font-size:2.2rem; font-weight:700;
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
        margin:.45rem 0 .3rem 0;
    }
    .page-head p { color:var(--muted); font-size:.97rem; margin:0; max-width:680px; }

    /* ── Streamlit overrides ── */
    .stTabs [data-baseweb="tab-list"] { background:var(--bg2)!important; border-radius:10px!important; gap:4px; padding:4px; }
    .stTabs [data-baseweb="tab"] { border-radius:8px!important; color:var(--muted)!important; font-family:'Inter',sans-serif!important; padding:.4rem 1rem!important; transition:all .18s ease!important; }
    .stTabs [data-baseweb="tab"]:hover { color:var(--text)!important; background:rgba(255,255,255,0.04)!important; }
    .stTabs [aria-selected="true"] { background:linear-gradient(135deg,rgba(0,212,255,0.14),rgba(124,58,237,0.14))!important; color:var(--cyan)!important; }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none!important; }

    .stButton>button { border-radius:10px!important; font-family:'Inter',sans-serif!important; font-weight:600!important; transition:all .2s ease!important; }
    .stButton>button[kind="primary"] { background:linear-gradient(135deg,var(--cyan),var(--purple))!important; border:none!important; color:#fff!important; }
    .stButton>button[kind="primary"]:hover { transform:translateY(-1px)!important; box-shadow:0 8px 24px rgba(0,212,255,0.28)!important; }
    .stButton>button[kind="primary"]:active { transform:translateY(0)!important; box-shadow:none!important; }
    .stDownloadButton>button { border-radius:10px!important; font-weight:600!important; transition:all .2s ease!important; }

    [data-testid="stMetric"] {
        background:var(--card); border:1px solid var(--border); border-radius:var(--r);
        padding:.9rem 1.1rem; transition:all .22s ease;
    }
    [data-testid="stMetric"]:hover { border-color:rgba(0,212,255,0.30); transform:translateY(-2px); }
    [data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif!important; color:var(--cyan)!important; }
    [data-testid="stMetricLabel"] { color:var(--muted)!important; font-size:.8rem!important; }

    .stSuccess { background:rgba(16,185,129,0.09)!important; border:1px solid rgba(16,185,129,0.28)!important; border-radius:10px!important; }
    .stError   { background:rgba(255,107,107,0.09)!important; border:1px solid rgba(255,107,107,0.28)!important; border-radius:10px!important; }
    .stInfo    { background:rgba(0,212,255,0.07)!important;  border:1px solid rgba(0,212,255,0.22)!important;  border-radius:10px!important; }
    .stWarning { background:rgba(245,158,11,0.09)!important; border:1px solid rgba(245,158,11,0.28)!important; border-radius:10px!important; }

    .stSelectbox>div>div, .stMultiSelect>div>div { border-radius:10px!important; background:var(--bg2)!important; border-color:var(--border)!important; }
    [data-testid="stDataFrame"] { border-radius:10px!important; overflow:hidden; }

    /* Sidebar polish */
    [data-testid="stSidebar"] hr { margin:0.6rem 0!important; }
    [data-testid="stSidebar"] h3 { font-size:0.86rem!important; text-transform:uppercase; letter-spacing:.09em; color:var(--muted)!important; margin-bottom:.4rem!important; }

    /* Nav links → pill cards */
    [data-testid="stSidebarNav"] { gap:.3rem!important; }
    [data-testid="stSidebarNav"] a {
        background:transparent; border:1px solid transparent; border-radius:10px;
        padding:.5rem .75rem!important; margin:.1rem .25rem;
        color:var(--muted)!important; font-size:.9rem!important; font-weight:500;
        transition:all .18s ease;
    }
    [data-testid="stSidebarNav"] a:hover {
        background:rgba(255,255,255,0.04); border-color:var(--border); color:var(--text)!important;
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background:linear-gradient(135deg,rgba(0,212,255,0.12),rgba(124,58,237,0.12));
        border-color:rgba(0,212,255,0.30); color:var(--cyan)!important; font-weight:600;
    }
    [data-testid="stSidebarNav"] a span { overflow:visible!important; }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background:var(--card); border:1px solid var(--border); border-radius:10px;
        padding:.45rem .7rem; margin-bottom:.35rem; transition:all .18s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { border-color:rgba(0,212,255,0.35); }

    /* Expander polish */
    details[data-testid="stExpander"] {
        background:var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden;
    }
    details[data-testid="stExpander"][open] { border-color:rgba(0,212,255,0.22); }

    /* Spinner accent */
    .stSpinner>div { border-top-color:var(--cyan)!important; }

    hr { border-color:var(--border)!important; }

    ::-webkit-scrollbar { width:5px; height:5px; }
    ::-webkit-scrollbar-track { background:var(--bg2); }
    ::-webkit-scrollbar-thumb { background:rgba(0,212,255,0.28); border-radius:3px; }
    ::-webkit-scrollbar-thumb:hover { background:rgba(0,212,255,0.5); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Navigation ────────────────────────────────────────────────────────────────
pg = st.navigation(
    [
        st.Page("pages/overview.py", title="Overview", icon="📊", default=True),
        st.Page("pages/churn_prediction.py", title="Churn Prediction", icon="🔮"),
        st.Page("pages/segmentation.py", title="Customer Segmentation", icon="🧩"),
        st.Page("pages/ai_insights.py", title="AI Insights", icon="🤖"),
    ]
)

# Sidebar branding
with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700;
                    background:linear-gradient(135deg,#00d4ff,#7c3aed);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🧠 NeuroScope
        </div>
        <div style="font-size:0.72rem; color:#64748b; margin-top:0.2rem; letter-spacing:.05em;">
            AI CUSTOMER INTELLIGENCE
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.divider()

pg.run()
