"""Visual layer for the dashboard: global CSS, module headers, cards and footer.

Everything here is presentation only; the tabs keep using plain Streamlit widgets so
behaviour (and the AppTest smoke tests) does not depend on the styling.
"""

from __future__ import annotations

from contextlib import contextmanager
from html import escape

import streamlit as st

from src.style import OKABE_ITO

REPO_URL = "https://github.com/TBark5/pandemica"

# One accent per module, taken from the Okabe-Ito palette used by every figure.
MODULES = {
    "m1": ("Compartmental", "How does an epidemic unfold in a well-mixed population?",
           "SIR / SEIR / SEIRD ODEs, vaccination, waning immunity", OKABE_ITO["blue"], "📈"),
    "m2": ("Fitting", "What R0 do real outbreak data imply, and how sure are we?",
           "Least squares on real flu data, bootstrap CIs", OKABE_ITO["vermillion"], "🎯"),
    "m3": ("Stochastic", "What does chance do when case numbers are small?",
           "Gillespie algorithm, extinction probability", OKABE_ITO["green"], "🎲"),
    "m4": ("Network", "How does who-contacts-whom change the epidemic?",
           "Erdos-Renyi / Watts-Strogatz / Barabasi-Albert, targeted vaccination",
           OKABE_ITO["orange"], "🕸️"),
    "m5": ("Interventions", "How do intervention timing and strength change outcomes?",
           "Lockdown, vaccination, test-and-isolate counterfactuals", OKABE_ITO["purple"], "🛡️"),
    "m6": ("Sensitivity", "Which parameters matter most?",
           "Latin hypercube sampling + partial rank correlation", OKABE_ITO["sky"], "🎚️"),
    "m7": ("Spatial", "How does an epidemic travel between regions?",
           "8-region metapopulation SEIR with gravity-model travel", "#B59F00", "🗺️"),
}

_CSS = """
<style>
/* ---------- page frame ---------- */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* ---------- tabs as a pill bar ---------- */
[data-testid="stTabs"] [role="tablist"] {
  gap: 4px; padding: 5px; background: #fff; border: 1px solid #E1E6EE; border-bottom-width: 1px;
  border-radius: 999px; width: fit-content; max-width: 100%; overflow-x: auto;
  box-shadow: 0 1px 2px rgba(16,32,46,.04); margin-bottom: 1.4rem;
}
[data-testid="stTabs"] [role="tab"] {
  padding: 7px 16px; border-radius: 999px; white-space: nowrap; cursor: pointer;
  transition: background .15s ease, color .15s ease;
}
[data-testid="stTabs"] [role="tab"] p { font-size: .87rem; font-weight: 600; color: #3A4A5E; }
[data-testid="stTabs"] [role="tab"]:hover { background: #EEF4FA; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: #0B3C5D; box-shadow: 0 4px 10px -4px rgba(11,60,93,.6);
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color: #fff; }
[data-testid="stTabs"] .react-aria-SelectionIndicator { display: none; }

/* ---------- hero ---------- */
.st-key-hero {
  position: relative; overflow: hidden; padding: 2.4rem 2.4rem 1.6rem;
  border-radius: 22px; color: #EAF2FA;
  background:
    radial-gradient(900px 380px at 88% -10%, rgba(86,180,233,.35), transparent 60%),
    radial-gradient(600px 300px at 0% 110%, rgba(213,94,0,.28), transparent 60%),
    linear-gradient(135deg, #0B2239 0%, #0B3C5D 55%, #0F4C75 100%);
  box-shadow: 0 18px 40px -18px rgba(11,34,57,.55);
}
.st-key-hero::after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: radial-gradient(rgba(255,255,255,.09) 1px, transparent 1px);
  background-size: 22px 22px;
  mask-image: linear-gradient(to left, #000 10%, transparent 70%);
}
.st-key-hero h1 {
  margin: .35rem 0 .5rem; color: #fff; font-size: 3.1rem; font-weight: 800; letter-spacing: -0.035em;
  padding: 0; line-height: 1.05;
}
.st-key-hero h1 a { display: none; }
.st-key-hero p { color: #C9DAEA; }
.hero-kicker {
  display: inline-flex; align-items: center; gap: 8px; font-size: .78rem; font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase; color: #9FD3F2;
}
.hero-kicker .dot {
  width: 8px; height: 8px; border-radius: 50%; background: #E69F00;
  box-shadow: 0 0 0 4px rgba(230,159,0,.22);
}
.hero-lead { font-size: 1.12rem; line-height: 1.6; max-width: 760px; margin: .2rem 0 .4rem; }
.hero-lead b { color: #fff; }
.hero-note {
  display: inline-block; font-size: .82rem; color: #FFD9A8; padding: 5px 12px;
  border: 1px solid rgba(255,217,168,.35); border-radius: 12px; background: rgba(230,159,0,.1);
}
.st-key-hero [data-testid="stMetric"] {
  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14);
  backdrop-filter: blur(6px); box-shadow: none;
}
.st-key-hero [data-testid="stMetricLabel"] p { color: #A9C4DC; }
.st-key-hero [data-testid="stMetricValue"] { color: #fff; }
.st-key-hero [data-testid="stCaptionContainer"] p { color: #8FB0CC; }

/* ---------- metric cards ---------- */
[data-testid="stMetric"] {
  background: #fff; border: 1px solid #E1E6EE; border-radius: 14px;
  padding: 14px 18px 12px; box-shadow: 0 1px 2px rgba(16,32,46,.04);
}
[data-testid="stMetricLabel"] p {
  font-size: .82rem; font-weight: 600; color: #5B6B7F; white-space: normal;
}
[data-testid="stMetricLabel"] div { overflow: visible; white-space: normal; }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.02em; font-size: 1.85rem; }

/* ---------- module header ---------- */
.mod-head { display: flex; gap: 16px; align-items: center; margin: .2rem 0 1.3rem; }
.mod-badge {
  flex: none; width: 54px; height: 54px; border-radius: 16px; display: grid; place-items: center;
  font-weight: 800; font-size: 1.05rem; color: #fff; background: var(--accent);
  box-shadow: 0 8px 18px -8px var(--accent);
}
.mod-kicker { font-size: .74rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .08em; color: var(--accent); }
.mod-title { font-size: 1.65rem; font-weight: 800; letter-spacing: -0.025em; color: #16202E;
  line-height: 1.2; }
.mod-q { color: #5B6B7F; font-size: .98rem; margin-top: 2px; }

/* ---------- parameter panel ---------- */
div[class*="st-key-controls"] {
  background: #fff; border: 1px solid #E1E6EE; border-radius: 16px;
  padding: 16px 16px 6px; box-shadow: 0 1px 2px rgba(16,32,46,.04);
}
.panel-title { display: flex; align-items: center; gap: 8px; font-size: .78rem; font-weight: 700;
  letter-spacing: .07em; text-transform: uppercase; color: #5B6B7F; margin-bottom: 2px; }
.panel-title::before { content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent, #0072B2); }

/* ---------- plots and images as cards ---------- */
[data-testid="stImage"], [data-testid="stImageContainer"] {
  background: #fff; border: 1px solid #E1E6EE; border-radius: 16px; padding: 10px;
  box-shadow: 0 1px 2px rgba(16,32,46,.04);
}
[data-testid="stImage"] img { border-radius: 10px; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-testid="stCaptionContainer"] p { color: #66768A; }

/* ---------- module cards on the overview ---------- */
.section-title { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em;
  margin: 2.2rem 0 .2rem; color: #16202E; }
.section-sub { color: #5B6B7F; margin-bottom: 1rem; }
.mod-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 14px; }
a.mod-card {
  position: relative; display: block; text-decoration: none !important; color: inherit !important;
  background: #fff; border: 1px solid #E1E6EE; border-radius: 16px; padding: 18px 18px 16px;
  box-shadow: 0 1px 2px rgba(16,32,46,.04); overflow: hidden;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
a.mod-card::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 4px;
  background: var(--accent); }
a.mod-card:hover { transform: translateY(-3px); border-color: var(--accent);
  box-shadow: 0 14px 28px -16px rgba(16,32,46,.35); }
.mod-card .top { display: flex; align-items: center; justify-content: space-between; }
.mod-card .code { font-size: .75rem; font-weight: 800; color: var(--accent);
  letter-spacing: .06em; }
.mod-card .icon { font-size: 1.35rem; }
.mod-card .name { font-weight: 800; font-size: 1.1rem; margin: 6px 0 4px; color: #16202E; }
.mod-card .q { color: #33445A; font-size: .92rem; line-height: 1.45; }
.mod-card .method { margin-top: 10px; font-size: .8rem; color: #66768A; }
.mod-card .go { margin-top: 12px; font-size: .82rem; font-weight: 700; color: var(--accent); }

.back-link a { font-size: .88rem; font-weight: 600; text-decoration: none; color: #0072B2 !important; }

/* ---------- footer ---------- */
.footer { margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid #E1E6EE;
  display: flex; flex-wrap: wrap; gap: 8px 20px; justify-content: space-between;
  font-size: .82rem; color: #7A8899; }
.footer a { color: #0072B2; text-decoration: none; font-weight: 600; }

@media (max-width: 640px) {
  .st-key-hero { padding: 1.6rem 1.2rem 1rem; }
  .st-key-hero h1 { font-size: 2.2rem; }
  .mod-title { font-size: 1.3rem; }
}
</style>
"""


def inject_css() -> None:
    """Add the dashboard stylesheet to the page (call once per run)."""
    st.html(_CSS)


def module_header(key: str, subtitle: str | None = None) -> None:
    """Coloured badge + title + guiding question for one module tab."""
    name, question, _method, accent, _icon = MODULES[key]
    code = key.upper()
    st.html(
        f'<div class="mod-head" style="--accent:{accent}">'
        f'<div class="mod-badge">{code}</div><div>'
        f'<div class="mod-kicker">Module {code[1:]}{" · " + escape(subtitle) if subtitle else ""}'
        f'</div><div class="mod-title">{escape(name)}</div>'
        f'<div class="mod-q">{escape(question)}</div></div></div>'
    )


@contextmanager
def controls(col, key: str, title: str = "Parameters"):
    """A white card holding a tab's sliders, placed inside column ``col``."""
    with col:
        box = st.container(key=f"controls_{key}")
        with box:
            st.html(f'<div class="panel-title" style="--accent:{MODULES[key][3]}">'
                    f'{escape(title)}</div>')
            yield box


def module_cards() -> None:
    """Grid of clickable cards, one per module (deep links to ?tab=mX)."""
    cards = "".join(
        f'<a class="mod-card" href="?tab={key}" target="_self" style="--accent:{accent}">'
        f'<div class="top"><span class="code">{key.upper()}</span>'
        f'<span class="icon">{icon}</span></div>'
        f'<div class="name">{escape(name)}</div><div class="q">{escape(question)}</div>'
        f'<div class="method">{escape(method)}</div><div class="go">Open module &rarr;</div></a>'
        for key, (name, question, method, accent, icon) in MODULES.items()
    ) + (
        f'<a class="mod-card" href="{REPO_URL}#what-is-inside" target="_blank" '
        'style="--accent:#5B6B7F"><div class="top"><span class="code">EXTRAS</span>'
        '<span class="icon">🧪</span></div><div class="name">Extensions</div>'
        '<div class="q">Bayesian MCMC fit with a negative-binomial likelihood, and an '
        'age-structured SEIR comparing vaccine-priority strategies.</div>'
        '<div class="method">emcee, contact matrices, next-generation R0</div>'
        '<div class="go">Read more &rarr;</div></a>'
    )
    st.html(
        '<div class="section-title">Seven modules, one outbreak toolkit</div>'
        '<div class="section-sub">Each module has live sliders. Use the tabs above, or open a '
        'module on its own page.</div>'
        f'<div class="mod-grid">{cards}</div>'
    )


def back_link() -> None:
    st.html('<div class="back-link"><a href="?" target="_self">&larr; All modules</a></div>')


def footer() -> None:
    st.html(
        '<div class="footer"><span>PANDEMICA · simplified educational models, not forecasts '
        'or policy advice.</span>'
        f'<span><a href="{REPO_URL}" target="_blank">Source on GitHub</a> · '
        f'<a href="{REPO_URL}/blob/main/METHODS.md" target="_blank">Methods</a></span></div>'
    )
