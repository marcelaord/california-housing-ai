import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    FEATURES,
    FEATURE_LABELS,
    LINEAR_DEGREE,
    POLYNOMIAL_DEGREES,
    RANDOM_STATE,
    WineSimulator,
)

st.set_page_config(
    page_title="Simulador Calidad de Vino",
    page_icon="🍷",
    layout="wide",
)

_BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_BASE, "assets", "style.css"), encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

MODEL_OPTIONS = [("Regresión Lineal Múltiple", LINEAR_DEGREE)] + [
    (f"Regresión Polinomial · grado {d}", d) for d in POLYNOMIAL_DEGREES
]
OPTION_NAMES = [name for name, _ in MODEL_OPTIONS]

COLOR_LINEAL = "#2e9e5b"
COLOR_POLI = "#e67e22"
COLOR_VINO = "#722F37"
MODEL_1 = "Regresión Lineal Múltiple"
MODEL_2 = None  # se fija según el grado elegido

VINO_GROUPS = [
    ("🍇 Perfil ácido", ["fixed acidity", "volatile acidity", "citric acid", "pH"]),
    ("🍬 Azúcar y cuerpo", ["residual sugar", "density"]),
    ("🧂 Sabor mineral", ["chlorides", "sulphates"]),
    ("🌬️ Frescura (SO₂)", ["free sulfur dioxide", "total sulfur dioxide"]),
    ("🍷 Alcohol", ["alcohol"]),
]

FEATURE_ICONS = {
    "fixed acidity": "🍋",
    "volatile acidity": "🌶️",
    "citric acid": "🍊",
    "residual sugar": "🍬",
    "chlorides": "🧂",
    "free sulfur dioxide": "💨",
    "total sulfur dioxide": "🌫️",
    "density": "⚖️",
    "pH": "🔬",
    "sulphates": "🧪",
    "alcohol": "🍷",
}


@st.cache_resource
def load_simulator():
    return WineSimulator()


sim = load_simulator()

QUALITY_LABELS = [
    (8.0, "Excelente", "★★★★★"),
    (7.0, "Muy buena", "★★★★☆"),
    (6.0, "Buena", "★★★☆☆"),
    (5.0, "Aceptable", "★★☆☆☆"),
    (4.0, "Regular", "★☆☆☆☆"),
    (0.0, "Mala", "☆☆☆☆☆"),
]


def quality_badge(pred):
    for threshold, label, stars in QUALITY_LABELS:
        if pred >= threshold:
            return label, stars
    return "Mala", "☆☆☆☆☆"


def quality_color(pred, lo=3.0, hi=8.0):
    t = max(0.0, min(1.0, (pred - lo) / (hi - lo)))
    if t < 0.5:
        u = t * 2
        r = 231 + (241 - 231) * u
        g = 76 + (196 - 76) * u
        b = 60 + (15 - 60) * u
    else:
        u = (t - 0.5) * 2
        r = 241 + (39 - 241) * u
        g = 196 + (174 - 196) * u
        b = 15 + (96 - 15) * u
    return f"rgb({int(r)},{int(g)},{int(b)})"


# ---------------------------------------------------------------------------
# Barra lateral: selección del modelo y datos del dataset
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🍷 Configuración")
    choice = st.selectbox("Modelo a utilizar para predecir", OPTION_NAMES, key="modelo")
    degree = dict(MODEL_OPTIONS)[choice]

    cmp_degree = degree if degree != LINEAR_DEGREE else POLYNOMIAL_DEGREES[0]
    MODEL_2 = f"Regresión Polinomial · grado {cmp_degree}"

    st.markdown("---")
    st.subheader("📊 Dataset utilizado")
    st.write(
        f"- Filas: **{len(sim.df)}**\n"
        f"- Features: **{len(FEATURES)}**\n"
        f"- Target: **{sim.y.name}** (3 – {int(sim.y.max())})\n"
        f"- Train/Test: **80/20** (`random_state={RANDOM_STATE}`)\n"
        "- Datos 100% reales de `WineQT.csv` (sin datos sintéticos)"
    )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:14px;">
        <div>
          <h1>🍷 Simulador de Calidad de Vino</h1>
          <p class="hero-sub">Regresión Lineal Múltiple vs Regresión Polinomial ·
          Modelado Predictivo · IA / Aprendizaje Automático · Caso Wine Quality</p>
        </div>
        <div class="hero-glass">🍷</div>
      </div>
    </div>
    <div class="wine-divider">🍇 🍷 🍾</div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sección 1: Simula tu vino
# ---------------------------------------------------------------------------
st.markdown("## 🧪 Simula tu vino")
st.caption("Ajusta las propiedades físico-químicas y mira la nota de calidad que predice cada modelo.")

ranges = sim.feature_ranges()
defaults = sim.X.mean().to_dict()

inputs = {}
with st.expander("✨ Valores físico-químicos de tu vino", expanded=True):
    for title, feats in VINO_GROUPS:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            cols = st.columns(2)
            for i, f in enumerate(feats):
                lo, hi = ranges[f]
                step = 0.0001 if f == "density" else 0.01
                fmt = "%.4f" if f == "density" else "%.2f"
                with cols[i % 2]:
                    inputs[f] = st.slider(
                        f"{FEATURE_ICONS[f]} {FEATURE_LABELS[f]}",
                        min_value=float(lo),
                        max_value=float(hi),
                        value=float(defaults[f]),
                        step=step,
                        format=fmt,
                        key=f"slider_{f}",
                    )

pred_sel = sim.predict(inputs, degree)
pred_cmp = sim.predict(inputs, cmp_degree)

label_sel, stars_sel = quality_badge(pred_sel)
label_cmp, stars_cmp = quality_badge(pred_cmp)
color_sel = quality_color(pred_sel)

sel_name = f"📈 {choice}"

if degree == LINEAR_DEGREE:
    linear_pred, linear_label, linear_stars = pred_sel, label_sel, stars_sel
    poly_pred, poly_label, poly_stars = pred_cmp, label_cmp, stars_cmp
    linear_role, poly_role = "Modelo seleccionado", "Modelo comparado"
else:
    linear_pred, linear_label, linear_stars = pred_cmp, label_cmp, stars_cmp
    poly_pred, poly_label, poly_stars = pred_sel, label_sel, stars_sel
    linear_role, poly_role = "Modelo comparado", "Modelo seleccionado"

st.markdown("---")

col_nota, col_modelos = st.columns([1, 1.25])

with col_nota:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, {color_sel}, {color_sel}cc);
                    color:#1a1a1a; border-radius:18px; padding:26px 18px; text-align:center;
                    box-shadow:0 8px 20px rgba(0,0,0,.18);">
          <div style="font-size:17px; opacity:.95;">💯 Nota de calidad estimada</div>
          <div style="font-size:74px; font-weight:800; line-height:1.1; color:#1a1a1a;">
            {round(pred_sel)}<span style="font-size:30px;">/10</span>
          </div>
          <div style="font-size:26px; font-weight:700;">{label_sel}</div>
          <div style="font-size:28px; letter-spacing:5px; margin-top:8px; color:#ffd700;">{stars_sel}</div>
          <div style="font-size:14px; opacity:.9; margin-top:10px;">
            {sel_name} · valor real predicho {pred_sel:.2f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_modelos:
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f"""
            <div style="border:2px solid {COLOR_LINEAL}55; border-radius:14px; padding:14px 12px;
                        text-align:center; background:#f7fbf8;">
              <div style="font-weight:700; color:{COLOR_LINEAL}; font-size:16px;">📈 Lineal Múltiple</div>
              <div style="font-size:34px; font-weight:800; color:{COLOR_LINEAL}; margin:6px 0;">
                {linear_pred:.2f}
              </div>
              <div style="background:#eee; border-radius:6px; height:10px; overflow:hidden;">
                <div style="background:{COLOR_LINEAL}; height:10px; width:{max(0, min(100, (linear_pred - 3) / 5 * 100)):.1f}%;"></div>
              </div>
              <div style="font-size:14px; margin-top:8px;"><b>{linear_label}</b></div>
              <div style="font-size:12px; color:#999;">{linear_role}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div style="border:2px solid {COLOR_POLI}55; border-radius:14px; padding:14px 12px;
                        text-align:center; background:#fdf8f3;">
              <div style="font-weight:700; color:{COLOR_POLI}; font-size:16px;">🎢 {MODEL_2}</div>
              <div style="font-size:34px; font-weight:800; color:{COLOR_POLI}; margin:6px 0;">
                {poly_pred:.2f}
              </div>
              <div style="background:#eee; border-radius:6px; height:10px; overflow:hidden;">
                <div style="background:{COLOR_POLI}; height:10px; width:{max(0, min(100, (poly_pred - 3) / 5 * 100)):.1f}%;"></div>
              </div>
              <div style="font-size:14px; margin-top:8px;"><b>{poly_label}</b></div>
              <div style="font-size:12px; color:#999;">{poly_role}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    delta = pred_sel - pred_cmp
    if abs(delta) < 0.05:
        mejor = "🧘 Ambas regresiones predicen lo mismo para este vino."
        tono = "#6c757d"
    elif delta > 0:
        mejor = f"🏆 La {choice} sugiere una nota {delta:+.2f} mayor. ¡Apuesta por el {choice.lower()}!"
        tono = COLOR_POLI if degree != LINEAR_DEGREE else COLOR_LINEAL
    else:
        mejor = f"🏆 La {MODEL_2} sugiere una nota {delta:+.2f} mejor. ¡Apuesta por la {MODEL_2}!"
        tono = COLOR_POLI if degree == LINEAR_DEGREE else COLOR_LINEAL

    st.markdown(
        f"""
        <div style="border-radius:12px; padding:12px 16px; margin-top:14px;
                    background:#fff8e1; border:1px solid #f0c93d; color:#7a5b00; font-size:15px;">
          {mejor}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "La nota sale del modelo seleccionado. Compara siempre con la otra regresión para ver cuál se acerca mejor."
    )

# ---------------------------------------------------------------------------
# Sección 2: La diferencia se ve en la curva (Plotly)
# ---------------------------------------------------------------------------
st.markdown("## 🎢 ¿Cómo cambia la nota según una variable?")
st.caption(
    "Elige una variable y mira cómo se mueve la nota. La **Lineal sube en línea recta**; "
    "la **Polinomial se dobla** según el valor. La franja gris es la zona donde las dos opiniones difieren."
)

vary_label = st.selectbox(
    "Variable a explorar (el resto de features quedan como están en los sliders)",
    [FEATURE_LABELS[f] for f in FEATURES],
    index=FEATURES.index("alcohol"),
    key="vary_feature",
)
vary = FEATURES[[FEATURE_LABELS[f] for f in FEATURES].index(vary_label)]

curves = sim.response_curves(inputs, vary, [LINEAR_DEGREE, cmp_degree])
col_lin = f"pred_{LINEAR_DEGREE}"
col_pol = f"pred_{cmp_degree}"

fig_curves = go.Figure()
band_x = curves[vary].to_numpy()
y_top = np.maximum(curves[col_lin].to_numpy(), curves[col_pol].to_numpy())
y_bot = np.minimum(curves[col_lin].to_numpy(), curves[col_pol].to_numpy())
fig_curves.add_trace(
    go.Scatter(x=band_x, y=y_top, mode="lines", line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False)
)
fig_curves.add_trace(
    go.Scatter(
        x=band_x,
        y=y_bot,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(127,140,141,.13)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
    )
)

for name, col, color, dash in [
    (MODEL_1, col_lin, COLOR_LINEAL, "solid"),
    (MODEL_2, col_pol, COLOR_POLI, "dash"),
]:
    fig_curves.add_trace(
        go.Scatter(
            x=curves[vary],
            y=curves[col],
            mode="lines",
            name=name,
            line=dict(color=color, width=3, dash=dash),
            hovertemplate=f"<b>{name}</b><br>{FEATURE_LABELS[vary]}: %{{x:.2f}}<br>Nota: %{{y:.2f}}<extra></extra>",
        )
    )

for name, val, color, sym in [
    (MODEL_1, pred_cmp, COLOR_LINEAL, "triangle-up"),
    (MODEL_2, pred_sel, COLOR_POLI, "diamond"),
]:
    fig_curves.add_trace(
        go.Scatter(
            x=[inputs[vary]],
            y=[val],
            mode="markers",
            name=name,
            marker=dict(size=13, color=color, symbol=sym, line=dict(width=1.5, color="#ffffff")),
            showlegend=False,
            hovertemplate=f"<b>{name}</b><br>Tu vino: %{{x:.2f}}<br>Nota: %{{y:.2f}}<extra></extra>",
        )
    )

fig_curves.add_vline(
    x=inputs[vary],
    line_dash="dash",
    line_color="#c0392b",
    annotation_text="Tu vino",
    annotation_position="top left",
    annotation_font=dict(size=12, color="#c0392b"),
)
fig_curves.update_layout(
    height=440,
    margin=dict(l=40, r=20, t=30, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Lato", color="#3B2B20", size=13),
    xaxis=dict(title=f"Valor de la variable ({FEATURE_LABELS[vary]})", gridcolor="#f0e4da"),
    yaxis=dict(title="Nota de calidad predicha", range=[3, 8], gridcolor="#f0e4da"),
    legend=dict(orientation="h", y=1.1, x=0),
    hovermode="x unified",
)
st.plotly_chart(fig_curves, width="stretch")

# ---------------------------------------------------------------------------
# Sección 3: Detalles técnicos (colapsados)
# ---------------------------------------------------------------------------
with st.expander("🔬 Así se entrena y se mide el modelo", expanded=False):
    tab_eq, tab_coef, tab_fit, tab_cmp = st.tabs(
        ["📝 Ecuación", "📊 Coeficientes", "📉 Real vs predicho", "⚖️ Los 4 modelos"]
    )

    with tab_eq:
        st.code(sim.equation(degree), language="text")

    with tab_coef:
        if degree == LINEAR_DEGREE:
            st.markdown("Regresión Lineal Múltiple: cada feature aporta un coeficiente independiente.")
            coef_df = (
                pd.DataFrame(
                    {
                        "Feature": [FEATURE_LABELS[f] for f in FEATURES],
                        "Coeficiente": sim.fitted(LINEAR_DEGREE)["model"].coef_,
                    }
                )
                .assign(Abs=lambda d: d["Coeficiente"].abs())
                .sort_values("Abs", ascending=True)
            )
            fig_coef = go.Figure(
                go.Bar(
                    x=coef_df["Coeficiente"],
                    y=coef_df["Feature"],
                    orientation="h",
                    marker=dict(
                        color=[COLOR_LINEAL if c > 0 else "#c0392b" for c in coef_df["Coeficiente"]]
                    ),
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                )
            )
            fig_coef.update_layout(
                height=380,
                margin=dict(l=120, r=20, t=20, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Lato", color="#3B2B20", size=13),
                xaxis=dict(title="Coeficiente β", gridcolor="#f0e4da"),
                yaxis=dict(title=None),
            )
            st.plotly_chart(fig_coef, width="stretch")
        else:
            st.markdown(
                f"Regresión Polinomial (grado {degree}): genera **{sim.n_polynomial_terms()[degree]} términos** "
                "(potencias e interacciones) de las 11 features estandarizadas. "
                "Ridge aplica λ = {:.2f} (cross-validation) para evitar el sobreajuste.".format(
                    sim.fitted(degree)["model"].alpha_
                )
            )
            st.code(sim.equation(degree), language="text")

    with tab_fit:
        y_test = np.asarray(sim.y_test)
        pred_lin = np.asarray(sim.predict_bulk(sim.X_test, LINEAR_DEGREE))
        pred_pol = np.asarray(sim.predict_bulk(sim.X_test, cmp_degree))
        lo, hi = y_test.min(), y_test.max()
        fig_fit = go.Figure()
        for name, col, color in [
            (MODEL_1, pred_lin, COLOR_LINEAL),
            (MODEL_2, pred_pol, COLOR_POLI),
        ]:
            fig_fit.add_trace(
                go.Scatter(
                    x=y_test,
                    y=col,
                    mode="markers",
                    name=name,
                    marker=dict(size=7, opacity=0.5, color=color, line=dict(width=0.5, color="#ffffff")),
                    hovertemplate=f"<b>{name}</b><br>Real: %{{x}}<br>Predicho: %{{y:.2f}}<extra></extra>",
                )
            )
        fig_fit.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="Ideal (perfecta)",
                line=dict(color="#c0392b", dash="dot", width=2),
            )
        )
        fig_fit.update_layout(
            height=380,
            margin=dict(l=40, r=20, t=30, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Lato", color="#3B2B20", size=13),
            xaxis=dict(title="Calidad real (test)", gridcolor="#f0e4da"),
            yaxis=dict(title="Calidad predicha (test)", gridcolor="#f0e4da"),
            legend=dict(orientation="h", y=1.1, x=0),
        )
        st.plotly_chart(fig_fit, width="stretch")

    with tab_cmp:
        st.dataframe(sim.comparison_table(), hide_index=True, width="stretch")

st.markdown("---")
st.caption(
    "Modelos entrenados únicamente con datos reales de `WineQT.csv`. "
    "Resultado integrado del Caso 2 · Taller 1.1."
)