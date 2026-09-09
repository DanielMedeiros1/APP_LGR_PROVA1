"""
Aplicativo Streamlit - Construção e Localização de Raízes no Lugar Geométrico das Raízes (LGR)
Disciplina: DCA 3701 - Projeto de Sistemas de Controle
Autor do site: Daniel Medeiros Soares Carneiro
"""

import os
import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False

# Diretorio base = pasta onde este arquivo app.py esta salvo.
# Usar caminho absoluto (em vez de "assets/...") evita erros de imagem nao
# encontrada quando o Streamlit Cloud roda o app com outro diretorio de trabalho.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ----------------------------------------------------------------------------------
# CONFIGURACAO DA PAGINA
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="LGR passo a passo | DCA 3701",
    page_icon="📈",
    layout="wide",
)

# Paleta de cores autoral (diferente do material de referencia)
COR_POLO = "#7C3AED"     # roxo
COR_ZERO = "#059669"     # verde esmeralda
COR_LOCUS = "#2563EB"    # azul
COR_ASSIN = "#EA580C"    # laranja
COR_CENTROIDE = "#111827"
COR_DESLOC = "#DB2777"   # rosa/magenta
COR_CRUZ = "#0EA5E9"     # ciano

st.markdown(
    """
    <style>
    .disciplina-badge {
        display:inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        background: linear-gradient(90deg,#2563EB,#7C3AED);
        color: white;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: .3px;
    }
    .autor-footer {
        text-align:center;
        color:#6B7280;
        font-size:0.85rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E5E7EB;
        margin-top: 2rem;
    }
    .passo-header {
        font-weight:700;
        font-size:1.05rem;
        color:#1F2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

s_sym, K_sym = sp.symbols("s K")
w_sym = sp.symbols("w", real=True)


# ----------------------------------------------------------------------------------
# FUNCOES AUXILIARES
# ----------------------------------------------------------------------------------
def parse_coeffs(text):
    """Converte texto 'a, b, c' em lista de floats (maior grau primeiro)."""
    text = text.strip()
    if not text:
        raise ValueError("Campo vazio.")
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip() != ""]
    coeffs = [float(sp.nsimplify(p)) for p in parts]
    # remove zeros a esquerda (mas mantem ao menos 1 coef)
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-14:
        coeffs.pop(0)
    return coeffs


def poly_expr(coeffs, var=s_sym):
    """Monta a expressao polinomial a partir dos coeficientes, preservando a
    representacao decimal (evita que o SymPy 'simplifique' 0.2 para 1/5)."""
    n = len(coeffs) - 1
    expr = 0
    for i, c in enumerate(coeffs):
        c_r = round(c, 10)
        if abs(c_r - round(c_r)) < 1e-9:
            coef = sp.Integer(int(round(c_r)))
        else:
            coef = sp.Float(c_r)
        expr += coef * var ** (n - i)
    return sp.expand(expr)


def coeffs_from_expr(expr, var=s_sym):
    p = sp.Poly(sp.expand(expr), var)
    return [complex(c) for c in p.all_coeffs()]


def fmt(x, nd=4):
    """Formata numero real."""
    if abs(x) < 10 ** (-nd - 1):
        x = 0.0
    return f"{x:.{nd}f}"


def fmt_trim(x, nd=4):
    """Formata numero real removendo zeros decimais desnecessarios
    (ex.: 13.0000 -> '13', 2.1900 -> '2.19')."""
    if abs(x) < 10 ** (-nd - 1):
        x = 0.0
    txt = f"{x:.{nd}f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return txt if txt not in ("", "-") else "0"


def fmt_complex(z, nd=4, var="s"):
    """Formata numero complexo estilo a+bj."""
    re = z.real
    im = z.imag
    if abs(im) < 1e-9:
        return f"{fmt(re, nd)}"
    sign = "+" if im >= 0 else "-"
    return f"{fmt(re, nd)} {sign} {fmt(abs(im), nd)}j"


def latex_complex(z, nd=4):
    re = z.real
    im = z.imag
    if abs(im) < 1e-9:
        return f"{fmt(re, nd)}"
    sign = "+" if im >= 0 else "-"
    return f"{fmt(re, nd)} {sign} {fmt(abs(im), nd)}j"


def angle_deg(z, ref="atan2"):
    return np.degrees(np.angle(z))


def group_roots(roots, tol=1e-4):
    """Agrupa raizes numericamente proximas, retornando lista de
    (raiz_representativa, multiplicidade), na ordem em que aparecem."""
    groups = []
    for r in roots:
        found = False
        for g in groups:
            if abs(g[0] - r) < tol:
                g[1] += 1
                found = True
                break
        if not found:
            groups.append([r, 1])
    return [(g[0], g[1]) for g in groups]


def latex_factor(root, mult, nd=4):
    """Retorna o fator latex (s - raiz)^mult de forma simplificada:
    - raiz real nula          -> 's' ou 's^m'
    - raiz real nao nula      -> '(s + a)' / '(s - a)' (sem parenteses duplicados,
                                   sem casas decimais redundantes)
    - raiz complexa           -> '(s - (a+bj))^m' (mantido explicito)
    """
    re = root.real
    im = root.imag

    if abs(im) < 1e-6:
        if abs(re) < 10 ** (-nd - 1):
            base = "s"
        else:
            val = fmt_trim(abs(re), nd)
            sign = "-" if re > 0 else "+"
            base = f"\\left(s {sign} {val}\\right)"
    else:
        base = r"\left(s - (" + latex_complex(root, nd) + r")\right)"

    if mult > 1:
        return base + f"^{{{mult}}}"
    return base


def routh_table_symbolic(coeffs):
    """Constrói a tabela de Routh-Hurwitz simbolicamente a partir dos coeficientes
    do polinômio característico (a_n, a_{n-1}, ..., a_0), em ordem decrescente de grau.

    Retorna:
        rows      - lista de linhas da tabela (cada uma, lista de expressões sympy)
        steps     - lista de tuplas (i, j, a1, a2, b1, b2, val) descrevendo o cálculo
                    de cada elemento a partir da 3a linha em diante, para exibição
                    passo a passo
        eps_used  - True se algum pivô nulo precisou ser substituído por épsilon
        eps_sym   - o símbolo usado para o épsilon (varepsilon), positivo e -> 0+
    """
    n = len(coeffs)
    width = (n + 1) // 2
    rows = [[sp.Integer(0)] * width for _ in range(n)]
    evens = coeffs[0::2]
    odds = coeffs[1::2]
    for i, c in enumerate(evens):
        rows[0][i] = sp.nsimplify(c)
    for i, c in enumerate(odds):
        rows[1][i] = sp.nsimplify(c)

    steps = []
    eps_sym = sp.symbols("varepsilon", positive=True)
    eps_used = False
    for i in range(2, n):
        pivot = sp.simplify(rows[i - 1][0])
        if pivot.is_number and pivot == 0:
            rows[i - 1][0] = eps_sym
            eps_used = True
        for j in range(width - 1):
            a1, a2 = rows[i - 1][0], rows[i - 1][j + 1]
            b1, b2 = rows[i - 2][0], rows[i - 2][j + 1]
            num = sp.expand(a1 * b2 - b1 * a2)
            val = sp.simplify(num / a1) if a1 != 0 else sp.nan
            rows[i][j] = val
            steps.append((i, j, a1, a2, b1, b2, val))
        rows[i][width - 1] = sp.Integer(0)
    return rows, steps, eps_used, eps_sym


def routh_row_label(power):
    """Rotulo latex da linha da tabela de Routh para a potencia de s indicada."""
    if power == 0:
        return "s^{0}"
    if power == 1:
        return "s"
    return f"s^{{{power}}}"


def new_fig(scale, w0=7.0, h0=5.0):
    fig, ax = plt.subplots(figsize=(w0 * scale, h0 * scale))
    return fig, ax


def style_axis(ax, title):
    ax.axhline(0, color="#9CA3AF", linewidth=1, zorder=0)
    ax.axvline(0, color="#9CA3AF", linewidth=1, zorder=0)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.set_xlabel(r"Real ($\sigma$)")
    ax.set_ylabel(r"Imaginário ($j\omega$)")
    ax.set_title(title)
    ax.set_facecolor("white")


def ensure_min_yspan(ax, min_abs=1.0):
    """Evita graficos 'achatados' quando todos os polos/zeros sao reais."""
    ymin, ymax = ax.get_ylim()
    if (ymax - ymin) < 2 * min_abs:
        mid = (ymax + ymin) / 2
        ax.set_ylim(mid - min_abs, mid + min_abs)


def plot_poles_zeros(ax, poles, zeros, label_poles="Pólos", label_zeros="Zeros", annotate=True):
    if len(poles):
        ax.scatter(
            [p.real for p in poles], [p.imag for p in poles],
            marker="x", s=170, linewidths=3, color=COR_POLO, label=label_poles, zorder=5,
        )
        if annotate:
            for i, p in enumerate(poles):
                ax.annotate(f"p{i+1}", (p.real, p.imag), textcoords="offset points",
                            xytext=(8, 8), color=COR_POLO, fontsize=10, fontweight="bold")
    if len(zeros):
        ax.scatter(
            [z.real for z in zeros], [z.imag for z in zeros],
            marker="o", s=140, linewidths=2.5, facecolors="none",
            edgecolors=COR_ZERO, label=label_zeros, zorder=5,
        )
        if annotate:
            for i, z in enumerate(zeros):
                ax.annotate(f"z{i+1}", (z.real, z.imag), textcoords="offset points",
                            xytext=(8, 8), color=COR_ZERO, fontsize=10, fontweight="bold")


def real_axis_segments(real_pts, xmin, xmax):
    """Retorna lista de segmentos (x0,x1) do eixo real pertencentes ao LGR."""
    pts = sorted(real_pts)
    if not pts:
        return []
    segs = []
    # regiao mais a esquerda -> ate xmin
    bounds = [xmin - 1] + pts + [xmax + 1]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        mid = (a + b) / 2
        count_right = sum(1 for p in pts if p > mid)
        if count_right % 2 == 1:
            seg_a = max(a, xmin)
            seg_b = min(b, xmax)
            if seg_b > seg_a:
                segs.append((seg_a, seg_b))
    return segs


def sweep_locus(den_coeffs, num_coeffs, kmax, n_points=1500):
    """Varre K de 0 a kmax e calcula raizes de den(s)+K*num(s)=0 para desenhar o LGR,
    rastreando cada ramo por correspondencia de proximidade (assignment) entre passos
    consecutivos de K para evitar embaralhamento entre ramos."""
    from scipy.optimize import linear_sum_assignment

    den_coeffs = np.array(den_coeffs, dtype=complex)
    num_coeffs = np.array(num_coeffs, dtype=complex)
    deg = len(den_coeffs) - 1
    if len(num_coeffs) < len(den_coeffs):
        num_pad = np.concatenate([np.zeros(len(den_coeffs) - len(num_coeffs), dtype=complex), num_coeffs])
    else:
        num_pad = num_coeffs

    # concentra mais pontos em K baixo (onde a curvatura e maior) usando espacamento nao linear
    frac = np.linspace(0, 1, n_points) ** 1.6
    ks = frac * kmax

    all_roots = np.zeros((n_points, deg), dtype=complex)
    prev = None
    for i, k in enumerate(ks):
        poly = den_coeffs + k * num_pad
        r = np.roots(poly)
        if prev is None:
            r_sorted = np.array(sorted(r, key=lambda z: (z.real, z.imag)))
        else:
            cost = np.abs(prev[:, None] - r[None, :])
            row_ind, col_ind = linear_sum_assignment(cost)
            r_sorted = np.zeros_like(r)
            r_sorted[row_ind] = r[col_ind]
        all_roots[i, :] = r_sorted
        prev = r_sorted
    return ks, all_roots


# ----------------------------------------------------------------------------------
# CABECALHO
# ----------------------------------------------------------------------------------
col_a, col_b = st.columns([3, 1])
with col_a:
    st.title("📈 Construção do Lugar Geométrico das Raízes (LGR)")
    st.markdown('<span class="disciplina-badge">DCA 3701 - Projeto de Sistemas de Controle</span>', unsafe_allow_html=True)
with col_b:
    st.markdown("&nbsp;")

st.caption(
    "Ferramenta que resolve, passo a passo, os 12 passos clássicos para a construção manual "
    "do LGR a partir de G(s), H(s) e de um ponto de teste no plano s."
)

st.divider()

# ----------------------------------------------------------------------------------
# ENTRADAS
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Dados de entrada")
    st.caption("Informe os coeficientes dos polinômios em ordem decrescente de grau, separados por vírgula.")

    st.subheader("G(s) = K · Num / Den")
    Gnum_txt = st.text_input("Numerador de G(s)", value="1, 0.2, 4", help="Ex.: s²+0.2s+4 → 1, 0.2, 4")
    Gden_txt = st.text_input("Denominador de G(s)", value="1, 2, 2, 1", help="Ex.: s³+2s²+2s+1 → 1, 2, 2, 1")

    st.subheader("H(s) = Num / Den")
    Hnum_txt = st.text_input("Numerador de H(s)", value="1")
    Hden_txt = st.text_input("Denominador de H(s)", value="1, 2")

    st.subheader("Ponto de teste s₀")
    c1, c2 = st.columns(2)
    with c1:
        s0_re = st.number_input("Parte real", value=-2.19, format="%.4f")
    with c2:
        s0_im = st.number_input("Parte imaginária", value=2.89, format="%.4f")

    st.subheader("🖼️ Tamanho dos gráficos")
    fig_scale = st.slider("Escala dos gráficos", min_value=0.5, max_value=2.5, value=1.0, step=0.1)

    calcular = st.button("▶️ Calcular os 12 passos", type="primary", use_container_width=True)

if "calculado" not in st.session_state:
    st.session_state["calculado"] = False

if calcular:
    st.session_state["calculado"] = True

if not st.session_state["calculado"]:
    st.info("⬅️ Preencha G(s), H(s) e o ponto de teste na barra lateral e clique em **Calcular os 12 passos**.")
    st.markdown(
        """
        ### Como usar
        1. Digite os coeficientes de **G(s)** (sem o ganho K, que é tratado separadamente) e de **H(s)**.
        2. Informe um **ponto de teste s₀** no plano complexo (usado nos Passos 11 e 12).
        3. Ajuste a **escala dos gráficos** conforme sua preferência.
        4. Clique em **Calcular os 12 passos**.
        """
    )
    st.markdown(
        '<div class="autor-footer">Disciplina: <b>DCA 3701 - Projeto de Sistemas de Controle</b> · '
        "Autor do site: <b>Daniel Medeiros Soares Carneiro</b></div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------------------------------------------------------------
# PROCESSAMENTO
# ----------------------------------------------------------------------------------
try:
    Gnum_c = parse_coeffs(Gnum_txt)
    Gden_c = parse_coeffs(Gden_txt)
    Hnum_c = parse_coeffs(Hnum_txt)
    Hden_c = parse_coeffs(Hden_txt)
except Exception as e:
    st.error(f"Erro ao interpretar os coeficientes: {e}")
    st.stop()

Gnum_e = poly_expr(Gnum_c)
Gden_e = poly_expr(Gden_c)
Hnum_e = poly_expr(Hnum_c)
Hden_e = poly_expr(Hden_c)

P_num_e = sp.expand(Gnum_e * Hnum_e)
P_den_e = sp.expand(Gden_e * Hden_e)

num_coeffs = coeffs_from_expr(P_num_e)
den_coeffs = coeffs_from_expr(P_den_e)

zeros = np.roots(num_coeffs) if len(num_coeffs) > 1 else np.array([])
poles = np.roots(den_coeffs) if len(den_coeffs) > 1 else np.array([])

nz_ = len(zeros)
np_ = len(poles)

s0 = complex(s0_re, s0_im)

st.success(f"✅ Sistema processado com sucesso: {np_} pólo(s) e {nz_} zero(s) finitos.")

tabs = st.tabs(
    [f"Passo {i}" for i in range(1, 13)]
    + ["📊 LGR completo", "📑 Tabela dos 12 Passos", "📚 O que é o LGR?"]
)

# ----------------------------------------------------------------------------------
# PASSO 1
# ----------------------------------------------------------------------------------
with tabs[0]:
    st.markdown('<p class="passo-header">Passo 1 — Equação característica</p>', unsafe_allow_html=True)
    st.latex(r"G(s) = K \cdot \dfrac{" + sp.latex(Gnum_e) + "}{" + sp.latex(Gden_e) + "}")
    st.latex(r"H(s) = \dfrac{" + sp.latex(Hnum_e) + "}{" + sp.latex(Hden_e) + "}")
    st.markdown("**Função de transferência de malha aberta:**")
    st.latex(
        r"G(s)H(s) = K \cdot \dfrac{" + sp.latex(P_num_e) + "}{" + sp.latex(P_den_e) + r"} = K \cdot P(s)"
    )
    st.markdown("**Equação característica:**")
    st.latex(
        r"1 + K \cdot P(s) = 0 \;\Longrightarrow\; " + sp.latex(P_den_e) +
        r" + K\left(" + sp.latex(P_num_e) + r"\right) = 0"
    )

# ----------------------------------------------------------------------------------
# PASSO 2
# ----------------------------------------------------------------------------------
with tabs[1]:
    st.markdown('<p class="passo-header">Passo 2 — Forma fatorada de P(s)</p>', unsafe_allow_html=True)
    st.markdown(
        "Fatorando numerador e denominador de P(s) em termos dos pólos e zeros de malha aberta "
        "(raízes repetidas são agrupadas com expoente de multiplicidade):"
    )
    zero_groups = group_roots(zeros)
    pole_groups = group_roots(poles)

    num_terms = " \\, ".join([latex_factor(r, m) for r, m in zero_groups]) or "1"
    den_terms = " \\, ".join([latex_factor(r, m) for r, m in pole_groups]) or "1"
    st.latex(r"P(s) = \dfrac{" + num_terms + "}{" + den_terms + "}")

# ----------------------------------------------------------------------------------
# PASSO 3
# ----------------------------------------------------------------------------------
with tabs[2]:
    st.markdown('<p class="passo-header">Passo 3 — Pólos e zeros no plano s</p>', unsafe_allow_html=True)
    fig, ax = new_fig(fig_scale)
    style_axis(ax, "Pólos e zeros de malha aberta")
    plot_poles_zeros(ax, poles, zeros)
    ax.legend(loc="best")
    ensure_min_yspan(ax)
    st.pyplot(fig, use_container_width=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Pólos (n_p = {np_})**")
        for i, p in enumerate(poles):
            st.latex(f"p_{{{i+1}}} = {latex_complex(p)}")
    with c2:
        st.markdown(f"**Zeros (n_z = {nz_})**")
        for i, z in enumerate(zeros):
            st.latex(f"z_{{{i+1}}} = {latex_complex(z)}")

# ----------------------------------------------------------------------------------
# PASSO 4
# ----------------------------------------------------------------------------------
with tabs[3]:
    st.markdown('<p class="passo-header">Passo 4 — Segmentos do eixo real pertencentes ao LGR</p>', unsafe_allow_html=True)
    real_crit = [p.real for p in poles if abs(p.imag) < 1e-6] + [z.real for z in zeros if abs(z.imag) < 1e-6]
    all_re = [p.real for p in poles] + [z.real for z in zeros] + [0]
    xmin, xmax = min(all_re) - 1.5, max(all_re) + 1.5
    segs = real_axis_segments(real_crit, xmin, xmax)

    fig, ax = new_fig(fig_scale)
    style_axis(ax, "Segmentos do eixo real pertencentes ao LGR")
    plot_poles_zeros(ax, poles, zeros, annotate=False)
    for i, (a, b) in enumerate(segs):
        ax.plot([a, b], [0, 0], color=COR_LOCUS, linewidth=5, alpha=0.75,
                label="Segmento LGR" if i == 0 else None, zorder=4)
    ax.legend(loc="best")
    ensure_min_yspan(ax)
    st.pyplot(fig, use_container_width=False)

    st.markdown("**Regra:** pertencem ao LGR os segmentos do eixo real à esquerda de um número ímpar de pólos e zeros reais.")
    if segs:
        segs_txt = ", ".join([f"[{fmt(a,4)}, {fmt(b,4)}]" for a, b in segs])
        st.latex(segs_txt)
    else:
        st.write("Nenhum segmento do eixo real pertence ao LGR.")

# ----------------------------------------------------------------------------------
# PASSO 5
# ----------------------------------------------------------------------------------
with tabs[4]:
    st.markdown('<p class="passo-header">Passo 5 — Número de lugares separados (ramos)</p>', unsafe_allow_html=True)
    Ls = max(np_, nz_)
    st.latex(f"n_p = {np_}, \\quad n_z = {nz_}")
    st.latex(r"L_s = \max(n_p,\, n_z) = \max(" + f"{np_},\\, {nz_}) = {Ls}")

# ----------------------------------------------------------------------------------
# PASSO 6
# ----------------------------------------------------------------------------------
with tabs[5]:
    st.markdown('<p class="passo-header">Passo 6 — Simetria</p>', unsafe_allow_html=True)
    st.markdown(
        "O LGR **é simétrico em relação ao eixo real**, pois raízes complexas de polinômios "
        "com coeficientes reais sempre ocorrem em pares conjugados."
    )

# ----------------------------------------------------------------------------------
# PASSO 7 (ASSINTOTAS)
# ----------------------------------------------------------------------------------
with tabs[6]:
    st.markdown('<p class="passo-header">Passo 7 — Assíntotas</p>', unsafe_allow_html=True)
    na = abs(np_ - nz_)
    sum_re_poles = sum(p.real for p in poles)
    sum_re_zeros = sum(z.real for z in zeros)

    st.markdown("**Número de assíntotas:**")
    st.latex(f"n_a = |n_p - n_z| = |{np_} - {nz_}| = {na}")

    sigma_a = None
    phis = []
    if na > 0:
        st.markdown("**Centroide (ponto de encontro das assíntotas):**")
        st.latex(r"\sigma_a = \dfrac{\sum \mathrm{Re}(p_i) - \sum \mathrm{Re}(z_j)}{n_p - n_z}")
        st.markdown("Soma das partes reais dos pólos:")
        st.latex(r"\sum \mathrm{Re}(p_i) = " + " + ".join([f"({fmt(p.real)})" for p in poles]) + f" = {fmt(sum_re_poles)}")
        st.markdown("Soma das partes reais dos zeros:")
        if nz_:
            st.latex(r"\sum \mathrm{Re}(z_j) = " + " + ".join([f"({fmt(z.real)})" for z in zeros]) + f" = {fmt(sum_re_zeros)}")
        else:
            st.latex(r"\sum \mathrm{Re}(z_j) = 0")
        sigma_a = (sum_re_poles - sum_re_zeros) / (np_ - nz_)
        st.markdown("Substituindo:")
        st.latex(
            r"\sigma_a = \dfrac{(" + fmt(sum_re_poles) + ") - (" + fmt(sum_re_zeros) + ")}{" +
            f"{np_ - nz_}" + r"} = " + fmt((sum_re_poles - sum_re_zeros)) + f" / {np_-nz_} = {fmt(sigma_a)}"
        )

        st.markdown("**Ângulos das assíntotas:**")
        st.latex(r"\phi_a = \dfrac{(2q+1)\cdot 180^\circ}{n_a}, \quad q = 0, 1, \dots, n_a-1")
        for q in range(na):
            phi = (2 * q + 1) * 180 / na
            phis.append(phi)
            st.latex(f"q={q}: \\quad \\phi_a = \\dfrac{{(2\\cdot{q}+1)\\cdot 180^\\circ}}{{{na}}} = {fmt(phi,2)}^\\circ")
    else:
        st.info("Não há assíntotas (n_p = n_z).")

    # grafico
    fig, ax = new_fig(fig_scale)
    style_axis(ax, "LGR - Assíntotas")
    plot_poles_zeros(ax, poles, zeros, annotate=False)
    for i, (a, b) in enumerate(segs):
        ax.plot([a, b], [0, 0], color=COR_LOCUS, linewidth=5, alpha=0.75,
                label="Segmento LGR" if i == 0 else None, zorder=4)
    if sigma_a is not None:
        ylim = max([abs(p.imag) for p in poles] + [abs(z.imag) for z in zeros] + [1]) * 1.6
        for i, phi in enumerate(phis):
            dx = np.cos(np.radians(phi))
            dy = np.sin(np.radians(phi))
            length = ylim * 3
            ax.plot([sigma_a, sigma_a + dx * length], [0, dy * length],
                    linestyle="--", color=COR_ASSIN, linewidth=1.6,
                    label="Assíntotas" if i == 0 else None, zorder=3)
        ax.plot(sigma_a, 0, marker="+", color=COR_CENTROIDE, markersize=16, markeredgewidth=3,
                label=f"Centroide (σₐ = {fmt(sigma_a,2)})", zorder=6)
        ax.set_ylim(-ylim, ylim)
    ax.legend(loc="best")
    ensure_min_yspan(ax)
    st.pyplot(fig, use_container_width=False)

# ----------------------------------------------------------------------------------
# PASSO 8 (BREAKAWAY)
# ----------------------------------------------------------------------------------
with tabs[7]:
    st.markdown('<p class="passo-header">Passo 8 — Pontos de saída/entrada (descolamento)</p>', unsafe_allow_html=True)
    st.markdown("**1) Isolar K na equação característica:**")
    st.latex(r"D(s) + K \cdot N(s) = 0 \;\Longrightarrow\; K = -\dfrac{D(s)}{N(s)}")
    st.latex(r"K = -\dfrac{" + sp.latex(P_den_e) + "}{" + sp.latex(P_num_e) + "}")

    st.markdown(r"**2) Condição de descolamento** $\dfrac{dK}{ds}=0$:")
    st.latex(r"\dfrac{dK}{ds} = -\dfrac{D'(s)\cdot N(s) - D(s)\cdot N'(s)}{N(s)^2} = 0")
    st.markdown("Para o numerador ser zero:")
    st.latex(r"D'(s)\cdot N(s) - D(s)\cdot N'(s) = 0")

    Dp_e = sp.diff(P_den_e, s_sym)
    Np_e = sp.diff(P_num_e, s_sym)
    st.markdown("**3) Calculando as derivadas:**")
    st.latex(r"N(s) = " + sp.latex(P_num_e))
    st.latex(r"N'(s) = " + sp.latex(sp.expand(Np_e)))
    st.latex(r"D(s) = " + sp.latex(P_den_e))
    st.latex(r"D'(s) = " + sp.latex(sp.expand(Dp_e)))

    breakaway_eq = sp.expand(Dp_e * P_num_e - P_den_e * Np_e)
    st.markdown("**4) Equação de descolamento:**")
    st.latex(sp.latex(breakaway_eq) + " = 0")

    st.markdown("**5) Raízes da equação de descolamento:**")
    st.caption(
        "Raízes que coincidem com um pólo/zero de malha aberta (comum quando há multiplicidade ≥ 2) "
        "são soluções triviais da equação acima e não representam pontos reais de descolamento — "
        "por isso são descartadas, assim como qualquer solução com K ≤ 0."
    )
    valid_points = []
    try:
        b_coeffs = coeffs_from_expr(breakaway_eq)
        if len(b_coeffs) > 1:
            b_roots = np.roots(b_coeffs)
            K_func = sp.lambdify(s_sym, -P_den_e / P_num_e, "numpy")
            tol_trivial = 1e-4
            for r in b_roots:
                try:
                    kv = complex(K_func(r))
                except Exception:
                    kv = complex("nan")
                is_real_s = abs(r.imag) < 1e-4
                is_real_k = abs(kv.imag) < 1e-3
                is_pos_k = kv.real > 1e-9
                is_trivial = (
                    any(abs(r - p) < tol_trivial for p in poles) or
                    any(abs(r - z) < tol_trivial for z in zeros)
                )
                pertence = is_real_s and is_real_k and is_pos_k and not is_trivial
                if is_trivial and is_real_s:
                    tag = r"[\text{trivial (coincide com pólo/zero) - descartado}]"
                elif pertence:
                    tag = r"[\text{pertence ao LGR}]"
                else:
                    tag = r"[\text{fora}]"
                st.latex(f"s = {latex_complex(r)}, \\quad K = {latex_complex(kv)} \\quad {tag}")
                if pertence:
                    valid_points.append((r.real, kv.real))
        else:
            st.write("Não há equação de descolamento não trivial (sistema sem termos suficientes).")
    except Exception as e:
        st.warning(f"Não foi possível resolver a equação de descolamento automaticamente: {e}")

    if valid_points:
        for sr, kv in valid_points:
            st.success(f"Ponto de descolamento válido: s = {fmt(sr)} com K = {fmt(kv)}")
    else:
        st.info("Nenhum ponto de descolamento real e válido (K > 0) foi encontrado.")

    fig, ax = new_fig(fig_scale)
    style_axis(ax, "LGR - Pontos de descolamento")
    plot_poles_zeros(ax, poles, zeros, annotate=False)
    for i, (a, b) in enumerate(segs):
        ax.plot([a, b], [0, 0], color=COR_LOCUS, linewidth=5, alpha=0.75,
                label="Segmento LGR" if i == 0 else None, zorder=4)
    if sigma_a is not None and na > 0:
        ylim = max([abs(p.imag) for p in poles] + [abs(z.imag) for z in zeros] + [1]) * 1.6
        for i, phi in enumerate(phis):
            dx = np.cos(np.radians(phi)); dy = np.sin(np.radians(phi))
            length = ylim * 3
            ax.plot([sigma_a, sigma_a + dx * length], [0, dy * length],
                    linestyle="--", color=COR_ASSIN, linewidth=1.4,
                    label="Assíntotas" if i == 0 else None, zorder=3)
        ax.set_ylim(-ylim, ylim)
    for i, (sr, kv) in enumerate(valid_points):
        ax.plot(sr, 0, marker="D", color=COR_DESLOC, markersize=11,
                label="Ponto de descolamento" if i == 0 else None, zorder=7)
        ax.annotate(f"s={fmt(sr,2)}, K={fmt(kv,2)}", (sr, 0), textcoords="offset points",
                    xytext=(10, 10), color=COR_DESLOC, fontsize=9)
    ax.legend(loc="best")
    ensure_min_yspan(ax)
    st.pyplot(fig, use_container_width=False)

# ----------------------------------------------------------------------------------
# PASSO 9 (ROUTH + CRUZAMENTO)
# ----------------------------------------------------------------------------------
with tabs[8]:
    st.markdown('<p class="passo-header">Passo 9 — Critério de Routh-Hurwitz e cruzamento com o eixo imaginário</p>', unsafe_allow_html=True)

    st.markdown("### 9.1 — Cruzamento com o eixo imaginário (substituição $s=j\\omega$)")
    st.markdown("**Método:** substituindo $s = j\\omega$ na equação característica e separando partes real e imaginária:")

    Djw = sp.expand(P_den_e.subs(s_sym, sp.I * w_sym))
    Njw = sp.expand(P_num_e.subs(s_sym, sp.I * w_sym))
    ReD = sp.expand(sp.re(Djw))
    ImD = sp.expand(sp.im(Djw))
    ReN = sp.expand(sp.re(Njw))
    ImN = sp.expand(sp.im(Njw))

    st.latex(r"\mathrm{Re}_D(\omega) = " + sp.latex(ReD))
    st.latex(r"\mathrm{Im}_D(\omega) = " + sp.latex(ImD))
    st.latex(r"\mathrm{Re}_N(\omega) = " + sp.latex(ReN))
    st.latex(r"\mathrm{Im}_N(\omega) = " + sp.latex(ImN))

    st.markdown("**Eliminando K entre as equações real e imaginária:**")
    st.latex(r"\mathrm{Re}_D \cdot \mathrm{Im}_N - \mathrm{Im}_D \cdot \mathrm{Re}_N = 0")
    cross_eq = sp.expand(ReD * ImN - ImD * ReN)
    st.latex(sp.latex(cross_eq) + " = 0")

    crossings = []
    try:
        cross_coeffs = coeffs_from_expr(cross_eq, w_sym)
        if len(cross_coeffs) > 1:
            w_roots = np.roots(cross_coeffs)
            st.markdown("**Soluções válidas** ($\\omega>0,\\ K>0$):")
            ImN_f = sp.lambdify(w_sym, ImN, "numpy")
            ImD_f = sp.lambdify(w_sym, ImD, "numpy")
            ReN_f = sp.lambdify(w_sym, ReN, "numpy")
            ReD_f = sp.lambdify(w_sym, ReD, "numpy")
            for wv in w_roots:
                if abs(wv.imag) < 1e-6 and wv.real > 1e-6:
                    wr_ = wv.real
                    imn_v = float(np.real(ImN_f(wr_)))
                    imd_v = float(np.real(ImD_f(wr_)))
                    if abs(imn_v) > 1e-9:
                        kv = -imd_v / imn_v
                    else:
                        ren_v = float(np.real(ReN_f(wr_)))
                        red_v = float(np.real(ReD_f(wr_)))
                        kv = -red_v / ren_v if abs(ren_v) > 1e-9 else float("nan")
                    if kv > 0:
                        crossings.append((wr_, kv))
                        st.latex(f"\\omega = {fmt(wr_,4)} \\;\\Rightarrow\\; K = {fmt(kv,4)}")
                        st.latex(f"\\therefore \\quad s = \\pm {fmt(wr_,4)}j, \\quad K = {fmt(kv,4)}")
        else:
            st.write("Não há cruzamento não trivial com o eixo imaginário.")
    except Exception as e:
        st.warning(f"Não foi possível calcular os cruzamentos automaticamente: {e}")

    if not crossings:
        st.info("Nenhum cruzamento válido com o eixo imaginário (ω>0, K>0) foi encontrado.")

    # ------------------------------------------------------------------------------
    # 9.2 - TABELA DE ROUTH-HURWITZ
    # ------------------------------------------------------------------------------
    st.divider()
    st.markdown("### 9.2 — Tabela de Routh-Hurwitz")
    st.markdown(
        "O critério de Routh-Hurwitz permite obter a faixa de valores de K para a qual o sistema "
        "em malha fechada é estável, sem a necessidade de calcular as raízes explicitamente."
    )

    st.markdown("**1) Equação característica, com K mantido genérico:**")
    char_eq_K = sp.expand(P_den_e + K_sym * P_num_e)
    st.latex(sp.latex(char_eq_K) + " = 0")

    poly_K = sp.Poly(char_eq_K, s_sym)
    coeffs_K = poly_K.all_coeffs()
    deg_K = poly_K.degree()
    powers_K = [deg_K - i for i in range(len(coeffs_K))]
    row_labels_K = [routh_row_label(p) for p in powers_K]

    rows_K, steps_K, eps_used_K, eps_sym_K = routh_table_symbolic(coeffs_K)

    st.markdown(
        "**2) Duas primeiras linhas** (coeficientes das potências pares e ímpares de s, respectivamente):"
    )
    st.latex(f"{row_labels_K[0]}: \\quad " + " \\quad ".join(sp.latex(sp.simplify(v)) for v in rows_K[0]))
    st.latex(f"{row_labels_K[1]}: \\quad " + " \\quad ".join(sp.latex(sp.simplify(v)) for v in rows_K[1]))

    st.markdown("**3) Linhas seguintes**, cada elemento calculado por:")
    st.latex(
        r"b_j = \dfrac{a_{i-1,0}\cdot a_{i-2,\,j+1} \;-\; a_{i-2,0}\cdot a_{i-1,\,j+1}}{a_{i-1,0}}"
    )

    if eps_used_K:
        st.info(
            f"Um pivô da primeira coluna se anulou; substituímos por "
            f"${sp.latex(eps_sym_K)} \\to 0^+$ para continuar a tabela (método do épsilon)."
        )

    _current_row_K = None
    for (i, j, a1, a2, b1, b2, val) in steps_K:
        if _current_row_K != i:
            st.markdown(f"Linha **{row_labels_K[i]}**:")
            _current_row_K = i
        st.latex(
            f"b_{{{j+1}}} = \\dfrac{{({sp.latex(a1)})({sp.latex(b2)}) - ({sp.latex(b1)})({sp.latex(a2)})}}"
            f"{{{sp.latex(a1)}}} = {sp.latex(sp.simplify(val))}"
        )

    st.markdown("**4) Tabela completa de Routh-Hurwitz:**")
    col_spec = "c|" + "c" * len(rows_K[0])
    array_rows = []
    for lbl, row in zip(row_labels_K, rows_K):
        cells = " & ".join(sp.latex(sp.simplify(v)) for v in row)
        array_rows.append(f"{lbl} & {cells}")
    st.latex(r"\begin{array}{" + col_spec + r"}" + r" \\ ".join(array_rows) + r"\end{array}")

    st.markdown("**5) Critério de estabilidade — primeira coluna:**")
    st.markdown(
        "O sistema é estável se, e somente se, **todos** os elementos da primeira coluna tiverem "
        "o mesmo sinal (nenhuma mudança de sinal). Cada mudança de sinal indica um pólo de malha "
        "fechada no semiplano direito do plano s."
    )
    first_col_K = [sp.simplify(row[0]) for row in rows_K]
    for lbl, val in zip(row_labels_K, first_col_K):
        st.latex(f"{lbl}: \\quad {sp.latex(val)}")

    st.markdown("**6) Faixa de K para estabilidade:**")
    conditions_K = []
    solved_ok_K = True
    for val in first_col_K:
        if val.free_symbols:
            try:
                sol = sp.solve_univariate_inequality(sp.simplify(val) > 0, K_sym, relational=False)
                conditions_K.append(sol)
            except Exception:
                solved_ok_K = False
        else:
            if val <= 0:
                st.error(
                    f"O termo {sp.latex(val)} é constante e não positivo: nessas condições o sistema "
                    "é instável independentemente de K (verifique os dados de entrada)."
                )
                solved_ok_K = False

    if solved_ok_K and conditions_K:
        interval_final_K = conditions_K[0]
        for c in conditions_K[1:]:
            interval_final_K = interval_final_K.intersect(c)
        interval_final_K = interval_final_K.intersect(sp.Interval.open(0, sp.oo))
        st.success(f"O sistema é estável para: $K \\in {sp.latex(interval_final_K)}$")
    elif solved_ok_K:
        st.info("A primeira coluna não depende de K; a estabilidade não é afetada pelo ganho.")
    else:
        st.warning(
            "Não foi possível resolver automaticamente a faixa de K a partir das inequações acima; "
            "analise o sinal de cada termo da primeira coluna manualmente."
        )

    if crossings:
        st.markdown("**7) Verificação cruzada com o cruzamento no eixo imaginário:**")
        for wr_, kv in crossings:
            near_zero_rows = []
            for lbl, val in zip(row_labels_K, first_col_K):
                if val.free_symbols:
                    try:
                        numeric = complex(val.subs(K_sym, kv))
                        if abs(numeric) < 1e-6 * max(1, abs(kv)):
                            near_zero_rows.append(lbl)
                    except Exception:
                        pass
            if near_zero_rows:
                st.latex(
                    f"\\text{{Em }} K = {fmt(kv,4)}\\text{{, a(s) linha(s) }}" +
                    ", ".join(near_zero_rows) +
                    r"\text{ da tabela se anula(m), confirmando os pólos em } s=\pm " +
                    f"{fmt(wr_,4)}" + r"j\text{ — mesmo resultado do método } s=j\omega\text{ (item 9.1).}"
                )

    fig, ax = new_fig(fig_scale)
    style_axis(ax, "LGR - Cruzamento com eixo imaginário")
    plot_poles_zeros(ax, poles, zeros, annotate=False)
    for i, (a, b) in enumerate(segs):
        ax.plot([a, b], [0, 0], color=COR_LOCUS, linewidth=4, alpha=0.5, zorder=3)
    for i, (wr_, kv) in enumerate(crossings):
        ax.plot(0, wr_, marker="s", color=COR_CRUZ, markersize=13,
                label=f"jω = {fmt(wr_,2)}j (K={fmt(kv,2)})", zorder=7)
        ax.plot(0, -wr_, marker="s", color=COR_CRUZ, markersize=13, zorder=7)
    ax.legend(loc="best", fontsize=8)
    ensure_min_yspan(ax)
    st.pyplot(fig, use_container_width=False)

# ----------------------------------------------------------------------------------
# PASSO 10 (ANGULOS DE PARTIDA E CHEGADA)
# ----------------------------------------------------------------------------------
with tabs[9]:
    st.markdown('<p class="passo-header">Passo 10 — Ângulos de partida e chegada</p>', unsafe_allow_html=True)

    complex_poles = [p for p in poles if p.imag > 1e-6]
    complex_zeros = [z for z in zeros if z.imag > 1e-6]

    st.markdown("### Ângulos de partida (pólos complexos)")
    st.markdown("**Fórmula:**")
    st.latex(r"\theta_d = 180^\circ - \sum_{j\neq k} \angle(p_k - p_j) + \sum_{j} \angle(p_k - z_j)")

    departure_results = []
    if complex_poles:
        for pk in complex_poles:
            st.markdown(f"**Pólo** $p_k = {latex_complex(pk)}$:")
            sum_theta = 0.0
            st.markdown("Ângulos dos outros pólos:")
            for pj in poles:
                if abs(pj - pk) < 1e-9:
                    continue
                ang = angle_deg(pk - pj)
                sum_theta += ang
                st.latex(f"\\angle(p_k - p_j) = \\angle({latex_complex(pk-pj)}) = {fmt(ang,2)}^\\circ")
            sum_phi = 0.0
            if zeros.size:
                st.markdown("Ângulos dos zeros:")
                for zj in zeros:
                    ang = angle_deg(pk - zj)
                    sum_phi += ang
                    st.latex(f"\\angle(p_k - z_j) = \\angle({latex_complex(pk-zj)}) = {fmt(ang,2)}^\\circ")
            st.markdown("Somatórios:")
            st.latex(f"\\sum \\angle(p_k - p_j) = {fmt(sum_theta,2)}^\\circ")
            st.latex(f"\\sum \\angle(p_k - z_j) = {fmt(sum_phi,2)}^\\circ")
            theta_d = (180 - sum_theta + sum_phi) % 360
            st.markdown("Resultado:")
            st.latex(f"\\theta_d = 180^\\circ - ({fmt(sum_theta,2)}^\\circ) + ({fmt(sum_phi,2)}^\\circ) = {fmt(theta_d,2)}^\\circ")
            st.latex(f"\\text{{Conjugado: }} {fmt((360-theta_d)%360,2)}^\\circ")
            departure_results.append((pk, theta_d))
    else:
        st.info("Não há pólos complexos (com parte imaginária positiva) para calcular ângulo de partida.")

    st.markdown("### Ângulos de chegada (zeros complexos)")
    st.markdown("**Fórmula:**")
    st.latex(r"\theta_a = 180^\circ - \sum_{j\neq k} \angle(z_k - z_j) + \sum_{j} \angle(z_k - p_j)")

    arrival_results = []
    if complex_zeros:
        for zk in complex_zeros:
            st.markdown(f"**Zero** $z_k = {latex_complex(zk)}$:")
            sum_theta = 0.0
            if len(zeros) > 1:
                st.markdown("Ângulos dos outros zeros:")
                for zj in zeros:
                    if abs(zj - zk) < 1e-9:
                        continue
                    ang = angle_deg(zk - zj)
                    sum_theta += ang
                    st.latex(f"\\angle(z_k - z_j) = {fmt(ang,2)}^\\circ")
            sum_phi = 0.0
            st.markdown("Ângulos dos pólos:")
            for pj in poles:
                ang = angle_deg(zk - pj)
                sum_phi += ang
                st.latex(f"\\angle(z_k - p_j) = {fmt(ang,2)}^\\circ")
            st.latex(f"\\sum \\angle(z_k - z_j) = {fmt(sum_theta,2)}^\\circ")
            st.latex(f"\\sum \\angle(z_k - p_j) = {fmt(sum_phi,2)}^\\circ")
            theta_a = (180 - sum_theta + sum_phi) % 360
            st.latex(f"\\theta_a = 180^\\circ - ({fmt(sum_theta,2)}^\\circ) + ({fmt(sum_phi,2)}^\\circ) = {fmt(theta_a,2)}^\\circ")
            st.latex(f"\\text{{Conjugado: }} {fmt((360-theta_a)%360,2)}^\\circ")
            arrival_results.append((zk, theta_a))
    else:
        st.info("Não há zeros complexos (com parte imaginária positiva) para calcular ângulo de chegada.")

    fig, ax = new_fig(fig_scale)
    style_axis(ax, "LGR - Ângulos de partida/chegada")
    plot_poles_zeros(ax, poles, zeros, annotate=False)
    for i, (a, b) in enumerate(segs):
        ax.plot([a, b], [0, 0], color=COR_LOCUS, linewidth=4, alpha=0.5, zorder=3)
    arrow_len = max([abs(p.imag) for p in poles] + [abs(z.imag) for z in zeros] + [1]) * 0.6
    extent_x = [p.real for p in poles] + [z.real for z in zeros]
    extent_y = [p.imag for p in poles] + [z.imag for z in zeros]
    for pk, th in departure_results:
        dx = arrow_len * np.cos(np.radians(th))
        dy = arrow_len * np.sin(np.radians(th))
        ax.annotate("", xy=(pk.real + dx, pk.imag + dy), xytext=(pk.real, pk.imag),
                    arrowprops=dict(arrowstyle="->", color=COR_POLO, lw=2))
        ax.text(pk.real + dx * 1.15, pk.imag + dy * 1.15, f"{fmt(th,1)}°", color=COR_POLO, fontsize=9)
        extent_x += [pk.real + dx * 1.3]
        extent_y += [pk.imag + dy * 1.3]
    for zk, th in arrival_results:
        dx = arrow_len * np.cos(np.radians(th))
        dy = arrow_len * np.sin(np.radians(th))
        ax.annotate("", xy=(zk.real + dx, zk.imag + dy), xytext=(zk.real, zk.imag),
                    arrowprops=dict(arrowstyle="->", color=COR_ZERO, lw=2))
        ax.text(zk.real + dx * 1.15, zk.imag + dy * 1.15, f"{fmt(th,1)}°", color=COR_ZERO, fontsize=9)
        extent_x += [zk.real + dx * 1.3]
        extent_y += [zk.imag + dy * 1.3]
    if extent_x:
        padx = max(0.5, (max(extent_x) - min(extent_x)) * 0.15)
        pady = max(0.5, (max(extent_y) - min(extent_y)) * 0.15)
        ax.set_xlim(min(extent_x) - padx, max(extent_x) + padx)
        ax.set_ylim(min(extent_y) - pady, max(extent_y) + pady)
    ax.legend(loc="best")
    ensure_min_yspan(ax)
    st.pyplot(fig, use_container_width=False)

# ----------------------------------------------------------------------------------
# PASSO 11 (CRITERIO DE ANGULO)
# ----------------------------------------------------------------------------------
with tabs[10]:
    st.markdown('<p class="passo-header">Passo 11 — Critério de ângulo</p>', unsafe_allow_html=True)
    st.markdown("**Condição de pertinência ao LGR:**")
    st.latex(r"\sum \angle(s_0 - p_i) - \sum \angle(s_0 - z_j) = \pm 180^\circ(2q+1)")
    st.markdown(f"**Ponto de teste:** $s_0 = {latex_complex(s0)}$")

    st.markdown("**Ângulos dos pólos** ($\\theta_i$):")
    sum_theta = 0.0
    for i, p in enumerate(poles):
        ang = angle_deg(s0 - p)
        sum_theta += ang
        st.latex(f"\\theta_{{{i+1}}} = \\angle(s_0 - p_{{{i+1}}}) = \\angle({latex_complex(s0-p)}) = {fmt(ang,2)}^\\circ")
    st.latex(f"\\sum \\theta_i = {fmt(sum_theta,2)}^\\circ")

    st.markdown("**Ângulos dos zeros** ($\\phi_j$):")
    sum_phi = 0.0
    for i, z in enumerate(zeros):
        ang = angle_deg(s0 - z)
        sum_phi += ang
        st.latex(f"\\phi_{{{i+1}}} = \\angle(s_0 - z_{{{i+1}}}) = \\angle({latex_complex(s0-z)}) = {fmt(ang,2)}^\\circ")
    if zeros.size:
        st.latex(f"\\sum \\phi_j = {fmt(sum_phi,2)}^\\circ")

    delta = sum_theta - sum_phi
    delta_norm = ((delta + 180) % 360) - 180
    st.markdown("**Avaliação:**")
    st.latex(f"\\Delta\\theta = \\sum \\theta_i - \\sum \\phi_j = {fmt(sum_theta,2)}^\\circ - {fmt(sum_phi,2)}^\\circ = {fmt(delta,2)}^\\circ")
    st.latex(f"\\text{{Ângulo normalizado: }} {fmt(abs(delta_norm),2)}^\\circ")

    pertence_angulo = abs(abs(delta_norm) - 180) < 5.0
    if pertence_angulo:
        st.success(f"O ponto pertence ao LGR (Δθ ≈ {fmt(abs(delta_norm),2)}° ≈ 180°)")
    else:
        st.error(f"O ponto NÃO pertence ao LGR (Δθ ≈ {fmt(abs(delta_norm),2)}°, esperado ≈ 180°)")

    fig, ax = new_fig(fig_scale)
    style_axis(ax, "Critério de Ângulo")
    plot_poles_zeros(ax, poles, zeros, annotate=False)
    for p in poles:
        ax.plot([s0.real, p.real], [s0.imag, p.imag], linestyle=":", color=COR_POLO, linewidth=1.3, zorder=2)
    for z in zeros:
        ax.plot([s0.real, z.real], [s0.imag, z.imag], linestyle=":", color=COR_ZERO, linewidth=1.3, zorder=2)
    marker_s0 = "*"
    ax.scatter([s0.real], [s0.imag], marker=marker_s0, s=350,
               color=(COR_LOCUS if pertence_angulo else "#DC2626"),
               edgecolors="black", linewidths=1,
               label=f"s₀ = {latex_complex(s0)} ({'pertence' if pertence_angulo else 'não pertence'})", zorder=8)
    ax.legend(loc="best", fontsize=8)
    ensure_min_yspan(ax)
    st.pyplot(fig, use_container_width=False)

# ----------------------------------------------------------------------------------
# PASSO 12 (CRITERIO DE MODULO)
# ----------------------------------------------------------------------------------
with tabs[11]:
    st.markdown('<p class="passo-header">Passo 12 — Cálculo de K</p>', unsafe_allow_html=True)
    st.markdown("**Fórmula do critério de módulo:**")
    st.latex(r"K = \dfrac{\prod_i |s_0 - p_i|}{\prod_j |s_0 - z_j|}")
    st.markdown(f"**Ponto:** $s_0 = {latex_complex(s0)}$")

    st.markdown("**Distâncias dos pólos:**")
    prod_p = 1.0
    for i, p in enumerate(poles):
        d = abs(s0 - p)
        prod_p *= d
        st.latex(f"|s_0 - p_{{{i+1}}}| = |{latex_complex(s0-p)}| = {fmt(d,4)}")
    st.markdown("**Produto das distâncias dos pólos:**")
    st.latex(f"\\prod_i |s_0 - p_i| = {fmt(prod_p,4)}")

    st.markdown("**Distâncias dos zeros:**")
    prod_z = 1.0
    for i, z in enumerate(zeros):
        d = abs(s0 - z)
        prod_z *= d
        st.latex(f"|s_0 - z_{{{i+1}}}| = |{latex_complex(s0-z)}| = {fmt(d,4)}")
    if zeros.size:
        st.markdown("**Produto das distâncias dos zeros:**")
        st.latex(f"\\prod_j |s_0 - z_j| = {fmt(prod_z,4)}")

    K_final = prod_p / prod_z if prod_z != 0 else float("inf")
    st.markdown("**Resultado:**")
    st.latex(f"K = \\dfrac{{{fmt(prod_p,4)}}}{{{fmt(prod_z,4)}}} = {fmt(K_final,4)}")

    if pertence_angulo:
        st.success(f"O ponto pertence ao LGR. K = {fmt(K_final,6)}")
    else:
        st.warning(
            f"O ponto não satisfaz rigorosamente o critério de ângulo (Passo 11); "
            f"o valor de K = {fmt(K_final,6)} é calculado, mas o ponto pode não pertencer ao LGR real."
        )

# ----------------------------------------------------------------------------------
# GRAFICO COMPLETO DO LGR
# ----------------------------------------------------------------------------------
with tabs[12]:
    st.markdown('<p class="passo-header">Gráfico completo do LGR</p>', unsafe_allow_html=True)

    all_re_full = [p.real for p in poles] + [z.real for z in zeros] + ([sigma_a] if sigma_a is not None else []) + [0]
    all_im_full = [p.imag for p in poles] + [z.imag for z in zeros] + [0]
    xr = max(all_re_full) - min(all_re_full) + 2
    kmax_guess = max([kv for _, kv in crossings] + [K_final if np.isfinite(K_final) else 0, 20]) * 1.6

    with st.spinner("Calculando os ramos do LGR..."):
        ks, roots_sweep = sweep_locus(den_coeffs, num_coeffs, kmax_guess, n_points=2500)

    fig, ax = new_fig(fig_scale, w0=8, h0=6.5)
    style_axis(ax, "Lugar Geométrico das Raízes")
    for branch in range(roots_sweep.shape[1]):
        ax.plot(roots_sweep[:, branch].real, roots_sweep[:, branch].imag,
                color=COR_LOCUS, linewidth=2, zorder=3)
    plot_poles_zeros(ax, poles, zeros, annotate=False)
    if sigma_a is not None:
        ax.plot(sigma_a, 0, marker="+", color=COR_CENTROIDE, markersize=16, markeredgewidth=3,
                label=f"Centroide (σₐ = {fmt(sigma_a,2)})", zorder=6)
    ax.legend(loc="best")

    xpad = max(1.0, xr * 0.15)
    ypad = max(1.0, (max(all_im_full) - min(all_im_full) + 2) * 0.15)
    ax.set_xlim(min(all_re_full) - xpad, max(all_re_full) + xpad)
    ax.set_ylim(min(all_im_full) - ypad - 1, max(all_im_full) + ypad + 1)

    st.pyplot(fig, use_container_width=False)
    st.caption(
        "O gráfico é gerado varrendo o ganho K de 0 até um valor suficientemente alto e "
        "resolvendo numericamente as raízes de D(s) + K·N(s) = 0 a cada passo."
    )

# ----------------------------------------------------------------------------------
# TABELA DOS 12 PASSOS (IMAGENS DE REFERENCIA)
# ----------------------------------------------------------------------------------
with tabs[13]:
    st.markdown('<p class="passo-header">Tabela-resumo dos 12 passos de construção do LGR</p>', unsafe_allow_html=True)
    st.markdown(
        "As tabelas abaixo resumem, em formato de quadro, os passos clássicos para a construção "
        "manual do Lugar Geométrico das Raízes — os itens 1 a 10 tratam do **traçado** do LGR "
        "(equação característica, pólos e zeros, eixo real, ramos, simetria, assíntotas, "
        "descolamento, Routh-Hurwitz e ângulos de partida/chegada), enquanto os itens 11 e 12 "
        "tratam da **localização de uma raiz específica** dentro do LGR já traçado e do cálculo "
        "do ganho K correspondente. São exatamente os mesmos 12 passos resolvidos automaticamente "
        "nas abas anteriores desta ferramenta."
    )

    img1_path = os.path.join(ASSETS_DIR, "tabela_passos_1_10.png")
    img2_path = os.path.join(ASSETS_DIR, "tabela_passos_11_12.png")

    st.markdown("**Passos 1 a 10 — Construção do LGR**")
    if os.path.exists(img1_path):
        st.image(
            img1_path,
            caption="Tabela 2.2 — Passos para a Construção do LGR (itens 1 a 10)",
            use_container_width=True,
        )
    else:
        st.error(
            f"Imagem não encontrada em `{img1_path}`. Verifique se a pasta `assets/` "
            "com as imagens foi enviada ao repositório, no mesmo diretório do app.py."
        )

    st.markdown("")
    st.markdown("**Passos 11 e 12 — Localizando raízes no LGR**")
    if os.path.exists(img2_path):
        st.image(
            img2_path,
            caption="Tabela 2.4 — Localizando Raízes no LGR (itens 11 e 12)",
            use_container_width=True,
        )
    else:
        st.error(
            f"Imagem não encontrada em `{img2_path}`. Verifique se a pasta `assets/` "
            "com as imagens foi enviada ao repositório, no mesmo diretório do app.py."
        )

    st.divider()
    st.caption(
        "📖 **Fonte / créditos:** Apostila de Sistemas de Controle, Prof. **Fábio Meneghetti "
        "Ugulino de Araújo** — Departamento de Engenharia de Computação e Automação (DCA), UFRN. "
        "Imagens reproduzidas exclusivamente para fins didáticos, como material de apoio à "
        "disciplina DCA 3701."
    )

# ----------------------------------------------------------------------------------
# O QUE E O LGR
# ----------------------------------------------------------------------------------
with tabs[14]:
    st.markdown('<p class="passo-header">O que é o Lugar Geométrico das Raízes (LGR)?</p>', unsafe_allow_html=True)

    st.markdown(
        """
O **Lugar Geométrico das Raízes** (LGR), também conhecido pelo nome em inglês *Root Locus*,
é um método gráfico desenvolvido por **Walter R. Evans em 1948** para analisar como as raízes
da equação característica de um sistema realimentado — ou seja, os **pólos de malha fechada** —
se deslocam no plano complexo **s** à medida que um parâmetro do sistema, tipicamente o **ganho K**
de um controlador, varia de 0 a infinito.
        """
    )

    st.markdown("### 🧭 Para que é usado")
    st.markdown(
        """
Em um sistema de controle com realimentação, a função de transferência de malha fechada depende
diretamente da posição dos seus pólos, que por sua vez dependem do ganho K através da equação
característica:
        """
    )
    st.latex(r"1 + K\cdot P(s) = 0")
    st.markdown(
        """
Resolver essa equação para cada valor de K seria extremamente trabalhoso. O LGR resolve esse
problema construindo, de forma gráfica, **todas as trajetórias possíveis** que os pólos de malha
fechada podem assumir conforme K varia — permitindo visualizar de uma só vez o comportamento do
sistema para qualquer ganho, sem precisar recalcular as raízes a cada tentativa.

O LGR é usado principalmente para:
- **Projetar e ajustar controladores** (proporcional, PID, compensadores de avanço/atraso de fase),
  escolhendo o ganho K ou a posição de pólos/zeros adicionais que levem o sistema ao comportamento desejado;
- **Analisar a estabilidade** do sistema em malha fechada para diferentes valores de K, incluindo
  os pontos exatos em que o sistema cruza para a instabilidade (cruzamento com o eixo imaginário);
- **Prever a resposta transitória** (tempo de acomodação, sobressinal, tempo de subida) a partir
  da posição dos pólos dominantes, já que essas posições estão diretamente ligadas ao fator de
  amortecimento (ζ) e à frequência natural (ωₙ) do sistema;
- **Determinar pontos de descolamento e de entrada** no eixo real, onde pólos reais se encontram
  e passam a se tornar complexos conjugados (ou vice-versa).
        """
    )

    st.markdown("### 🎯 Importância na área de Sistemas de Controle")
    st.markdown(
        """
O LGR é uma das ferramentas clássicas mais importantes da **Teoria de Controle Clássico**, ao lado
dos diagramas de Bode, do critério de Nyquist e do critério de Routh-Hurwitz. Sua importância vem de:

- **Visão gráfica e intuitiva**: em vez de uma resposta numérica isolada para um único K, o
  engenheiro enxerga o comportamento do sistema para *toda a faixa* de ganhos possíveis em um
  único gráfico;
- **Projeto por especificação de desempenho**: é possível traçar, sobre o próprio LGR, curvas de
  fator de amortecimento (ζ) e frequência natural (ωₙ) constantes, e escolher o ponto de operação
  (e o K correspondente) que atenda a requisitos de projeto como sobressinal máximo ou tempo de
  acomodação;
- **Robustez e margem de estabilidade**: mostra claramente até onde o ganho pode ser aumentado
  antes que o sistema se torne instável, complementando o critério de Routh-Hurwitz;
- **Base para técnicas mais avançadas**: o entendimento do LGR é pré-requisito para o projeto de
  compensadores em avanço/atraso de fase, controladores PID e outras técnicas de controle clássico
  amplamente usadas na indústria (processos industriais, robótica, aeroespacial, eletrônica de
  potência, entre outros).

Por isso, dominar a construção e a leitura do LGR — os 12 passos resolvidos nesta ferramenta —
é uma competência central na formação de quem trabalha com **Projeto de Sistemas de Controle**.
        """
    )

    st.caption(
        "📖 Referência conceitual: Apostila de Sistemas de Controle, Prof. Fábio Meneghetti "
        "Ugulino de Araújo — DCA/UFRN."
    )

st.markdown(
    '<div class="autor-footer">Disciplina: <b>DCA 3701 - Projeto de Sistemas de Controle</b> · '
    "Autor do site: <b>Daniel Medeiros Soares Carneiro</b></div>",
    unsafe_allow_html=True,
)
