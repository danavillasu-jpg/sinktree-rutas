"""
main.py - Enrutamiento adaptativo con Sink Tree y las K rutas más óptimas.

Ejecutar:  python -m streamlit run main.py
Instalar:  python -m pip install -r requirements.txt

Un solo archivo con dos partes, igual que la versión anterior:
  (1) MOTOR DE CÁLCULO: modelo de la red, costos, Dijkstra, Sink Tree,
      algoritmo de Yen (K rutas más cortas sin ciclos), simulación de tráfico.
  (2) INTERFAZ STREAMLIT: grafo interactivo (clic en nodos y en pesos),
      tablas editables, simulación en tiempo real, historial y exportación.

Cada camino (enlace) entre dos routers tiene:
  peso      w_ij  -> número que pones tú (ej. A-B = 150)
  retardo   d_ij  -> ms de propagación
  banda     b_ij  -> capacidad en Mbps
  activo          -> encendido / apagado manual
y en tiempo real además:
  carga     rho_ij(t) en [0, 0.95]  (utilización del enlace)
  falla     caída temporal aleatoria

Las 3 reglas de la simulación (se aplican en cada "tick"):
  R1 Tráfico : rho(t+1) = rho(t) + 0.15·(rho0 - rho(t)) + ruido + ráfagas ocasionales
  R2 Congestión: d_ef = d / (1 - rho)   ;   b_disp = b · (1 - rho)
                 (opcional) w_ef = w · (1 + rho)
  R3 Fallas  : un enlace cae con probabilidad p_falla y se recupera con p = 0.35
"""
from __future__ import annotations

import heapq
import json
import math
import zlib
from dataclasses import astuple, dataclass, replace
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# MOTOR DE CÁLCULO
# ============================================================================
INF = float("inf")

PESO, SALTOS, RETARDO, BANDA, ADAPT = "peso", "saltos", "retardo", "banda", "adaptativo"
NOMBRE_METRICA = {
    PESO: "Peso del camino (suma de pesos)",
    SALTOS: "Mínimo número de saltos",
    RETARDO: "Mínimo retardo efectivo",
    BANDA: "Mayor ancho de banda disponible",
    ADAPT: "Adaptativo (todas las variables)",
}
CORTO_METRICA = {PESO: "peso", SALTOS: "saltos", RETARDO: "ms", BANDA: "b_ref/b", ADAPT: "C"}

# Topología base: la de Johal (2008) + 3 enlaces extra (D-E, A-C, N-C) para que
# existan más caminos alternativos. Todo se puede cambiar desde la app.
NODOS_BASE = [("H", 0.0, 0.0), ("D", 1.2, 1.7), ("A", 2.8, 1.7), ("B", 4.2, 0.0),
              ("E", 1.1, 0.0), ("J", 2.2, 0.0), ("C", 3.2, 0.0), ("O", 1.1, -1.7),
              ("N", 2.3, -1.7)]
# (origen, destino, peso, retardo ms, banda Mbps, activo)
ENLACES_BASE = [
    ("H", "D", 120, 8, 100, True),
    ("D", "A", 90, 9, 100, True),
    ("A", "B", 150, 7, 100, True),
    ("H", "E", 60, 5, 50, True),
    ("E", "J", 110, 10, 50, True),
    ("J", "C", 70, 8, 100, True),
    ("C", "B", 80, 6, 100, True),
    ("H", "O", 75, 7, 50, True),
    ("O", "N", 95, 9, 50, True),
    ("N", "J", 85, 9, 50, True),
    ("D", "E", 40, 4, 100, True),
    ("A", "C", 65, 6, 100, True),
    ("N", "C", 130, 12, 50, True),
]
COLS_L = ["Origen", "Destino", "Peso", "Retardo_ms", "Banda_Mbps", "Activo"]
COLS_N = ["Nodo", "X", "Y"]


def clave(a: str, b: str) -> tuple[str, str]:
    """Identificador único de un enlace no dirigido (orden alfabético)."""
    return (a, b) if a <= b else (b, a)


def fmt(x) -> str:
    """Número corto: 150, 12.5, 0.27 (sin ceros sobrantes)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "∞"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"
    return s.rstrip("0").rstrip(".")


# --------------------------------------------------------------------------
# Modelo de datos
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Enlace:
    u: str
    v: str
    peso: float          # w_ij  (lo pone el usuario)
    retardo: float       # d_ij  [ms]
    banda: float         # b_ij  [Mbps]
    activo: bool = True  # encendido manual
    carga: float = 0.0   # rho_ij(t)  (simulación)
    falla: bool = False  # caída temporal (simulación)
    peso_ef: float | None = None

    @property
    def clave(self) -> tuple[str, str]:
        return clave(self.u, self.v)

    @property
    def etiqueta(self) -> str:
        return f"{self.u}-{self.v}"

    @property
    def operativo(self) -> bool:
        return self.activo and not self.falla and self.banda > 0

    @property
    def retardo_ef(self) -> float:          # R2: cola tipo M/M/1
        return self.retardo / (1.0 - self.carga)

    @property
    def banda_disp(self) -> float:          # R2: capacidad libre
        return self.banda * (1.0 - self.carga)

    @property
    def w(self) -> float:                   # peso que usa el cálculo
        return self.peso if self.peso_ef is None else self.peso_ef


@dataclass(frozen=True)
class Criterio:
    metrica: str = PESO
    alpha: float = 0.25   # saltos
    beta: float = 0.30    # retardo
    gamma: float = 0.25   # ancho de banda
    delta: float = 0.20   # peso del usuario
    banda_min: float = 0.0  # restricción QoS (Mbps disponibles mínimos)

    def pesos(self) -> tuple[float, float, float, float]:
        s = self.alpha + self.beta + self.gamma + self.delta
        if s <= 0:
            return (0.25, 0.25, 0.25, 0.25)
        return (self.alpha / s, self.beta / s, self.gamma / s, self.delta / s)


class Red:
    """Grafo no dirigido G = (V, E) con referencias para normalizar."""

    def __init__(self, enlaces: list[Enlace], nodos: list[str]):
        self.enlaces = list(enlaces)
        self.nodos = list(nodos)
        for e in self.enlaces:
            for n in (e.u, e.v):
                if n not in self.nodos:
                    self.nodos.append(n)
        self._idx = {e.clave: i for i, e in enumerate(self.enlaces)}
        self.d_ref = max((e.retardo for e in self.enlaces if e.retardo > 0), default=1.0)
        self.b_ref = max((e.banda for e in self.enlaces if e.banda > 0), default=1.0)
        self.w_ref = max((e.peso for e in self.enlaces if e.peso > 0), default=1.0)

    def enlace(self, a: str, b: str) -> Enlace:
        return self.enlaces[self._idx[clave(a, b)]]

    def tiene(self, a: str, b: str) -> bool:
        return clave(a, b) in self._idx

    def adyacencia(self, crit: Criterio):
        """Lista de adyacencia + diccionario de costos (ambos sentidos).
        Los enlaces con costo infinito (caídos o sin banda suficiente) no entran."""
        adj: dict[str, list[tuple[str, float]]] = {n: [] for n in self.nodos}
        costo: dict[tuple[str, str], float] = {}
        for e in self.enlaces:
            c = costo_enlace(e, crit, self)
            if c == INF:
                continue
            adj[e.u].append((e.v, c))
            adj[e.v].append((e.u, c))
            costo[(e.u, e.v)] = costo[(e.v, e.u)] = c
        for lista in adj.values():
            lista.sort()  # desempates deterministas
        return adj, costo


# --------------------------------------------------------------------------
# Costo de un enlace según la métrica elegida
# --------------------------------------------------------------------------
def costo_enlace(e: Enlace, crit: Criterio, red: Red) -> float:
    if not e.operativo or e.banda_disp < crit.banda_min:
        return INF
    m = crit.metrica
    if m == PESO:
        return e.w
    if m == SALTOS:
        return 1.0
    if m == RETARDO:
        return e.retardo_ef
    if m == BANDA:  # como OSPF: costo = referencia / ancho de banda
        return red.b_ref / max(e.banda_disp, 1e-9)
    a, b, g, d = crit.pesos()
    return (a * 1.0
            + b * e.retardo_ef / red.d_ref
            + g * red.b_ref / max(e.banda_disp, 1e-9)
            + d * e.w / red.w_ref)


# --------------------------------------------------------------------------
# Dijkstra, Sink Tree y K rutas (Yen)
# --------------------------------------------------------------------------
def dijkstra(adj, s, t=None, ban_n=frozenset(), ban_e=frozenset()):
    """Dijkstra con montículo binario: O((V + E) log V).
    ban_n / ban_e: nodos y arcos dirigidos prohibidos (los usa Yen)."""
    dist = {s: 0.0}
    prev: dict[str, str] = {}
    cola = [(0.0, s)]
    cerrados: set[str] = set()
    while cola:
        d, u = heapq.heappop(cola)
        if u in cerrados:
            continue
        cerrados.add(u)
        if u == t:
            break
        for v, w in adj.get(u, ()):
            if v in cerrados or v in ban_n or (u, v) in ban_e:
                continue
            nd = d + w
            if nd < dist.get(v, INF) - 1e-12:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(cola, (nd, v))
    return dist, prev


def reconstruir(prev, s, t):
    if s == t:
        return [s]
    if t not in prev:
        return None
    camino = [t]
    while camino[-1] != s:
        camino.append(prev[camino[-1]])
    return camino[::-1]


def sink_tree(adj, destino):
    """Dijkstra lanzado DESDE el destino. Como la red es no dirigida,
    prev[n] es justo el siguiente salto de n hacia el destino.
    Devuelve (costo mínimo de cada nodo al destino, siguiente salto)."""
    return dijkstra(adj, destino)


def camino_arbol(siguiente, origen, destino):
    """Sigue los siguientes saltos del Sink Tree desde el origen."""
    if origen == destino:
        return [origen]
    if origen not in siguiente:
        return None
    ruta = [origen]
    while ruta[-1] != destino and len(ruta) <= len(siguiente) + 1:
        ruta.append(siguiente[ruta[-1]])
    return ruta


def costo_ruta(ruta, costo) -> float:
    return sum(costo[(a, b)] for a, b in zip(ruta, ruta[1:]))


def k_rutas(adj, costo, origen, destino, k=3, primera=None):
    """Algoritmo de Yen (1971): las k rutas simples de menor costo.
    La 1ª ruta sale del Sink Tree (principio de optimalidad); las demás
    se obtienen desviando la ruta anterior en cada nodo ('spur node')."""
    if primera is None:
        _, prev = dijkstra(adj, origen, destino)
        primera = reconstruir(prev, origen, destino)
    if not primera or len(primera) < 2:
        return [primera] if primera else []
    A = [primera]
    B: list[tuple[float, int, list[str]]] = []
    vistos = {tuple(primera)}
    while len(A) < k:
        ultima = A[-1]
        for i in range(len(ultima) - 1):
            spur, raiz = ultima[i], ultima[: i + 1]
            ban_e = {(p[i], p[i + 1]) for p in A if len(p) > i + 1 and p[: i + 1] == raiz}
            ban_n = set(raiz[:-1])
            _, prev = dijkstra(adj, spur, destino, ban_n, ban_e)
            tramo = reconstruir(prev, spur, destino)
            if not tramo:
                continue
            nueva = raiz[:-1] + tramo
            t = tuple(nueva)
            if t in vistos:
                continue
            vistos.add(t)
            heapq.heappush(B, (costo_ruta(nueva, costo), len(nueva), nueva))
        if not B:
            break
        A.append(heapq.heappop(B)[2])
    return A


def metricas_ruta(red: Red, ruta, costo) -> dict:
    tramos = [red.enlace(a, b) for a, b in zip(ruta, ruta[1:])]
    pesos = [t.w for t in tramos]
    cs = [costo[(a, b)] for a, b in zip(ruta, ruta[1:])]
    return {
        "saltos": len(tramos),
        "peso": sum(pesos),
        "suma_pesos": " + ".join(fmt(p) for p in pesos) + f" = {fmt(sum(pesos))}",
        "suma_costos": " + ".join(fmt(c) for c in cs) + f" = {fmt(sum(cs))}",
        "retardo": sum(t.retardo_ef for t in tramos),
        "banda": min((t.banda_disp for t in tramos), default=0.0),
        "costo": sum(cs),
    }


# --------------------------------------------------------------------------
# Simulación en tiempo real: las 3 reglas
# --------------------------------------------------------------------------
def paso_trafico(dyn: dict, claves, rng, volatilidad=0.06, p_falla=0.01,
                 p_recupera=0.35, p_rafaga=0.03):
    """Avanza un tick. dyn[clave] = {'rho', 'rho0', 'caido'}."""
    for k in claves:
        s = dyn.get(k)
        if s is None:
            s = dyn[k] = {"rho": 0.0, "rho0": float(rng.uniform(0.10, 0.60)), "caido": False}
        # R1: tráfico con reversión a la media + ruido + ráfagas
        rho = s["rho"] + 0.15 * (s["rho0"] - s["rho"]) + rng.normal(0.0, volatilidad)
        if rng.random() < p_rafaga:
            rho += rng.uniform(0.20, 0.45)
        s["rho"] = float(min(max(rho, 0.0), 0.95))
        # R3: fallas y recuperaciones
        if s["caido"]:
            if rng.random() < p_recupera:
                s["caido"] = False
        elif rng.random() < p_falla:
            s["caido"] = True


def red_viva(base: list[Enlace], dyn: dict, trafico_afecta_peso: bool) -> list[Enlace]:
    """Aplica el estado dinámico (R2) sobre los enlaces base."""
    salida = []
    for e in base:
        s = dyn.get(e.clave)
        carga, caido = (s["rho"], s["caido"]) if s else (0.0, False)
        pe = e.peso * (1.0 + carga) if trafico_afecta_peso else e.peso
        salida.append(replace(e, carga=carga, falla=caido, peso_ef=pe))
    return salida


# --------------------------------------------------------------------------
# Posiciones de los nodos (dibujo)
# --------------------------------------------------------------------------
def layout_fr(nodos, aristas, semilla=7, iteraciones=300):
    """Fruchterman-Reingold vectorizado con numpy."""
    n = len(nodos)
    if n == 0:
        return {}
    if n == 1:
        return {nodos[0]: (0.0, 0.0)}
    rng = np.random.default_rng(semilla)
    P = rng.uniform(-1, 1, (n, 2))
    idx = {v: i for i, v in enumerate(nodos)}
    E = np.array([(idx[u], idx[v]) for u, v in aristas if u in idx and v in idx], dtype=int)
    k = math.sqrt(4.0 / n)
    temp = 0.25
    for _ in range(iteraciones):
        D = P[:, None, :] - P[None, :, :]
        dist2 = (D ** 2).sum(-1) + 1e-9
        disp = ((k * k / dist2)[..., None] * D).sum(1)            # repulsión
        if len(E):
            d = P[E[:, 0]] - P[E[:, 1]]
            largo = np.sqrt((d ** 2).sum(1))[:, None] + 1e-9
            f = d * largo / k                                      # atracción
            np.add.at(disp, E[:, 0], -f)
            np.add.at(disp, E[:, 1], f)
        disp -= 0.05 * P                                           # gravedad
        L = np.sqrt((disp ** 2).sum(1))[:, None] + 1e-9
        P += disp / L * np.minimum(L, temp)
        temp *= 0.985
    P -= P.mean(0)
    P /= max(np.abs(P).max(), 1e-9)
    P *= 3.0
    return {v: (float(P[i, 0]), float(P[i, 1])) for v, i in idx.items()}


def completar_posiciones(nodos, pos, aristas):
    """Ubica los nodos sin X/Y cerca de sus vecinos (determinista)."""
    faltan = [n for n in nodos if n not in pos]
    if not faltan:
        return pos
    if not pos:
        return layout_fr(nodos, aristas)
    pos = dict(pos)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    vecinos: dict[str, list[str]] = {}
    for u, v in aristas:
        vecinos.setdefault(u, []).append(v)
        vecinos.setdefault(v, []).append(u)
    for i, n in enumerate(faltan):
        h = zlib.crc32(n.encode("utf-8"))
        ang = (h % 360) * math.pi / 180
        r = 0.7 + ((h >> 9) % 100) / 200
        cerca = [pos[m] for m in vecinos.get(n, []) if m in pos]
        if cerca:
            cx = sum(p[0] for p in cerca) / len(cerca)
            cy = sum(p[1] for p in cerca) / len(cerca)
        else:
            cx, cy = max(xs) + 1.0, (min(ys) + max(ys)) / 2 + (i % 5 - 2) * 0.8
            r = 0.3
        pos[n] = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    return pos


def nombres(n: int) -> list[str]:
    """A, B, ..., Z, AA, AB, ... (como columnas de Excel)."""
    salida = []
    for i in range(n):
        s, j = "", i
        while True:
            s = chr(65 + j % 26) + s
            j = j // 26 - 1
            if j < 0:
                break
        salida.append(s)
    return salida


def red_aleatoria(n: int, grado: float, wmin: float, wmax: float, semilla: int):
    """Red 'realista': routers en el plano, árbol de expansión mínima (garantiza
    conexión) + enlaces cortos hasta el grado medio pedido.
    Peso y retardo proporcionales a la distancia (regla de tres) con ruido."""
    rng = np.random.default_rng(semilla if semilla > 0 else None)
    nom = nombres(n)
    lado = math.ceil(math.sqrt(n * 1.6))
    celdas = rng.choice(lado * lado, size=n, replace=False)
    P = np.array([(c % lado, c // lado) for c in celdas], dtype=float)
    P += rng.uniform(-0.2, 0.2, P.shape)
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    # Prim O(n^2)
    en_arbol = [0]
    mejor = D[0].copy()
    padre = np.zeros(n, dtype=int)
    usados = np.zeros(n, dtype=bool)
    usados[0] = True
    aristas = set()
    for _ in range(n - 1):
        m = np.where(usados, np.inf, mejor)
        j = int(np.argmin(m))
        aristas.add(clave(nom[j], nom[padre[j]]))
        usados[j] = True
        en_arbol.append(j)
        cambia = D[j] < mejor
        mejor = np.where(cambia, D[j], mejor)
        padre = np.where(cambia, j, padre)
    objetivo = max(n - 1, round(n * grado / 2))
    pares = sorted((D[i, j], i, j) for i in range(n) for j in range(i + 1, n))
    for _, i, j in pares:
        if len(aristas) >= objetivo:
            break
        aristas.add(clave(nom[i], nom[j]))
    idx = {v: i for i, v in enumerate(nom)}
    largos = [D[idx[u], idx[v]] for u, v in aristas]
    dmin, dmax = min(largos), max(largos)
    filas = []
    for u, v in sorted(aristas):
        d = D[idx[u], idx[v]]
        frac = (d - dmin) / (dmax - dmin) if dmax > dmin else 0.5
        frac = min(max(frac + rng.uniform(-0.15, 0.15), 0.0), 1.0)
        peso = float(round(wmin + (wmax - wmin) * frac))
        retardo = float(round(1 + 3 * d + rng.uniform(0, 3)))
        banda = float(rng.choice([10, 20, 50, 100, 1000], p=[0.1, 0.15, 0.3, 0.35, 0.1]))
        filas.append((u, v, peso, retardo, banda, True))
    nodos = [(nom[i], round(float(P[i, 0]) * 0.8, 3), round(-float(P[i, 1]) * 0.8, 3))
             for i in range(n)]
    origen = nom[0]
    destino = nom[int(np.argmax(D[0]))]
    return filas, nodos, origen, destino


# --------------------------------------------------------------------------
# Lectura y validación de las tablas
# --------------------------------------------------------------------------
def _vacio(x) -> bool:
    return x is None or x is pd.NA or (isinstance(x, float) and math.isnan(x))


def _txt(x) -> str:
    return "" if _vacio(x) else str(x).strip()


def _num(x, defecto: float) -> float:
    try:
        v = float(x)
        return defecto if math.isnan(v) else v
    except (TypeError, ValueError):
        return defecto


def leer_tablas(df_l: pd.DataFrame, df_n: pd.DataFrame):
    avisos: list[str] = []
    nodos: list[str] = []
    pos: dict[str, tuple[float, float]] = {}
    for fila in df_n.itertuples(index=False):
        n = _txt(fila.Nodo)
        if not n:
            continue
        if n in nodos:
            avisos.append(f"Nodo repetido «{n}»: se ignora la copia.")
            continue
        nodos.append(n)
        x, y = _num(fila.X, math.nan), _num(fila.Y, math.nan)
        if math.isfinite(x) and math.isfinite(y):
            pos[n] = (x, y)
    enlaces: list[Enlace] = []
    vistos = set()
    for fila in df_l.itertuples(index=False):
        u, v = _txt(fila.Origen), _txt(fila.Destino)
        if not u or not v:
            continue
        if u == v:
            avisos.append(f"Enlace {u}-{v} ignorado: un nodo no se conecta consigo mismo.")
            continue
        k = clave(u, v)
        if k in vistos:
            avisos.append(f"Enlace {u}-{v} repetido: se usa solo el primero.")
            continue
        vistos.add(k)
        peso = _num(fila.Peso, 1.0)
        if peso < 0:
            avisos.append(f"Peso negativo en {u}-{v}: Dijkstra exige pesos ≥ 0, se usa 0.")
            peso = 0.0
        act = True if _vacio(fila.Activo) else bool(fila.Activo)
        enlaces.append(Enlace(u, v, peso, max(_num(fila.Retardo_ms, 1.0), 0.0),
                              max(_num(fila.Banda_Mbps, 100.0), 0.0), act))
        for n in (u, v):
            if n not in nodos:
                nodos.append(n)
    pos = completar_posiciones(nodos, pos, [(e.u, e.v) for e in enlaces])
    return enlaces, nodos, pos, avisos


# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================
st.set_page_config(page_title="Sink Tree · Rutas óptimas", layout="wide")

C_NODO, C_ORIGEN, C_DESTINO, C_AISLADO = "#264653", "#3a86ff", "#ffb703", "#adb5bd"
C_ENLACE, C_CAIDO, C_QOS, C_ARBOL = "#5f6f7f", "#d62839", "#e07a1f", "#1b7f74"
C_RUTAS = ["#00a676", "#f77f00", "#7b2cbf", "#d00070", "#0077b6"]

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem;}
      div[data-testid="stMetric"] {background:#f4f7fa; border-left:5px solid #2a9d8f;
                                    padding:8px 12px; border-radius:6px;}
      div[data-testid="stMetric"] * {color:#10222e !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

ss = st.session_state


def df_enlaces(filas) -> pd.DataFrame:
    df = pd.DataFrame(filas, columns=COLS_L)
    return df.astype({"Peso": float, "Retardo_ms": float, "Banda_Mbps": float, "Activo": bool})


def df_nodos(filas) -> pd.DataFrame:
    return pd.DataFrame(filas, columns=COLS_N).astype({"X": float, "Y": float})


# --------------------------------------------------------------------------
# Estado de sesión
# --------------------------------------------------------------------------
if "links_src" not in ss:
    ss.links_src = df_enlaces(ENLACES_BASE)
    ss.nodes_src = df_nodos(NODOS_BASE)
    ss.ver = 0
    ss.dyn = {}
    ss.tick = 0
    ss.hist = []
    ss.eventos = []
    ss.ctx = None
    ss.rng = np.random.default_rng()
    ss.arista_sel = None
    ss.clics = 0
    ss.origen, ss.destino = "H", "B"

for _k, _v in {"metrica": PESO, "K": 3, "pa": 0.25, "pb": 0.30, "pg": 0.25, "pd": 0.20,
               "bmin": 0.0, "sim": True, "intervalo": 6.0, "vol": 0.04, "pfalla": 0.01,
               "afecta": False, "alto": 620, "tam": 40, "etiqueta": "Peso",
               "overlay": False, "modo_clic": "Destino"}.items():
    ss.setdefault(_k, _v)

# Cambios pedidos desde un clic en el grafo (se aplican antes de crear widgets)
for _k, _v in ss.pop("_pend", {}).items():
    ss[_k] = _v


def aplicar(links: pd.DataFrame, nodes: pd.DataFrame, reiniciar=False):
    """Guarda una nueva versión de la red y reconstruye los editores."""
    ss.links_src = df_enlaces(links[COLS_L].values.tolist()) if len(links) else df_enlaces([])
    ss.nodes_src = df_nodos(nodes[COLS_N].values.tolist()) if len(nodes) else df_nodos([])
    ss.ver += 1
    if reiniciar:
        ss.dyn, ss.hist, ss.eventos, ss.arista_sel = {}, [], [], None


def nodos_con_posicion() -> pd.DataFrame:
    return df_nodos([(n, round(ss.pos[n][0], 3), round(ss.pos[n][1], 3)) for n in ss.nodos])


# --------------------------------------------------------------------------
# Encabezado y pestañas
# --------------------------------------------------------------------------
st.title("Enrutamiento adaptativo · Sink Tree y rutas óptimas")
st.caption("Teoría de grafos: Dijkstra (Sink Tree) + Yen (K rutas más cortas) sobre una red "
           "con pesos, saltos, retardo y ancho de banda que cambian en tiempo real.")

tab_vivo, tab_red, tab_teoria, tab_export = st.tabs(
    ["Red en vivo", "Construir / editar red", "Teoría y fórmulas", "Exportar"])

# --------------------------------------------------------------------------
# Construir / editar red  (se ejecuta primero: el resto depende de ella)
# --------------------------------------------------------------------------
with tab_red:
    zona_herr = st.container()
    st.markdown("#### Tablas editables")
    st.caption("Escribe directamente en las celdas. Agrega filas con el botón + al final de cada tabla y "
               "bórralas seleccionándolas. Un nodo nuevo puede quedar sin X/Y: se ubica solo.")
    c_nod, c_enl = st.columns([1, 2.3])
    with c_nod:
        st.markdown("**Nodos (routers)**")
        ed_nodos = st.data_editor(
            ss.nodes_src, key=f"ed_nodos_{ss.ver}", num_rows="dynamic", hide_index=True,
            column_config={
                "Nodo": st.column_config.TextColumn("Nodo", required=True),
                "X": st.column_config.NumberColumn("X", step=0.1, format="%.2f"),
                "Y": st.column_config.NumberColumn("Y", step=0.1, format="%.2f"),
            })
    with c_enl:
        st.markdown("**Caminos (enlaces) y su peso**")
        ed_enlaces = st.data_editor(
            ss.links_src, key=f"ed_enlaces_{ss.ver}", num_rows="dynamic", hide_index=True,
            column_config={
                "Origen": st.column_config.TextColumn("Desde", required=True),
                "Destino": st.column_config.TextColumn("Hasta", required=True),
                "Peso": st.column_config.NumberColumn("Peso", min_value=0.0, step=1.0,
                                                      help="Número que se suma para elegir rutas"),
                "Retardo_ms": st.column_config.NumberColumn("Retardo (ms)", min_value=0.0, step=1.0),
                "Banda_Mbps": st.column_config.NumberColumn("Ancho de banda (Mbps)", min_value=0.0,
                                                            step=1.0),
                "Activo": st.column_config.CheckboxColumn("Activo", default=True),
            })

    ss.links_cur, ss.nodes_cur = ed_enlaces, ed_nodos
    ss.base_enlaces, ss.nodos, ss.pos, _avisos = leer_tablas(ed_enlaces, ed_nodos)
    for _a in _avisos:
        st.warning(_a)
    st.caption(f"Red actual: **{len(ss.nodos)} nodos** y **{len(ss.base_enlaces)} enlaces**.")

    # ---- Herramientas rápidas (se dibujan arriba de las tablas) ---------------
    with zona_herr:
        st.markdown("#### Herramientas")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            with st.form("f_nodo", clear_on_submit=True):
                st.markdown("**Agregar nodo**")
                sugerido = next(n for n in nombres(len(ss.nodos) + 30) if n not in ss.nodos)
                nuevo = st.text_input("Nombre", value=sugerido)
                conectar = st.multiselect("Conectar con", ss.nodos)
                w_n = st.number_input("Peso de cada camino nuevo", 0.0, value=100.0, step=1.0)
                r_n = st.number_input("Retardo (ms)", 0.0, value=5.0, step=1.0)
                b_n = st.number_input("Ancho de banda (Mbps)", 0.0, value=100.0, step=1.0)
                if st.form_submit_button("Agregar nodo", width="stretch"):
                    nuevo = nuevo.strip()
                    if not nuevo or nuevo in ss.nodos:
                        st.error("Nombre vacío o repetido.")
                    else:
                        nd = pd.concat([nodos_con_posicion(),
                                        df_nodos([(nuevo, math.nan, math.nan)])], ignore_index=True)
                        nl = pd.concat([ss.links_cur, df_enlaces(
                            [(nuevo, m, w_n, r_n, b_n, True) for m in conectar])], ignore_index=True)
                        aplicar(nl, nd)
                        st.rerun()
        with h2:
            with st.form("f_enlace"):
                st.markdown("**Crear o cambiar un camino**")
                ca, cb = st.columns(2)
                u_e = ca.selectbox("Desde", ss.nodos, key="fe_u")
                v_e = cb.selectbox("Hasta", ss.nodos, index=min(1, len(ss.nodos) - 1), key="fe_v")
                w_e = st.number_input("Peso", 0.0, value=150.0, step=1.0)
                r_e = st.number_input("Retardo (ms) ", 0.0, value=5.0, step=1.0)
                b_e = st.number_input("Ancho de banda (Mbps) ", 0.0, value=100.0, step=1.0)
                if st.form_submit_button("Guardar camino", width="stretch"):
                    if not u_e or u_e == v_e:
                        st.error("Elige dos nodos distintos.")
                    else:
                        nl = ss.links_cur.copy()
                        k = clave(u_e, v_e)
                        hit = [i for i, r in nl.iterrows()
                               if clave(_txt(r.Origen), _txt(r.Destino)) == k]
                        if hit:
                            nl.loc[hit[0], ["Peso", "Retardo_ms", "Banda_Mbps", "Activo"]] = \
                                [w_e, r_e, b_e, True]
                        else:
                            nl = pd.concat([nl, df_enlaces([(u_e, v_e, w_e, r_e, b_e, True)])],
                                           ignore_index=True)
                        aplicar(nl, nodos_con_posicion())
                        st.rerun()
        with h3:
            with st.form("f_rand"):
                st.markdown("**Red aleatoria realista**")
                n_r = st.slider("Número de nodos", 3, 60, 12)
                g_r = st.slider("Grado medio (conexiones por nodo)", 1.5, 5.0, 2.8, 0.1)
                cw1, cw2 = st.columns(2)
                wmin = cw1.number_input("Peso mín.", 0.0, value=10.0, step=5.0)
                wmax = cw2.number_input("Peso máx.", 1.0, value=200.0, step=5.0)
                sem = st.number_input("Semilla (0 = al azar)", 0, value=0, step=1)
                if st.form_submit_button("Generar red", width="stretch"):
                    filas, nods, o_r, d_r = red_aleatoria(n_r, g_r, min(wmin, wmax),
                                                          max(wmin, wmax), int(sem))
                    aplicar(df_enlaces(filas), df_nodos(nods), reiniciar=True)
                    ss._pend = {"origen": o_r, "destino": d_r}
                    st.rerun()
        with h4:
            st.markdown("**Quitar y ordenar**")
            quitar_n = st.selectbox("Nodo a quitar", ss.nodos, key="q_nodo")
            if st.button("Quitar nodo y sus caminos", width="stretch", disabled=not ss.nodos):
                nl = ss.links_cur[(ss.links_cur.Origen.map(_txt) != quitar_n)
                                  & (ss.links_cur.Destino.map(_txt) != quitar_n)]
                nd = nodos_con_posicion()
                aplicar(nl, nd[nd.Nodo != quitar_n])
                st.rerun()
            etqs = [e.clave for e in ss.base_enlaces]
            ku = st.selectbox("Camino a quitar", etqs, key="q_enlace",
                              format_func=lambda k: f"{k[0]}-{k[1]}")
            if st.button("Quitar camino", width="stretch", disabled=not etqs):
                nl = ss.links_cur[[clave(_txt(r.Origen), _txt(r.Destino)) != ku
                                   for r in ss.links_cur.itertuples()]]
                aplicar(nl, nodos_con_posicion())
                st.rerun()
            if st.button("Reorganizar dibujo automático", width="stretch"):
                ss.pos = layout_fr(ss.nodos, [(e.u, e.v) for e in ss.base_enlaces])
                aplicar(ss.links_cur, nodos_con_posicion())
                st.rerun()
            if st.button("Restaurar red de ejemplo", width="stretch"):
                aplicar(df_enlaces(ENLACES_BASE), df_nodos(NODOS_BASE), reiniciar=True)
                ss._pend = {"origen": "H", "destino": "B"}
                st.rerun()

# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
if ss.nodos:
    if ss.origen not in ss.nodos:
        ss.origen = ss.nodos[0]
    if ss.destino not in ss.nodos:
        ss.destino = ss.nodos[-1]


def limpiar_trafico():
    for s in ss.dyn.values():
        s["rho"], s["caido"] = 0.0, False


with st.sidebar:
    # ---- Editor del camino en el que se hizo clic ------------------------
    if ss.arista_sel:
        fila_sel = [i for i, r in ss.links_cur.iterrows()
                    if clave(_txt(r.Origen), _txt(r.Destino)) == tuple(ss.arista_sel)]
        if not fila_sel:
            ss.arista_sel = None
        else:
            i_sel = fila_sel[0]
            r_sel = ss.links_cur.loc[i_sel]
            u_s, v_s = ss.arista_sel
            st.markdown(f"### Editar camino {u_s} – {v_s}")
            with st.form(f"f_sel_{u_s}_{v_s}_{ss.ver}"):
                w_s = st.number_input("Peso", 0.0, value=_num(r_sel.Peso, 1.0), step=1.0)
                d_s = st.number_input("Retardo (ms)", 0.0, value=_num(r_sel.Retardo_ms, 1.0),
                                      step=1.0)
                b_s = st.number_input("Ancho de banda (Mbps)", 0.0,
                                      value=_num(r_sel.Banda_Mbps, 100.0), step=1.0)
                a_s = st.checkbox("Activo", value=True if _vacio(r_sel.Activo)
                                  else bool(r_sel.Activo))
                if st.form_submit_button("Guardar", type="primary", width="stretch"):
                    nl = ss.links_cur.copy()
                    nl.loc[i_sel, ["Peso", "Retardo_ms", "Banda_Mbps", "Activo"]] = \
                        [w_s, d_s, b_s, a_s]
                    aplicar(nl, nodos_con_posicion())
                    st.rerun()
            if st.button("Cerrar editor", width="stretch"):
                ss.arista_sel = None
                st.rerun()
            st.divider()

    st.title("Panel de control")
    if not ss.nodos:
        st.error("La red está vacía. Agrega nodos en «Construir / editar red».")
    else:
        c_o, c_d = st.columns(2)
        c_o.selectbox("Origen", ss.nodos, key="origen")
        c_d.selectbox("Destino", ss.nodos, key="destino")
        st.radio("Clic en un nodo del grafo lo vuelve:", ["Destino", "Origen"], key="modo_clic",
                 horizontal=True)
    st.number_input("Cantidad de rutas óptimas (K)", 1, 5, step=1, key="K")
    st.selectbox("Qué se optimiza", list(NOMBRE_METRICA), key="metrica",
                 format_func=NOMBRE_METRICA.get)
    if ss.metrica == ADAPT:
        st.caption("Importancia de cada variable (se normalizan entre su suma)")
        st.slider("α · saltos", 0.0, 1.0, step=0.05, key="pa")
        st.slider("β · retardo", 0.0, 1.0, step=0.05, key="pb")
        st.slider("γ · ancho de banda", 0.0, 1.0, step=0.05, key="pg")
        st.slider("δ · peso del camino", 0.0, 1.0, step=0.05, key="pd")
    st.number_input("Banda mínima exigida (Mbps)", 0.0, step=5.0, key="bmin",
                    help="Restricción de QoS: se descartan enlaces cuya banda DISPONIBLE "
                         "sea menor. Cambia las rutas cuando sube la carga.")

    st.divider()
    st.subheader("Tiempo real")
    st.toggle("Simulación de tráfico activa", key="sim")
    st.slider("Actualizar cada (segundos)", 2.0, 60.0, step=1.0, key="intervalo")
    st.slider("Volatilidad del tráfico (R1)", 0.0, 0.20, step=0.01, key="vol")
    st.slider("Probabilidad de falla por tick (R3)", 0.0, 0.20, step=0.005, key="pfalla",
              format="%.3f")
    st.checkbox("El tráfico también modifica el peso (w·(1+ρ))", key="afecta")
    cb1, cb2 = st.columns(2)
    if cb1.button("Un paso", width="stretch", disabled=ss.sim):
        ss._paso = True
    cb2.button("Quitar tráfico", width="stretch", on_click=limpiar_trafico)

    st.divider()
    st.subheader("Vista del grafo")
    st.slider("Alto del grafo (px)", 380, 1200, step=20, key="alto")
    st.slider("Tamaño de los nodos", 20, 70, step=2, key="tam")
    st.selectbox("Número sobre cada camino", ["Peso", "Costo actual", "Retardo", "Banda disponible",
                                              "Carga"], key="etiqueta")
    st.checkbox("Superponer el Sink Tree en el grafo", key="overlay")


# --------------------------------------------------------------------------
# Dibujo del grafo
# --------------------------------------------------------------------------
def texto_etiqueta(e: Enlace, costo_e: float, tipo: str) -> str:
    if not e.operativo:
        return "caído"
    if tipo == "Peso":
        return fmt(round(e.w, 1))
    if tipo == "Costo actual":
        return fmt(round(costo_e, 2)) if costo_e != INF else "QoS"
    if tipo == "Retardo":
        return f"{e.retardo_ef:.0f}ms"
    if tipo == "Banda disponible":
        return f"{e.banda_disp:.0f}M"
    return f"{e.carga * 100:.0f}%"


def color_carga(c: float) -> tuple[str, str]:
    if c < 0.3:
        return "#ffffff", "#6c757d"
    if c < 0.6:
        return "#fff4cc", "#c9a227"
    if c < 0.8:
        return "#ffe1c7", "#e07a1f"
    return "#ffd3d8", "#d62839"


def dibujar(red: Red, pos, origen, destino, rutas, sig, dist, costo, modo="rutas"):
    fig = go.Figure()
    mapa = {}
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    paso = span * 0.032
    en_arbol = {clave(n, s) for n, s in sig.items()}

    # 1) Enlaces base agrupados (pocas trazas = dibujo rápido)
    grupos: dict[tuple, list] = {}
    for e in red.enlaces:
        (x0, y0), (x1, y1) = pos[e.u], pos[e.v]
        if not e.operativo:
            estilo = ("Enlace caído", C_CAIDO, 2.5, "dash")
        elif (e.u, e.v) not in costo:
            estilo = ("Excluido por QoS", C_QOS, 2.5, "dot")
        elif modo == "sink":
            estilo = ("Rama del Sink Tree", C_ARBOL, 6, "solid") if e.clave in en_arbol \
                else ("No usado por el árbol", "#b0bac4", 2, "dot")
        else:
            r = e.banda / red.b_ref
            ancho = 2.5 if r <= 0.34 else (3.5 if r <= 0.67 else 5.0)
            estilo = ("Enlace", C_ENLACE, ancho, "solid")
        grupos.setdefault(estilo, [[], []])
        grupos[estilo][0] += [x0, x1, None]
        grupos[estilo][1] += [y0, y1, None]
    for (nombre, color, ancho, trazo), (gx, gy) in grupos.items():
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", hoverinfo="skip", name=nombre,
                                 line=dict(color=color, width=ancho, dash=trazo),
                                 showlegend=nombre not in ("Enlace",)))

    # 2) Rutas óptimas: líneas paralelas desplazadas para que se vean todas
    if modo == "rutas":
        desplaz = [0, 1, -1, 2, -2]
        for i in reversed(range(len(rutas))):
            r = rutas[i]
            off = desplaz[i] * paso
            rx, ry = [], []
            for a, b in zip(r, r[1:]):
                u, v = clave(a, b)
                (x0, y0), (x1, y1) = pos[u], pos[v]
                L = math.hypot(x1 - x0, y1 - y0) or 1.0
                nx, ny = -(y1 - y0) / L * off, (x1 - x0) / L * off
                rx += [x0 + nx, x1 + nx, None]
                ry += [y0 + ny, y1 + ny, None]
            fig.add_trace(go.Scatter(
                x=rx, y=ry, mode="lines", hoverinfo="skip", opacity=1.0, legendrank=i + 1,
                line=dict(color=C_RUTAS[i], width=[8, 6, 5, 4, 4][i]),
                name=f"Ruta {i + 1} · {fmt(round(costo_ruta(r, costo), 3))}"))

    # 3) Número sobre cada camino (clicable)
    lx, ly, tx, hv, fondo, borde, tam, claves_e = [], [], [], [], [], [], [], []
    for e in red.enlaces:
        (x0, y0), (x1, y1) = pos[e.u], pos[e.v]
        c_e = costo.get((e.u, e.v), INF)
        t = texto_etiqueta(e, c_e, ss.etiqueta)
        f, b = color_carga(e.carga) if e.operativo else ("#ffe3e8", C_CAIDO)
        lx.append((x0 + x1) / 2)
        ly.append((y0 + y1) / 2)
        tx.append(t)
        fondo.append(f)
        borde.append(b)
        tam.append(16 + 7.5 * len(t))
        claves_e.append(e.clave)
        estado = "activo" if e.operativo else ("caído (simulación)" if e.falla else "apagado")
        hv.append(
            f"<b>Camino {e.u}–{e.v}</b><br>Peso: {fmt(e.peso)}"
            + (f" → efectivo {fmt(round(e.w, 1))}" if abs(e.w - e.peso) > 1e-9 else "")
            + f"<br>Retardo: {fmt(e.retardo)} ms → efectivo {e.retardo_ef:.1f} ms"
            + f"<br>Banda: {fmt(e.banda)} Mbps → disponible {e.banda_disp:.1f} Mbps"
            + f"<br>Carga ρ: {e.carga * 100:.0f}%<br>Estado: {estado}"
            + f"<br>Costo ({CORTO_METRICA[ss.metrica]}): {fmt(round(c_e, 4))}"
            + "<br><i>Clic para editar</i>")
    fig.add_trace(go.Scatter(
        x=lx, y=ly, mode="markers+text", text=tx, textposition="middle center",
        textfont=dict(size=12, color="#10222e"), hovertext=hv, hoverinfo="text",
        marker=dict(symbol="square", size=tam, color=fondo, line=dict(color=borde, width=1.6)),
        customdata=[f"{a}|{b}" for a, b in claves_e], showlegend=False,
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
    mapa["etiquetas"], mapa["claves"] = len(fig.data) - 1, claves_e

    # 4) Nodos
    orden = [n for n in red.nodos if n in pos]
    colores, bordes, tamanos, hov = [], [], [], []
    for n in orden:
        if n == origen:
            colores.append(C_ORIGEN)
        elif n == destino:
            colores.append(C_DESTINO)
        elif n not in dist:
            colores.append(C_AISLADO)
        else:
            colores.append(C_NODO)
        bordes.append("#10222e" if n in (origen, destino) else "white")
        tamanos.append(ss.tam * (1.18 if n in (origen, destino) else 1))
        if n == destino:
            hov.append(f"<b>{n}</b> (destino, raíz del Sink Tree)")
        elif n in dist:
            hov.append(f"<b>{n}</b><br>Costo a {destino}: {fmt(round(dist[n], 4))}"
                       f"<br>Siguiente salto: {sig[n]}"
                       f"<br>Saltos: {len(camino_arbol(sig, n, destino)) - 1}")
        else:
            hov.append(f"<b>{n}</b><br>Sin camino a {destino}")
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in orden], y=[pos[n][1] for n in orden], mode="markers+text",
        text=orden, textposition="middle center",
        textfont=dict(color=["#10222e" if n == destino else "white" for n in orden],
                      size=max(10, int(ss.tam * 0.38))),
        marker=dict(size=tamanos, color=colores, line=dict(color=bordes, width=2.5)),
        hovertext=hov, hoverinfo="text", customdata=orden, showlegend=False,
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
    mapa["nodos"], mapa["orden"] = len(fig.data) - 1, orden

    # 5) Costo acumulado bajo cada nodo (vista Sink Tree)
    if modo == "sink":
        con = [n for n in orden if n in dist]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in con], y=[pos[n][1] - span * 0.075 for n in con],
            mode="text", text=[fmt(round(dist[n], 2)) for n in con], hoverinfo="skip",
            textfont=dict(size=11, color=C_ARBOL), showlegend=False))

    # 6) Flechas del Sink Tree (cada nodo apunta a su siguiente salto)
    if modo == "sink" or ss.overlay:
        col = C_ARBOL if modo == "sink" else "rgba(27,127,116,0.8)"
        for n, s in sig.items():
            (xa, ya), (xb, yb) = pos[n], pos[s]
            # la flecha arranca después del número del camino para no taparlo
            xa, ya = xa + 0.62 * (xb - xa), ya + 0.62 * (yb - ya)
            fig.add_annotation(x=xb, y=yb, ax=xa, ay=ya, xref="x", yref="y", axref="x",
                               ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3,
                               arrowwidth=3 if modo == "sink" else 2, arrowcolor=col,
                               standoff=ss.tam / 2 + 3)

    pad = span * 0.12
    fig.update_layout(
        height=ss.alto, margin=dict(l=5, r=5, t=10, b=5), dragmode="pan",
        hovermode="closest", clickmode="event+select", uirevision=f"v{ss.ver}",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad],
                     scaleanchor="x", scaleratio=1)
    return fig, mapa


CONFIG_PLOTLY = {"scrollZoom": True, "displaylogo": False,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# --------------------------------------------------------------------------
# Historial adaptativo: detecta cuándo y por qué cambia la mejor ruta
# --------------------------------------------------------------------------
def explicar_cambio(red: Red, vieja, nueva, costo, crit: Criterio, c_antes) -> str:
    if vieja is None:
        return "Se recuperó la conectividad hacia el destino"
    if nueva is None:
        return "El destino quedó inalcanzable"
    tramos = list(zip(vieja, vieja[1:]))
    caidos = [f"{a}-{b}" for a, b in tramos if red.tiene(a, b) and not red.enlace(a, b).operativo]
    if caidos:
        return "Falla en " + ", ".join(caidos)
    qos = [f"{a}-{b}" for a, b in tramos if (a, b) not in costo]
    if qos:
        return f"Banda disponible < {fmt(crit.banda_min)} Mbps en " + ", ".join(qos)
    c_vieja = costo_ruta(vieja, costo)
    c_nueva = costo_ruta(nueva, costo)
    if c_vieja > c_antes + 1e-9:
        return (f"Congestión: la ruta anterior subió de {fmt(round(c_antes, 3))} a "
                f"{fmt(round(c_vieja, 3))}; la nueva cuesta {fmt(round(c_nueva, 3))}")
    return (f"Apareció una ruta más barata ({fmt(round(c_nueva, 3))} < "
            f"{fmt(round(c_vieja, 3))}): enlace recuperado o con menos carga")


def registrar(red: Red, rutas, costo, crit: Criterio):
    # Se comparan valores (astuple): Streamlit redefine las clases en cada rerun
    ctx = (ss.origen, ss.destino, astuple(crit), ss.K, ss.ver, ss.afecta)
    if ss.ctx != ctx:
        ss.ctx, ss.hist, ss.eventos, ss.ult_tick = ctx, [], [], None
    if ss.get("ult_tick") == ss.tick:
        return
    ss.ult_tick = ss.tick
    r1 = rutas[0] if rutas else None
    fila = {"tick": ss.tick, "hora": datetime.now().strftime("%H:%M:%S"),
            "_r1": r1, "_c1": costo_ruta(r1, costo) if r1 else None}
    for i, r in enumerate(rutas):
        fila[f"Ruta {i + 1}"] = round(costo_ruta(r, costo), 4)
    if ss.hist and ss.hist[-1]["_r1"] != r1:
        prev = ss.hist[-1]
        ss.eventos.append({
            "Tick": ss.tick, "Hora": fila["hora"],
            "Ruta anterior": " → ".join(prev["_r1"]) if prev["_r1"] else "—",
            "Ruta nueva": " → ".join(r1) if r1 else "—",
            "Causa": explicar_cambio(red, prev["_r1"], r1, costo, crit, prev["_c1"] or 0.0)})
        ss.eventos = ss.eventos[-200:]
    ss.hist.append(fila)
    ss.hist = ss.hist[-400:]


def criterio_actual() -> Criterio:
    return Criterio(ss.metrica, ss.pa, ss.pb, ss.pg, ss.pd, ss.bmin)


# --------------------------------------------------------------------------
# Red en vivo  (fragmento que se refresca solo cada «intervalo» segundos)
# --------------------------------------------------------------------------
@st.fragment(run_every=float(ss.intervalo) if ss.sim else None)
def panel_vivo():
    if not ss.nodos:
        st.info("Construye la red en la pestaña «Construir / editar red».")
        return
    if ss.sim or ss.pop("_paso", False):
        paso_trafico(ss.dyn, [e.clave for e in ss.base_enlaces], ss.rng, ss.vol, ss.pfalla)
        ss.tick += 1

    red = Red(red_viva(ss.base_enlaces, ss.dyn, ss.afecta), ss.nodos)
    crit = criterio_actual()
    origen, destino, K = ss.origen, ss.destino, int(ss.K)
    adj, costo = red.adyacencia(crit)
    dist, sig = sink_tree(adj, destino)
    primera = camino_arbol(sig, origen, destino)
    rutas = k_rutas(adj, costo, origen, destino, K, primera) if primera and origen != destino \
        else []
    registrar(red, rutas, costo, crit)

    # ---- indicadores ----------------------------------------------------
    caidos = sum(not e.operativo for e in red.enlaces)
    carga_media = (sum(e.carga for e in red.enlaces) / len(red.enlaces)) if red.enlaces else 0
    m1, m2, m3, m4, m5 = st.columns([1.1, 2.2, 1, 1, 1])
    m1.metric("Origen → Destino", f"{origen} → {destino}")
    m2.metric(f"Mejor ruta · costo {fmt(round(costo_ruta(rutas[0], costo), 3))}" if rutas
              else "Mejor ruta", " → ".join(rutas[0]) if rutas else "sin camino")
    m3.metric("Actualización n.º" if ss.sim else "Pausado en n.º", ss.tick)
    m4.metric("Enlaces caídos", caidos)
    m5.metric("Carga media", f"{carga_media * 100:.0f}%")

    s1, s2, s3, s4 = st.tabs(["Grafo y rutas óptimas", "Sink Tree",
                              "Estado de los enlaces", "Historial adaptativo"])

    # ---- Grafo + K rutas -------------------------------------------------
    with s1:
        if origen == destino:
            st.warning("Origen y destino son el mismo nodo: elige otro destino.")
        elif not rutas:
            st.error(f"No existe camino {origen} → {destino} con los enlaces disponibles.")
        st.caption("Clic en un **nodo** → lo vuelve " + ss.modo_clic.lower()
                   + " · clic en el **número de un camino** → editar su peso · rueda = zoom · "
                     "arrastrar = mover. Grosor gris = ancho de banda; color del recuadro = carga.")
        fig, mapa = dibujar(red, ss.pos, origen, destino, rutas, sig, dist, costo, "rutas")
        ev = st.plotly_chart(fig, key=f"g_rutas_{ss.clics}", on_select="rerun",
                             selection_mode="points", config=CONFIG_PLOTLY)
        puntos = (ev or {}).get("selection", {}).get("points", []) if ev else []
        if puntos:
            p = puntos[0]
            cn = p.get("curve_number")
            pi = p.get("point_index", p.get("point_number"))
            if cn == mapa["nodos"] and pi is not None:
                campo = "destino" if ss.modo_clic == "Destino" else "origen"
                ss._pend = {campo: mapa["orden"][pi]}
            elif cn == mapa["etiquetas"] and pi is not None:
                ss.arista_sel = list(mapa["claves"][pi])
            ss.clics += 1
            st.rerun()

        if rutas:
            filas = []
            mejor = costo_ruta(rutas[0], costo)
            for i, r in enumerate(rutas):
                m = metricas_ruta(red, r, costo)
                fila = {"Ruta n.º": i + 1, "Recorrido": " → ".join(r),
                        "Suma de pesos": m["suma_pesos"], "Saltos": m["saltos"],
                        "Retardo (ms)": round(m["retardo"], 1),
                        "Banda cuello (Mbps)": round(m["banda"], 1)}
                if ss.metrica != PESO:
                    fila[f"Suma del costo ({CORTO_METRICA[ss.metrica]})"] = m["suma_costos"]
                fila["Costo total"] = round(m["costo"], 4)
                fila["Δ vs. 1ª"] = round(m["costo"] - mejor, 4)
                filas.append(fila)
            ss.exp_rutas = pd.DataFrame(filas)
            st.dataframe(ss.exp_rutas, hide_index=True, width="stretch")
            if len(rutas) < K:
                st.info(f"Solo existen {len(rutas)} ruta(s) sin ciclos entre {origen} y {destino} "
                        "con los enlaces disponibles.")
            st.caption("La ruta 1 sale del Sink Tree (Dijkstra desde el destino); las rutas 2…K "
                       "con el algoritmo de Yen. «Banda cuello» = mínima banda disponible del "
                       "camino (capacidad real de la ruta).")

    # ---- Sink Tree --------------------------------------------------------
    with s2:
        izq, der = st.columns([3, 2])
        with izq:
            fig_s, _ = dibujar(red, ss.pos, origen, destino, rutas, sig, dist, costo, "sink")
            st.plotly_chart(fig_s, key="g_sink", config=CONFIG_PLOTLY)
        with der:
            filas_s = []
            for n in red.nodos:
                if n == destino:
                    continue
                if n in sig:
                    camino = camino_arbol(sig, n, destino)
                    filas_s.append({"Router": n, "Siguiente salto": sig[n],
                                    f"Costo a {destino}": round(dist[n], 4),
                                    "Saltos": len(camino) - 1, "Camino": " → ".join(camino)})
                else:
                    filas_s.append({"Router": n, "Siguiente salto": "sin ruta",
                                    f"Costo a {destino}": None, "Saltos": None, "Camino": "—"})
            ss.exp_sink = pd.DataFrame(filas_s)
            st.dataframe(ss.exp_sink, hide_index=True, width="stretch")
            st.markdown(
                f"**Sink Tree con raíz en {destino}.** Cada router guarda un único siguiente "
                "salto. Por el **principio de optimalidad**, si la ruta óptima de X pasa por Y, "
                "el tramo Y→destino también es óptimo; por eso la unión de todas las rutas "
                "óptimas forma un árbol (sin ciclos) y basta un salto por nodo. El número verde "
                "bajo cada nodo es su costo mínimo hasta el destino.")

    # ---- Estado de los enlaces --------------------------------------------
    with s3:
        en_ruta = {}
        for i, r in enumerate(rutas):
            for a, b in zip(r, r[1:]):
                en_ruta.setdefault(clave(a, b), []).append(str(i + 1))
        filas_e = []
        for e in red.enlaces:
            c_e = costo.get((e.u, e.v), INF)
            filas_e.append({
                "Camino": e.etiqueta, "Peso": e.peso, "Peso efectivo": round(e.w, 2),
                "Retardo (ms)": e.retardo, "Retardo efectivo": round(e.retardo_ef, 2),
                "Banda (Mbps)": e.banda, "Banda disponible": round(e.banda_disp, 1),
                "Carga": e.carga,
                "Estado": "Activo" if e.operativo else ("Caído" if e.falla else "Apagado"),
                "Costo": None if c_e == INF else round(c_e, 4),
                "En rutas": ", ".join(en_ruta.get(e.clave, [])) or "—"})
        ss.exp_estado = pd.DataFrame(filas_e)
        st.dataframe(ss.exp_estado, hide_index=True, width="stretch",
                     column_config={"Carga": st.column_config.ProgressColumn(
                         "Carga ρ", min_value=0.0, max_value=1.0, format="percent")})
        st.caption("R1 mueve la carga ρ; R2 calcula retardo efectivo d/(1−ρ) y banda disponible "
                   "b·(1−ρ); R3 provoca caídas temporales. Costo vacío = enlace excluido.")

    # ---- Historial ----------------------------------------------------------
    with s4:
        if len(ss.hist) > 1:
            dfh = pd.DataFrame(ss.hist).set_index("tick")
            cols = [c for c in dfh.columns if c.startswith("Ruta ")]
            st.markdown("**Costo de la 1ª, 2ª y 3ª mejor ruta en cada instante**")
            st.line_chart(dfh[cols], color=C_RUTAS[:len(cols)], height=280)
        else:
            st.info("Activa la simulación para ver cómo evolucionan los costos.")
        st.markdown(f"**Cambios de la mejor ruta ({len(ss.eventos)})** · enrutamiento adaptativo")
        if ss.eventos:
            st.dataframe(pd.DataFrame(ss.eventos[::-1]), hide_index=True, width="stretch")
        else:
            st.caption("Aún no ha cambiado la mejor ruta en este escenario.")


with tab_vivo:
    panel_vivo()

# --------------------------------------------------------------------------
# Teoría y fórmulas
# --------------------------------------------------------------------------
with tab_teoria:
    st.markdown("### La red como grafo")
    st.markdown("La red es un grafo no dirigido y ponderado $G=(V,E,w)$: los routers son los "
                "vértices $V$ y cada camino entre dos routers es una arista $(i,j)\\in E$ con "
                "un costo $c_{ij}\\ge 0$. Una ruta $P$ es una secuencia de aristas y su costo es "
                "la **suma** de los costos de sus aristas:")
    st.latex(r"C(P)=\sum_{(i,j)\in P} c_{ij}")
    st.markdown("### Variables de cada enlace y qué cambia las rutas")
    st.latex(r"\rho_{ij}(t+1)=\rho_{ij}(t)+0.15\,(\rho^{0}_{ij}-\rho_{ij}(t))+\varepsilon,"
             r"\quad \varepsilon\sim\mathcal N(0,\sigma)\qquad\text{(R1 tráfico)}")
    st.latex(r"d^{ef}_{ij}=\frac{d_{ij}}{1-\rho_{ij}}\qquad b^{disp}_{ij}=b_{ij}(1-\rho_{ij})"
             r"\qquad w^{ef}_{ij}=w_{ij}(1+\rho_{ij})\ \text{(opcional)}\qquad\text{(R2 congestión)}")
    st.markdown("**R3 fallas:** en cada tick un enlace cae con probabilidad $p_{falla}$ y vuelve "
                "con probabilidad 0.35. Un enlace caído, apagado o con $b^{disp}$ menor que la "
                "banda mínima exigida (QoS) sale del grafo: $c_{ij}=\\infty$.")
    st.markdown("### Funciones de costo $c_{ij}$ disponibles")
    st.markdown(
        "| Métrica | $c_{ij}$ |\n|---|---|\n"
        "| Peso del camino | $w_{ij}$ (el número que escribes) |\n"
        "| Saltos | $1$ |\n"
        "| Retardo | $d^{ef}_{ij}$ |\n"
        "| Ancho de banda | $b_{ref}/b^{disp}_{ij}$ (como el costo de OSPF) |\n"
        "| Adaptativo | $\\alpha+\\beta\\,d^{ef}_{ij}/d_{ref}+\\gamma\\,b_{ref}/b^{disp}_{ij}"
        "+\\delta\\,w_{ij}/w_{ref}$ |")
    st.caption("d_ref, b_ref y w_ref son los máximos de la tabla: dividir entre ellos deja cada "
               "término sin unidades para poder sumarlos. α+β+γ+δ se normaliza a 1.")
    st.markdown("### Algoritmos")
    st.markdown(
        "1. **Dijkstra desde el destino → Sink Tree.** Calcula el costo mínimo de *todos* los "
        "routers hacia el destino y el siguiente salto de cada uno. Con montículo binario: "
        "$O((|V|+|E|)\\log|V|)$.\n"
        "2. **Principio de optimalidad (Bellman).** Si la ruta óptima I→K pasa por J, entonces "
        "J→K también es óptima. Por eso todas las rutas óptimas hacia un destino forman un "
        "árbol: el *sink tree*.\n"
        "3. **Algoritmo de Yen → K rutas.** Toma la ruta 1 y, para cada nodo de ella, prohíbe "
        "la arista que ya usaron las rutas encontradas con el mismo prefijo y vuelve a correr "
        "Dijkstra. El mejor candidato es la ruta siguiente. Costo: $O(K\\,|V|\\,(|E|+|V|)\\log|V|)$; "
        "no enumera todos los caminos (eso sería exponencial).\n"
        "4. **Enrutamiento adaptativo.** En cada tick cambian $\\rho$, $d^{ef}$, $b^{disp}$ y las "
        "fallas → se recalculan los costos, el sink tree y las K rutas. El historial registra "
        "cada vez que la mejor ruta cambia y la causa.")

# --------------------------------------------------------------------------
# Exportar
# --------------------------------------------------------------------------
with tab_export:
    st.markdown("Descarga la red para guardarla o cárgala otra vez más tarde.")
    topologia = {
        "nodos": [{"nodo": n, "x": round(ss.pos[n][0], 3), "y": round(ss.pos[n][1], 3)}
                  for n in ss.nodos],
        "enlaces": [{"origen": e.u, "destino": e.v, "peso": e.peso, "retardo_ms": e.retardo,
                     "banda_mbps": e.banda, "activo": e.activo} for e in ss.base_enlaces],
    }
    e1, e2, e3 = st.columns(3)
    e1.download_button("Descargar Red completa (JSON)", json.dumps(topologia, indent=2, ensure_ascii=False),
                       "red.json", "application/json", width="stretch")
    e2.download_button("Descargar Caminos y pesos (CSV)",
                       ss.links_cur.to_csv(index=False).encode("utf-8"),
                       "caminos.csv", "text/csv", width="stretch")
    if ss.get("exp_rutas") is not None:
        e3.download_button("Descargar Rutas óptimas (CSV)", ss.exp_rutas.to_csv(index=False).encode("utf-8"),
                           "rutas_optimas.csv", "text/csv", width="stretch")
    e4, e5, e6 = st.columns(3)
    if ss.get("exp_sink") is not None:
        e4.download_button("Descargar Sink Tree (CSV)", ss.exp_sink.to_csv(index=False).encode("utf-8"),
                           "sink_tree.csv", "text/csv", width="stretch")
    if ss.get("exp_estado") is not None:
        e5.download_button("Descargar Estado de enlaces (CSV)",
                           ss.exp_estado.to_csv(index=False).encode("utf-8"),
                           "estado_enlaces.csv", "text/csv", width="stretch")
    if ss.hist:
        dfh = pd.DataFrame(ss.hist).drop(columns=["_r1", "_c1"])
        e6.download_button("Descargar Historial (CSV)", dfh.to_csv(index=False).encode("utf-8"),
                           "historial.csv", "text/csv", width="stretch")

    st.divider()
    archivo = st.file_uploader("Cargar una red guardada (JSON)", type="json")
    if archivo is not None and ss.get("ult_archivo") != archivo.file_id:
        try:
            data = json.load(archivo)
            nl = df_enlaces([(d["origen"], d["destino"], d.get("peso", 1), d.get("retardo_ms", 1),
                              d.get("banda_mbps", 100), d.get("activo", True))
                             for d in data["enlaces"]])
            nd = df_nodos([(d["nodo"], d.get("x"), d.get("y")) for d in data.get("nodos", [])])
            ss.ult_archivo = archivo.file_id
            aplicar(nl, nd, reiniciar=True)
            st.rerun()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as err:
            st.error(f"El archivo no tiene el formato esperado: {err}")
