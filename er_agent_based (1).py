"""
Agent-Based Simulation: Emergency Room Resource Allocation
Course: Modelling & Simulation  (follow-up to the Monte Carlo model)
---------------------------------------------------------------------
Same ER, same synthetic demand data as the Monte Carlo model, but now every
person is an agent with their own state and decision rules, stepped minute by
minute. Outcomes EMERGE from interactions instead of being fixed by a queue
formula.

AGENTS AND RULES
  Patient       arrives, is triaged, waits for a bed, waits for a doctor,
                is treated, then goes home or is admitted (boards in the ED
                bed until a ward bed opens).
                - may DETERIORATE while waiting (ESI level rises)
                - ESI 4-5 patients LEAVE when they run out of patience;
                  patience shrinks when the waiting room is crowded
  Triage nurse  takes one patient at a time (ESI 1 ambulance cases bypass).
  Doctor        chooses the next patient by a POLICY (acuity / fifo / hybrid),
                gets FATIGUED (slower consults) and decides to take a break
                when tired and the queue is short.
  Ward nurses   each supports a few bedside patients; if too many patients are
                in treatment, everybody's treatment slows down.
  Ward          releases beds at random; admitted patients wait (board) until
                one opens. Boarders block ED beds (bed blocking).

Run:  python er_agent_based.py   -> abm_dashboard.html + abm_*.csv
"""
import base64, csv, heapq, io, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

SEED = 7
BASE = dict(doctors=4, beds=16, nurses=4, triage_nurses=2, policy="acuity")
N_REPS, N_POLICY, N_GRID = 300, 150, 40
DOC_GRID, NURSE_GRID = [3, 4, 5], [3, 4, 5, 6]
POLICIES = ["acuity", "fifo", "hybrid"]
TARGET = dict(mean_wait=30.0, lwbs=5.0)
COST = dict(doctor=1800, bed=250, nurse=700)   # synthetic $/day

# ---- synthetic inputs (same as the Monte Carlo model) -----------------------
HOURLY_RATE = np.array([3.0, 2.5, 2.2, 2.0, 2.2, 3.0, 4.0, 5.5, 7.0, 8.0, 8.5, 8.5,
                        8.0, 8.0, 8.5, 9.0, 9.5, 9.5, 9.0, 8.0, 7.0, 5.5, 4.5, 3.5])
ESI_P = np.array([0.03, 0.12, 0.38, 0.32, 0.15])
CONSULT_MEAN = np.array([45, 35, 25, 15, 10])
TREAT_MEAN = np.array([240, 180, 120, 60, 30])
PATIENCE_MEAN, DEMAND_SIGMA = 120.0, 0.15
# ---- extra ABM inputs -------------------------------------------------------
TRIAGE_MEAN = 6.0                                   # minutes
ADMIT_P = np.array([0.70, 0.45, 0.20, 0.02, 0.0])   # by ESI 1..5
DET = np.array([0, 0, 0.02, 0.03, 0.02, 0.01]) / 60  # per-minute escalation hazard, index = current ESI
WARD_MEAN = 40.0                                    # minutes between ward bed releases
NURSE_LOAD = 3                                      # bedside patients per nurse
FATIGUE_PER_MIN, FATIGUE_SLOWDOWN, BREAK_MIN = 1 / 480, 0.30, 20
HYBRID_W = 20.0                                     # 1 ESI level = 20 min of waiting
T_DAY = 1440
WAIT_TRI, TRIAGING, WAIT_BED, WAIT_DOC, CONSULT, TREAT, BOARD, DONE, LWBS = range(9)


def lognormal(rng, mean, sigma):
    return rng.lognormal(np.log(mean) - sigma ** 2 / 2, sigma)


class Patient:
    __slots__ = ("pid", "esi", "esi0", "t_arr", "patience", "consult", "treat", "tri", "admit",
                 "state", "t_doc", "t_board", "t_done", "esc", "left", "t_tri", "t_bed", "t_consult_end")

    def __init__(self, pid, t, rng):
        self.pid = pid
        self.esi = self.esi0 = int(rng.choice(5, p=ESI_P)) + 1
        self.t_arr = t
        self.consult = lognormal(rng, CONSULT_MEAN[self.esi - 1], 0.4)
        self.treat = lognormal(rng, TREAT_MEAN[self.esi - 1], 0.5)
        self.patience = rng.exponential(PATIENCE_MEAN)
        self.tri = max(1, round(lognormal(rng, TRIAGE_MEAN, 0.4)))
        self.admit = rng.random() < ADMIT_P[self.esi - 1]
        self.state = WAIT_TRI
        self.t_doc = self.t_board = self.t_done = self.t_tri = -1
        self.t_bed = self.t_consult_end = -1
        self.esc = 0
        self.left = 0.0


class Doctor:
    __slots__ = ("p", "left", "fatigue", "brk")
    def __init__(self): self.p, self.left, self.fatigue, self.brk = None, 0, 0.0, 0


class TriageNurse:
    __slots__ = ("p", "left")
    def __init__(self): self.p, self.left = None, 0


def run_day(seed, cfg, detail=False):
    """One simulated 24h day. Two RNG streams: `pop` (demand + patient traits, identical
    across policies for fair comparison) and `dyn` (deterioration, ward beds)."""
    pop, dyn = np.random.default_rng([seed, 0]), np.random.default_rng([seed, 1])
    B, N, policy = cfg["beds"], cfg["nurses"], cfg["policy"]
    mult = pop.lognormal(-DEMAND_SIGMA ** 2 / 2, DEMAND_SIGMA)
    ward_rate = dyn.lognormal(0, 0.25) / WARD_MEAN
    counts = pop.poisson(np.repeat(HOURLY_RATE, 60) * mult / 60)
    arrivals, everyone = {}, []
    for m in np.nonzero(counts)[0]:
        arrivals[int(m)] = [Patient(len(everyone) + k + 1, int(m) + pop.random(), pop) for k in range(counts[m])]
        everyone.extend(arrivals[int(m)])
    key = {"acuity": lambda p: (p.esi, p.t_arr), "fifo": lambda p: p.t_arr,
           "hybrid": lambda p: p.t_arr + HYBRID_W * (p.esi - 1)}[policy]

    docs = [Doctor() for _ in range(cfg["doctors"])]
    tns = [TriageNurse() for _ in range(cfg["triage_nurses"])]
    triage_q, wait_bed, wait_doc, treating, boarding = [], [], [], [], []
    tl = np.zeros((T_DAY, 10))
    snaps = []
    t = 0
    while t < 4000:
        for p in arrivals.get(t, []):                      # 1. arrivals
            if p.esi == 1:
                p.state = WAIT_BED; wait_bed.append(p)     # ambulance bypasses triage
            else:
                triage_q.append(p)
        for n in tns:                                      # 2. triage nurses
            if n.p:
                n.left -= 1
                if n.left <= 0:
                    n.p.state = WAIT_BED; wait_bed.append(n.p); n.p = None
            if n.p is None and triage_q:
                n.p = triage_q.pop(0); n.p.state = TRIAGING; n.p.t_tri = t; n.left = n.p.tri
        waiting = triage_q + wait_bed + wait_doc           # 3. patience + deterioration
        if waiting:
            u = dyn.random(len(waiting))
            cf = 1 / (1 + max(0, len(triage_q) + len(wait_bed) - 8) / 20)
            gone = False
            for p, x in zip(waiting, u):
                if p.esi >= 4 and t - p.t_arr > p.patience * cf:
                    p.state = LWBS; p.t_done = t; gone = True
                elif x < DET[p.esi]:
                    p.esi -= 1; p.esc += 1; p.treat *= 1.25
            if gone:
                triage_q = [p for p in triage_q if p.state != LWBS]
                wait_bed = [p for p in wait_bed if p.state != LWBS]
                wait_doc = [p for p in wait_doc if p.state != LWBS]
        n_consult = sum(d.p is not None for d in docs)
        free = B - (len(wait_doc) + n_consult + len(treating) + len(boarding))
        if free > 0 and wait_bed:                          # 4. bed allocation
            wait_bed.sort(key=key)
            for p in wait_bed[:free]:
                p.state = WAIT_DOC; p.t_bed = t; wait_doc.append(p)
            wait_bed = wait_bed[free:]
        for d in docs:                                     # 5. doctors decide
            if d.p:
                d.fatigue = min(1.0, d.fatigue + FATIGUE_PER_MIN); d.left -= 1
                if d.left <= 0:
                    d.p.t_consult_end = t; d.p.state = TREAT; d.p.left = d.p.treat; treating.append(d.p); d.p = None
            elif d.brk > 0:
                d.brk -= 1; d.fatigue = max(0.0, d.fatigue - 1 / 30)
            elif d.fatigue > 0.6 and len(wait_doc) < 3:
                d.brk = BREAK_MIN
            elif wait_doc:
                p = min(wait_doc, key=key); wait_doc.remove(p)
                p.t_doc = t; p.state = CONSULT; d.p = p
                d.left = max(1, round(p.consult * (1 + FATIGUE_SLOWDOWN * d.fatigue)))
        rate = min(1.0, N * NURSE_LOAD / len(treating)) if treating else 1.0   # 6. treatment
        still = []
        for p in treating:
            p.left -= rate
            if p.left > 0: still.append(p)
            elif p.admit: p.state = BOARD; p.t_board = t; boarding.append(p)
            else: p.state = DONE; p.t_done = t
        treating = still
        if boarding and dyn.random() < ward_rate:          # 7. ward releases a bed
            b = boarding.pop(0); b.state = DONE; b.t_done = t
        n_consult = sum(d.p is not None for d in docs)
        if t < T_DAY:
            tl[t] = (len(triage_q), len(wait_bed), len(wait_doc), n_consult, len(treating), len(boarding),
                     len(wait_doc) + n_consult + len(treating) + len(boarding), n_consult,
                     np.mean([d.fatigue for d in docs]), min(1.0, len(treating) / (N * NURSE_LOAD)))
            if detail and t % 60 == 0:
                snaps.append(1)
        if t >= T_DAY and not (triage_q or wait_bed or wait_doc or treating or boarding or any(d.p for d in docs)):
            break
        t += 1

    seen = [p for p in everyone if p.t_doc >= 0]
    w = np.array([p.t_doc - p.t_arr for p in seen])
    e0 = np.array([p.esi0 for p in seen])
    g = lambda mask: w[mask].mean() if mask.any() else np.nan
    fin = [p for p in everyone if p.state == DONE]
    adm = [p for p in fin if p.t_board >= 0]
    m = dict(arrivals=len(everyone), mean_wait=w.mean(), p95_wait=np.percentile(w, 95),
             wait_esi12=g(e0 <= 2), wait_esi3=g(e0 == 3), wait_esi45=g(e0 >= 4),
             lwbs_pct=100 * np.mean([p.state == LWBS for p in everyone]),
             mean_los=np.mean([p.t_done - p.t_arr for p in fin]),
             mean_boarding=np.mean([p.t_done - p.t_board for p in adm]) if adm else np.nan,
             bed_block_pct=100 * sum(p.t_done - p.t_board for p in adm) / (B * T_DAY),
             escalations=sum(p.esc for p in everyone), bed_util=100 * tl[:, 6].mean() / B,
             nurse_util=100 * tl[:, 9].mean(), doctor_util=100 * tl[:, 7].mean() / cfg["doctors"],
             fatigue=tl[:, 8].mean(), hourly=tl.reshape(24, 60, 10).mean(1))
    if detail:
        m["patients"] = [dict(patient_id=p.pid, arrival_min=round(p.t_arr, 2), esi_at_arrival=p.esi0, esi_final=p.esi,
                              escalations=p.esc, wait_for_doctor_min=None if p.t_doc < 0 else round(p.t_doc - p.t_arr, 2),
                              left_without_being_seen=int(p.state == LWBS), admitted=int(p.admit),
                              boarding_min=None if p.t_board < 0 else round(p.t_done - p.t_board, 2),
                              length_of_stay_min=None if p.state != DONE else round(p.t_done - p.t_arr, 2)) for p in everyone]
        m["timeline"] = tl
        m["raw_patients"] = everyone
    return m


def run_many(cfg, reps, seed0):
    return [run_day(seed0 + i, cfg) for i in range(reps)]


col = lambda res, k: np.array([r[k] for r in res], float)


def ci(x):
    x = x[~np.isnan(x)]; h = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


# ---- plots ------------------------------------------------------------------
INK, TEAL, RED, AMBER, GREY = "#14262E", "#0F7B78", "#C8402F", "#D9A21B", "#8A9BA3"
ESI_COL = {1: "#6E1423", 2: RED, 3: AMBER, 4: TEAL, 5: GREY}

ICON_SVG = {
    "users": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="8" r="3.2"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/><circle cx="17" cy="9" r="2.6"/><path d="M15.5 14.3c2.6.4 4.5 2.6 4.5 5.2"/></svg>',
    "clock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3.5 2"/></svg>',
    "alert": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.5 2.5 20h19L12 3.5Z"/><path d="M12 9.5v4.5"/><circle cx="12" cy="17" r=".6" fill="currentColor" stroke="none"/></svg>',
    "door": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M13 3H6v18h7"/><path d="M13 3l6 2v14l-6 2"/><path d="M15.5 12h5M18 9.5l2.5 2.5-2.5 2.5"/></svg>',
    "bed": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 18v-6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v6"/><path d="M2 18v2M22 18v2"/><path d="M4 12V8a2 2 0 0 1 2-2h5v6"/><circle cx="7.5" cy="9" r="1"/></svg>',
    "pulse": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h4l2-7 4 14 2-9 2 2h6"/></svg>',
    "flask": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 2h6M10 2v6.5L4.5 19a1.6 1.6 0 0 0 1.4 2.4h12.2a1.6 1.6 0 0 0 1.4-2.4L14 8.5V2"/><path d="M7.2 15h9.6"/></svg>',
}

DESIGN_CSS = """
:root{--bg:#EFF4F5;--ink:#122026;--mut:#5B6E77;--teal:#0E8C86;--teal-ink:#07504C;--red:#C8402F;--amber:#D9A21B;--chip:#122026;
--card:#fff;--card2:#E7F1F1;--line:#D7E3E4;--radius:16px;--radius-pill:999px;
--shadow-sm:0 1px 2px rgba(18,32,38,.06),0 1px 1px rgba(18,32,38,.05);
--shadow-md:0 10px 28px -12px rgba(18,32,38,.28);
box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);color-scheme:light}
:root[data-theme="dark"]{--bg:#0B1518;--ink:#E7F1F1;--mut:#93ABB0;--card:#132226;--card2:#16302E;--line:#223C3E;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;transition:background .2s,color .2s}
main{max-width:1180px;margin:0 auto;padding:28px 20px 48px}
h1{font-family:"Space Grotesk",system-ui,sans-serif;font-size:29px;margin:2px 0 2px;letter-spacing:-.01em}
h2{font-family:"Space Grotesk",system-ui,sans-serif;font-size:19px;margin:38px 0 12px;display:flex;align-items:center;gap:8px}
p.sub{color:var(--mut);margin:0;max-width:70ch}
.hero{padding-bottom:2px}
.hero-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:2px}
.kicker{font:600 11px/1 "IBM Plex Mono",monospace;letter-spacing:.12em;color:var(--teal);text-transform:uppercase}
#theme-toggle,#theme-toggle-abm{background:var(--card);border:1px solid var(--line);color:var(--ink);border-radius:var(--radius-pill);
width:34px;height:34px;padding:0;display:inline-flex;align-items:center;justify-content:center;font-size:15px;cursor:pointer;box-shadow:var(--shadow-sm)}
.ekg{width:100%;height:24px;margin:2px 0 6px;color:var(--teal);opacity:.55;display:block}
.ekg polyline{fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:900;stroke-dashoffset:900;animation:ekgdraw 4s ease-in-out infinite}
@keyframes ekgdraw{0%{stroke-dashoffset:900}55%{stroke-dashoffset:0}100%{stroke-dashoffset:-900}}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin-top:20px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;box-shadow:var(--shadow-sm);position:relative;overflow:hidden}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--teal),transparent)}
.kpi-top{display:flex;align-items:center;gap:10px}
.kpi-ic{width:34px;height:34px;border-radius:10px;background:var(--card2);color:var(--teal);display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto}
.kpi-ic svg{width:18px;height:18px}
.kpi .v{font-size:26px;font-weight:650;font-family:"Space Grotesk",system-ui,sans-serif}
.kpi small{font-size:13px;margin-left:2px;color:var(--mut);font-family:"IBM Plex Sans",sans-serif;font-weight:400}
.kpi .l{font-weight:600;margin-top:8px;font-size:13.5px} .kpi .c{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
figure{margin:0;background:#fff;border:1px solid var(--line);border-radius:var(--radius);padding:10px;box-shadow:var(--shadow-sm)}
figure img{max-width:100%;height:auto;display:block;border-radius:8px}
figcaption{font-size:12.5px;color:var(--mut);padding:6px 4px 0}
.rec{background:linear-gradient(135deg,var(--teal-ink),var(--chip));color:#fff;padding:18px 20px;border-radius:var(--radius);margin-top:22px;box-shadow:var(--shadow-md)}
.rec b{color:#7FE0DA}
.tw{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow-sm)}
table{border-collapse:collapse;width:100%;min-width:560px}
th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--line)}
th{background:var(--card2);font-size:12.5px;text-transform:uppercase;letter-spacing:.03em;color:var(--mut);font-weight:600}
tr.ok td{background:rgba(14,140,134,.09)} tr.best td{background:rgba(14,140,134,.22);font-weight:650}
.meta{color:var(--mut);font-size:13px;margin-top:8px}
.arcade{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:16px;box-shadow:var(--shadow-sm)}
.arcade-bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between;margin-bottom:10px}
.seg{display:flex;gap:4px;flex-wrap:wrap;background:var(--card2);padding:4px;border-radius:var(--radius-pill)}
.seg button{background:transparent;color:var(--ink);padding:6px 13px;font-size:13px;border-radius:var(--radius-pill);box-shadow:none}
.seg button.active{background:var(--teal);color:#fff;box-shadow:var(--shadow-sm)}
button{background:var(--teal);color:#fff;border:0;border-radius:var(--radius-pill);padding:7px 16px;font:inherit;font-weight:600;cursor:pointer;box-shadow:var(--shadow-sm)}
button:hover{filter:brightness(1.05)}
button:focus-visible,input:focus-visible{outline:3px solid var(--teal);outline-offset:2px}
canvas{width:100%;height:auto;display:block;background:#0B1C20;border:3px solid #0B1C20;border-radius:12px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.06)}
.scorebar{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;font-size:13.5px;color:var(--mut)}
.scorebar b{font-size:19px;color:var(--ink);font-family:"IBM Plex Mono",monospace;font-weight:500}
.grade{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:50%;background:var(--chip);color:#fff;font-weight:700;font-family:"Space Grotesk",sans-serif}
.toasts{position:fixed;top:14px;right:14px;display:flex;flex-direction:column;gap:6px;z-index:50;max-width:260px}
.toast{background:var(--chip);color:#fff;padding:9px 13px;border-radius:10px;font-size:13px;box-shadow:var(--shadow-md);animation:abmfade .2s ease-out}
@keyframes abmfade{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}
#mc-clock,#abm-clock{font-variant-numeric:tabular-nums;font-family:"IBM Plex Mono",monospace}
.legend{display:flex;flex-wrap:wrap;gap:10px 16px;margin-top:12px;font-size:12.5px;color:var(--mut);align-items:center;
background:var(--card2);padding:8px 12px;border-radius:10px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend span::before{content:'';width:14px;height:14px;border-radius:50%;display:inline-block;flex:0 0 auto;box-sizing:border-box}
.lg-esi::before{background:var(--c)}
.lg-ring::before{background:#fff;border:2px solid var(--ink)}
.lg-ring.dash::before{border-style:dashed}
.lg-ring.green::before{border-color:#1a9c3a;border-width:3px}
.lg-ring.amber::before{border-color:#D9A21B;border-width:3px}
.lg-ring.teal::before{border-color:#0E8C86;border-width:2.5px}
"""
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK})


def b64(fig, dpi=130):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def hist(x, title, xlabel, color=TEAL):
    x = x[~np.isnan(x)]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.hist(x, bins=40, color=color, edgecolor="white")
    ax.axvline(x.mean(), color=INK, lw=1.6, label=f"mean {x.mean():.1f}")
    ax.axvline(np.percentile(x, 95), color=RED, lw=1.4, ls="--", label=f"95th pct {np.percentile(x, 95):.1f}")
    ax.set(title=title, xlabel=xlabel, ylabel="Simulated days"); ax.legend(frameon=False, fontsize=8)
    return b64(fig)


def assign_slots(intervals, capacity):
    """Stable seat/bed/doctor index for a patient's whole stay (see er_monte_carlo.py)."""
    ev = sorted(intervals, key=lambda x: x[0])
    free = list(range(capacity)); heapq.heapify(free)
    active, out = [], {}
    for start, end, pid in ev:
        while active and active[0][0] <= start:
            _, s = heapq.heappop(active)
            heapq.heappush(free, s)
        s = heapq.heappop(free) if free else capacity
        out[pid] = s
        heapq.heappush(active, (end, s))
    return out


GAME_WAIT_SLOTS = 24
GAME_PRESETS = [("Understaffed", 3, 3), ("Baseline", 4, 4), ("Well-staffed", 5, 5), ("Maxed out", 6, 6)]
GAME_SEED = 55555


def build_game_preset(doctors, nurses, seed):
    cfg = dict(BASE, doctors=doctors, nurses=nurses)
    d = run_day(seed, cfg, detail=True)
    people = d["raw_patients"]
    wait_iv, bed_iv, doc_iv, tri_iv = [], [], [], []
    for i, p in enumerate(people):
        wait_end = p.t_bed if p.t_bed >= 0 else p.t_done
        wait_iv.append((p.t_arr, wait_end, i))
        if p.t_bed >= 0:
            bed_iv.append((p.t_bed, p.t_done, i))
        if p.t_doc >= 0:
            doc_iv.append((p.t_doc, p.t_consult_end, i))
        if p.t_tri >= 0:
            tri_iv.append((p.t_tri, p.t_tri + p.tri, i))
    wslot = assign_slots(wait_iv, GAME_WAIT_SLOTS)
    bslot = assign_slots(bed_iv, cfg["beds"])
    dslot = assign_slots(doc_iv, doctors)
    tslot = assign_slots(tri_iv, cfg["triage_nurses"])
    patients = [dict(
        esi=p.esi0, arr=round(float(p.t_arr), 2),
        triStart=round(float(p.t_tri), 2) if p.t_tri >= 0 else None,
        triEnd=round(float(p.t_tri + p.tri), 2) if p.t_tri >= 0 else None, triSlot=tslot.get(i),
        bedStart=round(float(p.t_bed), 2) if p.t_bed >= 0 else None,
        pickup=round(float(p.t_doc), 2) if p.t_doc >= 0 else None,
        consultEnd=round(float(p.t_consult_end), 2) if p.t_consult_end >= 0 else None,
        boardStart=round(float(p.t_board), 2) if p.t_board >= 0 else None,
        left=round(float(p.t_done), 2), lwbs=bool(p.state == LWBS), admitted=bool(p.admit),
        waitSlot=wslot[i], bedSlot=bslot.get(i), docSlot=dslot.get(i)) for i, p in enumerate(people)]
    return dict(doctors=doctors, nurses=nurses, beds=cfg["beds"], triageNurses=cfg["triage_nurses"],
               waitSlots=GAME_WAIT_SLOTS, arrivals=d["arrivals"], meanWait=round(float(d["mean_wait"]), 1),
               lwbsPct=round(float(d["lwbs_pct"]), 1), patients=patients)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    out = os.path.dirname(os.path.abspath(__file__)); P = lambda n: os.path.join(out, n)
    print("Baseline ...")
    base = run_many(BASE, N_REPS, SEED * 1000)
    print("Policy comparison ...")
    pol = {p: run_many(dict(BASE, policy=p), N_POLICY, 5000) for p in POLICIES}
    print("Staffing grid ...")
    grid = []
    for d in DOC_GRID:
        for n in NURSE_GRID:
            r = run_many(dict(BASE, doctors=d, nurses=n), N_GRID, 9000)
            mw, lw = col(r, "mean_wait"), col(r, "lwbs_pct")
            grid.append(dict(doctors=d, nurses=n, cost=d * COST["doctor"] + n * COST["nurse"] + BASE["beds"] * COST["bed"],
                             mean_wait=mw.mean(), wait_esi12=np.nanmean(col(r, "wait_esi12")), lwbs=lw.mean(),
                             boarding=np.nanmean(col(r, "mean_boarding")), fatigue=col(r, "fatigue").mean(),
                             p_meet=100 * np.mean((mw <= TARGET["mean_wait"]) & (lw <= TARGET["lwbs"]))))
    ok = [g for g in grid if g["mean_wait"] <= TARGET["mean_wait"] and g["lwbs"] <= TARGET["lwbs"]]
    best = min(ok, key=lambda g: g["cost"]) if ok else min(grid, key=lambda g: g["mean_wait"])
    sample = run_day(SEED * 1000 + 3, BASE, detail=True)

    keys = ["arrivals", "mean_wait", "p95_wait", "wait_esi12", "lwbs_pct", "mean_los", "mean_boarding",
            "bed_block_pct", "escalations", "bed_util", "nurse_util", "doctor_util", "fatigue"]
    st = {k: ci(col(base, k)) for k in keys}
    mw, lw = col(base, "mean_wait"), col(base, "lwbs_pct")
    p_meet = 100 * np.mean((mw <= TARGET["mean_wait"]) & (lw <= TARGET["lwbs"]))

    F = {}
    F["wait"] = hist(mw, "Mean door-to-doctor wait", "Minutes")
    F["lwbs"] = hist(lw, "Patients who leave without being seen", "% of arrivals", RED)
    F["board"] = hist(col(base, "mean_boarding"), "Mean boarding time of admitted patients", "Minutes", AMBER)
    F["esc"] = hist(col(base, "escalations"), "Patients who deteriorated while waiting", "Patients per day", GREY)
    H = np.array([r["hourly"] for r in base]); hrs = np.arange(24)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, (lab, c, cols_) in enumerate([("Waiting room", RED, H[:, :, 0] + H[:, :, 1]), ("Boarders in ED beds", AMBER, H[:, :, 5]),
                                          ("Beds occupied", TEAL, H[:, :, 6])]):
        ax.fill_between(hrs, np.percentile(cols_, 5, 0), np.percentile(cols_, 95, 0), color=c, alpha=0.15)
        ax.plot(hrs, cols_.mean(0), color=c, lw=2, label=lab)
    ax.set(title="Crowding through the day (mean, 5-95% band)", xlabel="Hour", ylabel="Patients", xticks=range(0, 24, 3))
    ax.legend(frameon=False, fontsize=8); F["crowd"] = b64(fig)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(hrs, H[:, :, 8].mean(0), color=RED, lw=2, label="doctor fatigue (0-1)")
    ax.plot(hrs, H[:, :, 9].mean(0), color=TEAL, lw=2, label="ward-nurse load (0-1)")
    ax.set(title="Fatigue and nurse load by hour", xlabel="Hour", ylabel="Level", xticks=range(0, 24, 3)); ax.legend(frameon=False, fontsize=8)
    F["fat"] = b64(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.4)); wd = 0.26
    grp = [("ESI 1-2", "wait_esi12"), ("ESI 3", "wait_esi3"), ("ESI 4-5", "wait_esi45")]
    for i, p in enumerate(POLICIES):
        ax.bar(np.arange(3) + (i - 1) * wd, [np.nanmean(col(pol[p], k)) for _, k in grp], wd, label=p,
               color=[TEAL, AMBER, RED][i])
    ax.set(xticks=range(3), xticklabels=[g for g, _ in grp], ylabel="Mean wait (min)", title="Who waits under each doctor policy?")
    ax.legend(frameon=False, fontsize=8); F["pol_wait"] = b64(fig)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(POLICIES, [np.mean(col(pol[p], "lwbs_pct")) for p in POLICIES], color=RED, width=0.5)
    ax.set(ylabel="% leaving unseen", title="Walk-outs by doctor policy"); F["pol_lwbs"] = b64(fig)

    Z = np.array([[next(g["mean_wait"] for g in grid if g["doctors"] == d and g["nurses"] == n) for n in NURSE_GRID] for d in DOC_GRID])
    fig, ax = plt.subplots(figsize=(5.4, 3.4)); im = ax.imshow(Z, cmap="RdYlGn_r", aspect="auto", vmin=0)
    ax.set(xticks=range(len(NURSE_GRID)), xticklabels=NURSE_GRID, yticks=range(len(DOC_GRID)), yticklabels=DOC_GRID,
           xlabel="Ward nurses", ylabel="Doctors", title="Mean wait (min): doctors x nurses")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]): ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8); F["heat"] = b64(fig)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.scatter(col(base, "mean_boarding"), mw, s=8, alpha=0.35, color=TEAL)
    ok_ = ~np.isnan(col(base, "mean_boarding"))
    ax.set(title=f'Boarding vs wait (corr {np.corrcoef(col(base, "mean_boarding")[ok_], mw[ok_])[0, 1]:.2f})',
           xlabel="Mean boarding time (min)", ylabel="Mean wait (min)"); F["bb"] = b64(fig)

    print("Building arcade presets ...")
    game_presets = {label: build_game_preset(d, n, GAME_SEED) for label, d, n in GAME_PRESETS}
    game_json, labels_json, target_json = json.dumps(game_presets), json.dumps([l for l, _, _ in GAME_PRESETS]), json.dumps(TARGET)

    # comparison with the Monte Carlo baseline (if its CSV is present)
    mc_rows = ""
    mcp = P("replication_results.csv")
    if os.path.exists(mcp):
        with open(mcp) as f: rows = list(csv.DictReader(f))
        mcm = lambda k: np.mean([float(r[k]) for r in rows])
        for lab, mck, abk in [("Arrivals per day", "arrivals", "arrivals"), ("Mean wait (min)", "mean_wait_min", "mean_wait"),
                              ("95th pct wait (min)", "p95_wait_min", "p95_wait"), ("Left unseen (%)", "lwbs_pct", "lwbs_pct"),
                              ("Bed utilisation (%)", "bed_util_pct", "bed_util"), ("Length of stay (min)", "mean_los_min", "mean_los")]:
            mc_rows += f"<tr><td style='text-align:left'>{lab}</td><td>{mcm(mck):.1f}</td><td>{st[abk][0]:.1f}</td></tr>"

    kp = lambda lab, k, u="", d=1, icon="pulse": (f'<div class="kpi"><div class="kpi-top"><span class="kpi-ic">{ICON_SVG[icon]}</span>'
                                     f'<div class="v">{st[k][0]:.{d}f}<small>{u}</small></div></div><div class="l">{lab}</div>'
                                     f'<div class="c">95% CI {st[k][1]:.{d}f} to {st[k][2]:.{d}f}</div></div>')
    img = lambda k, cap: f'<figure><img alt="{k}" src="data:image/png;base64,{F[k]}"><figcaption>{cap}</figcaption></figure>'
    trs = "".join(f'<tr class="{"best" if g is best else ("ok" if g in ok else "")}"><td>{g["doctors"]}</td><td>{g["nurses"]}</td>'
                  f'<td>${g["cost"]:,}</td><td>{g["mean_wait"]:.1f}</td><td>{g["wait_esi12"]:.1f}</td><td>{g["lwbs"]:.1f}%</td>'
                  f'<td>{g["boarding"]:.0f}</td><td>{g["fatigue"]:.2f}</td><td>{g["p_meet"]:.0f}%</td></tr>' for g in grid)
    pt = "".join(f'<tr><td style="text-align:left">{p}</td><td>{np.mean(col(pol[p], "mean_wait")):.1f}</td>'
                 f'<td>{np.nanmean(col(pol[p], "wait_esi12")):.1f}</td><td>{np.nanmean(col(pol[p], "wait_esi45")):.1f}</td>'
                 f'<td>{np.mean(col(pol[p], "lwbs_pct")):.1f}%</td></tr>' for p in POLICIES)

    html = HTML.replace("__KPIS__", kp("Arrivals per day", "arrivals", "", 0, "users") + kp("Mean wait to see doctor", "mean_wait", " min", 1, "clock") +
                        kp("Critical (ESI 1-2) wait", "wait_esi12", " min", 1, "alert") + kp("Left without being seen", "lwbs_pct", "%", 1, "door") +
                        kp("Boarding time (admitted)", "mean_boarding", " min", 0, "bed") + kp("Beds blocked by boarders", "bed_block_pct", "%", 1, "bed") +
                        kp("Deteriorated while waiting", "escalations", "", 1, "alert") + kp("Doctor fatigue (0-1)", "fatigue", "", 2, "pulse"))
    rep = {"__NREPS__": f"{N_REPS:,}", "__BEST__": f'{best["doctors"]} doctors and {best["nurses"]} ward nurses (with {BASE["beds"]} beds) at ${best["cost"]:,} per day',
           "__PMEET__": f'{best["p_meet"]:.0f}', "__BASEMEET__": f"{p_meet:.0f}", "__MCROWS__": mc_rows, "__TRS__": trs, "__PT__": pt, "__DESIGN_CSS__": DESIGN_CSS,
           "__GAME_JSON__": game_json, "__LABELS_JSON__": labels_json, "__TARGET_JSON__": target_json, "__WAITSLOTS__": str(GAME_WAIT_SLOTS),
           **{f"__IMG_{k}__": img(k, c) for k, c in dict(
               wait="Skewed by overload days, like the Monte Carlo result.", lwbs="Walk-outs rise sharply when the waiting room is crowded.",
               board="Admitted patients wait for ward beds while occupying ED beds.", esc="Long waits cause deterioration, a feedback loop the queue model lacked.",
               crowd="Boarders and waiting-room crowds build up together in the evening.", fat="Fatigue slows consults late in the day, deepening the evening peak.",
               pol_wait="Acuity-first protects critical patients but low-acuity patients wait longest.", pol_lwbs="Fifo keeps queues fair but sends more low-acuity walk-outs... or fewer? Read the table.",
               heat="Nurses matter as much as doctors once beds fill with treatment.", bb="Days with long boarding are also days with long waits.").items()}}
    rep["__IMG_pol_lwbs__"] = img("pol_lwbs", "Same patients, same days, only the policy differs.")
    for k, v in rep.items(): html = html.replace(k, v)
    open(P("abm_dashboard.html"), "w", encoding="utf-8").write(html)

    # ---- CSV exports
    write_csv(P("abm_replication_results.csv"), [dict(replication=i + 1, **{k: round(float(r[k]), 3) for k in keys}) for i, r in enumerate(base)])
    write_csv(P("abm_policy_comparison.csv"), [dict(policy=p, replication=i + 1, **{k: round(float(r[k]), 3) for k in keys})
                                                for p in POLICIES for i, r in enumerate(pol[p])])
    write_csv(P("abm_staffing_grid.csv"), [{k: round(float(v), 3) for k, v in g.items()} for g in grid])
    write_csv(P("abm_sample_day_patients.csv"), sample["patients"])
    names = ["triage_queue", "waiting_for_bed", "waiting_for_doctor", "in_consult", "in_treatment", "boarding", "beds_occupied", "doctors_busy", "doctor_fatigue", "nurse_load"]
    write_csv(P("abm_sample_day_timeline.csv"), [dict(minute=m, **{n: round(float(v), 3) for n, v in zip(names, row)}) for m, row in enumerate(sample["timeline"])])
    print("Best:", best)
    for k in keys: print(f"{k:14s} {st[k][0]:8.2f}  CI [{st[k][1]:.2f}, {st[k][2]:.2f}]")
    print("P(meet targets) baseline", round(p_meet, 1))
    for p in POLICIES: print(p, {k: round(float(np.nanmean(col(pol[p], k))), 2) for k in ["mean_wait", "wait_esi12", "wait_esi45", "lwbs_pct"]})


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet">
<title>ER Agent-Based Simulation Dashboard</title><style>
__DESIGN_CSS__
</style></head><body><main>
<div class="hero">
<div class="hero-top"><span class="kicker">ER OPERATIONS &middot; AGENT-BASED LAB</span>
<button id="theme-toggle-abm" type="button" aria-label="Toggle dark mode">&#127769;</button></div>
<h1>Emergency room: agent-based simulation</h1>
<svg class="ekg" viewBox="0 0 600 30" preserveAspectRatio="none" aria-hidden="true"><polyline points="0,16 55,16 68,4 82,27 95,16 150,16 165,7 180,25 195,16 600,16"/></svg>
<p class="sub">Every patient, triage nurse and doctor is an agent with its own state and decision rules. Results below come from __NREPS__ simulated days at 4 doctors, 16 beds and 4 ward nurses, plus policy and staffing experiments.</p>
</div>
<div class="kpis">__KPIS__</div>
<div class="rec"><b>Finding:</b> the cheapest staffing meeting both targets (mean wait &le; 30 min, walk-outs &le; 5%) is <b>__BEST__</b>, meeting them on __PMEET__% of simulated days. The baseline meets them on __BASEMEET__% of days.</div>
<h2>Play: staff the ER</h2>
<p class="sub" style="margin:0 0 10px">The same patients arrive every round &mdash; only doctors and ward nurses change (beds fixed at 16). Watch fatigue, boarding and deterioration play out.</p>
<div class="arcade">
<div class="arcade-bar">
<div class="seg" id="abm-presets" role="group" aria-label="Staffing preset"></div>
<div class="seg" id="abm-speeds" role="group" aria-label="Playback speed"></div>
<button id="abm-pl" type="button">Play</button>
</div>
<canvas id="abm-cv" width="944" height="380" role="img" aria-label="Animated view of the emergency room"></canvas>
<input id="abm-sl" type="range" min="0" max="1440" value="0" step="1" aria-label="Minute of the day" style="width:100%;margin-top:8px">
<div class="legend">
<span class="lg-esi" style="--c:#6E1423">ESI 1</span><span class="lg-esi" style="--c:#C8402F">ESI 2</span>
<span class="lg-esi" style="--c:#D9A21B">ESI 3</span><span class="lg-esi" style="--c:#0F7B78">ESI 4</span>
<span class="lg-esi" style="--c:#8A9BA3">ESI 5</span>
<span class="lg-ring teal">At triage</span><span class="lg-ring dash">Waiting for doctor</span>
<span class="lg-ring green">With doctor</span><span class="lg-ring amber">Boarding (admitted)</span>
</div>
<div class="scorebar">
<div>Time <b id="abm-clock">00:00</b></div>
<div>Score <b id="abm-score">0</b></div>
<div>Treated <b id="abm-treated">0</b></div>
<div>Walked out <b id="abm-lwbs">0</b></div>
<div>Wait so far <b id="abm-wait">0</b> min</div>
<div>Grade <span class="grade" id="abm-grade">-</span></div>
</div>
<p class="sr-only" id="abm-live" aria-live="polite"></p>
</div>
<div class="toasts" id="abm-toasts"></div>
<h2>Outcomes across days</h2><div class="grid">__IMG_wait____IMG_lwbs____IMG_board____IMG_esc____IMG_crowd____IMG_fat__</div>
<h2>Experiment 1: who should the doctor see next?</h2>
<div class="grid">__IMG_pol_wait____IMG_pol_lwbs__</div>
<div class="tw"><table><thead><tr><th style="text-align:left">Policy</th><th>Mean wait</th><th>ESI 1-2 wait</th><th>ESI 4-5 wait</th><th>Walk-outs</th></tr></thead><tbody>__PT__</tbody></table></div>
<h2>Experiment 2: doctors and ward nurses</h2><div class="grid">__IMG_heat____IMG_bb__</div>
<div class="tw"><table><thead><tr><th>Doctors</th><th>Nurses</th><th>Daily cost</th><th>Mean wait</th><th>ESI 1-2 wait</th><th>Walk-outs</th><th>Boarding (min)</th><th>Fatigue</th><th>Days meeting targets</th></tr></thead><tbody>__TRS__</tbody></table></div>
<h2>Agent-based vs Monte Carlo (same baseline)</h2>
<div class="tw"><table style="min-width:360px"><thead><tr><th style="text-align:left">Metric</th><th>Monte Carlo</th><th>Agent-based</th></tr></thead><tbody>__MCROWS__</tbody></table></div>
<p class="meta">All data are synthetic; costs are illustrative. Policy runs reuse identical patient days so differences come from the policy alone.</p>
</main><script>(function(){
const GAME=__GAME_JSON__, LABELS=__LABELS_JSON__, TARGET=__TARGET_JSON__, WAITSLOTS=__WAITSLOTS__;
const SPEEDS=[["Slow",15],["Normal",60],["Fast",180],["Turbo",600]];
const ESI_COL={1:'#6E1423',2:'#C8402F',3:'#D9A21B',4:'#0F7B78',5:'#8A9BA3'};
const cv=document.getElementById('abm-cv'), ctx=cv.getContext('2d'), sl=document.getElementById('abm-sl'), pl=document.getElementById('abm-pl'),
presetsEl=document.getElementById('abm-presets'), speedsEl=document.getElementById('abm-speeds'), toastsEl=document.getElementById('abm-toasts'),
clockEl=document.getElementById('abm-clock'), scoreEl=document.getElementById('abm-score'), treatedEl=document.getElementById('abm-treated'),
lwbsEl=document.getElementById('abm-lwbs'), waitEl=document.getElementById('abm-wait'), gradeEl=document.getElementById('abm-grade'), liveEl=document.getElementById('abm-live');
let label=LABELS[1], speed=SPEEDS[1][1], playing=false, simT=0, lastT=0, disp={}, floaties=[], data, L, audioCtx=null, reported=false;

function beep(freq,dur,type){ try{ if(!audioCtx) audioCtx=new (window.AudioContext||window.webkitAudioContext)();
  const o=audioCtx.createOscillator(), g=audioCtx.createGain(); o.type=type||'sine'; o.frequency.value=freq;
  o.connect(g); g.connect(audioCtx.destination); g.gain.setValueAtTime(0.08,audioCtx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.0001,audioCtx.currentTime+dur); o.start(); o.stop(audioCtx.currentTime+dur); }catch(e){} }
function toast(msg){ const d=document.createElement('div'); d.className='toast'; d.textContent=msg; toastsEl.appendChild(d); setTimeout(()=>d.remove(),2600); }

function layout(){
  data=GAME[label];
  const bedCols=Math.min(6,Math.max(3,Math.ceil(Math.sqrt(data.beds*1.4)))), bedRows=Math.ceil(data.beds/bedCols);
  const docCols=data.doctors>4?2:1;
  L={ wait:{x0:20,y0:44,gap:30,cols:6,rows:4},
      tri:{x0:257,y0:60,gap:44},
      bed:{x0:303,y0:40,w:506/bedCols,h:Math.min(56,300/bedRows),cols:bedCols},
      doc:{x0:850,y0:44,gap:44,cols:docCols} };
  disp={}; floaties=[]; reported=false;
}
function waitXY(slot){ if(slot>=WAITSLOTS) return {x:L.wait.x0+(L.wait.cols-1)*L.wait.gap+16,y:L.wait.y0+(L.wait.rows-1)*L.wait.gap+16};
  const c=slot%L.wait.cols, r=Math.floor(slot/L.wait.cols); return {x:L.wait.x0+c*L.wait.gap,y:L.wait.y0+r*L.wait.gap}; }
function triXY(slot){ return {x:L.tri.x0, y:L.tri.y0+slot*L.tri.gap}; }
function bedXY(slot){ const c=slot%L.bed.cols, r=Math.floor(slot/L.bed.cols); return {x:L.bed.x0+c*L.bed.w+L.bed.w/2,y:L.bed.y0+r*(L.bed.h+8)+L.bed.h/2}; }
function docXY(slot){ const c=slot%L.doc.cols, r=Math.floor(slot/L.doc.cols); return {x:L.doc.x0+c*L.doc.gap,y:L.doc.y0+r*L.doc.gap}; }

function stateAt(p,t){
  if(t<p.arr) return null;
  if(p.triStart!=null && t<p.triStart) return {phase:'wait',pos:waitXY(p.waitSlot)};
  if(p.triStart!=null && t<p.triEnd) return {phase:'triage',pos:triXY(p.triSlot)};
  if(p.bedStart==null) return t<p.left ? {phase:'wait',pos:waitXY(p.waitSlot)} : null;
  if(t<p.bedStart) return {phase:'wait',pos:waitXY(p.waitSlot)};
  if(p.lwbs) return t<p.left ? {phase:'bedwait',pos:bedXY(p.bedSlot)} : null;
  if(p.pickup!=null && t<p.pickup) return {phase:'bedwait',pos:bedXY(p.bedSlot)};
  if(p.consultEnd!=null && t<p.consultEnd) return {phase:'consult',pos:bedXY(p.bedSlot),doc:p.docSlot};
  if(p.boardStart!=null){ if(t<p.boardStart) return {phase:'treat',pos:bedXY(p.bedSlot)};
    return t<p.left ? {phase:'board',pos:bedXY(p.bedSlot)} : null; }
  return t<p.left ? {phase:'treat',pos:bedXY(p.bedSlot)} : null;
}
function fireEvents(prevT,t){
  if(t<=prevT) return;
  data.patients.forEach(p=>{
    if(p.left>prevT && p.left<=t){
      const xy=p.bedSlot!=null?bedXY(p.bedSlot):waitXY(p.waitSlot);
      if(p.lwbs){ floaties.push({x:xy.x,y:xy.y,txt:'-15 left',col:'#C8402F',life:1}); toast('A patient left without being seen (ESI '+p.esi+')'); beep(220,0.3,'sawtooth'); }
      else { floaties.push({x:xy.x,y:xy.y,txt:'+10',col:'#0F7B78',life:1}); beep(660,0.12,'sine'); }
    }
  });
}
function statsAt(t){
  let treated=0,lw=0,seen=0,sumW=0,arrived=0;
  data.patients.forEach(p=>{
    if(p.arr<=t) arrived++;
    if(p.left<=t){ if(p.lwbs) lw++; else treated++; }
    if(p.pickup!=null && p.pickup<=t){ seen++; sumW+=(p.pickup-p.arr); }
  });
  const meanWait=seen?sumW/seen:0, lwbsPct=arrived?100*lw/arrived:0;
  return {treated,lw,meanWait,lwbsPct,score:treated*10-lw*15,arrived};
}
function grade(s){
  if(s.meanWait<=TARGET.mean_wait*0.6 && s.lwbsPct<=TARGET.lwbs*0.5) return 'A';
  if(s.meanWait<=TARGET.mean_wait && s.lwbsPct<=TARGET.lwbs) return 'B';
  if(s.meanWait<=TARGET.mean_wait*1.6 && s.lwbsPct<=TARGET.lwbs*2) return 'C';
  if(s.meanWait<=TARGET.mean_wait*2.5) return 'D';
  return 'F';
}
const gradeColor={A:'#0F7B78',B:'#2f8f6b',C:'#D9A21B',D:'#c97a1b',F:'#C8402F'};
function fmtClock(t){ t=Math.max(0,Math.min(1439,Math.floor(t))); return String(Math.floor(t/60)).padStart(2,'0')+':'+String(t%60).padStart(2,'0'); }

function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.font='12px Segoe UI, Arial'; ctx.fillStyle='#5B6E77';
  ctx.fillText('Waiting room',10,20); ctx.fillText('Triage',224,20); ctx.fillText('ED beds ('+data.beds+')',300,20); ctx.fillText('Doctors',824,20);
  ctx.strokeStyle='#D3DCE0'; ctx.lineWidth=1;
  ctx.strokeRect(4,26,214,cv.height-40); ctx.strokeRect(222,26,72,cv.height-40); ctx.strokeRect(298,26,516,cv.height-40); ctx.strokeRect(818,26,120,cv.height-40);
  for(let i=0;i<data.triageNurses;i++){ const xy=triXY(i); ctx.strokeRect(xy.x-13,xy.y-13,26,26); }
  for(let i=0;i<data.beds;i++){ const xy=bedXY(i); ctx.strokeRect(xy.x-L.bed.w/2+4,xy.y-L.bed.h/2+4,L.bed.w-8,L.bed.h-8); }
  const docBusy=new Array(data.doctors).fill(false);
  data.patients.forEach((p,i)=>{
    const st=stateAt(p,simT);
    if(!disp[i]) disp[i]={x:0,y:cv.height/2,alpha:0};
    const d=disp[i];
    if(st){ d.x+=(st.pos.x-d.x)*0.18; d.y+=(st.pos.y-d.y)*0.18; d.alpha+=(1-d.alpha)*0.2; if(st.phase==='consult'&&st.doc!=null) docBusy[st.doc]=true; }
    else { d.alpha+=(0-d.alpha)*0.25; }
  });
  for(let i=0;i<data.doctors;i++){ const xy=docXY(i); ctx.beginPath(); ctx.arc(xy.x,xy.y,15,0,7);
    ctx.fillStyle=docBusy[i]?'#0F7B78':'#DDE4E7'; ctx.fill(); ctx.strokeStyle='#14262E'; ctx.stroke(); }
  data.patients.forEach((p,i)=>{
    const d=disp[i]; if(!d||d.alpha<0.02) return; const st=stateAt(p,simT);
    ctx.globalAlpha=d.alpha; ctx.beginPath(); ctx.arc(d.x,d.y,9,0,7); ctx.fillStyle=ESI_COL[p.esi]; ctx.fill();
    if(st&&st.phase==='consult'){ ctx.lineWidth=2.5; ctx.strokeStyle='#1a9c3a'; ctx.stroke(); }
    else if(st&&st.phase==='board'){ ctx.lineWidth=2.5; ctx.strokeStyle='#D9A21B'; ctx.stroke(); }
    else if(st&&st.phase==='bedwait'){ ctx.lineWidth=1.5; ctx.strokeStyle='#14262E'; ctx.setLineDash([3,2]); ctx.stroke(); ctx.setLineDash([]); }
    else if(st&&st.phase==='triage'){ ctx.lineWidth=2; ctx.strokeStyle='#0F7B78'; ctx.stroke(); }
    else { ctx.lineWidth=1; ctx.strokeStyle='#14262E'; ctx.stroke(); }
    ctx.globalAlpha=1;
  });
  floaties.forEach(f=>{ ctx.globalAlpha=Math.max(0,f.life); ctx.fillStyle=f.col; ctx.font='bold 12px Segoe UI';
    ctx.fillText(f.txt,f.x,f.y-(1-f.life)*30); ctx.globalAlpha=1; });
  floaties=floaties.filter(f=>f.life>0.02); floaties.forEach(f=>f.life-=0.02);
}
function frame(ts){
  if(!lastT) lastT=ts; const dt=Math.min(0.25,(ts-lastT)/1000); lastT=ts;
  if(playing){ const newT=Math.min(1440,simT+speed*dt); fireEvents(simT,newT); simT=newT; sl.value=Math.round(simT);
    if(simT>=1440 && !reported){ playing=false; pl.textContent='Play'; reported=true; const s=statsAt(1440);
      toast('Shift complete: grade '+grade(s)+', score '+s.score); } }
  draw();
  const s=statsAt(simT);
  clockEl.textContent=fmtClock(simT); scoreEl.textContent=s.score; treatedEl.textContent=s.treated;
  lwbsEl.textContent=s.lw; waitEl.textContent=s.meanWait.toFixed(1);
  const g=grade(s); gradeEl.textContent=g; gradeEl.style.background=gradeColor[g];
  if(Math.floor(simT)%30===0) liveEl.textContent=fmtClock(simT)+', score '+s.score+', '+s.treated+' treated, '+s.lw+' left without being seen';
  requestAnimationFrame(frame);
}
LABELS.forEach((lb,idx)=>{ const b=document.createElement('button'); b.type='button';
  b.textContent=lb+' ('+GAME[lb].doctors+'D/'+GAME[lb].nurses+'N)';
  b.onclick=()=>{ label=lb; layout(); simT=0; sl.value=0; playing=false; pl.textContent='Play';
    [...presetsEl.children].forEach(x=>x.classList.remove('active')); b.classList.add('active'); };
  presetsEl.appendChild(b); if(idx===1) b.classList.add('active');
});
SPEEDS.forEach(([lb,val],idx)=>{ const b=document.createElement('button'); b.type='button'; b.textContent=lb;
  b.onclick=()=>{ speed=val; [...speedsEl.children].forEach(x=>x.classList.remove('active')); b.classList.add('active'); };
  speedsEl.appendChild(b); if(idx===1) b.classList.add('active');
});
pl.onclick=()=>{ if(!audioCtx){ try{audioCtx=new (window.AudioContext||window.webkitAudioContext)();}catch(e){} }
  if(simT>=1440) simT=0; playing=!playing; pl.textContent=playing?'Pause':'Play'; };
sl.oninput=()=>{ simT=+sl.value; reported=false; };
layout(); requestAnimationFrame(frame);
})();
(function(){ const btn=document.getElementById('theme-toggle-abm'); if(!btn) return;
  const mq=window.matchMedia('(prefers-color-scheme: dark)');
  function apply(dark){ document.documentElement.dataset.theme=dark?'dark':'light'; btn.innerHTML=dark?'&#9728;&#65039;':'&#127769;'; }
  apply(mq.matches); btn.onclick=()=>apply(document.documentElement.dataset.theme!=='dark');
})();</script></body></html>"""

if __name__ == "__main__":
    main()
