"""
Monte Carlo Simulation: Emergency Room Resource Allocation
Course: Modelling & Simulation
-----------------------------------------------------------
Question: How many doctors and beds does an ER need so that patients are seen
quickly and few leave without being seen, at reasonable cost?

Model (one simulated 24h day = one Monte Carlo replication)
  * Arrivals: non-homogeneous Poisson (hourly rate profile) x random daily
    demand multiplier (lognormal; captures flu waves / bad days).
  * Triage: ESI level 1 (critical) .. 5 (minor), sampled per patient.
  * Flow: patient waits for a BED -> waits for a DOCTOR -> consult
    (lognormal) -> stays in bed for treatment (lognormal) -> discharge.
  * Queues are priority queues (lower ESI first, then first-come).
  * ESI 4-5 patients have exponential patience; if the wait to see a doctor
    exceeds it, they leave without being seen (LWBS).
  * All input data are synthetic.

Run:  python er_monte_carlo.py        -> writes er_dashboard.html + PNGs
"""
import base64, csv, heapq, io, itertools, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_REPS = 2000            # replications for the baseline scenario
N_REPS_GRID = 400        # replications per scenario in the staffing grid
BASE = dict(doctors=4, beds=16)
DOCTOR_GRID = [3, 4, 5, 6]
BED_GRID = [12, 16, 20, 24]
COST = dict(doctor=1800, bed=250)      # synthetic $ per slot per day
TARGET = dict(mean_wait=30.0, lwbs=5.0)  # minutes, percent

# ---- synthetic input data ---------------------------------------------------
HOURLY_RATE = np.array([3.0, 2.5, 2.2, 2.0, 2.2, 3.0, 4.0, 5.5, 7.0, 8.0, 8.5, 8.5,
                        8.0, 8.0, 8.5, 9.0, 9.5, 9.5, 9.0, 8.0, 7.0, 5.5, 4.5, 3.5])
ESI_P = np.array([0.03, 0.12, 0.38, 0.32, 0.15])
CONSULT_MEAN = np.array([45, 35, 25, 15, 10])      # minutes of doctor time
TREAT_MEAN = np.array([240, 180, 120, 60, 30])     # minutes in bed
PATIENCE_MEAN = 120.0                              # minutes (ESI 4-5 only)
DEMAND_SIGMA = 0.15


def lognormal(rng, mean, sigma, size=None):
    mu = np.log(mean) - sigma ** 2 / 2
    return rng.lognormal(mu, sigma, size)


def simulate_day(rng, doctors, beds, detail=False):
    mult = rng.lognormal(-DEMAND_SIGMA ** 2 / 2, DEMAND_SIGMA)
    counts = rng.poisson(HOURLY_RATE * mult)
    n = int(counts.sum())
    hourly = counts.copy()
    if n == 0:
        return None
    t_arr = np.concatenate([h * 60 + rng.uniform(0, 60, c) for h, c in enumerate(counts)])
    t_arr.sort()
    esi = rng.choice(5, size=n, p=ESI_P)
    consult = lognormal(rng, CONSULT_MEAN[esi], 0.4)
    treat = lognormal(rng, TREAT_MEAN[esi], 0.5)
    patience = rng.exponential(PATIENCE_MEAN, n)

    ev, cnt = [], itertools.count()
    for i in range(n):
        heapq.heappush(ev, (t_arr[i], next(cnt), 0, i))     # 0=arrival 1=doc free 2=bed free
    free_b, free_d = beds, doctors
    bedq, docq = [], []
    bed_start = np.full(n, np.nan)
    wait = np.full(n, np.nan)
    lwbs = np.zeros(n, bool)
    lwbs_time = np.full(n, np.nan)
    bed_minutes = 0.0
    los = np.full(n, np.nan)

    def dispatch(t):
        nonlocal free_b, free_d, bed_minutes
        progress = True
        while progress:
            progress = False
            while free_b and bedq:
                _, ta, i = heapq.heappop(bedq)
                free_b -= 1
                bed_start[i] = t
                heapq.heappush(docq, (esi[i], t_arr[i], i))
                progress = True
            while free_d and docq:
                _, ta, i = heapq.heappop(docq)
                w = t - t_arr[i]
                if esi[i] >= 3 and w > patience[i]:          # ESI 4-5 (idx 3,4)
                    lwbs[i] = True
                    lwbs_time[i] = t
                    free_b += 1
                    bed_minutes += t - bed_start[i]
                    progress = True
                    continue
                free_d -= 1
                wait[i] = w
                heapq.heappush(ev, (t + consult[i], next(cnt), 1, i))
                heapq.heappush(ev, (t + consult[i] + treat[i], next(cnt), 2, i))
                progress = True

    while ev:
        t, _, kind, i = heapq.heappop(ev)
        if kind == 0:
            heapq.heappush(bedq, (esi[i], t, i))
        elif kind == 1:
            free_d += 1
        else:
            free_b += 1
            bed_minutes += t - bed_start[i]
            los[i] = t - t_arr[i]
        dispatch(t)

    seen = ~lwbs
    patients = None
    raw = None
    if detail:
        patients = [dict(patient_id=i + 1, arrival_min=round(t_arr[i], 2), arrival_clock=f"{int(t_arr[i]//60):02d}:{int(t_arr[i]%60):02d}",
                         esi_level=int(esi[i]) + 1, consult_min=round(consult[i], 2), treatment_min=round(treat[i], 2),
                         patience_min=round(patience[i], 2), wait_for_doctor_min=None if lwbs[i] else round(wait[i], 2),
                         left_without_being_seen=int(lwbs[i]),
                         length_of_stay_min=None if lwbs[i] else round(los[i], 2)) for i in range(n)]
        pickup = t_arr + wait                       # NaN for lwbs patients
        consult_end = pickup + consult
        release = t_arr + los                       # NaN for lwbs patients
        left_time = np.where(lwbs, lwbs_time, release)
        raw = dict(esi=esi, t_arr=t_arr, bed_start=bed_start, pickup=pickup,
                   consult_end=consult_end, left_time=left_time, lwbs=lwbs)
    w = wait[seen]
    crit = wait[seen & (esi <= 1)]
    return dict(
        arrivals=n,
        mean_wait=w.mean(),
        p95_wait=np.percentile(w, 95),
        crit_wait=crit.mean() if len(crit) else np.nan,
        lwbs_pct=100 * lwbs.mean(),
        bed_util=100 * min(bed_minutes / (beds * 1440), 1.5),
        mean_los=np.nanmean(los),
        hourly=hourly,
        patients=patients,
        raw=raw,
        doctors=doctors,
        beds=beds,
    )


def run(doctors, beds, reps, seed):
    rng = np.random.default_rng(seed)
    out = [simulate_day(rng, doctors, beds) for _ in range(reps)]
    return [o for o in out if o]


def col(res, k):
    return np.array([r[k] for r in res], float)


def ci(x):
    x = x[~np.isnan(x)]
    m, s = x.mean(), x.std(ddof=1)
    h = 1.96 * s / np.sqrt(len(x))
    return m, m - h, m + h


# ---- plotting ---------------------------------------------------------------
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
.toast{background:var(--chip);color:#fff;padding:9px 13px;border-radius:10px;font-size:13px;box-shadow:var(--shadow-md);animation:mcfade .2s ease-out}
@keyframes mcfade{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}
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


def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def hist(x, title, xlabel, color=TEAL, ref=None, ref_label=None):
    x = x[~np.isnan(x)]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.hist(x, bins=40, color=color, edgecolor="white", alpha=0.95)
    ax.axvline(x.mean(), color=INK, lw=1.6, label=f"mean {x.mean():.1f}")
    ax.axvline(np.percentile(x, 95), color=RED, lw=1.4, ls="--", label=f"95th pct {np.percentile(x, 95):.1f}")
    if ref is not None:
        ax.axvline(ref, color=AMBER, lw=1.6, ls=":", label=ref_label)
    ax.set(title=title, xlabel=xlabel, ylabel="Simulated days")
    ax.legend(frameon=False, fontsize=8)
    return to_b64(fig)



def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def export_csvs(outdir, base, grid):
    P = lambda n: os.path.join(outdir, n)
    params = [dict(parameter=f"hourly_rate_h{h:02d}", value=v, unit="patients/hour", description="Poisson arrival rate") for h, v in enumerate(HOURLY_RATE)]
    for i in range(5):
        params += [dict(parameter=f"esi{i+1}_probability", value=ESI_P[i], unit="prob", description="Triage mix"),
                   dict(parameter=f"esi{i+1}_consult_mean", value=CONSULT_MEAN[i], unit="min", description="Doctor time, lognormal sigma 0.4"),
                   dict(parameter=f"esi{i+1}_treatment_mean", value=TREAT_MEAN[i], unit="min", description="Bed time, lognormal sigma 0.5")]
    params += [dict(parameter="patience_mean", value=PATIENCE_MEAN, unit="min", description="Exponential; ESI 4-5 only"),
               dict(parameter="daily_demand_sigma", value=DEMAND_SIGMA, unit="", description="Lognormal daily demand multiplier"),
               dict(parameter="cost_per_doctor_day", value=COST["doctor"], unit="$", description="Synthetic"),
               dict(parameter="cost_per_bed_day", value=COST["bed"], unit="$", description="Synthetic"),
               dict(parameter="target_mean_wait", value=TARGET["mean_wait"], unit="min", description="Service target"),
               dict(parameter="target_lwbs", value=TARGET["lwbs"], unit="%", description="Service target"),
               dict(parameter="baseline_doctors", value=BASE["doctors"], unit="", description=""),
               dict(parameter="baseline_beds", value=BASE["beds"], unit="", description=""),
               dict(parameter="seed", value=SEED, unit="", description="numpy default_rng")]
    write_csv(P("input_parameters.csv"), params)

    reps = []
    for k, r in enumerate(base, 1):
        row = dict(replication=k, arrivals=r["arrivals"], mean_wait_min=round(r["mean_wait"], 3),
                   p95_wait_min=round(r["p95_wait"], 3), critical_wait_min=round(r["crit_wait"], 3),
                   lwbs_pct=round(r["lwbs_pct"], 3), bed_util_pct=round(r["bed_util"], 3), mean_los_min=round(r["mean_los"], 3))
        row.update({f"arrivals_h{h:02d}": int(c) for h, c in enumerate(r["hourly"])})
        reps.append(row)
    write_csv(P("replication_results.csv"), reps)

    write_csv(P("scenario_grid.csv"), [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()} for g in grid])

    day = simulate_day(np.random.default_rng(SEED + 999), BASE["doctors"], BASE["beds"], detail=True)
    write_csv(P("sample_day_patients.csv"), day["patients"])

from matplotlib.patches import Rectangle, Circle


from matplotlib.patches import Rectangle, Circle


def assign_slots(intervals, capacity):
    """Greedy stable-slot assignment so an animated patient keeps one visual
    seat/bed/doctor index for its whole stay instead of jumping around."""
    ev = sorted(intervals, key=lambda x: x[0])
    free = list(range(capacity)); heapq.heapify(free)
    active = []
    out = {}
    for start, end, pid in ev:
        while active and active[0][0] <= start:
            _, s = heapq.heappop(active)
            heapq.heappush(free, s)
        s = heapq.heappop(free) if free else capacity   # capacity = overflow marker
        out[pid] = s
        heapq.heappush(active, (end, s))
    return out


GAME_WAIT_SLOTS = 24
GAME_PRESETS = [("Understaffed", 3, 12), ("Baseline", 4, 16), ("Well-staffed", 5, 20), ("Maxed out", 6, 24)]
GAME_SEED = 777


def build_game_preset(doctors, beds, seed):
    """One simulated day, same arriving patients for every preset (same seed,
    drawn before capacity-dependent logic runs) so only staffing differs."""
    d = None
    while d is None:
        d = simulate_day(np.random.default_rng(seed), doctors, beds, detail=True)
    r = d["raw"]
    n = len(r["esi"])
    wait_iv = [(r["t_arr"][i], r["bed_start"][i], i) for i in range(n)]
    bed_iv = [(r["bed_start"][i], r["left_time"][i], i) for i in range(n)]
    doc_iv = [(r["pickup"][i], r["consult_end"][i], i) for i in range(n) if not r["lwbs"][i]]
    wslot = assign_slots(wait_iv, GAME_WAIT_SLOTS)
    bslot = assign_slots(bed_iv, beds)
    dslot = assign_slots(doc_iv, doctors)
    patients = []
    for i in range(n):
        patients.append(dict(
            esi=int(r["esi"][i]) + 1, arr=round(float(r["t_arr"][i]), 2),
            bedStart=round(float(r["bed_start"][i]), 2), left=round(float(r["left_time"][i]), 2),
            lwbs=bool(r["lwbs"][i]), waitSlot=wslot[i], bedSlot=bslot[i],
            pickup=None if r["lwbs"][i] else round(float(r["pickup"][i]), 2),
            consultEnd=None if r["lwbs"][i] else round(float(r["consult_end"][i]), 2),
            docSlot=dslot.get(i)))
    return dict(doctors=doctors, beds=beds, waitSlots=GAME_WAIT_SLOTS,
               arrivals=d["arrivals"], meanWait=round(float(d["mean_wait"]), 1),
               lwbsPct=round(float(d["lwbs_pct"]), 1), patients=patients)


def main():
    print(f"Baseline: {BASE}, {N_REPS} replications ...")
    base = run(BASE["doctors"], BASE["beds"], N_REPS, SEED)
    arr, mw, p95, lw, bu, cw, los = (col(base, k) for k in
        ["arrivals", "mean_wait", "p95_wait", "lwbs_pct", "bed_util", "crit_wait", "mean_los"])

    stats = {k: ci(v) for k, v in dict(arrivals=arr, mean_wait=mw, p95_wait=p95,
             lwbs=lw, bed_util=bu, crit_wait=cw, los=los).items()}
    prob_meet = 100 * np.mean((mw <= TARGET["mean_wait"]) & (lw <= TARGET["lwbs"]))
    prob_overload = 100 * np.mean(bu >= 90)

    print("Scenario grid ...")
    grid = []
    for d in DOCTOR_GRID:
        for b in BED_GRID:
            r = run(d, b, N_REPS_GRID, SEED + d * 100 + b)
            m, l_ = col(r, "mean_wait"), col(r, "lwbs_pct")
            grid.append(dict(doctors=d, beds=b,
                             cost=d * COST["doctor"] + b * COST["bed"],
                             mean_wait=float(m.mean()), lwbs=float(l_.mean()),
                             util=float(col(r, "bed_util").mean()),
                             p_meet=float(100 * np.mean((m <= TARGET["mean_wait"]) & (l_ <= TARGET["lwbs"])))))
    ok = [g for g in grid if g["mean_wait"] <= TARGET["mean_wait"] and g["lwbs"] <= TARGET["lwbs"]]
    best = min(ok, key=lambda g: g["cost"]) if ok else min(grid, key=lambda g: g["mean_wait"])

    print("Building arcade presets ...")
    game_presets = {label: build_game_preset(d, b, GAME_SEED) for label, d, b in GAME_PRESETS}
    game_json = json.dumps(game_presets)
    labels_json = json.dumps([label for label, _, _ in GAME_PRESETS])
    target_json = json.dumps(TARGET)

    # ---- figures
    figs = {}
    figs["arrivals"] = hist(arr, "Daily patient arrivals", "Patients per day", GREY)
    figs["wait"] = hist(mw, "Mean door-to-doctor wait", "Minutes", TEAL, TARGET["mean_wait"], "target 30 min")
    figs["p95"] = hist(p95, "95th percentile wait (worst-case patient)", "Minutes", AMBER)
    figs["lwbs"] = hist(lw, "Patients who leave without being seen", "% of arrivals", RED, TARGET["lwbs"], "target 5%")
    figs["util"] = hist(bu, "Bed utilisation", "% of bed-minutes used", TEAL, 90, "overload 90%")

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    run_mean = np.cumsum(mw) / np.arange(1, len(mw) + 1)
    ax.plot(run_mean, color=TEAL)
    ax.axhline(mw.mean(), color=GREY, ls="--", lw=1)
    ax.set(title="Convergence of estimated mean wait", xlabel="Replications", ylabel="Running mean (min)")
    figs["conv"] = to_b64(fig)

    hourly = np.array([r["hourly"] for r in base])
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    h = np.arange(24)
    ax.fill_between(h, np.percentile(hourly, 5, 0), np.percentile(hourly, 95, 0), color=TEAL, alpha=0.25, label="5th-95th pct")
    ax.plot(h, hourly.mean(0), color=TEAL, lw=2, label="mean")
    ax.set(title="Arrivals by hour of day", xlabel="Hour", ylabel="Patients per hour", xticks=range(0, 24, 3))
    ax.legend(frameon=False, fontsize=8)
    figs["hourly"] = to_b64(fig)

    Z = np.array([[next(g["mean_wait"] for g in grid if g["doctors"] == d and g["beds"] == b)
                   for b in BED_GRID] for d in DOCTOR_GRID])
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    im = ax.imshow(Z, cmap="RdYlGn_r", aspect="auto", vmin=0)
    ax.set(xticks=range(len(BED_GRID)), xticklabels=BED_GRID, yticks=range(len(DOCTOR_GRID)),
           yticklabels=DOCTOR_GRID, xlabel="Beds", ylabel="Doctors", title="Mean wait (min) by staffing level")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    figs["heat"] = to_b64(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for g in grid:
        good = g in ok
        ax.scatter(g["cost"], g["mean_wait"], s=60, color=TEAL if good else GREY, zorder=3)
        ax.annotate(f'{g["doctors"]}D/{g["beds"]}B', (g["cost"], g["mean_wait"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.scatter(best["cost"], best["mean_wait"], s=200, facecolors="none", edgecolors=RED, lw=2, zorder=4)
    ax.axhline(TARGET["mean_wait"], color=AMBER, ls=":")
    ax.set(title="Cost vs wait trade-off", xlabel="Daily staffing + bed cost ($)", ylabel="Mean wait (min)")
    figs["cost"] = to_b64(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.scatter(arr, mw, s=6, alpha=0.3, color=TEAL)
    z = np.polyfit(arr, mw, 1)
    xs = np.linspace(arr.min(), arr.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=RED)
    ax.set(title=f"Demand vs wait (corr {np.corrcoef(arr, mw)[0, 1]:.2f})", xlabel="Arrivals per day", ylabel="Mean wait (min)")
    figs["scatter"] = to_b64(fig)

    # ---- dashboard
    rows = "".join(
        f'<tr class="{"best" if g is best else ("ok" if g in ok else "")}"><td>{g["doctors"]}</td><td>{g["beds"]}</td>'
        f'<td>${g["cost"]:,}</td><td>{g["mean_wait"]:.1f}</td><td>{g["lwbs"]:.1f}%</td>'
        f'<td>{g["util"]:.0f}%</td><td>{g["p_meet"]:.0f}%</td></tr>' for g in grid)

    def kpi(label, key, unit, digits=1, icon="pulse"):
        m, lo, hi = stats[key]
        return (f'<div class="kpi"><div class="kpi-top"><span class="kpi-ic">{ICON_SVG[icon]}</span>'
                f'<div class="v">{m:.{digits}f}<small>{unit}</small></div></div>'
                f'<div class="l">{label}</div><div class="c">95% CI {lo:.{digits}f} to {hi:.{digits}f}</div></div>')

    def card(title, key, note=""):
        return f'<figure><img alt="{title}" src="data:image/png;base64,{figs[key]}"><figcaption>{note}</figcaption></figure>'

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet">
<title>ER Resource Allocation: Monte Carlo Dashboard</title><style>
{DESIGN_CSS}
</style></head><body><main>
<div class="hero">
<div class="hero-top"><span class="kicker">ER OPERATIONS &middot; MONTE CARLO LAB</span>
<button id="theme-toggle" type="button" aria-label="Toggle dark mode">&#127769;</button></div>
<h1>Emergency room resource allocation</h1>
<svg class="ekg" viewBox="0 0 600 30" preserveAspectRatio="none" aria-hidden="true"><polyline points="0,16 55,16 68,4 82,27 95,16 150,16 165,7 180,25 195,16 600,16"/></svg>
<p class="sub">Monte Carlo simulation of {N_REPS:,} synthetic 24-hour days at {BASE["doctors"]} doctors and {BASE["beds"]} beds, plus a staffing grid of {len(grid)} scenarios ({N_REPS_GRID} days each).</p>
</div>
<div class="kpis">
{kpi("Arrivals per day","arrivals","",1,'users')}{kpi("Mean wait to see doctor","mean_wait"," min",1,'clock')}
{kpi("Worst-case wait (95th pct)","p95_wait"," min",1,'clock')}{kpi("Critical (ESI 1-2) wait","crit_wait"," min",1,'alert')}
{kpi("Left without being seen","lwbs","%",1,'door')}{kpi("Bed utilisation","bed_util","%",0,'bed')}
</div>
<div class="rec"><b>Recommendation:</b> the cheapest configuration meeting both targets (mean wait &le; {TARGET["mean_wait"]:.0f} min, LWBS &le; {TARGET["lwbs"]:.0f}%)
is <b>{best["doctors"]} doctors and {best["beds"]} beds</b> at ${best["cost"]:,} per day, meeting the targets on {best["p_meet"]:.0f}% of simulated days.
The baseline meets them on {prob_meet:.0f}% of days and runs at 90%+ bed use on {prob_overload:.0f}% of days.</div>
<h2>Play: staff the ER</h2>
<p class="sub" style="margin:0 0 10px">The same patients arrive every round &mdash; only the staffing changes. Pick a shift, hit play, and see what happens.</p>
<div class="arcade">
<div class="arcade-bar">
<div class="seg" id="mc-presets" role="group" aria-label="Staffing preset"></div>
<div class="seg" id="mc-speeds" role="group" aria-label="Playback speed"></div>
<button id="mc-pl" type="button">Play</button>
</div>
<canvas id="mc-cv" width="920" height="380" role="img" aria-label="Animated view of the emergency room"></canvas>
<input id="mc-sl" type="range" min="0" max="1440" value="0" step="1" aria-label="Minute of the day" style="width:100%;margin-top:8px">
<div class="legend">
<span class="lg-esi" style="--c:#6E1423">ESI 1</span><span class="lg-esi" style="--c:#C8402F">ESI 2</span>
<span class="lg-esi" style="--c:#D9A21B">ESI 3</span><span class="lg-esi" style="--c:#0F7B78">ESI 4</span>
<span class="lg-esi" style="--c:#8A9BA3">ESI 5</span>
<span class="lg-ring dash">Waiting for doctor</span><span class="lg-ring green">With doctor</span>
</div>
<div class="scorebar">
<div>Time <b id="mc-clock">00:00</b></div>
<div>Score <b id="mc-score">0</b></div>
<div>Treated <b id="mc-treated">0</b></div>
<div>Walked out <b id="mc-lwbs">0</b></div>
<div>Wait so far <b id="mc-wait">0</b> min</div>
<div>Grade <span class="grade" id="mc-grade">-</span></div>
</div>
<p class="sr-only" id="mc-live" aria-live="polite"></p>
</div>
<div class="toasts" id="mc-toasts"></div>
<h2>Distribution of outcomes (baseline)</h2><div class="grid">
{card("","arrivals","Demand varies day to day because of the lognormal demand multiplier.")}
{card("","wait","Right-skewed: most days are fine, a few overloaded days drive the tail.")}
{card("","p95","What the unluckiest patients experience.")}
{card("","lwbs","Low-acuity patients who give up before being seen.")}
{card("","util","Days past 90% use are where queues explode.")}
{card("","hourly","The afternoon-evening peak drives the congestion.")}
</div>
<h2>Model checks and drivers</h2><div class="grid">
{card("","conv","The running mean stabilises, so the replication count is adequate.")}
{card("","scatter","Wait grows faster than linearly with demand (queueing effect).")}
</div>
<h2>Staffing scenarios</h2><div class="grid">
{card("","heat","Adding beds helps only once doctors are sufficient, and vice versa.")}
{card("","cost","Ringed point is the cheapest scenario meeting both targets.")}
</div>
<div class="tw" style="margin-top:14px"><table><thead><tr><th>Doctors</th><th>Beds</th><th>Daily cost</th><th>Mean wait (min)</th><th>LWBS</th><th>Bed use</th><th>Days meeting targets</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="meta">Shaded rows meet both targets; the darkest row is the cheapest. All data are synthetic; costs are illustrative. Seed {SEED}.</p>
</main>
<script>(function(){{
const GAME={game_json}, LABELS={labels_json}, TARGET={target_json}, WAITSLOTS={GAME_WAIT_SLOTS};
const SPEEDS=[["Slow",15],["Normal",60],["Fast",180],["Turbo",600]];
const ESI_COL={{1:'#6E1423',2:'#C8402F',3:'#D9A21B',4:'#0F7B78',5:'#8A9BA3'}};
const cv=document.getElementById('mc-cv'), ctx=cv.getContext('2d'), sl=document.getElementById('mc-sl'), pl=document.getElementById('mc-pl'),
presetsEl=document.getElementById('mc-presets'), speedsEl=document.getElementById('mc-speeds'), toastsEl=document.getElementById('mc-toasts'),
clockEl=document.getElementById('mc-clock'), scoreEl=document.getElementById('mc-score'), treatedEl=document.getElementById('mc-treated'),
lwbsEl=document.getElementById('mc-lwbs'), waitEl=document.getElementById('mc-wait'), gradeEl=document.getElementById('mc-grade'), liveEl=document.getElementById('mc-live');
let label=LABELS[1], speed=SPEEDS[1][1], playing=false, simT=0, lastT=0, disp={{}}, floaties=[], data, L, audioCtx=null, reported=false;

function beep(freq,dur,type){{ try{{ if(!audioCtx) audioCtx=new (window.AudioContext||window.webkitAudioContext)();
  const o=audioCtx.createOscillator(), g=audioCtx.createGain(); o.type=type||'sine'; o.frequency.value=freq;
  o.connect(g); g.connect(audioCtx.destination); g.gain.setValueAtTime(0.08,audioCtx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.0001,audioCtx.currentTime+dur); o.start(); o.stop(audioCtx.currentTime+dur); }}catch(e){{}} }}
function toast(msg){{ const d=document.createElement('div'); d.className='toast'; d.textContent=msg; toastsEl.appendChild(d); setTimeout(()=>d.remove(),2600); }}

function layout(){{
  data=GAME[label];
  const bedCols=Math.min(6,Math.max(3,Math.ceil(Math.sqrt(data.beds*1.4)))), bedRows=Math.ceil(data.beds/bedCols);
  const docCols=data.doctors>4?2:1, docRows=Math.ceil(data.doctors/docCols);
  L={{ wait:{{x0:26,y0:44,gap:34,cols:6,rows:4}},
      bed:{{x0:300,y0:40,w:486/bedCols,h:Math.min(56,300/bedRows),cols:bedCols}},
      doc:{{x0:837,y0:44,gap:44,cols:docCols}} }};
  disp={{}}; floaties=[]; reported=false;
}}
function waitXY(slot){{ if(slot>=WAITSLOTS) return {{x:L.wait.x0+(L.wait.cols-1)*L.wait.gap+18,y:L.wait.y0+(L.wait.rows-1)*L.wait.gap+18}};
  const c=slot%L.wait.cols, r=Math.floor(slot/L.wait.cols); return {{x:L.wait.x0+c*L.wait.gap,y:L.wait.y0+r*L.wait.gap}}; }}
function bedXY(slot){{ const c=slot%L.bed.cols, r=Math.floor(slot/L.bed.cols);
  return {{x:L.bed.x0+c*L.bed.w+L.bed.w/2,y:L.bed.y0+r*(L.bed.h+8)+L.bed.h/2}}; }}
function docXY(slot){{ const c=slot%L.doc.cols, r=Math.floor(slot/L.doc.cols); return {{x:L.doc.x0+c*L.doc.gap,y:L.doc.y0+r*L.doc.gap}}; }}

function stateAt(p,t){{
  if(t<p.arr) return null;
  if(t<p.bedStart) return {{phase:'wait',pos:waitXY(p.waitSlot)}};
  if(p.lwbs){{ if(t<p.left) return {{phase:'bedwait',pos:bedXY(p.bedSlot)}}; return null; }}
  if(p.pickup!=null && t<p.pickup) return {{phase:'bedwait',pos:bedXY(p.bedSlot)}};
  if(p.consultEnd!=null && t<p.consultEnd) return {{phase:'consult',pos:bedXY(p.bedSlot),doc:p.docSlot}};
  if(t<p.left) return {{phase:'treat',pos:bedXY(p.bedSlot)}};
  return null;
}}
function fireEvents(prevT,t){{
  if(t<=prevT) return;
  data.patients.forEach(p=>{{
    if(p.left>prevT && p.left<=t){{
      const xy=bedXY(p.bedSlot);
      if(p.lwbs){{ floaties.push({{x:xy.x,y:xy.y,txt:'-15 left',col:'#C8402F',life:1}}); toast('A patient left without being seen (ESI '+p.esi+')'); beep(220,0.3,'sawtooth'); }}
      else {{ floaties.push({{x:xy.x,y:xy.y,txt:'+10',col:'#0F7B78',life:1}}); beep(660,0.12,'sine'); }}
    }}
  }});
}}
function statsAt(t){{
  let treated=0,lw=0,seen=0,sumW=0,arrived=0;
  data.patients.forEach(p=>{{
    if(p.arr<=t) arrived++;
    if(p.left<=t){{ if(p.lwbs) lw++; else treated++; }}
    if(p.pickup!=null && p.pickup<=t){{ seen++; sumW+=(p.pickup-p.arr); }}
  }});
  const meanWait=seen?sumW/seen:0, lwbsPct=arrived?100*lw/arrived:0;
  return {{treated,lw,meanWait,lwbsPct,score:treated*10-lw*15,arrived}};
}}
function grade(s){{
  if(s.meanWait<=TARGET.mean_wait*0.6 && s.lwbsPct<=TARGET.lwbs*0.5) return 'A';
  if(s.meanWait<=TARGET.mean_wait && s.lwbsPct<=TARGET.lwbs) return 'B';
  if(s.meanWait<=TARGET.mean_wait*1.6 && s.lwbsPct<=TARGET.lwbs*2) return 'C';
  if(s.meanWait<=TARGET.mean_wait*2.5) return 'D';
  return 'F';
}}
const gradeColor={{A:'#0F7B78',B:'#2f8f6b',C:'#D9A21B',D:'#c97a1b',F:'#C8402F'}};
function fmtClock(t){{ t=Math.max(0,Math.min(1439,Math.floor(t))); return String(Math.floor(t/60)).padStart(2,'0')+':'+String(t%60).padStart(2,'0'); }}

function draw(){{
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.font='12px Segoe UI, Arial'; ctx.fillStyle='#5B6E77';
  ctx.fillText('Waiting room',12,20); ctx.fillText('ED beds ('+data.beds+')',300,20); ctx.fillText('Doctors',810,20);
  ctx.strokeStyle='#D3DCE0'; ctx.lineWidth=1;
  ctx.strokeRect(6,26,270,cv.height-40); ctx.strokeRect(288,26,502,cv.height-40); ctx.strokeRect(804,26,110,cv.height-40);
  for(let i=0;i<data.beds;i++){{ const xy=bedXY(i); ctx.strokeRect(xy.x-L.bed.w/2+4,xy.y-L.bed.h/2+4,L.bed.w-8,L.bed.h-8); }}
  const docBusy=new Array(data.doctors).fill(false);
  data.patients.forEach((p,i)=>{{
    const st=stateAt(p,simT);
    if(!disp[i]) disp[i]={{x:0,y:cv.height/2,alpha:0}};
    const d=disp[i];
    if(st){{ d.x+=(st.pos.x-d.x)*0.18; d.y+=(st.pos.y-d.y)*0.18; d.alpha+=(1-d.alpha)*0.2; if(st.phase==='consult'&&st.doc!=null) docBusy[st.doc]=true; }}
    else {{ d.alpha+=(0-d.alpha)*0.25; }}
  }});
  for(let i=0;i<data.doctors;i++){{ const xy=docXY(i); ctx.beginPath(); ctx.arc(xy.x,xy.y,15,0,7);
    ctx.fillStyle=docBusy[i]?'#0F7B78':'#DDE4E7'; ctx.fill(); ctx.strokeStyle='#14262E'; ctx.stroke(); }}
  data.patients.forEach((p,i)=>{{
    const d=disp[i]; if(!d||d.alpha<0.02) return; const st=stateAt(p,simT);
    ctx.globalAlpha=d.alpha; ctx.beginPath(); ctx.arc(d.x,d.y,9,0,7); ctx.fillStyle=ESI_COL[p.esi]; ctx.fill();
    if(st&&st.phase==='consult'){{ ctx.lineWidth=2.5; ctx.strokeStyle='#1a9c3a'; ctx.stroke(); }}
    else if(st&&st.phase==='bedwait'){{ ctx.lineWidth=1.5; ctx.strokeStyle='#14262E'; ctx.setLineDash([3,2]); ctx.stroke(); ctx.setLineDash([]); }}
    else {{ ctx.lineWidth=1; ctx.strokeStyle='#14262E'; ctx.stroke(); }}
    ctx.globalAlpha=1;
  }});
  floaties.forEach(f=>{{ ctx.globalAlpha=Math.max(0,f.life); ctx.fillStyle=f.col; ctx.font='bold 12px Segoe UI';
    ctx.fillText(f.txt,f.x,f.y-(1-f.life)*30); ctx.globalAlpha=1; }});
  floaties=floaties.filter(f=>f.life>0.02);
  floaties.forEach(f=>f.life-=0.02);
}}
function frame(ts){{
  if(!lastT) lastT=ts; const dt=Math.min(0.25,(ts-lastT)/1000); lastT=ts;
  if(playing){{ const newT=Math.min(1440,simT+speed*dt); fireEvents(simT,newT); simT=newT; sl.value=Math.round(simT);
    if(simT>=1440 && !reported){{ playing=false; pl.textContent='Play'; reported=true; const s=statsAt(1440);
      toast('Shift complete: grade '+grade(s)+', score '+s.score); }} }}
  draw();
  const s=statsAt(simT);
  clockEl.textContent=fmtClock(simT); scoreEl.textContent=s.score; treatedEl.textContent=s.treated;
  lwbsEl.textContent=s.lw; waitEl.textContent=s.meanWait.toFixed(1);
  const g=grade(s); gradeEl.textContent=g; gradeEl.style.background=gradeColor[g];
  if(Math.floor(simT)%30===0) liveEl.textContent=fmtClock(simT)+', score '+s.score+', '+s.treated+' treated, '+s.lw+' left without being seen';
  requestAnimationFrame(frame);
}}
LABELS.forEach((lb,idx)=>{{ const b=document.createElement('button'); b.type='button';
  b.textContent=lb+' ('+GAME[lb].doctors+'D/'+GAME[lb].beds+'B)';
  b.onclick=()=>{{ label=lb; layout(); simT=0; sl.value=0; playing=false; pl.textContent='Play';
    [...presetsEl.children].forEach(x=>x.classList.remove('active')); b.classList.add('active'); }};
  presetsEl.appendChild(b); if(idx===1) b.classList.add('active');
}});
SPEEDS.forEach(([lb,val],idx)=>{{ const b=document.createElement('button'); b.type='button'; b.textContent=lb;
  b.onclick=()=>{{ speed=val; [...speedsEl.children].forEach(x=>x.classList.remove('active')); b.classList.add('active'); }};
  speedsEl.appendChild(b); if(idx===1) b.classList.add('active');
}});
pl.onclick=()=>{{ if(!audioCtx){{ try{{audioCtx=new (window.AudioContext||window.webkitAudioContext)();}}catch(e){{}} }}
  if(simT>=1440) simT=0; playing=!playing; pl.textContent=playing?'Pause':'Play'; }};
sl.oninput=()=>{{ simT=+sl.value; reported=false; }};
layout(); requestAnimationFrame(frame);
}})();
(function(){{ const btn=document.getElementById('theme-toggle'); if(!btn) return;
  const mq=window.matchMedia('(prefers-color-scheme: dark)');
  function apply(dark){{ document.documentElement.dataset.theme=dark?'dark':'light'; btn.innerHTML=dark?'&#9728;&#65039;':'&#127769;'; }}
  apply(mq.matches); btn.onclick=()=>apply(document.documentElement.dataset.theme!=='dark');
}})();</script></body></html>"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "er_dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    export_csvs(os.path.dirname(os.path.abspath(__file__)), base, grid)
    print("Best:", best)
    print("Dashboard written to", path)
    for k, v in stats.items():
        print(f"{k:10s} mean {v[0]:8.2f}  CI [{v[1]:.2f}, {v[2]:.2f}]")
    print(f"P(meet targets) baseline {prob_meet:.1f}%")


if __name__ == "__main__":
    main()
