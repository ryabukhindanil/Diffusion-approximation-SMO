import tkinter as tk
from tkinter import messagebox
import numpy as np
import math
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

# Цветовая схема
BG       = "#F4F6FB"
PANEL    = "#FFFFFF"
BORDER   = "#C8CFDF"
ACCENT   = "#3B5BDB"
TEXT     = "#1A1F36"
SUBTEXT  = "#5C6280"
GREEN    = "#2B8A3E"
RED      = "#C92A2A"
YELLOW   = "#E67700"
TEAL     = "#0B7285"
PLOT_BG  = "#FFFFFF"
GRID_C   = "#E9ECF0"
ENTRY_BG = "#F0F2F8"

plt.rcParams.update({
    "figure.facecolor": PLOT_BG,
    "axes.facecolor":   PLOT_BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "text.color":       TEXT,
    "grid.color":       GRID_C,
    "legend.facecolor": PANEL,
    "legend.edgecolor": BORDER,
    "legend.labelcolor": TEXT,
    "font.family":      "Segoe UI",
    "font.size":        10
})

# Условиям устойчивости
STABILITY_HINTS = {
    "M/G/1": (
        ("Условие устойчивости:", ACCENT, "bold"),
        ("   ρ = λ/μ  < 1  →  λ  < μ", TEXT, "normal"),
        ("", None, "normal"),
        ("Параметр диффузии:", ACCENT, "bold"),
        ("  γ = 2·(μ−λ) / (1 + cS²), cS² = μ²·σS²", SUBTEXT, "normal"),
    ),
    "M/M/1 с приоритетами": (
        ("Условие устойчивости:", ACCENT, "bold"),
        ("  ρ = (λ₁ + λ₂)/μ  < 1  →  λ₁ + λ₂  < μ", TEXT, "bold"),
        ("", None, "normal"),
        ("Средние времена ожидания:", ACCENT, "bold"),
        ("  E[W₁] = ρ / (μ·(1−ρ₁)·(1−ρ))", SUBTEXT, "normal"),
        ("  E[W₂] = ρ / (μ·(1−ρ)²)", SUBTEXT, "normal"),
    ),
    "M/M/c/K с нетерпением": (
        ("Система всегда устойчива!", ACCENT, "bold"),
        ("    ρ = λ / (c·μ)", TEXT, "normal"),
        ("", None, "normal"),
        ("При ρ > 1 потери растут из-за переполнения", SUBTEXT, "normal"),
    ),
    "G/G/2": (
        ("Условие устойчивости:", ACCENT, "bold"),
        ("  ρ = λ / (2·μ)  < 1  →  λ  < 2·μ", TEXT, "normal"),
        ("", None, "normal"),
        ("Параметр диффузии:", ACCENT, "bold"),
        ("  γ = 2·(2μ−λ) / (cA² + cS²), ", SUBTEXT, "normal"),
        ("  cA² = λ²·σA², cS² = μ²·σS²", SUBTEXT, "normal"),
    ),
}

# Рамка с заголовком
def labeled_frame(parent, title, **kwargs):
    outer = tk.Frame(parent, bg=PANEL, bd=1, relief="solid",
                     highlightbackground=BORDER, highlightthickness=1)
    tk.Label(outer, text=title, bg=PANEL, fg=ACCENT,
             font=("Segoe UI", 10, "bold"), pady=4).pack(fill="x", padx=8)
    sep = tk.Frame(outer, height=1, bg=BORDER)
    sep.pack(fill="x")
    inner = tk.Frame(outer, bg=PANEL, **kwargs)
    inner.pack(fill="both", expand=True, padx=8, pady=6)
    return outer, inner

# Симулятор событийного моделирования
class QueueSimulator:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def exp_rv(self, rate):
        if rate <= 0: return float('inf')
        return -math.log(self.rng.random()) / rate

    def normal_rv(self, mu, sigma):
        u1 = max(1e-10, self.rng.random())
        u2 = self.rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def simulate_mg1(self, T, lam, mu, cs2):
        mean_s = 1.0 / mu
        var_s = cs2 * (mean_s ** 2)
        sigma_s = math.sqrt(var_s) if var_s > 0 else 0
        t = 0.0; q = 0
        next_arrival = self.exp_rv(lam)
        next_departure = float('inf')
        ts = [0.0]; qs = [0.0]
        while t < T:
            if next_arrival < next_departure:
                t = next_arrival; q += 1
                if q == 1:
                    s_time = self.normal_rv(mean_s, sigma_s)
                    while s_time <= 0: s_time = self.normal_rv(mean_s, sigma_s)
                    next_departure = t + s_time
                next_arrival = t + self.exp_rv(lam)
            else:
                t = next_departure; q = max(0, q - 1)
                if q > 0:
                    s_time = self.normal_rv(mean_s, sigma_s)
                    while s_time <= 0: s_time = self.normal_rv(mean_s, sigma_s)
                    next_departure = t + s_time
                else:
                    next_departure = float('inf')
            ts.append(t); qs.append(q)
        return np.array(ts), np.array(qs)

    def simulate_mm1_priority(self, T, lam1, lam2, mu):
        t = 0.0; q1 = 0; q2 = 0; in_service_class = 0
        next_arr1 = self.exp_rv(lam1)
        next_arr2 = self.exp_rv(lam2)
        next_dep = float('inf')
        ts = [0.0]; qs_total = [0.0]
        while t < T:
            next_event = min(next_arr1, next_arr2, next_dep); t = next_event
            if abs(t - next_arr1) < 1e-12:
                q1 += 1; next_arr1 = t + self.exp_rv(lam1)
                if in_service_class == 0:
                    in_service_class = 1; next_dep = t + self.exp_rv(mu)
            elif abs(t - next_arr2) < 1e-12:
                q2 += 1; next_arr2 = t + self.exp_rv(lam2)
                if in_service_class == 0:
                    in_service_class = 2; next_dep = t + self.exp_rv(mu)
            else:
                if in_service_class == 1: q1 = max(0, q1 - 1)
                elif in_service_class == 2: q2 = max(0, q2 - 1)
                in_service_class = 0
                if q1 > 0:
                    in_service_class = 1; next_dep = t + self.exp_rv(mu)
                elif q2 > 0:
                    in_service_class = 2; next_dep = t + self.exp_rv(mu)
                else:
                    next_dep = float('inf')
            qs_total.append(q1 + q2); ts.append(t)
        return np.array(ts), np.array(qs_total)

    def simulate_mmck_impatient(self, T, lam, mu, c, K, alpha):
        t = 0.0; n = 0
        next_arrival = self.exp_rv(lam)
        ts = [0.0]; qs = [0.0]
        while t < T:
            rate_dep = min(n, c) * mu if n > 0 else 0
            waiters = max(0, n - c); rate_abandon = waiters * alpha
            t_arr = next_arrival
            t_dep = t + self.exp_rv(rate_dep) if rate_dep > 0 else float('inf')
            t_abn = t + self.exp_rv(rate_abandon) if rate_abandon > 0 else float('inf')
            next_event_time = min(t_arr, t_dep, t_abn)
            if next_event_time >= T: break
            t = next_event_time
            if abs(t - t_arr) < 1e-12:
                if n < K: n += 1
                next_arrival = t + self.exp_rv(lam)
            elif abs(t - t_dep) < 1e-12:
                n = max(0, n - 1)
            else:
                n = max(0, n - 1)
            ts.append(t); qs.append(n)
        return np.array(ts), np.array(qs)

    def simulate_gg2(self, T, lam, mu, ca2, cs2):
        mean_a = 1.0 / lam
        sigma_a = math.sqrt(ca2) * mean_a if ca2 > 0 else 0.0
        mean_s = 1.0 / mu
        sigma_s = math.sqrt(cs2) * mean_s if cs2 > 0 else 0.0
        def gen_ia():
            while True:
                v = self.normal_rv(mean_a, sigma_a)
                if v > 0: return v
        def gen_svc():
            while True:
                v = self.normal_rv(mean_s, sigma_s)
                if v > 0: return v
        t = 0.0
        n = 0
        dep_times = [float('inf'), float('inf')]
        next_arrival = t + gen_ia()
        ts = [0.0]; qs = [0.0]
        while t < T:
            next_dep = min(dep_times)
            next_event = min(next_arrival, next_dep)
            if next_event >= T:
                break
            t = next_event
            if next_arrival <= next_dep: # Приход
                n += 1
                for i in range(2):
                    if dep_times[i] == float('inf'):
                        dep_times[i] = t + gen_svc()
                        break
                next_arrival = t + gen_ia()
            else: # Уход
                idx = 0 if dep_times[0] <= dep_times[1] else 1
                dep_times[idx] = float('inf')
                n -= 1
                # Если есть ждущий клиент — назначить на освободившийся канал
                busy = sum(1 for d in dep_times if d < float('inf'))
                if n > busy:
                    dep_times[idx] = t + gen_svc()
            ts.append(t)
            qs.append(float(n))
        return np.array(ts), np.array(qs)

# Диффузия и метрики ошибок
def solve_fokker_planck_variable_drift(x_grid, theta_func, sigma_sq):
    h = x_grid[1] - x_grid[0]
    integrand = np.array([2 * theta_func(x) / sigma_sq for x in x_grid])
    exponent = np.zeros_like(x_grid)
    for i in range(1, len(x_grid)):
        exponent[i] = exponent[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * h
    p_raw = np.maximum(np.exp(exponent), 0)
    area = np.trapezoid(p_raw, x_grid)
    if area <= 0: return x_grid, np.zeros_like(x_grid)
    return x_grid, p_raw / area


def _ou_trajectory(T, dt, gamma, sigma_norm, seed=None):
    N = max(1, int(T / dt))
    t = np.linspace(0, T, N)
    z = np.zeros(N)
    rng = np.random.default_rng(seed if seed is not None else 42)
    sq = math.sqrt(dt)
    for i in range(1, N):
        z[i] = z[i-1] - gamma * z[i-1] * dt + sigma_norm * sq * rng.standard_normal()
    return t, z

def calculate_error_metrics(model_func, params, T_sim=200, dt=0.05, M=15):
    rhos_to_test = np.concatenate([
        np.linspace(0.1, 0.75, 14),
        np.linspace(0.76, 0.95, 10)
    ])
    mae_list = []; mse_list = []; rho_list = []
    base_params = list(params)

    for rho_target in rhos_to_test:
        try:
            if model_func.__name__ == 'run_mg1':
                lam_new = rho_target * base_params[1]
                cur_params = (lam_new, base_params[1], base_params[2], T_sim, dt)
            elif model_func.__name__ == 'run_mm1_priority':
                lam1 = base_params[0]; mu = base_params[2]
                lam2_new = rho_target * mu - lam1
                if lam2_new <= 0: continue
                cur_params = (lam1, lam2_new, mu, T_sim, dt)
            elif model_func.__name__ == 'run_mmck':
                lam_new = rho_target * base_params[1] * base_params[2]
                cur_params = (lam_new, base_params[1], base_params[2],
                              base_params[3], base_params[4], T_sim, dt)
            elif model_func.__name__ == 'run_gg2':
                mu_b = base_params[1]; ca2_b = base_params[2]; cs2_b = base_params[3]
                lam_new = rho_target * 2 * mu_b
                cur_params = (lam_new, mu_b, ca2_b, cs2_b, T_sim, dt)
            else:
                continue

            accum_mae = 0.0; accum_mse = 0.0; count = 0
            for _ in range(M):
                res, _ = model_func(cur_params)
                if res is None: continue
                z_sim_i = np.interp(res["t_diff"], res["ts_sim"], res["zs_sim"])
                diff = z_sim_i - res["zs_diff"]
                accum_mae += float(np.mean(np.abs(diff)))
                accum_mse += float(np.mean(diff ** 2))
                count += 1
            if count > 0:
                rho_list.append(rho_target)
                mae_list.append(accum_mae / count)
                mse_list.append(accum_mse / count)
        except Exception:
            continue

    return np.array(rho_list), np.array(mae_list), np.array(mse_list)

# Функции обработки моделей
def run_mg1(params):
    lam, mu, cs2, T_sim, dt = params
    rho = lam / mu
    if rho >= 1:
        return None, f"ρ = {rho:.3f} ≥ 1. Система неустойчива!"

    theta = mu - lam
    sigma_sq = lam + mu * cs2
    sigma = math.sqrt(sigma_sq)
    gamma = 2.0 * theta / sigma_sq
    mean_q_exact = rho + (rho**2 * (1.0 + cs2)) / (2.0 * (1.0 - rho))
    scale = 1.0 / gamma
    
    # Симуляция
    sim = QueueSimulator(seed=42)
    ts_sim, qs_sim = sim.simulate_mg1(T_sim, lam, mu, cs2)
    zs_sim = (qs_sim - mean_q_exact) / scale

    sigma_norm = math.sqrt(2.0 * gamma)
    t_diff, zs_diff = _ou_trajectory(T_sim, dt, gamma, sigma_norm, seed=42)

    z_sim_curr = np.interp(t_diff, ts_sim, zs_sim)
    curr_mae = float(np.mean(np.abs(z_sim_curr - zs_diff)))
    curr_mse = float(np.mean((z_sim_curr - zs_diff) ** 2))

    stats = {
        "ρ = λ/μ": f"{rho:.4f}",
        "γ = 2(μ−λ)/(λ+μ·cS²)": f"{gamma:.4f}",
        "E[Q] точное (П-Х)": f"{mean_q_exact:.4f}",
        "E[Q] диффузия (1/γ)": f"{1.0/gamma:.4f}",
        "MAE (текущее)": f"{curr_mae:.4f}",
        "MSE (текущее)": f"{curr_mse:.4f}",
    }
    return {
        "ts_sim": ts_sim, "zs_sim": zs_sim,
        "t_diff": t_diff, "zs_diff": zs_diff,
        "stats": stats,
        "params": (lam, mu, cs2, rho, gamma),
    }, None


def run_mm1_priority(params):
    lam1, lam2, mu, T_sim, dt = params
    lam  = lam1 + lam2
    rho1 = lam1 / mu; rho2 = lam2 / mu; rho = rho1 + rho2
    if rho >= 1:
        return None, f"ρ = {rho:.3f} ≥ 1. Система неустойчива!"

    theta    = mu - lam
    sigma_sq = lam + mu
    gamma    = 2.0 * theta / sigma_sq
    mean_q_exact = rho / (1.0 - rho)
    scale    = 1.0 / gamma

    sim = QueueSimulator(seed=42)
    ts_sim, qs_sim = sim.simulate_mm1_priority(T_sim, lam1, lam2, mu)
    zs_sim = (qs_sim - mean_q_exact) / scale

    sigma_norm = math.sqrt(2.0 * gamma)
    t_diff, zs_diff = _ou_trajectory(T_sim, dt, gamma, sigma_norm, seed=42)

    z_sim_curr = np.interp(t_diff, ts_sim, zs_sim)
    curr_mae = float(np.mean(np.abs(z_sim_curr - zs_diff)))
    curr_mse = float(np.mean((z_sim_curr - zs_diff) ** 2))

    Ew1 = rho / (mu * (1 - rho1) * (1 - rho)) if (1 - rho) > 1e-9 else float('inf')
    Ew2 = rho / (mu * (1 - rho) ** 2) if (1 - rho) > 1e-9 else float('inf')

    stats = {
        "ρ₁ = λ₁/μ": f"{rho1:.4f}",
        "ρ₂ = λ₂/μ": f"{rho2:.4f}",
        "ρ = ρ₁+ρ₂": f"{rho:.4f}",
        "E[W₁] (приоритетный)": f"{Ew1:.4f}",
        "E[W₂] (обычный)": f"{Ew2:.4f}",
        "MAE (текущее)": f"{curr_mae:.4f}",
        "MSE (текущее)": f"{curr_mse:.4f}",
    }
    return {
        "ts_sim": ts_sim, "zs_sim":  zs_sim,
        "t_diff": t_diff, "zs_diff": zs_diff,
        "stats": stats,
        "params": (lam1, lam2, mu, rho1, rho2),
    }, None

def run_mmck(params):
    lam, mu, c, K, alpha, T_sim, dt = params
    rho = lam / (c * mu)

    pi = np.zeros(K + 1); pi[0] = 1.0
    for n in range(1, K + 1):
        if n <= c:
            pi[n] = pi[n-1] * lam / (n * mu)
        else:
            denom = c * mu + (n - c) * alpha
            pi[n] = pi[n-1] * lam / denom if denom > 0 else 0
    s = pi.sum(); pi = pi / s if s > 0 else pi

    mean_q_exact = float(sum(n * pi[n] for n in range(K + 1)))
    var_q = float(sum((n - mean_q_exact)**2 * pi[n] for n in range(K + 1)))
    scale = math.sqrt(var_q) if var_q > 0 else 1.0

    sim = QueueSimulator(seed=42)
    ts_sim, qs_sim = sim.simulate_mmck_impatient(T_sim, lam, mu, c, K, alpha)
    zs_sim = (qs_sim - mean_q_exact) / scale

    N_steps = max(1, int(T_sim / dt))
    t_diff = np.linspace(0, T_sim, N_steps)
    q_diff = np.zeros(N_steps); q_diff[0] = mean_q_exact
    rng = np.random.default_rng(42); sqrt_dt = math.sqrt(dt)
    for i in range(1, N_steps):
        curr_q = max(0.0, min(float(K), q_diff[i-1]))
        if curr_q <= c:
            drift = lam - curr_q * mu
            diff_coeff = math.sqrt(max(lam + curr_q * mu, 1e-9))
        else:
            waiters = curr_q - c
            drift = lam - c * mu - waiters * alpha
            diff_coeff = math.sqrt(max(lam + c * mu + waiters * alpha, 1e-9))
        dq = drift * dt + diff_coeff * sqrt_dt * rng.standard_normal()
        q_diff[i] = q_diff[i-1] + dq
        if q_diff[i] < 0: q_diff[i] = -q_diff[i]
        if q_diff[i] > K: q_diff[i] = 2 * K - q_diff[i]
    zs_diff = (q_diff - mean_q_exact) / scale

    x_grid = np.linspace(0, K, 400)
    def theta_func(x):
        if x <= c: return lam - x * mu
        else: return lam - c * mu - (x - c) * alpha
    sigma_sq_avg = lam + c * mu
    x_th_raw, p_th_raw = solve_fokker_planck_variable_drift(x_grid, theta_func, sigma_sq_avg)
    x_th = (x_th_raw - mean_q_exact) / scale
    p_th = p_th_raw * scale

    z_sim_curr = np.interp(t_diff, ts_sim, zs_sim)
    curr_mae = float(np.mean(np.abs(z_sim_curr - zs_diff)))
    curr_mse = float(np.mean((z_sim_curr - zs_diff) ** 2))

    stats = {
        "ρ = λ/(c·μ)": f"{rho:.4f}",
        "P(потеря)": f"{pi[K]:.4f}",
        "E[N] (в системе)": f"{mean_q_exact:.4f}",
        "MAE (текущее)": f"{curr_mae:.4f}",
        "MSE (текущее)": f"{curr_mse:.4f}",
    }
    return {
        "ts_sim": ts_sim, "zs_sim": zs_sim,
        "t_diff": t_diff, "zs_diff": zs_diff,
        "x_th": x_th, "p_th": p_th,
        "stats": stats,
        "params": (lam, mu, c, K, alpha, rho),
    }, None


def run_gg2(params):
    lam, mu, ca2, cs2, T_sim, dt = params
    c = 2
    rho = lam / (c * mu)
    if rho >= 1:
        return None, f"ρ = {rho:.3f} ≥ 1. Система неустойчива!"

    theta    = c * mu - lam
    sigma_sq = lam * ca2 + c * mu * cs2
    sigma    = math.sqrt(sigma_sq)
    gamma    = 2.0 * theta / sigma_sq if sigma_sq > 0 else 1.0
    mean_q_approx = 1.0 / gamma if gamma > 0 else 1.0
    scale = mean_q_approx

    # Симуляция
    sim = QueueSimulator(seed=42)
    ts_sim, qs_sim = sim.simulate_gg2(T_sim, lam, mu, ca2, cs2)
    zs_sim = (qs_sim - mean_q_approx) / scale

    sigma_norm = math.sqrt(2.0 * gamma)
    t_diff, zs_diff = _ou_trajectory(T_sim, dt, gamma, sigma_norm, seed=42)

    z_sim_curr = np.interp(t_diff, ts_sim, zs_sim)
    curr_mae   = float(np.mean(np.abs(z_sim_curr - zs_diff)))
    curr_mse   = float(np.mean((z_sim_curr - zs_diff) ** 2))

    # Матрица ковариации Σ
    sig_diag = lam * ca2 + mu * cs2
    sig_off  = lam * ca2 / 2.0
    Sigma = np.array([[sig_diag, sig_off],
                      [sig_off,  sig_diag]])

    stats = {
        "ρ = λ/(2·μ)": f"{rho:.4f}",
        "γ = 2·θ/(cA²+cS²)": f"{gamma:.4f}",
        "E[Q] (диффузия, 1/γ)": f"{mean_q_approx:.4f}",
        "σ²₁₁ (диагональ)": f"{Sigma[0,0]:.4f}",
        "σ₁₂ (ковариация)": f"{Sigma[0,1]:.4f}",
        "MAE (текущее)": f"{curr_mae:.4f}",
        "MSE (текущее)": f"{curr_mse:.4f}",
    }
    return {
        "ts_sim": ts_sim, "zs_sim": zs_sim,
        "t_diff": t_diff, "zs_diff": zs_diff,
        "stats": stats,
        "params": (lam, mu, ca2, cs2, rho, gamma),
        "Sigma": Sigma,
    }, None

# Графики
def plot_trajectories(ax, ts_sim, zs_sim, t_diff, zs_diff, model_name):
    step = max(1, len(ts_sim) // 3000)
    ax.plot(ts_sim[::step], zs_sim[::step],
            color=ACCENT, alpha=0.8, linewidth=1.0, label="Симуляция")
    ax.plot(t_diff, zs_diff,
            color=RED, linewidth=1.6, linestyle='-', label="Диффузия")
    ax.axhline(0, color=SUBTEXT, linewidth=0.7, linestyle=':', alpha=0.7)
    ax.set_title(f"{model_name}: Сравнение траекторий Z(t)",
                 fontsize=11, color=TEXT, fontweight='bold', pad=10)
    ax.set_xlabel("Время t"); ax.set_ylabel("Z(t)")
    ax.legend(fontsize=9, loc='best'); ax.grid(True, alpha=0.4)


def plot_stationary_density(ax, x_theory, p_theory, model_name):
    if x_theory is not None and p_theory is not None:
        ax.plot(x_theory, p_theory, color=RED, linewidth=2.2, label="Диффузионная плотность")
        ax.fill_between(x_theory, p_theory, alpha=0.15, color=RED)
    ax.set_title(f"{model_name}: Стационарное распределение",
                 fontsize=10, color=TEXT, pad=8)
    ax.set_xlabel("Нормированная длина x"); ax.set_ylabel("Плотность p(x)")
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.4, axis='y')


def plot_error_metrics(ax_mae, ax_mse, model_func, params, model_name):
    rhos, maes, mses = calculate_error_metrics(model_func, params, T_sim=200, dt=0.05, M=15)
    if len(rhos) > 0:
        ax_mae.plot(rhos, maes, color=GREEN, lw=2.5, marker='o', markersize=4)
        ax_mae.set_title(f"{model_name}: Зависимость MAE от загрузки ρ",
                         fontsize=10, color=TEXT)
        ax_mae.set_xlabel("ρ"); ax_mae.set_ylabel("MAE")
        ax_mae.grid(True, alpha=0.4)

        ax_mse.plot(rhos, mses, color=RED, lw=2.5, marker='s', markersize=4)
        ax_mse.set_title(f"{model_name}: Зависимость MSE от загрузки ρ",
                         fontsize=10, color=TEXT)
        ax_mse.set_xlabel("ρ"); ax_mse.set_ylabel("MSE")
        ax_mse.grid(True, alpha=0.4)
    else:
        for ax in (ax_mae, ax_mse):
            ax.text(0.5, 0.5, "Недостаточно данных",
                    ha='center', va='center', transform=ax.transAxes)

# GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Диффузионная аппроксимация нормированных очередей СМО")
        self.geometry("1450x950")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.entries = {}
        self.active_model = tk.StringVar()
        self.tab_btns = {}
        self._build_ui()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=ACCENT)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Диффузионная аппроксимация нормированных очередей СМО",
                 bg=ACCENT, fg="white", font=("Segoe UI", 13, "bold"),
                 pady=10, padx=14).pack(anchor="center")

        tab_bar = tk.Frame(self, bg=BG, pady=6)
        tab_bar.pack(fill="x", padx=12)
        self.models = ["M/G/1", "M/M/1 с приоритетами", "M/M/c/K с нетерпением", "G/G/2"]
        self.active_model.set(self.models[0])

        for m in self.models:
            btn = tk.Button(tab_bar, text=m, bg=PANEL, fg=SUBTEXT,
                            relief="flat", font=("Segoe UI", 10),
                            bd=1, padx=14, pady=6,
                            highlightbackground=BORDER, highlightthickness=1,
                            command=lambda m=m: self._switch_model(m))
            btn.pack(side="left", padx=3)
            self.tab_btns[m] = btn

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        left_col = tk.Frame(main, bg=BG, width=320)
        left_col.pack(side="left", fill="y", padx=(0, 10))
        left_col.pack_propagate(False)

        self.sim_outer, self.sim_frame = labeled_frame(left_col, "Параметры симуляции")
        self.sim_outer.pack(fill="x", pady=(0, 8))

        self.param_outer, self.param_frame = labeled_frame(left_col, "Параметры модели")
        self.param_outer.pack(fill="x", pady=(0, 8))

        self.hint_outer, self.hint_frame = labeled_frame(left_col, "Условия устойчивости и формулы")
        self.hint_outer.pack(fill="x", pady=(0, 8))

        calc_btn = tk.Button(left_col, text="▶   Рассчитать и Построить",
                             bg=ACCENT, fg="white", font=("Segoe UI", 11, "bold"),
                             relief="flat", pady=9, command=self._calculate)
        calc_btn.pack(fill="x", pady=(0, 8))

        self.stats_outer, self.stats_frame = labeled_frame(left_col, "Результаты расчёта")
        self.stats_outer.pack(fill="both", expand=True)

        right_col = tk.Frame(main, bg=PANEL, bd=1, relief="solid",
                             highlightbackground=BORDER, highlightthickness=1)
        right_col.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(10, 8), facecolor=PLOT_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_col)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        tb_frame = tk.Frame(right_col, bg=PANEL)
        tb_frame.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, tb_frame)

        self._switch_model(self.models[0])

    def _switch_model(self, model):
        self.active_model.set(model)
        for m, btn in self.tab_btns.items():
            btn.configure(bg=ACCENT if m == model else PANEL,
                          fg="white" if m == model else SUBTEXT)
        for w in self.param_frame.winfo_children(): w.destroy()
        for w in self.hint_frame.winfo_children(): w.destroy()
        for w in self.stats_frame.winfo_children(): w.destroy()
        self._build_sim_params()
        self._build_params(model)
        self._build_hints(model)

    def _build_sim_params(self):
        if self.sim_frame.winfo_children(): return
        self._add_param_to_frame(self.sim_frame, "T (время):", "T_sim", "100")
        self._add_param_to_frame(self.sim_frame, " dt (шаг):", "dt", "0.05")

    def _add_param_to_frame(self, frame, label, key, default):
        row = tk.Frame(frame, bg=PANEL)
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label, bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 10), anchor="w").pack(side="left", fill="x", expand=True)
        var = tk.StringVar(value=str(default))
        ent = tk.Entry(row, textvariable=var, width=7, bg=ENTRY_BG, fg=TEXT,
                       insertbackground=TEXT, relief="flat",
                       font=("Segoe UI", 10), bd=1,
                       highlightbackground=BORDER, highlightthickness=1)
        ent.pack(side="right")
        self.entries[key] = var

    def _add_param(self, label, key, default):
        self._add_param_to_frame(self.param_frame, label, key, default)

    def _build_hints(self, model):
        hints = STABILITY_HINTS.get(model, ())
        for line, color, style in hints:
            if color is None:
                tk.Label(self.hint_frame, text="", bg=PANEL, height=1).pack(fill="x")
                continue
            weight = "bold" if style == "bold" else ("italic" if style == "italic" else "normal")
            tk.Label(self.hint_frame, text=line, bg=PANEL, fg=color,
                     font=("Segoe UI", 10, weight),
                     anchor="w", justify="left").pack(fill="x")

    def _build_params(self, model):
        if model == "M/G/1":
            self._add_param("λ (интенсивность входного потока)", "lam", 0.7)
            self._add_param("μ (интенсивность обслуживания)", "mu", 1.0)
            self._add_param("cS² (вариац. времени обслуживания)", "cs2", 1.0)
        elif model == "M/M/1 с приоритетами":
            self._add_param("λ₁ (интенсив. (высший приоритет))", "lam1", 0.3)
            self._add_param("λ₂ (интенсив. (низший приоритет))", "lam2", 0.4)
            self._add_param("μ (интенсивность обслуживания)", "mu", 1.0)
        elif model == "M/M/c/K с нетерпением":
            self._add_param("λ (интенсивность входного потока)", "lam", 2.0)
            self._add_param("μ (интенсивность обслуживания)", "mu", 1.0)
            self._add_param("c (число каналов обслуживания)", "c", 3)
            self._add_param("K (ёмкость буфера)", "K", 10)
            self._add_param("α (нетерпение)", "alpha", 0.2)
        elif model == "G/G/2":
            self._add_param("λ (интенсивность входного потока)", "lam", 0.8)
            self._add_param("μ (интенсивность обслуж. канала)", "mu", 1.0)
            self._add_param("cA² (вариац. интервалов поступления)", "ca2", 1.0)
            self._add_param("cS² (вариац. времени обслуживания)", "cs2", 1.0)

    def _get(self, key, cast=float):
        try: return cast(self.entries[key].get())
        except: return 0

    def _calculate(self):
        model = self.active_model.get()
        T_sim = self._get("T_sim"); dt = self._get("dt")
        res = None; err = None
        self.config(cursor="watch"); self.update_idletasks()
        try:
            if model == "M/G/1":
                params = (self._get("lam"), self._get("mu"), self._get("cs2"), T_sim, dt)
                res, err = run_mg1(params)
                if res: self._plot_mg1(res, params)
            elif model == "M/M/1 с приоритетами":
                params = (self._get("lam1"), self._get("lam2"), self._get("mu"), T_sim, dt)
                res, err = run_mm1_priority(params)
                if res: self._plot_mm1p(res, params)
            elif model == "M/M/c/K с нетерпением":
                params = (self._get("lam"), self._get("mu"),
                          int(self._get("c")), int(self._get("K")),
                          self._get("alpha"), T_sim, dt)
                res, err = run_mmck(params)
                if res: self._plot_mmck(res, params)
            elif model == "G/G/2":
                params = (self._get("lam"), self._get("mu"),
                          self._get("ca2"), self._get("cs2"), T_sim, dt)
                res, err = run_gg2(params)
                if res: self._plot_gg2(res, params)
        except Exception as e:
            messagebox.showerror("Ошибка!", str(e)); return
        finally:
            self.config(cursor="")
        if err: messagebox.showerror("Недопустимые параметры!", err); return
        if res: self._show_stats(res["stats"])

    # Дополнительные функции построения

    def _plot_mg1(self, res, params):
        self.fig.clear()
        gs = gridspec.GridSpec(3, 2, figure=self.fig, height_ratios=[1.5, 1, 1], hspace=0.45, wspace=0.3)
        ax_traj = self.fig.add_subplot(gs[0, :])
        plot_trajectories(ax_traj, res["ts_sim"], res["zs_sim"], res["t_diff"], res["zs_diff"], "M/G/1")
        ax_dist = self.fig.add_subplot(gs[1, 0])
        _, _, _, _, gamma = res["params"]
        x_th = np.linspace(0, 6, 200)
        p_th = gamma * np.exp(-gamma * x_th)
        plot_stationary_density(ax_dist, x_th, p_th, "M/G/1")
        ax_mae = self.fig.add_subplot(gs[1, 1])
        ax_mse = self.fig.add_subplot(gs[2, 1])
        plot_error_metrics(ax_mae, ax_mse, run_mg1, params, "M/G/1")
        ax_empty = self.fig.add_subplot(gs[2, 0])
        ax_empty.axis('off')
        self.fig.canvas.draw()

    def _plot_mm1p(self, res, params):
        self.fig.clear()
        gs = gridspec.GridSpec(3, 2, figure=self.fig, height_ratios=[1.5, 1, 1], hspace=0.45, wspace=0.3)
        ax_traj = self.fig.add_subplot(gs[0, :])
        plot_trajectories(ax_traj, res["ts_sim"], res["zs_sim"], res["t_diff"], res["zs_diff"], "M/M/1 Priority")
        ax_dist = self.fig.add_subplot(gs[1, 0])
        lam1, lam2, mu, _, _ = res["params"]
        theta = mu - (lam1 + lam2)
        sigma_sq = (lam1 + lam2) + mu
        gamma = 2.0 * theta / sigma_sq
        x_th = np.linspace(0, 6, 200)
        p_th = gamma * np.exp(-gamma * x_th)
        plot_stationary_density(ax_dist, x_th, p_th, "M/M/1 P")
        ax_marg = self.fig.add_subplot(gs[1, 1])
        g1 = 2*theta/(lam1+mu) if theta > 0 else 1
        g2 = 2*theta/(lam2+mu) if theta > 0 else 1
        x = np.linspace(0, 6, 200)
        ax_marg.plot(x, g1*np.exp(-g1*x), color=ACCENT, lw=2, label="Класс 1")
        ax_marg.plot(x, g2*np.exp(-g2*x), color=RED, lw=2, ls='--', label="Класс 2")
        ax_marg.set_title("Маргинальные плотности", fontsize=10, color=TEXT)
        ax_marg.legend(); ax_marg.grid(True, alpha=0.4)
        ax_mae = self.fig.add_subplot(gs[2, 0])
        ax_mse = self.fig.add_subplot(gs[2, 1])
        plot_error_metrics(ax_mae, ax_mse, run_mm1_priority, params, "M/M/1 P")
        self.fig.canvas.draw()

    def _plot_mmck(self, res, params):
        self.fig.clear()
        gs = gridspec.GridSpec(3, 2, figure=self.fig, height_ratios=[1.5, 1, 1], hspace=0.45, wspace=0.3)
        ax_traj = self.fig.add_subplot(gs[0, :])
        plot_trajectories(ax_traj, res["ts_sim"], res["zs_sim"], res["t_diff"], res["zs_diff"], "M/M/c/K")
        ax_dist = self.fig.add_subplot(gs[1, 0])
        plot_stationary_density(ax_dist, res["x_th"], res["p_th"], "M/M/c/K")
        ax_mae = self.fig.add_subplot(gs[1, 1])
        ax_mse = self.fig.add_subplot(gs[2, 1])
        plot_error_metrics(ax_mae, ax_mse, run_mmck, params, "M/M/c/K")
        ax_empty = self.fig.add_subplot(gs[2, 0])
        ax_empty.axis('off')
        self.fig.canvas.draw()

    def _plot_gg2(self, res, params):
        self.fig.clear()
        gs = gridspec.GridSpec(3, 2, figure=self.fig, height_ratios=[1.5, 1, 1], hspace=0.45, wspace=0.3)
        ax_traj = self.fig.add_subplot(gs[0, :])
        plot_trajectories(ax_traj, res["ts_sim"], res["zs_sim"], res["t_diff"], res["zs_diff"], "G/G/2")
        ax_mat = self.fig.add_subplot(gs[1, 0])
        Sigma = res["Sigma"]
        vmax = max(Sigma[0, 0], Sigma[1, 1])
        im = ax_mat.imshow(Sigma, cmap="Blues", aspect="auto", vmin=0, vmax=vmax * 1.1)
        ax_mat.set_title("Матрица ковариации Σ", fontsize=10, color=TEXT)
        ax_mat.set_xticks([0, 1]); ax_mat.set_yticks([0, 1])
        ax_mat.set_xticklabels(["Канал 1", "Канал 2"])
        ax_mat.set_yticklabels(["Канал 1", "Канал 2"])
        for i in range(2):
            for j in range(2):
                val = Sigma[i, j]
                txt_color = "white" if val > vmax * 0.6 else "black"
                ax_mat.text(j, i, f"{val:.3f}", ha="center", va="center", color=txt_color, fontsize=12, fontweight='bold')
        self.fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
        ax_mae = self.fig.add_subplot(gs[1, 1])
        ax_mse = self.fig.add_subplot(gs[2, 1])
        plot_error_metrics(ax_mae, ax_mse, run_gg2, params, "G/G/2")
        ax_info = self.fig.add_subplot(gs[2, 0])
        ax_info.axis('off')
        self.fig.canvas.draw()

    def _show_stats(self, stats):
        for w in self.stats_frame.winfo_children(): w.destroy()
        for k, v in stats.items():
            row = tk.Frame(self.stats_frame, bg=PANEL)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=k, bg=PANEL, fg=SUBTEXT,
                     font=("Segoe UI", 10), anchor="w").pack(side="left", fill="x", expand=True)
            tk.Label(row, text=v, bg=PANEL, fg=GREEN,
                     font=("Segoe UI", 10, "bold"), anchor="e").pack(side="right")


if __name__ == "__main__":
    app = App()
    app.mainloop()
