"""Command-line experiment runner producing CSV/PNG benchmark outputs."""

import argparse, csv, os, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.pinn import (
    OscillatorConfig, MLP, FourierPINN, PositiveParameter,
    exact_solution, losses_forward, losses_inverse, make_observations, relative_l2
)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def train_forward(model, cfg, steps=4000, n_phys=1000, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t = torch.linspace(0, cfg.t_end, n_phys).view(-1,1).requires_grad_(True)
    hist=[]
    for i in range(steps):
        opt.zero_grad()
        loss, parts = losses_forward(model, t, cfg)
        loss.backward(); opt.step()
        if i % 100 == 0:
            hist.append((i, float(loss), float(parts["physics"]), float(parts["boundary"])))
    return hist

def train_inverse(model, cfg, noise, steps=5000, n_phys=500, n_obs=30, seed=0):
    t_phys = torch.linspace(0, cfg.t_end, n_phys).view(-1,1).requires_grad_(True)
    t_obs, u_obs = make_observations(cfg, n_obs, noise, seed)
    mu_p = PositiveParameter(1.0)
    opt = torch.optim.Adam(list(model.parameters()) + list(mu_p.parameters()), lr=1e-3)
    hist=[]
    for i in range(steps):
        opt.zero_grad()
        loss, parts = losses_inverse(model, t_phys, t_obs, u_obs, cfg.m, cfg.k, mu_p)
        loss.backward(); opt.step()
        if i % 100 == 0:
            hist.append((i, float(loss), float(parts["mu"])))
    return mu_p().item(), hist, t_obs, u_obs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["forward","inverse","noise","frequency"], default="inverse")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    args=ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    seed_all(args.seed)

    if args.experiment=="forward":
        cfg=OscillatorConfig(mu=4,k=400)
        model=MLP()
        hist=train_forward(model,cfg,args.steps)
        t=torch.linspace(0,1,1000).view(-1,1)
        pred=model(t).detach(); truth=exact_solution(t,cfg)
        err=relative_l2(pred,truth).item()
        plt.figure(figsize=(8,4))
        plt.plot(t,truth,label="Analytical")
        plt.plot(t,pred,label="PINN")
        plt.xlabel("t"); plt.ylabel("u(t)"); plt.legend(); plt.tight_layout()
        plt.savefig("figures/forward_solution.png",dpi=180); plt.close()
        print(f"Forward relative L2 error: {err:.4e}")

    elif args.experiment=="inverse":
        cfg=OscillatorConfig(mu=4,k=400)
        model=MLP()
        mu_hat,hist,t_obs,u_obs=train_inverse(model,cfg,0.05,args.steps,seed=args.seed)
        print(f"True mu={cfg.mu:.4f}, estimated mu={mu_hat:.4f}")

    elif args.experiment=="noise":
        cfg=OscillatorConfig(mu=4,k=400)
        rows=[]
        for noise in [0.0,0.01,0.05,0.1,0.2]:
            model=MLP()
            mu_hat,_,_,_=train_inverse(model,cfg,noise,args.steps,seed=args.seed)
            rows.append((noise,cfg.mu,mu_hat,abs(mu_hat-cfg.mu)/cfg.mu))
            print(rows[-1])
        with open("results/noise_sweep.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["noise_std","mu_true","mu_est","relative_error"]); w.writerows(rows)

    elif args.experiment=="frequency":
        rows=[]
        for w0 in [20,40,80,120]:
            k=w0*w0
            cfg=OscillatorConfig(mu=4,k=k)
            for name,Model in [("vanilla",MLP),("fourier",FourierPINN)]:
                model=Model()
                mu_hat,_,_,_=train_inverse(model,cfg,0.05,args.steps,seed=args.seed)
                rows.append((w0,name,mu_hat,abs(mu_hat-cfg.mu)/cfg.mu))
                print(rows[-1])
        with open("results/frequency_sweep.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["omega0","model","mu_est","relative_error"]); w.writerows(rows)

if __name__=="__main__":
    main()
