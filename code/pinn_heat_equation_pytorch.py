"""
PINN for the one-dimensional heat equation

    u_t = alpha u_xx,  x in (0, 1), t in (0, 1]
    u(x, 0) = sin(pi x)
    u(0, t) = u(1, t) = 0

Exact solution:

    u(x,t) = exp(-alpha*pi^2*t) sin(pi*x)

This script is intended for a PyTorch environment. It is included as the
reproducible code part of the course report.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


def exact_solution(x, t, alpha=1.0):
    return torch.exp(-alpha * torch.pi**2 * t) * torch.sin(torch.pi * x)


class MLP(nn.Module):
    def __init__(self, width=50, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def pde_residual(model, x, t, alpha):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t - alpha * u_xx


def sample_points(n_f, n_ic, n_bc, device):
    x_f = torch.rand(n_f, 1, device=device)
    t_f = torch.rand(n_f, 1, device=device)

    x_ic = torch.rand(n_ic, 1, device=device)
    t_ic = torch.zeros(n_ic, 1, device=device)
    u_ic = torch.sin(torch.pi * x_ic)

    t_bc = torch.rand(n_bc, 1, device=device)
    x_left = torch.zeros(n_bc // 2, 1, device=device)
    x_right = torch.ones(n_bc - n_bc // 2, 1, device=device)
    x_bc = torch.cat([x_left, x_right], dim=0)
    t_bc = torch.cat([t_bc[: n_bc // 2], t_bc[n_bc // 2 :]], dim=0)
    u_bc = torch.zeros_like(x_bc)

    return x_f, t_f, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = MLP(width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        x_f, t_f, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc = sample_points(
            args.n_f, args.n_ic, args.n_bc, device
        )

        r = pde_residual(model, x_f, t_f, args.alpha)
        loss_pde = torch.mean(r**2)
        loss_ic = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
        loss_bc = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)
        loss = loss_pde + loss_ic + loss_bc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.print_every == 0 or epoch == 1:
            history.append(
                [
                    epoch,
                    float(loss.detach().cpu()),
                    float(loss_pde.detach().cpu()),
                    float(loss_ic.detach().cpu()),
                    float(loss_bc.detach().cpu()),
                ]
            )
            print(
                f"epoch={epoch:05d} loss={loss.item():.4e} "
                f"pde={loss_pde.item():.4e} ic={loss_ic.item():.4e} bc={loss_bc.item():.4e}"
            )

    nx, nt = args.n_test_x, args.n_test_t
    x = torch.linspace(0, 1, nx, device=device).reshape(-1, 1)
    t = torch.linspace(0, 1, nt, device=device).reshape(-1, 1)
    xx, tt = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
    x_test = xx.reshape(-1, 1)
    t_test = tt.reshape(-1, 1)
    with torch.no_grad():
        pred = model(x_test, t_test)
        exact = exact_solution(x_test, t_test, args.alpha)
        err = pred - exact
        rmse = torch.sqrt(torch.mean(err**2)).item()
        linf = torch.max(torch.abs(err)).item()
        rel_l2 = torch.linalg.norm(err).item() / torch.linalg.norm(exact).item()
    print(f"RMSE={rmse:.6e}, L_inf={linf:.6e}, relative_L2={rel_l2:.6e}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "loss_history.csv", np.array(history), delimiter=",",
               header="epoch,total,pde,ic,bc", comments="")
    torch.save(model.state_dict(), out / "pinn_heat_model.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_f", type=int, default=2000)
    parser.add_argument("--n_ic", type=int, default=100)
    parser.add_argument("--n_bc", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_test_x", type=int, default=101)
    parser.add_argument("--n_test_t", type=int, default=101)
    parser.add_argument("--output_dir", type=str, default="data/pinn_outputs")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
