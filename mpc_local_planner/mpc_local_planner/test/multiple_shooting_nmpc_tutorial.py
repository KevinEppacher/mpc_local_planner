#!/usr/bin/env python3
import torch
import pypose as pp
import matplotlib.pyplot as plt

# Diskretisierung
T = 0.2
N = 10  # prediction horizon

# Bounds
v_max, v_min = 0.6, -0.6
omega_max, omega_min = torch.pi/4, -torch.pi/4

# Gewichtungen
Q = torch.diag(torch.tensor([1.0, 5.0, 0.1]))
R = torch.diag(torch.tensor([0.5, 0.05]))

class DiffDrive(pp.module.NLS):
    """Nichtlineares Differentialfahrzeugmodell"""
    def __init__(self, dt=T):
        super().__init__()
        self.register_buffer('dt', torch.tensor(dt))

    def state_transition(self, state, u, t=None):
        # state: [...,3] = [x,y,theta]
        # u: [...,2] = [v,omega]
        x, y, theta = state[...,0], state[...,1], state[...,2]
        v, omega = u[...,0], u[...,1]
        x_next = x + self.dt * v * torch.cos(theta)
        y_next = y + self.dt * v * torch.sin(theta)
        th_next = theta + self.dt * omega
        return torch.stack((x_next, y_next, th_next), dim=-1)

    def observation(self, state, u, t=None):
        return state

def solve_mpc(x0, xs, horizon=N, iters=20):
    model = DiffDrive()
    U = torch.zeros((horizon, 2), requires_grad=True)  # Decision vars
    opt = torch.optim.Adam([U], lr=0.1)

    for i in range(iters):
        state = x0.clone().detach().float().unsqueeze(0)  # [1,3]
        cost = 0.0
        for k in range(horizon):
            u = U[k]
            u_clamped = torch.stack([
                v_max * torch.tanh(u[0]),
                omega_max * torch.tanh(u[1])
            ])
            state, _ = model(state, u_clamped.unsqueeze(0))  # <-- fix: take only next_state
            diff = (state.squeeze(0) - xs)
            cost = cost + diff @ Q @ diff + u_clamped @ R @ u_clamped

        opt.zero_grad()
        cost.backward()
        opt.step()

    with torch.no_grad():
        u0 = torch.stack([
            v_max * torch.tanh(U[0,0]),
            omega_max * torch.tanh(U[0,1])
        ]).numpy()
    return u0


def main():
    x0 = torch.tensor([0.0, 0.0, 0.0])
    xs = torch.tensor([1.5, 1.5, 0.0])
    sim_time = 20
    steps = int(sim_time / T)

    xx = [x0.numpy()]
    u_cl = []

    for k in range(steps):
        u = solve_mpc(x0, xs, horizon=N, iters=50)
        u_cl.append(u)
        # Propagate system
        v, omega = u
        x0 = x0 + T*torch.tensor([
            v*torch.cos(x0[2]),
            v*torch.sin(x0[2]),
            omega
        ])
        xx.append(x0.numpy())
        if torch.linalg.norm(x0 - xs) < 1e-2:
            break

    xx = torch.tensor(xx).T.numpy()
    u_cl = torch.tensor(u_cl).T.numpy()

    # Plots
    plt.figure()
    plt.plot(xx[0,:], xx[1,:], 'b.-')
    plt.plot(xs[0], xs[1], 'ro')
    plt.xlabel('x'); plt.ylabel('y'); plt.grid()
    plt.title('Trajectory')

    plt.figure()
    plt.plot(u_cl[0,:], label='v')
    plt.plot(u_cl[1,:], label='omega')
    plt.xlabel('step'); plt.ylabel('control'); plt.legend(); plt.grid()
    plt.title('Control inputs')
    plt.show()

if __name__ == "__main__":
    main()
