#!/usr/bin/env python3
import torch
import pypose as pp
import matplotlib.pyplot as plt

class CartPole(pp.module.NLS):
    def __init__(self, dt=0.02, m_cart=1.0, m_pole=0.1, length=0.5, g=9.81):
        super().__init__()
        # Modellparameter
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('m_cart', torch.tensor(m_cart))
        self.register_buffer('m_pole', torch.tensor(m_pole))
        self.register_buffer('length', torch.tensor(length))
        self.register_buffer('g', torch.tensor(g))

    def state_transition(self, state, u, t=None):
        """
        Zustand: [x, dx, theta, dtheta]
        Eingabe: [force]
        """
        x, dx, theta, dtheta = state[...,0], state[...,1], state[...,2], state[...,3]
        force = u[...,0]

        m_c = self.m_cart
        m_p = self.m_pole
        l = self.length
        g = self.g

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        total_mass = m_c + m_p
        temp = (force + m_p*l*dtheta**2*sin_t) / total_mass

        theta_acc = (g*sin_t - cos_t*temp) / (l*(4.0/3.0 - m_p*cos_t**2/total_mass))
        x_acc = temp - m_p*l*theta_acc*cos_t/total_mass

        # Euler Integration
        x = x + dx * self.dt
        dx = dx + x_acc * self.dt
        theta = theta + dtheta * self.dt
        dtheta = dtheta + theta_acc * self.dt

        return torch.stack((x, dx, theta, dtheta), dim=-1)

    def observation(self, state, u, t=None):
        return state

def simulate_cartpole(steps=200, u_value=0.0, dt=0.02):
    model = CartPole(dt=dt)
    state = torch.tensor([[0.0, 0.0, 0.1, 0.0]], dtype=torch.float32) # [B,4]
    control = torch.tensor([[u_value]], dtype=torch.float32)          # [B,1]

    traj = []
    for _ in range(steps):
        state, _ = model(state, control)
        traj.append(state.detach().clone())
    return torch.cat(traj, dim=0)  # [T,4]

def plot_cartpole(traj, dt=0.02):
    T = traj.shape[0]
    time = torch.arange(T) * dt
    x = traj[:,0].numpy()
    theta = traj[:,2].numpy()

    fig, axs = plt.subplots(2,1,figsize=(8,6),sharex=True)
    axs[0].plot(time, x, label="Cart Position x [m]")
    axs[0].legend(); axs[0].grid()

    axs[1].plot(time, theta, label="Pole Angle θ [rad]")
    axs[1].legend(); axs[1].grid()

    plt.xlabel("Time [s]")
    plt.suptitle("CartPole Simulation")
    plt.show()

def main():
    traj = simulate_cartpole(steps=200, u_value=0.0, dt=0.02)
    plot_cartpole(traj, dt=0.02)

if __name__ == "__main__":
    main()
