#!/usr/bin/env python3
import torch
import pypose as pp
import matplotlib.pyplot as plt

# Params
T = 0.5
N = 5
rob_diam = 0.3
obs_x, obs_y, obs_diam = 6.0, 0.8, 0.5

v_max, v_min = 0.6, -0.6
omega_max, omega_min = torch.pi/4, -torch.pi/4

Q = torch.diag(torch.tensor([1.0, 1.0, 0.5]))
R = torch.diag(torch.tensor([0.5, 0.05]))

class DiffDrive(pp.module.NLS):
    def __init__(self, dt=T):
        super().__init__()
        self.register_buffer('dt', torch.tensor(dt))

    def state_transition(self, state, u, t=None):
        x, y, theta = state[...,0], state[...,1], state[...,2]
        v, omega = u[...,0], u[...,1]
        x_next = x + self.dt * v * torch.cos(theta)
        y_next = y + self.dt * v * torch.sin(theta)
        th_next = theta + self.dt * omega
        return torch.stack((x_next, y_next, th_next), dim=-1)

    def observation(self, state, u, t=None):
        return state

def obstacle_penalty(state):
    dx = state[...,0] - obs_x
    dy = state[...,1] - obs_y
    dist = torch.sqrt(dx**2 + dy**2)
    min_dist = (rob_diam/2 + obs_diam/2)
    return torch.clamp(min_dist - dist, min=0.0)**2 * 200.0

def solve_mpc(x0, ref_traj, horizon=N, iters=30):
    model = DiffDrive()
    U = torch.zeros((horizon, 2), requires_grad=True)
    opt = torch.optim.Adam([U], lr=0.1)

    for _ in range(iters):
        state = x0.clone().detach().float().unsqueeze(0) # [1,3]
        cost = torch.tensor(0.0)
        for k in range(horizon):
            u = U[k]
            u_clamped = torch.stack([
                torch.clamp(u[0], v_min, v_max),
                torch.clamp(u[1], omega_min, omega_max)
            ])
            state, _ = model(state, u_clamped.unsqueeze(0))
            diff = (state.squeeze(0) - ref_traj[k])
            cost = cost + diff @ Q @ diff + u_clamped @ R @ u_clamped
            cost = cost + obstacle_penalty(state.squeeze(0))
        opt.zero_grad()
        cost.backward()
        opt.step()

    with torch.no_grad():
        u0 = torch.stack([
            torch.clamp(U[0,0], v_min, v_max),
            torch.clamp(U[0,1], omega_min, omega_max)
        ]).detach().cpu().numpy()
    return u0

def main():
    x0 = torch.tensor([0.0, 0.0, 0.0])
    sim_time = 30
    steps = int(sim_time / T)

    # --- Prepare interactive plots ---
    plt.ion()

    # Trajectory + obstacle
    fig_xy, ax_xy = plt.subplots()
    traj_line, = ax_xy.plot([], [], 'b.-', label="Trajectory")
    circle = plt.Circle((obs_x, obs_y), obs_diam/2, color='r', fill=False, linestyle='--', label="Obstacle")
    ax_xy.add_patch(circle)
    ax_xy.set_xlabel('x'); ax_xy.set_ylabel('y'); ax_xy.grid(True); ax_xy.legend()
    ax_xy.set_title("Trajectory with Obstacle Avoidance")
    # Reasonable axis limits (adjust if needed)
    ax_xy.set_xlim(-0.5, 12.5)
    ax_xy.set_ylim(-0.5, 2.5)

    # Control inputs
    fig_u, ax_u = plt.subplots()
    line_v, = ax_u.plot([], [], label='v')
    line_w, = ax_u.plot([], [], label='omega')
    ax_u.set_xlabel('step'); ax_u.set_ylabel('control'); ax_u.grid(True); ax_u.legend()
    ax_u.set_title("Control inputs")

    xs, ys = [x0[0].item()], [x0[1].item()]
    v_hist, w_hist, t_hist = [], [], []

    try:
        for mpciter in range(steps):
            # Build reference trajectory
            ref_list = []
            for k in range(N):
                t_predict = (mpciter + k) * T
                x_ref = min(0.5 * t_predict, 12.0)
                y_ref = 1.0
                theta_ref = 0.0
                ref_list.append(torch.tensor([x_ref, y_ref, theta_ref]))
            ref_traj = torch.stack(ref_list)

            # Solve MPC
            u = solve_mpc(x0, ref_traj, horizon=N, iters=40)
            v, omega = float(u[0]), float(u[1])

            # Forward integrate system (closed loop)
            x0 = x0 + T*torch.tensor([
                v*torch.cos(x0[2]),
                v*torch.sin(x0[2]),
                omega
            ])

            # Collect data
            xs.append(float(x0[0])); ys.append(float(x0[1]))
            v_hist.append(v); w_hist.append(omega); t_hist.append(mpciter)

            # --- Live update ---
            traj_line.set_data(xs, ys)
            line_v.set_data(t_hist, v_hist)
            line_w.set_data(t_hist, w_hist)

            # Grow x/y limits if needed
            if xs[-1] > ax_xy.get_xlim()[1] - 0.5:
                ax_xy.set_xlim(ax_xy.get_xlim()[0], xs[-1] + 0.5)

            # Grow control plot x-axis
            ax_u.set_xlim(0, max(10, mpciter + 1))
            # Slightly pad control y-limits
            ymin = min(v_hist + w_hist) - 0.1
            ymax = max(v_hist + w_hist) + 0.1
            ax_u.set_ylim(ymin, ymax)

            fig_xy.canvas.draw(); fig_xy.canvas.flush_events()
            fig_u.canvas.draw();  fig_u.canvas.flush_events()
            plt.pause(0.001)  # small pause for UI update

    # At the end, turn off interactive mode and keep the window open
        plt.ioff()
        plt.show()

    except KeyboardInterrupt:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
