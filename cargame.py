import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 1e-3
GRID_SIZE = 5
GAMMA = 0.9
TARGET_UPDATE_EVERY = 200
GRAD_CLIP_NORM = 1.0
BATCH_SIZE = 32
MIN_BUFFER_SIZE = 64
GRADIENT_STEPS = 4
CRASH_PENALTY = -1
STAY_PENALTY = -0.5
LIVING_COST = 0.1
GRID_REWARD_MAX = 0.5
ACTIONS = ['U', 'D', 'L', 'R']
A = list(range(len(ACTIONS)))


def encode_state(s, grid_size):
    """Normalizes state coordinates for the Neural Network."""
    return torch.tensor(
        [s[0] / (grid_size - 1), s[1] / (grid_size - 1),
         s[2] / (grid_size - 1), s[3] / (grid_size - 1)],
        dtype=torch.float32, device=device
    )

class DQN(nn.Module):
    """Neural Network that approximates Q(s, a1, a2)."""
    def __init__(self, state_dim=4, action_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, s):
        return self.net(s)


class ReplayBuffer:
    """Fixed-size buffer to store (state_tensor, target_q) tuples."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_tensor, target_q):
        self.buffer.append((state_tensor, target_q))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.stack([x[0] for x in batch])
        targets = torch.stack([x[1] for x in batch])
        return states, targets

    def __len__(self):
        return len(self.buffer)


def neural_planning(env, iterations=8000):
    """
    Instead of a tabular dictionary, we train the DQN by sampling 
    states and using the environment rules to compute targets.
    """
    net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=10000)

    losses = []

    print("Starting Neural Planning (Function Approximation)...")
    for i in range(iterations):
        # 1. Sample any random state from the environment
        s = random.choice(env.states)
        if (s[0], s[1]) == (s[2], s[3]): continue 

        s_tensor = encode_state(s, env.grid_size)
        
        # 2. Compute Target for all 16 joint actions using the Model (env)
        with torch.no_grad():
            target_q_all = torch.zeros(16, device=device)
            for a_idx in range(16):
                a1, a2 = a_idx // 4, a_idx % 4
                
                s_next = env.transition(s, a1, a2)
                r = env.reward(s, a1, a2)
                
                if (s_next[0], s_next[1]) == (s_next[2], s_next[3]):
                    v_next = 0.0
                else:
                    sn_tensor = encode_state(s_next, env.grid_size)
                    q_next = target_net(sn_tensor).view(4, 4)
                    # Minimax: Maximize the guaranteed value (min over opponent)
                    v_next = torch.max(torch.min(q_next, dim=1)[0]).item()
                
                target_q_all[a_idx] = r + GAMMA * v_next

        # Store in replay buffer
        replay_buffer.push(s_tensor, target_q_all)

        # 3. Update the Network (sample mini-batch from buffer)
        if len(replay_buffer) >= MIN_BUFFER_SIZE:
            for _ in range(GRADIENT_STEPS):
                batch_states, batch_targets = replay_buffer.sample(BATCH_SIZE)
                q_pred = net(batch_states)
                loss = loss_fn(q_pred, batch_targets)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            losses.append(loss.item())

            if (i + 1) % TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(net.state_dict())

        if i % 1000 == 0 and losses:
            print(f"Planning Step {i} | Loss: {losses[-1]:.6f}")

    return net, losses

# Environment & Utilities

class CarGame:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.states = [(x1,y1,x2,y2) for x1 in range(grid_size) for y1 in range(grid_size) 
                       for x2 in range(grid_size) for y2 in range(grid_size)]
        self.grid_reward = self._make_grid_reward(grid_size)

    def _make_grid_reward(self, n):
        c = (n - 1) / 2
        grid = np.zeros((n, n))
        for x in range(n):
            for y in range(n):
                dist = abs(x - c) + abs(y - c)
                grid[x, y] = GRID_REWARD_MAX * (1 - dist / (2 * c))
        return grid

    def move(self, x, y, a):
        if a == 0: y_new = min(self.grid_size - 1, y + 1); x_new = x
        elif a == 1: y_new = max(0, y - 1); x_new = x
        elif a == 2: x_new = max(0, x - 1); y_new = y
        elif a == 3: x_new = min(self.grid_size - 1, x + 1); y_new = y
        return x_new, y_new

    def transition(self, s, a1, a2):
        x1n, y1n = self.move(s[0], s[1], a1)
        x2n, y2n = self.move(s[2], s[3], a2)
        return (x1n, y1n, x2n, y2n)

    def reward(self, s, a1, a2):
        x1, y1, x2, y2 = s
        sn = self.transition(s, a1, a2)
        if (sn[0], sn[1]) == (sn[2], sn[3]): return CRASH_PENALTY
        r = self.grid_reward[sn[0], sn[1]] - LIVING_COST
        if (sn[0], sn[1]) == (x1, y1): r += STAY_PENALTY
        if (sn[2], sn[3]) == (x2, y2): r -= STAY_PENALTY
        return r

def get_policy(net, env):
    policy = {}
    for s in env.states:
        s_t = encode_state(s, env.grid_size)
        with torch.no_grad():
            q = net(s_t).view(4, 4)
        a1 = torch.argmax(torch.min(q, dim=1)[0]).item()
        a2 = torch.argmin(q[a1]).item()
        policy[s] = (lambda a=a1: a, lambda a=a2: a)
    return policy

def rollout(env, s0, policy, T=20):
    traj = [s0]
    s = s0
    for _ in range(T):
        if (s[0], s[1]) == (s[2], s[3]): break
        a1, a2 = policy[s][0](), policy[s][1]()
        s = env.transition(s, a1, a2)
        traj.append(s)
    return traj
def draw_trajectory(ax, traj, grid_size, title="", subtitle=""):
    ax.clear()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_title(subtitle, fontsize=10, color="gray", pad=6)

    offset = 0.12
    off1 = np.array([ offset,  offset])   # Player 1 (red)
    off2 = np.array([-offset, -offset])   # Player 2 (blue)

    total_steps = len(traj) - 1

    for i, (s, s_next) in enumerate(zip(traj[:-1], traj[1:])):

        # alpha grows over time
        if i == total_steps - 1:
            alpha = 1.0
        else:
            alpha = 0.2 + 0.6 * (i / total_steps)

        # Player 1
        p1_start = np.array([s[0] + 0.5, s[1] + 0.5]) + off1
        p1_end   = np.array([s_next[0] + 0.5, s_next[1] + 0.5]) + off1
        d1 = p1_end - p1_start

        if np.linalg.norm(d1) > 1e-6:
            ax.arrow(
                p1_start[0], p1_start[1],
                d1[0], d1[1],
                color="red",
                alpha=alpha,
                head_width=0.15,
                length_includes_head=True
            )

        # Player 2
        p2_start = np.array([s[2] + 0.5, s[3] + 0.5]) + off2
        p2_end   = np.array([s_next[2] + 0.5, s_next[3] + 0.5]) + off2
        d2 = p2_end - p2_start

        if np.linalg.norm(d2) > 1e-6:
            ax.arrow(
                p2_start[0], p2_start[1],
                d2[0], d2[1],
                color="blue",
                alpha=alpha,
                head_width=0.15,
                length_includes_head=True
            )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimax DQN Car Game")
    parser.add_argument("--iterations", type=int, default=8000, help="Number of planning iterations")
    args = parser.parse_args()

    env = CarGame(grid_size=GRID_SIZE)

    # Neural planning
    net, loss = neural_planning(env, iterations=args.iterations)
    policy = get_policy(net, env)

    # Filter to non-crash states (players not at same position)
    valid_states = [s for s in env.states if (s[0], s[1]) != (s[2], s[3])]

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    is_headless = plt.get_backend().lower() == "agg" or not has_display

    fig, ax = plt.subplots()
    idx = [0]

    def update_plot(event=None):
        s0 = valid_states[idx[0]]
        traj = rollout(env, s0, policy)

        # Compute Stats
        total_moves = len(traj) - 1
        p1_moves = sum(
            (s[0], s[1]) != (sn[0], sn[1])
            for s, sn in zip(traj[:-1], traj[1:])
        )
        p2_moves = sum(
            (s[2], s[3]) != (sn[2], sn[3])
            for s, sn in zip(traj[:-1], traj[1:])
        )
        unique_states = len(set(traj))

        draw_trajectory(ax, traj, GRID_SIZE)

        fig.suptitle(
            f"Planning DQN Rollout from {s0}",
            fontsize=14,
            y=0.97
        )

        ax.set_title(
            f"Total moves: {total_moves} | "
            f"P1 moves: {p1_moves} | "
            f"P2 moves: {p2_moves} | "
            f"Unique states: {unique_states}",
            fontsize=10,
            color="gray",
            pad=6
        )

        fig.canvas.draw_idle()

    if not is_headless:
        from matplotlib.widgets import Button

        plt.subplots_adjust(bottom=0.2)

        axprev = plt.axes([0.25, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.65, 0.05, 0.1, 0.075])

        bprev = Button(axprev, "Prev")
        bnext = Button(axnext, "Next")

        def prev_state(event):
            idx[0] = (idx[0] - 1) % len(valid_states)
            update_plot()

        def next_state(event):
            idx[0] = (idx[0] + 1) % len(valid_states)
            update_plot()

        bprev.on_clicked(prev_state)
        bnext.on_clicked(next_state)

        update_plot()
        plt.show()
    else:
        # Save 5 random rollouts in headless mode
        sample_states = random.sample(valid_states, min(5, len(valid_states)))
        print(f"Headless mode detected (backend={plt.get_backend()}). Saving 5 rollouts...")
        for i, s0 in enumerate(sample_states):
            idx[0] = valid_states.index(s0)
            update_plot()
            rollout_path = f"rollout_{i+1}.png"
            fig.savefig(rollout_path, dpi=150, bbox_inches="tight")
            print(f"  Saved {rollout_path} from {s0}")

    plt.figure()
    plt.plot(loss)
    plt.title("Planning Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")

    if not is_headless:
        plt.show()
    else:
        loss_path = "planning_loss.png"
        plt.savefig(loss_path, dpi=150, bbox_inches="tight")
        print(f"Saved {loss_path}")

