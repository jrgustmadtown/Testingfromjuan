"""
General-Sum Car Game - Nash DQN

Two-player general-sum game where each player has their own reward function.
Requires Nash equilibrium computation instead of minimax.
"""
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
STEP_SIZE = 1 / GRID_SIZE  # Each move is a step of this size in [0, 1) space
GAMMA = 0.9
TARGET_UPDATE_EVERY = 200
GRAD_CLIP_NORM = 1.0
BATCH_SIZE = 32
MIN_BUFFER_SIZE = 64
GRADIENT_STEPS = 4
# Rewards normalized: scale factor 1/0.9 to get max reward = +1
# Original ratios preserved: crash:stay:living:grid = 10:5:1:5
CRASH_PENALTY = -10/9    # -1.111 (was -1)
STAY_PENALTY = -5/9      # -0.556 (was -0.5)
LIVING_COST = 1/9        #  0.111 (was 0.1)
GRID_REWARD_MAX = 5/9    #  0.556 (was 0.5)
ACTIONS = ['U', 'D', 'L', 'R']
A = list(range(len(ACTIONS)))


def encode_state(s, grid_size):
    """
    Normalizes state coordinates for the Neural Network.
    Positions are in [0, 1-STEP_SIZE] space, where each grid cell = STEP_SIZE.
    """
    step = 1 / grid_size
    return torch.tensor(
        [s[0] * step, s[1] * step,
         s[2] * step, s[3] * step],
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


def export_weights(net, filepath, player_info=""):
    """
    Export network weights in standard format:
    - W_ij is weight from unit i in previous layer to unit j in next layer
    - Columns separated by commas, rows by newlines
    - Layer matrices separated by "-----"
    """
    with open(filepath, 'w') as f:
        layer_idx = 0
        for name, module in net.net.named_children():
            if isinstance(module, nn.Linear):
                # PyTorch stores as (out, in), transpose to get W_ij from i to j
                W = module.weight.data.cpu().numpy().T
                b = module.bias.data.cpu().numpy()
                
                if layer_idx > 0:
                    f.write("-----\n")
                
                f.write(f"# Layer {layer_idx}: Linear({W.shape[0]} -> {W.shape[1]})\n")
                f.write("# Weight matrix W (W_ij = weight from unit i to unit j):\n")
                for row in W:
                    f.write(",".join(f"{v:.6f}" for v in row) + "\n")
                
                f.write("# Bias vector:\n")
                f.write(",".join(f"{v:.6f}" for v in b) + "\n")
                
                layer_idx += 1
        
        f.write("-----\n")
        f.write("# Metadata\n")
        f.write(f"# Architecture: 4 -> 64 -> 64 -> 16\n")
        f.write(f"# Activation: ReLU (after layers 0 and 1)\n")
        f.write(f"# Output: 16 Q-values for joint actions (a1*4 + a2)\n")
        f.write(f"# Actions: 0=Up, 1=Down, 2=Left, 3=Right\n")
        if player_info:
            f.write(f"# {player_info}\n")
    
    print(f"Saved weights to {filepath}")


def neural_planning(env, iterations=8000):
    """
    Instead of a tabular dictionary, we train the DQN by sampling 
    states and using the environment rules to compute targets.
    """
    net1 = DQN().to(device)
    net2 = DQN().to(device)
    target_net1 = DQN().to(device)
    target_net2 = DQN().to(device)
    target_net1.load_state_dict(net1.state_dict())
    target_net2.load_state_dict(net2.state_dict())
    optimizer1 = optim.Adam(net1.parameters(), lr=LR)
    optimizer2 = optim.Adam(net2.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer1 = ReplayBuffer(capacity=10000)
    replay_buffer2 = ReplayBuffer(capacity=10000)

    losses1 = []
    losses2 = []

    print("Starting Neural Planning (Independent Learners)...")
    for i in range(iterations):
        # 1. Sample any random state from the environment
        s = random.choice(env.states)
        if (s[0], s[1]) == (s[2], s[3]): continue 

        s_tensor = encode_state(s, env.grid_size)
        
        # 2. Compute Target for all 16 joint actions using the Model (env)
        # Batched: compute all transitions and rewards, then one forward pass
        with torch.no_grad():
            next_states = []
            rewards1 = []
            rewards2 = []
            terminal_mask = []
            
            for a_idx in range(16):
                a1, a2 = a_idx // 4, a_idx % 4
                s_next = env.transition(s, a1, a2)
                r1, r2 = env.reward(s, a1, a2)
                next_states.append(s_next)
                rewards1.append(r1)
                rewards2.append(r2)
                terminal_mask.append((s_next[0], s_next[1]) == (s_next[2], s_next[3]))
            
            rewards1 = torch.tensor(rewards1, dtype=torch.float32, device=device)
            rewards2 = torch.tensor(rewards2, dtype=torch.float32, device=device)
            terminal_mask = torch.tensor(terminal_mask, dtype=torch.bool, device=device)
            
            # Encode all 16 next states in one batch
            next_tensors = torch.stack([
                encode_state(ns, env.grid_size) for ns in next_states
            ])
            
            # Forward pass on both target networks
            q1_next_all = target_net1(next_tensors).view(16, 4, 4)  # [16, 4, 4]
            q2_next_all = target_net2(next_tensors).view(16, 4, 4)  # [16, 4, 4]
            
            # Get greedy actions from current networks (for opponent modeling)
            q1_current = net1(next_tensors).view(16, 4, 4)
            q2_current = net2(next_tensors).view(16, 4, 4)
            
            # P1 greedy: argmax_a1 of max_a2 Q1 (best a1 assuming best response)
            # P2 greedy: argmax_a2 of max_a1 Q2 (best a2 assuming best response)
            a1_greedy = q1_current.max(dim=2)[0].argmax(dim=1)  # [16]
            a2_greedy = q2_current.max(dim=1)[0].argmax(dim=1)  # [16]
            
            # V1(s') = max_a1 Q1(s', a1, π2(s')) - P1 picks best given P2's greedy
            # V2(s') = max_a2 Q2(s', π1(s'), a2) - P2 picks best given P1's greedy
            v1_next = torch.zeros(16, device=device)
            v2_next = torch.zeros(16, device=device)
            for idx in range(16):
                v1_next[idx] = q1_next_all[idx, :, a2_greedy[idx]].max()
                v2_next[idx] = q2_next_all[idx, a1_greedy[idx], :].max()
            
            v1_next[terminal_mask] = 0.0
            v2_next[terminal_mask] = 0.0
            
            target_q1 = rewards1 + GAMMA * v1_next
            target_q2 = rewards2 + GAMMA * v2_next

        # Store in replay buffers
        replay_buffer1.push(s_tensor, target_q1)
        replay_buffer2.push(s_tensor, target_q2)

        # 3. Update both networks (sample mini-batch from each buffer)
        if len(replay_buffer1) >= MIN_BUFFER_SIZE:
            for _ in range(GRADIENT_STEPS):
                # Update P1's network
                batch_states, batch_targets = replay_buffer1.sample(BATCH_SIZE)
                q_pred = net1(batch_states)
                loss1 = loss_fn(q_pred, batch_targets)
                optimizer1.zero_grad()
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(net1.parameters(), GRAD_CLIP_NORM)
                optimizer1.step()
                
                # Update P2's network
                batch_states, batch_targets = replay_buffer2.sample(BATCH_SIZE)
                q_pred = net2(batch_states)
                loss2 = loss_fn(q_pred, batch_targets)
                optimizer2.zero_grad()
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(net2.parameters(), GRAD_CLIP_NORM)
                optimizer2.step()

            losses1.append(loss1.item())
            losses2.append(loss2.item())

            if (i + 1) % TARGET_UPDATE_EVERY == 0:
                target_net1.load_state_dict(net1.state_dict())
                target_net2.load_state_dict(net2.state_dict())

        if i % 1000 == 0 and losses1:
            print(f"Step {i} | P1 Loss: {losses1[-1]:.6f} | P2 Loss: {losses2[-1]:.6f}")

    return (net1, net2), (losses1, losses2)

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
        if (sn[0], sn[1]) == (sn[2], sn[3]): return CRASH_PENALTY, CRASH_PENALTY
        r1 = self.grid_reward[sn[0], sn[1]] - LIVING_COST
        r2 = self.grid_reward[sn[2], sn[3]] - LIVING_COST
        if (sn[0], sn[1]) == (x1, y1): r1 += STAY_PENALTY
        if (sn[2], sn[3]) == (x2, y2): r2 += STAY_PENALTY
        return r1, r2

def get_policy(nets, env):
    net1, net2 = nets
    policy = {}
    for s in env.states:
        s_t = encode_state(s, env.grid_size)
        with torch.no_grad():
            q1 = net1(s_t).view(4, 4)
            q2 = net2(s_t).view(4, 4)
        # Iterate to find mutual best response
        a1, a2 = 0, 0
        for _ in range(3):
            a1_new = q1[:, a2].argmax().item()
            a2_new = q2[a1, :].argmax().item()
            if a1_new == a1 and a2_new == a2:
                break
            a1, a2 = a1_new, a2_new
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
    parser = argparse.ArgumentParser(description="General-Sum Car Game (Independent Learners)")
    parser.add_argument("--iterations", type=int, default=8000, help="Number of planning iterations")
    parser.add_argument("--grid-size", type=int, default=5, help="Size of the grid (default: 5)")
    args = parser.parse_args()

    env = CarGame(grid_size=args.grid_size)

    # Neural planning
    nets, losses = neural_planning(env, iterations=args.iterations)
    net1, net2 = nets
    losses1, losses2 = losses
    policy = get_policy(nets, env)

    # Export weights for both players
    export_weights(net1, "weights_player1.txt", "Player 1 Q-network (independent learner)")
    export_weights(net2, "weights_player2.txt", "Player 2 Q-network (independent learner)")

    # Plot loss curves
    plt.figure()
    plt.plot(losses1, label="P1 Loss", alpha=0.7)
    plt.plot(losses2, label="P2 Loss", alpha=0.7)
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Independent Learners Training Loss")
    plt.legend()
    plt.savefig("planning_loss.png")
    plt.close()
    print("Saved planning_loss.png")

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

        draw_trajectory(ax, traj, args.grid_size)

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

