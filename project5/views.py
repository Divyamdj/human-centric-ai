import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import torch.nn.functional as F
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from django.utils.html import escape
from . import mouse


# Global state
TRAINER = None
GRID = None

# Define file paths for models
base_dir = os.path.dirname(__file__)
feedback_file = os.path.join(base_dir, "feedback_log.json")
policy_base_path = os.path.join(base_dir, "policy_base.pth")
reward_model_path = os.path.join(base_dir, "reward_model.pth")


def initialize_global_state():
    """Initialize global variables if they don't exist."""
    global TRAINER, GRID
    if TRAINER is None:
        TRAINER = ReinforceTrainer()
        # Load the base policy if it exists, otherwise train one
        if os.path.exists(policy_base_path):
            TRAINER.policy.load_state_dict(torch.load(policy_base_path))
    if GRID is None:
        GRID, _, _, _ = mouse.initialize_grid_with_cheese_types()


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * mouse.GRID_SIZE * mouse.GRID_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.net(x)


class ReinforceTrainer:
    def __init__(self, gamma=0.99, lr=3e-3, device="cpu"):
        self.policy = PolicyNet().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device

    @staticmethod
    def obs_from_grid(grid):
        planes = np.zeros(
            (6, mouse.GRID_SIZE, mouse.GRID_SIZE), dtype=np.float32)
        for v in (mouse.EMPTY, mouse.MOUSE, mouse.CHEESE,
                  mouse.TRAP, mouse.WALL, mouse.ORGANIC_CHEESE):
            planes[v] = (grid == v)
        return planes

    def select_action(self, obs):
        with torch.no_grad():
            logits = self.policy(torch.from_numpy(
                obs).unsqueeze(0).to(self.device))
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            return action

    def generate_episode(self, max_steps=50):
        grid, _, _, _ = mouse.initialize_grid_with_cheese_types()
        obs = self.obs_from_grid(grid)
        logps, rewards, steps_log, observations = [], [], [], []
        done, steps = False, 0
        while not done and steps < max_steps:
            observations.append(obs)
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            logits = self.policy(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            logp = dist.log_prob(action)
            logps.append(logp)

            action_str = mouse.ACTIONS[action.item()]
            new_grid, reward = mouse.move(action_str, grid)
            rewards.append(reward)

            explanation = ("Collected cheese!" if reward == 10 else
                           "Fell in trap!" if reward == -50 else
                           "Moved to empty cell / bump.")

            steps_log.append(
                {
                    "grid": render_board(new_grid),
                    "action": action_str,
                    "reward": reward,
                    "explanation": explanation,
                }
            )

            grid = new_grid
            obs = self.obs_from_grid(grid)
            done = reward == 10 or reward == -50
            steps += 1
        # Return observations for fine-tuning
        return logps, rewards, steps_log, observations

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        t = torch.tensor(returns, dtype=torch.float32)
        if t.std() > 1e-6:
            t = (t - t.mean()) / (t.std() + 1e-8)
        return t

    def train(self, epochs=20, episodes_per_epoch=20, max_steps=50):
        history = []
        for ep in range(epochs):
            batch_loss, batch_return = 0.0, 0.0
            self.optimizer.zero_grad()
            total_loss = torch.tensor(0.0).to(self.device)

            for _ in range(episodes_per_epoch):

                logps, rewards, _, _ = self.generate_episode(
                    max_steps=max_steps)
                returns = self.compute_returns(rewards).to(self.device)
                logps_t = torch.stack(logps).to(self.device)

                loss = -(logps_t * returns).sum()
                total_loss += loss
                batch_return += sum(rewards)

            total_loss /= episodes_per_epoch
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            batch_loss = total_loss.item()
            history.append({
                "epoch": ep + 1,
                "loss": batch_loss,
                "avg_return": batch_return / episodes_per_epoch
            })

        torch.save(self.policy.state_dict(), policy_base_path)

        return {"history": history}

    def fine_tune(self, ref_policy, reward_model, epochs=20, episodes_per_epoch=10, kl_beta=0.1):
        history = []
        ref_policy.to(self.device)
        reward_model.to(self.device)

        for ep in range(epochs):
            self.optimizer.zero_grad()
            total_policy_loss = 0.0
            total_kl_div = 0.0
            avg_learned_reward = 0.0

            for _ in range(episodes_per_epoch):
                logps, _, steps_log, observations = self.generate_episode()

                # 1. Compute learned reward for the trajectory
                features = trajectory_features(steps_log)
                features_tensor = torch.tensor(
                    list(features.values()), dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    learned_reward = reward_model(features_tensor).item()

                avg_learned_reward += learned_reward

                # 2. Compute the policy loss using the learned reward

                policy_loss = -torch.stack(logps).sum() * learned_reward
                total_policy_loss += policy_loss

                # 3. Compute KL-divergence penalty
                obs_tensor = torch.from_numpy(
                    np.array(observations)).to(self.device)
                with torch.no_grad():
                    ref_logits = ref_policy(obs_tensor)

                current_logits = self.policy(obs_tensor)

                kl_div = F.kl_div(
                    F.log_softmax(current_logits, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                    reduction='batchmean'
                )
                total_kl_div += kl_div

            # Combine policy loss and KL penalty
            total_policy_loss /= episodes_per_epoch
            total_kl_div /= episodes_per_epoch

            loss = total_policy_loss + kl_beta * total_kl_div

            loss.backward()
            self.optimizer.step()

            history.append({
                "epoch": ep + 1,
                "loss": loss.item(),
                "policy_loss": total_policy_loss.item(),
                "kl_div": total_kl_div.item(),
                "avg_learned_reward": avg_learned_reward / episodes_per_epoch,
            })

        return {"history": history}


def render_board(grid):
    symbols = {
        mouse.EMPTY: '▫️', mouse.MOUSE: '🐭', mouse.CHEESE: '🧀',
        mouse.TRAP: '💀', mouse.WALL: '🧱', mouse.ORGANIC_CHEESE: '🥦'
    }
    return [[symbols[c] for c in row] for row in grid]


@csrf_protect
@require_http_methods(["GET"])
def dashboard(request):
    initialize_global_state()

    can_finetune = os.path.exists(
        policy_base_path) and os.path.exists(reward_model_path)
    return render(request, "dashboard.html", {
        "board_lines": render_board(GRID),
        "can_finetune": can_finetune
    })


@csrf_protect
@require_http_methods(["POST"])
def sample_policy(request):
    initialize_global_state()
    global GRID
    obs = TRAINER.obs_from_grid(GRID)
    action = TRAINER.select_action(obs)
    action_str = mouse.ACTIONS[action]
    GRID, reward = mouse.move(action_str, GRID)
    return render(request, "dashboard.html", {
        "board_lines": render_board(GRID),
        "last_reward": reward,
        "last_info": escape(f"Action={action_str}")
    })


@csrf_protect
@require_http_methods(["POST"])
def train(request):
    initialize_global_state()
    epochs = int(request.POST.get("epochs", "10"))
    episodes = int(request.POST.get("episodes_per_epoch", "16"))
    result = TRAINER.train(epochs=epochs, episodes_per_epoch=episodes)
    return render(request, "training_summary.html", {"history": result["history"]})
    return render(request, "training_summary.html", {"history": result["history"]})


@csrf_protect
@require_http_methods(["POST"])
def reset_env(request):
    initialize_global_state()
    global GRID
    GRID, _, _, _ = mouse.initialize_grid_with_cheese_types()
    return redirect("project5:index")


@csrf_protect
@require_http_methods(["POST"])
def run_episode(request):
    initialize_global_state()

    _, _, steps_log, _ = TRAINER.generate_episode(max_steps=30)
    return render(request, "episode_run.html", {"steps_log": steps_log})


# --- Trajectory Generator & Bradley-Terry ---
def trajectory_features(traj_log):
    """Extract features from a trajectory log."""

    total_reward = sum(step["reward"] for step in traj_log)

    # Check if cheese was collected in the final step
    final_step = traj_log[-1] if traj_log else {}
    final_grid_flat = sum(final_step.get("grid", []), [])

    normal_cheese = 1 if '🧀' in final_grid_flat and final_step.get(
        "reward") == 10 else 0
    organic_cheese = 1 if '🥦' in final_grid_flat and final_step.get(
        "reward") == 10 else 0

    return {
        "total_reward": total_reward,
        "normal_cheese": normal_cheese,
        "organic_cheese": organic_cheese,
        "steps": len(traj_log),
    }


@csrf_protect
@require_http_methods(["GET", "POST"])
def compare_trajectories(request):
    initialize_global_state()
    if request.method == "POST":
        choice = request.POST.get("choice")
        traj1 = request.session.get("traj1")
        traj2 = request.session.get("traj2")

        if traj1 and traj2 and choice:
            record = {
                "traj1": trajectory_features(traj1),
                "traj2": trajectory_features(traj2),
                "choice": choice,
            }

            data = []
            if os.path.exists(feedback_file):
                with open(feedback_file, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
            data.append(record)
            with open(feedback_file, "w") as f:
                json.dump(data, f, indent=2)

        request.session.pop("traj1", None)
        request.session.pop("traj2", None)
        return render(request, "compare.html", {"feedback_saved": True})

    traj1 = TRAINER.generate_episode(max_steps=30)[2]
    traj2 = TRAINER.generate_episode(max_steps=30)[2]

    request.session["traj1"] = traj1
    request.session["traj2"] = traj2

    return render(request, "compare.html", {"traj1": traj1, "traj2": traj2})


class RewardModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(
            ), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


@require_http_methods(["GET", "POST"])
def train_reward_model(request):
    if not os.path.exists(feedback_file):
        return render(request, "reward_training.html", {"message": "No feedback data yet."})

    with open(feedback_file, "r") as f:
        data = json.load(f)

    if not data:
        return render(request, "reward_training.html", {"message": "Feedback log is empty."})

    if request.method == "POST":
        X1, X2, y = [], [], []
        for record in data:
            f1 = list(record["traj1"].values())
            f2 = list(record["traj2"].values())
            X1.append(f1)
            X2.append(f2)
            y.append(1 if record["choice"] == "traj1" else 0)

        X1 = torch.tensor(X1, dtype=torch.float32)
        X2 = torch.tensor(X2, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        model = RewardModel(input_dim=4)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        epochs = 200

        for _ in range(epochs):
            r1 = model(X1)
            r2 = model(X2)
            logits = r1 - r2
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save the trained reward model
        torch.save(model.state_dict(), reward_model_path)

        weights = {f"w{i}": p.item()
                   for i, p in enumerate(model.net[2].weight.squeeze())}
        bias = model.net[2].bias.detach().item()

        return render(request, "reward_training.html", {
            "message": f"Successfully trained and saved reward model on {len(data)} preferences.",
            "weights": weights,
            "bias": bias,
            "data_count": len(data)
        })
    return render(request, "reward_training.html", {
        "message": f"Ready to train reward model on {len(data)} preferences.",
        "data_count": len(data)
    })


@csrf_protect
@require_http_methods(["GET", "POST"])
def fine_tune_policy(request):
    initialize_global_state()
    if not os.path.exists(policy_base_path) or not os.path.exists(reward_model_path):

        return redirect("project5:index")

    if request.method == "POST":
        # Load the reference policy
        ref_policy = PolicyNet()
        ref_policy.load_state_dict(torch.load(policy_base_path))
        ref_policy.eval()

        # Load the reward model
        reward_model = RewardModel(input_dim=4)
        reward_model.load_state_dict(torch.load(reward_model_path))
        reward_model.eval()

        epochs = int(request.POST.get("epochs", "10"))
        kl_beta = float(request.POST.get("kl_beta", "0.1"))

        result = TRAINER.fine_tune(
            ref_policy, reward_model, epochs=epochs, kl_beta=kl_beta)

        return render(request, "finetuning_summary.html", {"history": result["history"]})

    return render(request, 'finetune.html')
