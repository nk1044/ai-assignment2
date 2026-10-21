"""Advantage actor-critic (README §7.3).

Synchronous A2C: ``n_envs`` simulators run in lock-step (the parallel actors of A3C) and one batched
update is made per iteration from their complete videos.

    actor : maximize  E[ log pi(a|s) * A ]  +  beta * H(pi(.|s))
    critic: minimize  (G - V(s))^2
with advantages A from GAE(gamma, lambda) and beta decayed linearly over training.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor, Critic


@dataclass
class A2CConfig:
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_start: float = 1.0
    entropy_end: float = 0.05
    entropy_decay_iters: int = 20_000
    reward_scale: float = 1.0           # rewards are divided by this before learning
    max_grad_norm: float = 5.0
    normalize_advantages: bool = True
    extra_dim: int = 0                  # 1 when the agent is conditioned on the data-saver lambda


class A2CAgent:
    def __init__(self, cfg=A2CConfig(), device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = Actor(cfg.extra_dim).to(self.device)
        self.critic = Critic(cfg.extra_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.iteration = 0

    # ---------------------------------------------------------------- acting
    def _tensors(self, states, extras):
        s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        e = None if extras is None else torch.as_tensor(np.asarray(extras), dtype=torch.float32,
                                                        device=self.device).reshape(len(s), -1)
        return s, e

    @torch.no_grad()
    def probs(self, states, extras=None):
        s, e = self._tensors(states, extras)
        return F.softmax(self.actor(s, e), dim=-1).cpu().numpy()

    def act(self, states, extras=None, greedy=False, rng=None):
        p = self.probs(states, extras)
        if greedy:
            return p.argmax(axis=1)
        rng = rng or np.random.default_rng()
        u = rng.random((len(p), 1))
        return np.minimum((p.cumsum(axis=1) < u).sum(axis=1), p.shape[1] - 1)

    # ---------------------------------------------------------------- learning
    def entropy_weight(self):
        frac = min(self.iteration / max(self.cfg.entropy_decay_iters, 1), 1.0)
        return self.cfg.entropy_start + frac * (self.cfg.entropy_end - self.cfg.entropy_start)

    @torch.no_grad()
    def advantages(self, states, extras, rewards, dones):
        """GAE over a (T, N) rollout made of complete episodes. Returns (advantages, returns), (T, N)."""
        T, N = rewards.shape
        s, e = self._tensors(states.reshape(T * N, *states.shape[2:]),
                             None if extras is None else extras.reshape(T * N, -1))
        values = self.critic(s, e).cpu().numpy().reshape(T, N)
        r = rewards / self.cfg.reward_scale
        adv = np.zeros((T, N), dtype=np.float32)
        last = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            not_done = 1.0 - dones[t]
            next_v = values[t + 1] if t + 1 < T else np.zeros(N, dtype=np.float32)
            delta = r[t] + self.cfg.gamma * next_v * not_done - values[t]
            last = delta + self.cfg.gamma * self.cfg.gae_lambda * not_done * last
            adv[t] = last
        return adv, adv + values

    def update(self, rollout):
        """One gradient step for actor and critic. ``rollout`` holds (T, N, ...) arrays."""
        states, actions, rewards, dones = (rollout[k] for k in ("states", "actions", "rewards", "dones"))
        extras = rollout.get("extras")
        adv, returns = self.advantages(states, extras, rewards, dones)

        T, N = rewards.shape
        s, e = self._tensors(states.reshape(T * N, *states.shape[2:]),
                             None if extras is None else extras.reshape(T * N, -1))
        a = torch.as_tensor(actions.reshape(-1), dtype=torch.long, device=self.device)
        adv_t = torch.as_tensor(adv.reshape(-1), device=self.device)
        ret_t = torch.as_tensor(returns.reshape(-1), device=self.device)
        if self.cfg.normalize_advantages:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        logits = self.actor(s, e)
        log_p = F.log_softmax(logits, dim=-1)
        entropy = -(log_p.exp() * log_p).sum(-1).mean()
        policy_loss = -(log_p.gather(1, a[:, None]).squeeze(1) * adv_t).mean()
        beta = self.entropy_weight()
        actor_loss = policy_loss - beta * entropy
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()

        critic_loss = F.mse_loss(self.critic(s, e), ret_t)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_opt.step()

        self.iteration += 1
        return {"policy_loss": policy_loss.item(), "critic_loss": critic_loss.item(),
                "entropy": entropy.item(), "entropy_weight": beta}

    # ---------------------------------------------------------------- checkpoints
    def save(self, path, **meta):
        torch.save({"config": asdict(self.cfg), "iteration": self.iteration,
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                    "meta": meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(A2CConfig(**ckpt["config"]), device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.actor_opt.load_state_dict(ckpt["actor_opt"])
        agent.critic_opt.load_state_dict(ckpt["critic_opt"])
        agent.iteration = ckpt["iteration"]
        agent.meta = ckpt.get("meta", {})
        return agent
