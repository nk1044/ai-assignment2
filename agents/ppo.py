"""PPO (proximal policy optimisation) with an asymmetric critic.

Per iteration: ``n_envs`` simulators stream one full video each with the current policy (the rollout,
48 x n_envs transitions). Then ``ppo_epochs`` passes of minibatch SGD over that rollout:

    ratio_t  = pi_new(a_t|s_t) / pi_old(a_t|s_t)
    actor  : maximize  E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]  +  beta * H(pi(.|s))
    critic : minimize  (G - V(s, future_bw))^2

Advantages A come from GAE(gamma, lambda) computed with the privileged critic V(s, future_bw), which sees
the true bandwidth of the next 32 s (training only). The actor only sees the normal observation s.
Clipping keeps each update close to the policy that collected the data, which makes it safe to reuse a
rollout for several epochs (more learning per simulated video than A2C's single step).
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from abrsim import config as C

from .networks import Actor, Critic


@dataclass
class PPOConfig:
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    minibatches: int = 4                # per epoch; 16 envs x 48 chunks = 768 -> 192 per minibatch
    target_kl: float = 0.03             # stop the epochs early if the policy moved further than this
    entropy_start: float = 0.1
    entropy_end: float = 0.01
    entropy_decay_iters: int = 6_000
    reward_scale: float = 1.0           # rewards are divided by this before learning
    max_grad_norm: float = 5.0
    normalize_advantages: bool = True
    extra_dim: int = 0                  # 1 when the agent is conditioned on the data-saver lambda
    priv_len: int = C.PRIV_WINDOWS      # privileged critic input length (future bandwidth windows)


class PPOAgent:
    def __init__(self, cfg=PPOConfig(), device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = Actor(cfg.extra_dim).to(self.device)
        self.critic = Critic(cfg.extra_dim, cfg.priv_len).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.iteration = 0

    # ---------------------------------------------------------------- acting (actor only)
    def _t(self, x, shape=None):
        if x is None:
            return None
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        return t if shape is None else t.reshape(shape)

    @torch.no_grad()
    def probs(self, states, extras=None):
        s = self._t(states)
        return F.softmax(self.actor(s, self._t(extras, (len(s), -1)) if extras is not None else None),
                         dim=-1).cpu().numpy()

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

    def _flat(self, rollout):
        """Flatten a (T, N, ...) rollout into (T*N, ...) tensors."""
        T, N = rollout["rewards"].shape
        s = self._t(rollout["states"], (T * N, C.S_INFO, C.S_LEN))
        e = self._t(rollout.get("extras"), (T * N, -1)) if rollout.get("extras") is not None else None
        p = self._t(rollout["privileged"], (T * N, -1))
        return s, e, p

    @torch.no_grad()
    def advantages(self, rollout):
        """GAE over a (T, N) rollout of complete episodes, using the privileged critic.
        Returns (advantages, returns), each (T, N)."""
        rewards, dones = rollout["rewards"], rollout["dones"]
        T, N = rewards.shape
        s, e, p = self._flat(rollout)
        values = self.critic(s, e, p).cpu().numpy().reshape(T, N)
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
        """PPO update: several epochs of clipped-surrogate minibatch steps on one rollout."""
        cfg = self.cfg
        adv, returns = self.advantages(rollout)
        s, e, p = self._flat(rollout)
        a = torch.as_tensor(rollout["actions"].reshape(-1), dtype=torch.long, device=self.device)
        adv_t = torch.as_tensor(adv.reshape(-1), device=self.device)
        ret_t = torch.as_tensor(returns.reshape(-1), device=self.device)
        if cfg.normalize_advantages:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        with torch.no_grad():                               # pi_old = the policy that collected the data
            old_logp = F.log_softmax(self.actor(s, e), dim=-1).gather(1, a[:, None]).squeeze(1)

        n = len(a)
        mb = max(n // cfg.minibatches, 1)
        beta = self.entropy_weight()
        stats = {"policy_loss": [], "critic_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
        epochs_done = 0
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            kls = []
            for start in range(0, n, mb):
                i = perm[start:start + mb]
                ei = e[i] if e is not None else None
                log_p = F.log_softmax(self.actor(s[i], ei), dim=-1)
                logp_a = log_p.gather(1, a[i, None]).squeeze(1)
                ratio = torch.exp(logp_a - old_logp[i])
                surr = torch.min(ratio * adv_t[i], torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t[i])
                entropy = -(log_p.exp() * log_p).sum(-1).mean()
                policy_loss = -surr.mean()
                self.actor_opt.zero_grad()
                (policy_loss - beta * entropy).backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()

                critic_loss = F.mse_loss(self.critic(s[i], ei, p[i]), ret_t[i])
                self.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_opt.step()

                with torch.no_grad():
                    kl = (old_logp[i] - logp_a).mean().item()
                kls.append(kl)
                stats["policy_loss"].append(policy_loss.item())
                stats["critic_loss"].append(critic_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(kl)
                stats["clip_frac"].append(((ratio - 1).abs() > cfg.clip_eps).float().mean().item())
            epochs_done += 1
            if cfg.target_kl and np.mean(kls) > cfg.target_kl:
                break

        self.iteration += 1
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        return out | {"entropy_weight": beta, "epochs": epochs_done}

    # ---------------------------------------------------------------- checkpoints
    def save(self, path, **meta):
        torch.save({"config": asdict(self.cfg), "iteration": self.iteration,
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                    "meta": meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(PPOConfig(**ckpt["config"]), device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.actor_opt.load_state_dict(ckpt["actor_opt"])
        agent.critic_opt.load_state_dict(ckpt["critic_opt"])
        agent.iteration = ckpt["iteration"]
        agent.meta = ckpt.get("meta", {})
        return agent
