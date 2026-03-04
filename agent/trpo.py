"""
TRPO — Trust Region Policy Optimization (practical implementation, one update per episode).

В этой версии мы НЕ используем valid_actions_mask, потому что в вашей среде
valid_actions всегда содержит все действия.

Совместимость:
- select_action(state, valid_actions, training) — принимает valid_actions, но игнорирует
- store_transition(...) — как у BaseAgent
- update() — раз в эпизод, как ожидает trainer.py

CG: scipy.sparse.linalg.cg + LinearOperator (matrix-free).
Value: табличная оценка V(s) по истории последних history_len траекторий.
"""

from __future__ import annotations

from typing import Dict, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .base_agent import BaseAgent
from .policy import PolicyNetwork
from env.actions import index_to_action
from env.state import observation_to_state

try:
    from scipy.sparse.linalg import cg as scipy_cg
    from scipy.sparse.linalg import LinearOperator
except Exception as e:  # pragma: no cover
    scipy_cg = None
    LinearOperator = None
    _SCIPY_IMPORT_ERROR = e
else:
    _SCIPY_IMPORT_ERROR = None


class TRPOAgent(BaseAgent):
    """
    TRPO Agent.

    Config keys (defaults):
      - discount_factor: 0.99
      - hidden_dims: [64, 64]
      - max_kl: 0.01
      - cg_iters: 10
      - cg_tol: 1e-10
      - backtrack_iters: 10
      - backtrack_coef: 0.5
      - damping: 1e-2
      - advantage_norm: True
      - max_grad_norm: 10.0      (клиппинг нормы градиента перед CG)
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict, baseline):
        super().__init__(observation_dim, action_space, config)

        if scipy_cg is None:
            raise ImportError(
                "SciPy is required for TRPOAgent (requested scipy.sparse.linalg.cg). "
                "Install: pip install scipy"
            ) from _SCIPY_IMPORT_ERROR

        self.action_space = action_space
        self.baseline = baseline
        action_dim = len(action_space)

        hidden_dims = config.get("hidden_dims", [64, 64])
        self.policy_network = PolicyNetwork(observation_dim, action_dim, hidden_dims)

        self.gamma = float(config.get("discount_factor", 0.99))

        self.max_kl = float(config.get("max_kl", config.get("TRPO_MAX_KL", 0.01)))
        self.cg_iters = int(config.get("cg_iters", config.get("TRPO_CG_ITERS", 10)))
        self.cg_tol = float(config.get("cg_tol", 1e-10))
        self.backtrack_iters = int(config.get("backtrack_iters", config.get("TRPO_BACKTRACK_ITERS", 10)))
        self.backtrack_coef = float(config.get("backtrack_coef", config.get("TRPO_BACKTRACK_COEF", 0.5)))
        self.damping = float(config.get("damping", 1e-2))
        self.advantage_norm = bool(config.get("advantage_norm", True))
        self.max_grad_norm = float(config.get("max_grad_norm", 10.0))

    # ---------------- acting ----------------

    def select_action(self, state, valid_actions: list, training: bool = True):
        """
        valid_actions принимаем для совместимости с trainer.py, но игнорируем,
        т.к. в вашей среде всегда доступны все действия.
        """
        device = next(self.policy_network.parameters()).device

        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            logits = self.policy_network.forward(state_t, None)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)

            if training:
                action_idx = int(dist.sample().item())
            else:
                action_idx = int(torch.argmax(probs, dim=-1).item())

            action = index_to_action(action_idx, self.action_space)
            log_prob = float(self.policy_network.get_log_probs(state_t, [action_idx], None).item())

        return action, log_prob

    # ---------------- utilities ----------------

    @staticmethod
    def _flat_params(model: torch.nn.Module) -> torch.Tensor:
        return parameters_to_vector(model.parameters()).detach()

    @staticmethod
    def _set_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
        vector_to_parameters(flat, model.parameters())

    def _build_batch_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.policy_network.parameters()).device

        states = torch.as_tensor(np.asarray(self.saved_states), dtype=torch.float32, device=device)
        actions_idx = torch.as_tensor(
            [self.action_space.index(a) for a in self.saved_actions],
            dtype=torch.long,
            device=device,
        )
        old_logp = torch.as_tensor(np.asarray(self.saved_log_probs), dtype=torch.float32, device=device)

        return states, actions_idx, old_logp

    def _compute_mc_returns(self) -> torch.Tensor:
        device = next(self.policy_network.parameters()).device
        rewards = self.saved_rewards
        T = len(rewards)

        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = float(rewards[t]) + self.gamma * G
            returns[t] = G

        return torch.as_tensor(returns, dtype=torch.float32, device=device)

    def _compute_advantages(
        self,
        states: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Advantages = G_t - b(s_t) через self.baseline.predict().
        Возвращает (advantages_tensor, value_mse) для логирования.
        """
        states_list = list(states.detach().cpu().numpy())
        baseline_values = self.baseline.predict(states_list).astype(np.float32)

        returns_np = returns.detach().cpu().numpy()
        value_mse = float(np.mean((baseline_values - returns_np) ** 2))

        baseline_t = torch.as_tensor(baseline_values, dtype=torch.float32, device=states.device)
        adv = returns - baseline_t
        if self.advantage_norm and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return adv, value_mse

    def _surrogate_and_kl(
        self,
        states: torch.Tensor,
        actions_idx: torch.Tensor,
        old_logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_logp = self.policy_network.get_log_probs(states, actions_idx, None)
        ratio = torch.exp(new_logp - old_logp)
        surrogate = (ratio * adv).mean()
        kl = (old_logp - new_logp).mean()
        return surrogate, kl

    def _fisher_vector_product(self, kl: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        params = tuple(self.policy_network.parameters())
        grads = torch.autograd.grad(kl, params, create_graph=True, retain_graph=True)
        flat_grads = torch.cat([g.reshape(-1) for g in grads])

        g_v = torch.dot(flat_grads, v)
        hv = torch.autograd.grad(g_v, params, retain_graph=True)
        flat_hv = torch.cat([h.reshape(-1) for h in hv]).detach()

        # Если Hessian содержит NaN (вырожденная политика), возвращаем чистый damping
        if not torch.isfinite(flat_hv).all():
            return self.damping * v

        return flat_hv + self.damping * v

    def _scipy_cg_solve(self, fvp: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor) -> torch.Tensor:
        device = b.device
        b_np = b.detach().cpu().double().numpy()
        n = b_np.shape[0]

        def matvec(v_np: np.ndarray) -> np.ndarray:
            v = torch.from_numpy(v_np).to(device=device, dtype=torch.float32)
            out = fvp(v)
            return out.detach().cpu().double().numpy()

        A = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        x_np, _info = scipy_cg(A, b_np, maxiter=self.cg_iters, atol=0.0, rtol=self.cg_tol)
        return torch.from_numpy(x_np).to(device=device, dtype=torch.float32)

    def _line_search(
        self,
        old_params: torch.Tensor,
        full_step: torch.Tensor,
        old_surrogate: float,
        states: torch.Tensor,
        actions_idx: torch.Tensor,
        old_logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        metrics: Dict[str, float] = {"accepted": 0.0, "alpha": 0.0, "new_kl": 0.0, "new_surrogate": old_surrogate}

        for i in range(self.backtrack_iters):
            alpha = self.backtrack_coef ** i
            new_params = old_params + alpha * full_step
            self._set_flat_params(self.policy_network, new_params)

            with torch.no_grad():
                surr, kl = self._surrogate_and_kl(states, actions_idx, old_logp, adv)

            surr_v = float(surr.item())
            kl_v = float(kl.item())

            if (surr_v > old_surrogate) and (kl_v <= self.max_kl):
                metrics.update({"accepted": 1.0, "alpha": float(alpha), "new_kl": kl_v, "new_surrogate": surr_v})
                return new_params, metrics

        self._set_flat_params(self.policy_network, old_params)
        return old_params, metrics

    # ---------------- training ----------------

    def update(self) -> Dict[str, float]:
        states, actions_idx, old_logp = self._build_batch_tensors()
        returns = self._compute_mc_returns()

        # 1. Добавляем текущий эпизод в историю
        returns_np = returns.detach().cpu().numpy()
        episode_traj = [
            (observation_to_state(self.saved_states[t]), float(returns_np[t]))
            for t in range(len(self.saved_states))
        ]
        #self.baseline.add_trajectory(episode_traj)

        # 2. Advantages = G_t - b(s_t)
        adv, value_loss = self._compute_advantages(states, returns)
        self.baseline.add_trajectory(episode_traj)

        old_params = self._flat_params(self.policy_network)

        with torch.no_grad():
            surr_old, kl_old = self._surrogate_and_kl(states, actions_idx, old_logp, adv)
        old_surrogate = float(surr_old.item())
        old_kl = float(kl_old.item())

        # g = ∇ surrogate
        self.policy_network.zero_grad(set_to_none=True)
        surrogate_for_grad, kl_for_fvp = self._surrogate_and_kl(states, actions_idx, old_logp, adv)
        grads = torch.autograd.grad(surrogate_for_grad, tuple(self.policy_network.parameters()), retain_graph=True)
        flat_g = torch.cat([g.reshape(-1) for g in grads]).detach()

        # Клиппинг нормы градиента — защита от NaN/inf в CG
        if not torch.isfinite(flat_g).all():
            self._set_flat_params(self.policy_network, old_params)
            self.reset_trajectory()
            return {
                "policy_loss": old_surrogate,
                "policy_surrogate_old": old_surrogate,
                "policy_kl_old": old_kl,
                "policy_update_accepted": 0.0,
                "policy_alpha": 0.0,
                "policy_surrogate_new": old_surrogate,
                "policy_kl_new": old_kl,
                "value_loss": value_loss,
                "cg_shs": 0.0,
            }
        grad_norm = flat_g.norm()
        if grad_norm > self.max_grad_norm:
            flat_g = flat_g * (self.max_grad_norm / grad_norm)

        def fvp(v: torch.Tensor) -> torch.Tensor:
            return self._fisher_vector_product(kl_for_fvp, v)

        step_dir = self._scipy_cg_solve(fvp, flat_g)

        # Если CG вернул NaN (вырожденная Fisher-матрица) — пропускаем обновление
        if not torch.isfinite(step_dir).all():
            self._set_flat_params(self.policy_network, old_params)
            self.reset_trajectory()
            return {
                "policy_loss": old_surrogate,
                "policy_surrogate_old": old_surrogate,
                "policy_kl_old": old_kl,
                "policy_update_accepted": 0.0,
                "policy_alpha": 0.0,
                "policy_surrogate_new": old_surrogate,
                "policy_kl_new": old_kl,
                "value_loss": value_loss,
                "cg_shs": 0.0,
            }

        F_step = fvp(step_dir)
        shs = torch.dot(step_dir, F_step)
        shs_val = float(shs.item())

        if (not np.isfinite(shs_val)) or (shs_val <= 0.0):
            self._set_flat_params(self.policy_network, old_params)
            self.reset_trajectory()
            return {
                "policy_loss": old_surrogate,
                "policy_surrogate_old": old_surrogate,
                "policy_kl_old": old_kl,
                "policy_update_accepted": 0.0,
                "policy_alpha": 0.0,
                "policy_surrogate_new": old_surrogate,
                "policy_kl_new": old_kl,
                "value_loss": value_loss,
                "cg_shs": shs_val,
            }

        max_step = np.sqrt((2.0 * self.max_kl) / (shs_val + 1e-12))
        full_step = step_dir * float(max_step)

        _, ls = self._line_search(
            old_params=old_params,
            full_step=full_step,
            old_surrogate=old_surrogate,
            states=states,
            actions_idx=actions_idx,
            old_logp=old_logp,
            adv=adv,
        )

        with torch.no_grad():
            surr_new, kl_new = self._surrogate_and_kl(states, actions_idx, old_logp, adv)

        metrics = {
            "policy_loss": float(surr_new.item()),
            "policy_surrogate_old": old_surrogate,
            "policy_kl_old": old_kl,
            "policy_update_accepted": float(ls["accepted"]),
            "policy_alpha": float(ls["alpha"]),
            "policy_surrogate_new": float(surr_new.item()),
            "policy_kl_new": float(kl_new.item()),
            "value_loss": value_loss,
            "cg_shs": shs_val,
        }

        self.reset_trajectory()
        return metrics
