"""
TRPO — Trust Region Policy Optimization.
Наследуется от REINFORCEBaselineAgent: тот же табличный V(s) по истории траекторий,
advantages A_t = G_t - V(s_t) без нормализации. Обновление политики: natural gradient
(CG) + line search по KL.
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple, Callable

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .reinforce import _valid_actions_mask
from .reinforce_baseline import REINFORCEBaselineAgent
from env.actions import action_to_index, index_to_action
from env.state import observation_to_state

try:
    from scipy.sparse.linalg import cg as scipy_cg
    from scipy.sparse.linalg import LinearOperator
except Exception as e:
    scipy_cg = None
    LinearOperator = None
    _SCIPY_IMPORT_ERROR = e
else:
    _SCIPY_IMPORT_ERROR = None


class TRPOAgent(REINFORCEBaselineAgent):
    """
    TRPO: суррогат L = E[ratio * A], ограничение KL(π_old || π_new) ≤ δ.
    Value и advantages — как в REINFORCE baseline (табличный V(s), A_t = G_t - V(s_t)).
    Шаг: (F + λI)x = g через CG, затем line search по α.
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict):
        super().__init__(observation_dim, action_space, config)

        if scipy_cg is None:
            raise ImportError(
                "SciPy required for TRPO (scipy.sparse.linalg.cg). pip install scipy"
            ) from _SCIPY_IMPORT_ERROR

        self.max_kl = float(config.get("max_kl", config.get("TRPO_MAX_KL", 0.01)))
        self.cg_iters = int(config.get("cg_iters", config.get("TRPO_CG_ITERS", 10)))
        self.cg_tol = float(config.get("cg_tol", 1e-10))
        self.backtrack_iters = int(config.get("backtrack_iters", config.get("TRPO_BACKTRACK_ITERS", 10)))
        self.backtrack_coef = float(config.get("backtrack_coef", config.get("TRPO_BACKTRACK_COEF", 0.5)))
        self.damping = float(config.get("damping", 1e-2))
        # Ограничение advantages: иначе ratio*A взрывается и политика коллапсирует
        self.max_abs_advantage = float(config.get("max_abs_advantage", config.get("TRPO_MAX_ABS_ADVANTAGE", 10.0)))
        # Ограничение нормы градиента до CG (чтобы шаг не улетал)
        self.max_grad_norm_cg = float(config.get("max_grad_norm_cg", config.get("TRPO_MAX_GRAD_NORM_CG", 50.0)))

    def _build_batch_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """states [T, obs], actions_idx [T], old_logp [T], valid_masks [T, A]."""
        device = next(self.policy_network.parameters()).device
        states = torch.as_tensor(np.asarray(self.saved_states), dtype=torch.float32, device=device)
        actions_idx = torch.tensor(
            [action_to_index(a, self.action_space) for a in self.saved_actions],
            dtype=torch.long,
            device=device,
        )
        old_logp = torch.as_tensor(np.asarray(self.saved_log_probs), dtype=torch.float32, device=device)
        masks = []
        for t in range(len(self.saved_states)):
            va = self.saved_valid_actions[t] if t < len(self.saved_valid_actions) else self.action_space
            m = _valid_actions_mask(va, self.action_space, device)
            masks.append(m)
        valid_masks = torch.stack(masks)
        return states, actions_idx, old_logp, valid_masks

    def _get_advantages_tensor(self, returns: list, V: np.ndarray) -> torch.Tensor:
        """A_t = G_t - V(s_t), без нормализации."""
        baseline = self._predict_baseline(self.saved_states, V)
        returns_np = np.array(returns, dtype=np.float64)
        adv = returns_np - baseline
        device = next(self.policy_network.parameters()).device
        return torch.as_tensor(adv, dtype=torch.float32, device=device)

    def _surrogate_and_kl(
        self,
        states: torch.Tensor,
        actions_idx: torch.Tensor,
        old_logp: torch.Tensor,
        adv: torch.Tensor,
        valid_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """L = mean(ratio * A), KL ≈ mean(old_logp - new_logp). ratio и A ограничены для стабильности."""
        new_logp = self.policy_network.get_log_probs(states, actions_idx, valid_masks)
        log_ratio = new_logp - old_logp
        # Жёсткое ограничение ratio: иначе при больших A суррогат и градиент взрываются
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
        ratio = torch.exp(log_ratio)
        surr = (ratio * adv).mean()
        kl = (old_logp - new_logp).mean()
        return surr, kl

    def _fisher_vector_product(self, kl: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        params = list(self.policy_network.parameters())
        grads = torch.autograd.grad(kl, params, create_graph=True, retain_graph=True)
        flat_g = torch.cat([g.reshape(-1) for g in grads])
        gv = torch.dot(flat_g, v)
        hv = torch.autograd.grad(gv, params, retain_graph=True)
        flat_hv = torch.cat([h.reshape(-1) for h in hv]).detach()
        if not torch.isfinite(flat_hv).all():
            return self.damping * v
        return flat_hv + self.damping * v

    def _scipy_cg_solve(
        self, fvp: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor
    ) -> torch.Tensor:
        device = b.device
        b_np = b.detach().cpu().double().numpy()
        n = b_np.shape[0]
        zero = torch.zeros_like(b)

        def matvec(v_np: np.ndarray) -> np.ndarray:
            v = torch.from_numpy(v_np).to(device=device, dtype=torch.float32)
            out = fvp(v)
            return out.detach().cpu().double().numpy()

        try:
            # atol=1e-8 вместо 0: избегаем бесконечной точности и численных сбоев CG
            atol = max(1e-8, float(self.cg_tol))
            A = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                x_np, _ = scipy_cg(A, b_np, maxiter=self.cg_iters, atol=atol, rtol=max(1e-10, float(self.cg_tol)))
            if not np.isfinite(x_np).all():
                return zero
            return torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
        except Exception:
            return zero

    @staticmethod
    def _flat_params(model: torch.nn.Module) -> torch.Tensor:
        return parameters_to_vector(model.parameters()).detach()

    @staticmethod
    def _set_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
        vector_to_parameters(flat, model.parameters())

    def _line_search(
        self,
        old_params: torch.Tensor,
        full_step: torch.Tensor,
        old_surrogate: float,
        states: torch.Tensor,
        actions_idx: torch.Tensor,
        old_logp: torch.Tensor,
        adv: torch.Tensor,
        valid_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = {"accepted": 0.0, "alpha": 0.0, "new_kl": 0.0, "new_surrogate": old_surrogate}
        for i in range(self.backtrack_iters):
            alpha = self.backtrack_coef ** i
            new_params = old_params + alpha * full_step
            self._set_flat_params(self.policy_network, new_params)
            with torch.no_grad():
                surr, kl = self._surrogate_and_kl(states, actions_idx, old_logp, adv, valid_masks)
            surr_v = float(surr.item())
            kl_v = float(kl.item())
            if surr_v > old_surrogate and kl_v <= self.max_kl:
                out = {"accepted": 1.0, "alpha": float(alpha), "new_kl": kl_v, "new_surrogate": surr_v}
                return new_params, out
        self._set_flat_params(self.policy_network, old_params)
        return old_params, out

    def _mean_entropy(self, states: torch.Tensor, valid_masks: torch.Tensor) -> torch.Tensor:
        return self.policy_network.get_entropy(states, valid_masks).mean()

    def update(self) -> dict:
        if not self.saved_states:
            return {"policy_loss": 0.0, "baseline_mse": 0.0, "entropy": 0.0, "policy_update_accepted": 0.0}

        device = next(self.policy_network.parameters()).device
        returns = self._compute_returns()

        # 1. Текущий эпизод в историю (как в baseline)
        episode_traj = [
            (observation_to_state(self.saved_states[t]), float(returns[t]))
            for t in range(len(self.saved_states))
        ]
        self.trajectory_history.append(episode_traj)

        # 2. V(s) по истории, advantages без нормализации
        V = self._estimate_value_table()
        baseline_values = self._predict_baseline(self.saved_states, V)
        returns_np = np.array(returns, dtype=np.float64)
        baseline_mse = float(np.mean((baseline_values - returns_np) ** 2))
        adv = self._get_advantages_tensor(returns, V)
        # Клиппинг advantages: иначе ratio*A даёт взрыв градиента и коллапс энтропии
        adv = torch.clamp(adv, -self.max_abs_advantage, self.max_abs_advantage)

        states, actions_idx, old_logp, valid_masks = self._build_batch_tensors()
        old_params = self._flat_params(self.policy_network)

        with torch.no_grad():
            surr_old, kl_old = self._surrogate_and_kl(states, actions_idx, old_logp, adv, valid_masks)
        old_surrogate = float(surr_old.item())
        old_kl = float(kl_old.item())
        mean_entropy = float(self._mean_entropy(states, valid_masks).item())
        if not np.isfinite(mean_entropy):
            mean_entropy = 0.0

        # 3. g = ∇ surrogate
        self.policy_network.zero_grad(set_to_none=True)
        surr_for_grad, kl_for_fvp = self._surrogate_and_kl(states, actions_idx, old_logp, adv, valid_masks)
        grads = torch.autograd.grad(surr_for_grad, list(self.policy_network.parameters()), retain_graph=True)
        flat_g = torch.cat([g.reshape(-1) for g in grads]).detach()

        if not torch.isfinite(flat_g).all():
            self._set_flat_params(self.policy_network, old_params)
            self.reset_trajectory()
            return {
                "policy_loss": -old_surrogate,
                "baseline_mse": baseline_mse,
                "value_loss": baseline_mse,
                "entropy": mean_entropy,
                "policy_update_accepted": 0.0,
                "policy_alpha": 0.0,
                "cg_shs": 0.0,
            }

        # Нулевой градиент — не вызываем CG, шаг нулевой
        grad_norm = flat_g.norm().item()
        if grad_norm < 1e-12:
            self._set_flat_params(self.policy_network, old_params)
            self.reset_trajectory()
            return {
                "policy_loss": -old_surrogate,
                "baseline_mse": baseline_mse,
                "value_loss": baseline_mse,
                "entropy": mean_entropy,
                "policy_update_accepted": 0.0,
                "policy_alpha": 0.0,
                "cg_shs": 0.0,
            }
        # Ограничение нормы g до CG: слишком большой g даёт огромный шаг и нестабильность
        if grad_norm > self.max_grad_norm_cg:
            flat_g = flat_g * (self.max_grad_norm_cg / grad_norm)

        def fvp(v: torch.Tensor) -> torch.Tensor:
            return self._fisher_vector_product(kl_for_fvp, v)

        step_dir = self._scipy_cg_solve(fvp, flat_g)
        if not torch.isfinite(step_dir).all():
            self._set_flat_params(self.policy_network, old_params)
            self.reset_trajectory()
            return {
                "policy_loss": -old_surrogate,
                "baseline_mse": baseline_mse,
                "value_loss": baseline_mse,
                "entropy": mean_entropy,
                "policy_update_accepted": 0.0,
                "policy_alpha": 0.0,
                "cg_shs": 0.0,
            }

        F_step = fvp(step_dir)
        shs = torch.dot(step_dir, F_step)
        shs_val = float(shs.item())

        if not np.isfinite(shs_val) or shs_val <= 0.0:
            self._set_flat_params(self.policy_network, old_params)
            self.reset_trajectory()
            return {
                "policy_loss": -old_surrogate,
                "baseline_mse": baseline_mse,
                "value_loss": baseline_mse,
                "entropy": mean_entropy,
                "policy_update_accepted": 0.0,
                "policy_alpha": 0.0,
                "cg_shs": shs_val,
            }

        # Нижняя граница знаменателя: не делить на слишком маленькое (взрыв шага)
        shs_safe = max(shs_val, 1e-10)
        max_step = np.sqrt((2.0 * self.max_kl) / shs_safe)
        full_step = step_dir * float(max_step)

        _, ls = self._line_search(
            old_params, full_step, old_surrogate,
            states, actions_idx, old_logp, adv, valid_masks,
        )

        with torch.no_grad():
            surr_new, kl_new = self._surrogate_and_kl(states, actions_idx, old_logp, adv, valid_masks)
            mean_entropy = float(self._mean_entropy(states, valid_masks).item())
        if not np.isfinite(mean_entropy):
            mean_entropy = 0.0

        self.reset_trajectory()
        return {
            "policy_loss": -float(surr_new.item()),
            "baseline_mse": baseline_mse,
            "value_loss": baseline_mse,
            "entropy": mean_entropy,
            "policy_update_accepted": float(ls["accepted"]),
            "policy_alpha": float(ls["alpha"]),
            "cg_shs": shs_val,
        }
