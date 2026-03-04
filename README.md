# Towers of Hanoi (REINFORCE vs TRPO)

## Introduction

The goal of the game is to **move all disks from the first peg (stick) to the last peg** while following the main rule of the Tower of Hanoi puzzle:

> A larger disk can never be placed on top of a smaller disk.

At every step, the agent selects a move that transfers the top disk from one peg to another.  
The episode ends when the puzzle is solved or when the maximum number of steps is reached.

<p align="center">
  <img src="game_phase2.gif" width="420">
</p>

---

## 1) Game Formulation (MDP)

We model the Tower of Hanoi puzzle as a **Markov Decision Process (MDP)**.

### State Space

We will play with 3 pegs. The last peg is the third one.
Let $n$ be the number of disks.  
A state is represented by the vector:

$$
s = (p_1, h_1, \dots, p_n, h_n),
$$

where $p_i$ is the peg index holding disk $i$. Disk indices correspond to sizes: disk $1$ is the smallest, disk $n$ is the largest. $h_i$ is the height of the disk $i$ on its peg $p_i$. $h_i=0$ means that disk $i$ lies on the bottom of the peg $p_i$

### Action Space

An action corresponds to moving the top disk from source peg $i$ to target peg $j$:

$$
a = (i \rightarrow j), \qquad i,j \in \{1,2,3\}, \quad i \neq j.
$$

An action is **valid** only if the moved disk is smaller than the top disk on peg $j$. If a source peg $i$ is empty, no disk moves with this action.

### Transition Function

The transition function is deterministic:

$$
s_{t+1} = T(s_t, a_t).
$$

If $a_t$ is valid, the environment applies the corresponding disk move (or no move if a source peg is empty). If $a_t$ is invalid, the state remains unchanged.

### Reward Function

We use three types of rewards:
- step penalty $r_{\text{step}}$,
- invalid move penalty $r_{\text{invalid}}$,
- victory reward $r_{\text{victory}}$.

In Phase 1 (initial experiments):

$$
r(s,a)=
\begin{cases}
r_{\text{victory}} & \text{if all the disks are correctly place on the third peg},\\
r_{\text{invalid}} & \text{if the move is invalid},\\
r_{\text{step}} & \text{otherwise}.
\end{cases}
$$

In every episode the agent plays till he wins (collects all the disks on the third peg in correct order) or he reaches the maximum number of steps (200).

Here we measure 2 metrics to describe success of a training:
- Success rate: the frequency of an agent to win the game. (The bigger, the better)
- Steps per episode: How many steps an agent made in the episode. (The lower, the better in case of 100% success rate) The optimal number of steps is equal to $2^n-1$.
---

## 2) Training with REINFORCE

We train a stochastic policy $\pi_\theta(a\mid S_t)$ (neural network parameters $\theta$ using the REINFORCE policy gradient method.

### Objective

The goal is to maximize the expected discounted return:

$$
v^{\pi^\theta}(s_{start})=
\mathbb{E}_{\pi_\theta}
\left[
\sum_{t=0}^{\tau-1} \gamma^t R_t | S_0=s_{start}
\right].
$$

where $s_{start}=(0, 0, 0, 1, 0, 2)$


For each episode we collect a trajectory:

$$
(s_0, a_0), (s_1, a_1), ..., (s_\tau, a_\tau)
$$


And make a step to maximize the following loss:

$$
L(\theta) = -\sum_{t=0}^{\tau-1} G_t \log \pi_\theta(a_t|s_t)
$$

where $G_t = \sum_{k=t}^{\tau-1} \gamma^{k-t} r_k$

---

### Initial training configuration

We first trained the agent with:

$$
r_{victory} = +100, \,\, r_{step} = -1, \,\, r_{invalid} = -5
$$


With **three disks**, the agent learned to solve the puzzle successfully. (success rate=100% and n_steps=2^3-1)

However, when we increased the difficulty to **four disks**, the training behaviour changed significantly. (success rate=0% and n_steps=200)

Instead of learning to solve the puzzle, the agent learned a strategy to **make only valid moves**.  
The reason is that the agent **almost never received the reward for winning** by following the initial policy, so he has no insentive to place all the disks on the third peg.

As a result, the agent optimized for survival instead of completing the task.

<p align="center">
<img src="images/original_reinforce.png" width="500">
</p>
<p align="center">
  <img src="game_3_disks.gif" width="350">
  <img src="game_4_disks.gif" width="350">
</p>


---

### Random Initial States

To improve exploration during training, we introduced **random initial states**.

Instead of always starting from the classical Tower of Hanoi configuration, the environment is initialized with a **random valid configuration of disks**.

This modification improves the learning process in several ways:

- the agent encounters **a larger variety of states** early in training
- the probability of starting **closer to the goal state increases**
- the agent receives the **goal reward more often**

As a result, the policy gradient signal becomes stronger and learning leads the agent to better results (success rate in (0.4, 0.65), n_steps in (75, 125)). However, these results are far from optimal.

We also **increased the reward for winning**, which further strengthens the learning signal. Overall, with random inisialisation of starting state and higher reward for winning the agent managed to train almost perfectly (success rate is equal to 100% and n_steps is around 22).  


<p align="center">
<img src="images/random_init.png" width="500">
</p>

---

### Fine-tuning for the Standard Initial State

Even though the agent trained almost perfectly to play the game from a random starting state, it needs to be trained additionally only with standard initial state (where all disks are located on the first peg). 

When training with random initial states, the agent rarely encounters the **original starting configuration** of the Tower of Hanoi puzzle.

Therefore we performed an additional **fine-tuning phase**, where every episode starts from the classical initial state.

This allows the agent to specialize its policy for the original puzzle setup.


<p align="center">
  <img src="images/finetune.png" width="350">
  <img src="game_phase2.gif" width="350">
</p>

---


## 3) Training with REINFORCE + Baseline

### Idea

REINFORCE with baseline reduces the variance of the policy gradient estimator without introducing bias.
For any state-dependent function $b(s_t)$ the following identity holds:

$$
\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(A_t\mid S_t)\, b(S_t)\right] = 0
$$

so subtracting $b(S_t)$ from the return leaves the gradient expectation unchanged while reducing its variance (hopefully).

We use the **value function estimate** $\hat{V}(s)$ as the baseline.
The resulting random variable

$$
A_t = G_t - \hat{V}(S_t)
$$

is called the **advantage** and measures how much better the actual return from step $t$ is compared to the expected return from state $s_t$.

So for gradient like in the theory for REINFORCE with the fact that substracting from $G_t$ any function of state $S_t$ random variable $-$ $b(S_t)$ leaves the gradient expectation unchanged:

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(A_t\mid S_t)\,A_t\right]
$$

### State Encoding

Because disk heights are uniquely determined by peg assignments, the full state
$s = ((p_0, h_0), \dots, (p_{n-1}, h_{n-1}))$ is characterised solely by the peg assignment vector
$(p_0, \ldots, p_{n-1})$ (0-indexed, $p_i \in \{0,1,2\}$).
We map it to a unique integer index via a mixed-radix encoding:

$$
\text{idx}(s) = \sum_{i=0}^{n-1} p_i \cdot P^{i}
$$

where $P = 3$ is the number of pegs.
The total number of distinct states is $P^n = 3^n$.
Inverse (decoding) is obviously can be uptained by iterative division $\text{idx}(s)$ by $P$.

### Value Function Estimation

The baseline $\hat{V}$ is a table of size $P^n$, recomputed from scratch before every gradient step using the **last $N$ episodes** stored in a circular buffer of length $N$ by the following algorithm:

**Algorithm:**

1. Set $V[i] = 0$, $ c[i] = 0$ for all $i \in \{0, \ldots, P^n - 1\}$
2. For each trajectory $\tau$ in the history buffer (oldest to newest):
   - For each pair $(s_t,\, G_t)$ in $\tau$:

$$
V[\text{idx}(s_t)]
\mathrel{+}=
\frac{G_t - V[\text{idx}(s_t)]}{c[\text{idx}(s_t)] + 1} \newline
$$

$$
c[\text{idx}(s_t)]
\mathrel{+}= 1
$$

3. Return $V$.

This is an **incremental mean**: on completion, $V[\text{idx}(s)]$ equals the unweighted average of all Monte Carlo returns $G_t$ collected from state $s$ across the history buffer.

### Reward Design and Training Setup

The reward function, discount factor $\gamma$, policy network architecture, optimizer, entropy regularisation coefficient, and all other hyperparameters are **identical** to those used for the plain REINFORCE agent (Section 2).
The only structural difference is the subtraction of $\hat{V}(s_t)$ from each return realization $g_t$ before computing the policy loss.


## 4) TRPO — Trust Region Policy Optimization

### 1. General Theory

#### 1.1 Surrogate Objective

The goal of policy optimization is to find parameters $\theta$ that maximize the expected discounted return:

$$V(\pi) = \mathbb{E}_{S_0, A_0, \ldots} \left[\sum_{t=0}^{\infty} \gamma^t R_t\right]$$

A key identity relates the performance of a new policy $\tilde{\pi}$ to the current policy $\pi$ through the advantage function:

$$V(\tilde{\pi}) = V(\pi) + \sum_s \rho_{\tilde{\pi}}(s) \sum_a \tilde{\pi}(a|s)\, A_\pi(s, a)$$

where $\rho_{\tilde{\pi}}(s) = \sum_{t=0}^{\infty} \gamma^t P(S_t = s \mid \tilde{\pi})$ is the discounted state visitation frequency under the new policy, and $A_\pi(S,A) = Q_\pi(S,A) - V_\pi(S)$ is the advantage of the current policy.

This identity implies that if the new policy $\tilde{\pi}$ has non-negative expected advantage at every state, then $V(\tilde{\pi}) \geq V(\pi)$. However, direct optimization of this expression is difficult because $\rho_{\tilde{\pi}}$ depends on $\tilde{\pi}$.

To address this, the **surrogate objective** is introduced:

$$L_\pi(\tilde{\pi}) = V(\pi) + \sum_s \rho_\pi(s) \sum_a \tilde{\pi}(a|s) A_\pi(s, a)$$

The surrogate $L_\pi$ differs from the exact expression for $V(\tilde{\pi})$ in that it uses the visitation frequencies $\rho_\pi$ from the **old** policy instead of $\rho_{\tilde{\pi}}$ from the new one.

#### 1.2 Approximation: Replacing $\rho_{\tilde{\pi}}$ with $\rho_\pi$

Replacing $\rho_{\tilde{\pi}}$ with $\rho_\pi$ is the key approximation in TRPO. It is valid in the following sense: the surrogate $L_\pi$ matches $V$ to first order in the parameters:

$$L_{\pi_{\theta_0}}(\pi_{\theta_0}) = V(\pi_{\theta_0}), \qquad \nabla_\theta L_{\pi_{\theta_0}}(\pi_\theta)\big|_{\theta = \theta_0} = \nabla_\theta V(\pi_\theta)\big|_{\theta = \theta_0}$$

That is, at the current parameter values the function values and gradients coincide. This means that a **sufficiently small** step improving $L_\pi$ is guaranteed to improve $V$ as well.

In summary: the approximation $\rho_{\tilde{\pi}} \approx \rho_\pi$ is justified when the KL divergence between the old and new policies is small. The closer the new policy is to the old one (in the KL sense), the more accurately the surrogate $L_\pi$ reflects the actual change in $V$.

### 1.3 Step Selection: From Penalty to Trust Region

The theory suggests maximizing $L_\pi(\tilde{\pi}) - C \cdot D_{KL}^{\max}(\pi, \tilde{\pi})$, which guarantees monotonic improvement. However, the constant $C = 4\gamma/(1-\gamma)^2$ is enormous in practice and leads to negligibly small steps.

Instead of a penalty, TRPO uses a **constraint** on the KL divergence:

$$\max_\theta L_{\theta_\text{old}}(\theta) \qquad \text{s.t.}\quad \bar{D}_{KL}(\theta_\text{old}, \theta) \leq \delta$$

where $\bar{D}_{KL}$ is the average KL divergence over states (rather than the maximum, which would be computationally intractable). The parameter $\delta$ (in the original paper and in our implementation, $\delta = 0.01$ by default) defines the size of the trust region — the region of parameter space where the surrogate remains a reliable approximation.

To solve this constrained problem, the **natural gradient** is used. Linearizing $L$ and quadratically approximating the KL constraint yields:

$$\max_\theta \nabla_\theta L\big|_{\theta_\text{old}} \cdot (\theta - \theta_\text{old}) \qquad \text{s.t.}\quad \tfrac{1}{2}(\theta - \theta_\text{old})^T F (\theta - \theta_\text{old}) \leq \delta$$

where $F$ is the Fisher information matrix (the Hessian of the KL divergence). The solution is $\theta_\text{new} = \theta_\text{old} + \beta F^{-1} g$, where $g = \nabla_\theta L\big|_{\theta_\text{old}}$ is the gradient of the surrogate, and the maximum step size is:

$$\beta = \sqrt{\frac{2\delta}{g^T F^{-1} g}}$$

Computing $F^{-1}g$ directly for networks with thousands of parameters is infeasible. Instead, **conjugate gradient (CG)** is used: the system $Fx = g$ is solved iteratively using only Fisher-vector products $F \cdot v$, without ever forming the matrix $F$ explicitly.

---

### 2. Practical Algorithm and Our Implementation

#### Overview of a Single TRPO Update Step

1. **Trajectory collection.** The agent runs an episode, storing states, actions, log-probabilities, and valid actions.

2. **Computing returns and advantages.**
   Returns $g_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$ are computed in the standard way. The baseline $V_\pi(s)$ is estimated using a **tabular** method over episode history (inherited from `REINFORCEBaselineAgent` and described in section 3 for baseline REINFORCE): for each discrete state $V_\pi(s)$ is the mean of observed returns. The advantage is then: $a_t = g_t - V_\pi(s_t)$.

3. **Fixing the old policy.** The probabilities $\pi_\text{old}(\cdot|s)$ — the distribution over all actions for each state in the batch — are saved. They are used for computing the full KL divergence and Fisher-vector products.

4. **Computing the surrogate gradient.** The surrogate is expressed via the importance sampling ratio:

   $$L(\theta) = \frac{1}{T}\sum_{t=1}^{T} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)} A_t$$

   If the gradient norm exceeds `max_grad_norm_cg` (default 50.0), the gradient is scaled down before being passed to CG.

5. **Conjugate Gradient (CG).** We solve $(F + \lambda I) x = g$, where $\lambda$ is the damping coefficient (default 0.01) added to regularize the Fisher matrix. The Fisher-vector product is computed via double automatic differentiation:
   - First `autograd.grad`: obtain the gradient of KL with respect to the parameters.
   - Dot product of this gradient with the vector $v$.
   - Second `autograd.grad`: obtain $Fv$.

   The implementation uses `scipy.sparse.linalg.cg` (when available) with `maxiter = 10` iterations by default.

6. **Computing the full step.** After obtaining $x \approx F^{-1}g$ from CG:
   - Verify that $x^T F x > 0$ (the direction is valid).
   - Maximum multiplier: $\beta = \sqrt{2\delta / (x^T F x)}$.
   - Full step: $\Delta\theta = \beta \cdot x$.
```

# Running the Project 

Watch LAUNCH.md 🐳🐳🐳