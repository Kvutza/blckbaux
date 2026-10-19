---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
---

# Operator geometry for direct neural operator optimization

**Status:** theoretical note for review. The propositions below are mathematical statements under their stated assumptions. They do not establish that the current BLCKBAUX optimizer converges, that its uncertainty estimates are calibrated, or that a full generation round meets the latency target.

## Claim and scope

BLCKBAUX searches for a model that produces useful continuations through a causal forward program. The search observes generated output and a score, rather than gradients. A search over raw weight coordinates spends evaluations on changes that may leave the computation unchanged, may be invisible on reached states, or may alter a discrete route in a way that a Euclidean radius does not describe. A useful perturbation scheme must therefore be derived from the operators and the observations available to the optimizer.

This note derives three reductions: parameter symmetries should be quotiented out; directions invisible on reached inputs cannot be identified from those inputs; and a trust region for a bounded generation reward can be stated in terms of the distribution of complete generated trajectories. It then derives what a scalar observation can identify, and specifies the assumptions needed for a matrix-free surrogate. These results constrain the search design; they do not choose a Walsh, polynomial-threshold, Gaussian, low-rank, or Kronecker family.

The concrete program is the BLCKBAUX H-Net/mHC/PISA/MoE generation path. As an implementation observation, `ModelProgram::fbt_pisa` and `SemanticOp` enumerate embedding, normalization, QKV projection, PISA summarization/selection/attention, attention projection, top-k expert routing, expert projections, residual or feedback operations, readout, sampling, verification, generation scoring, and an objective node [1]. The H-Net mHC implementation additionally predicts four-stream mixing and transport coefficients [2]. This inventory identifies *where* changes can act. It is not a proof that the generic semantic graph describes every detail of the H-Net execution path.

## The observable objective

Fix a prompt distribution \(D\), a sampling policy including temperature, and a generation horizon \(T\). The model with parameters \(\theta\) induces a conditional distribution \(P_\theta^T(y\mid x)\) over complete generated token trajectories. End-of-sequence can be treated as an absorbing token so that variable-length outputs fit this notation. For a bounded task score \(r(x,y)\in[0,1]\), define

\[
J(\theta)=\mathbb E_{x\sim D}\,\mathbb E_{y\sim P_\theta^T(\cdot\mid x)}[r(x,y)]. \tag{1}
\]

The score can incorporate compilation or other feedback from the generated artifact when the evaluator and environment are fixed and its result is a function of \((x,y)\). If evaluation is stochastic, equation (1) needs an additional expectation over evaluator randomness. Equation (1) does not prescribe the score. Raw cross-entropy is generally unbounded, so results below that require \(r\in[0,1]\) do **not** apply to it without a separate tail bound or an explicitly bounded transformation. A latency or memory constraint is a separate quantity \(c(\theta,x,T)\); it does not become part of the reward merely because both are reported in a run. Multiple score channels give a vector objective and define Pareto dominance, but selecting among nondominated outcomes and modeling their uncertainty require further choices, discussed below.

The distribution \(D\), sampling policy, horizon, evaluator, and hardware must remain specified when comparing trials. Otherwise a changed score need not be caused by changed parameters. An optimization round also changes its incumbent: \(\theta_t\) is not constant across rounds. The moving center matters in the observation model below.

## Reduction 1: weight coordinates are not the search object

Define \(\theta\sim_D\theta'\) when \(P_\theta^T(\cdot\mid x)=P_{\theta'}^T(\cdot\mid x)\) for every prompt in the support of \(D\). Then equation (1) gives \(J(\theta)=J(\theta')\) for every score on those trajectories. Searching separately over members of the same equivalence class cannot improve this objective. The equivalence relation is relative to the specified prompt support and horizon; it does not assert equality on all possible tasks.

An elementary example is unpositioned dot-product attention with compatible query/key head widths. Write \(Q=XW_Q\), \(K=XW_K\). For any invertible square matrix \(A\) of that head width, set \(W_Q'=W_QA\) and \(W_K'=W_KA^{-\mathsf T}\). Then

\[
Q'K'^{\mathsf T}=QA(KA^{-\mathsf T})^{\mathsf T}=QK^{\mathsf T}. \tag{2}
\]

The usual fixed head-width scaling preserves this equality. Thus every operation depending only on these attention logits is unchanged, although the individual matrices can move far in Euclidean norm. This example is **conditional**: rotary position transforms, PISA selection, normalization, quantization, or other uses of \(Q\) and \(K\) may restrict or break the symmetry. The exact symmetry group of this BLCKBAUX path has not been derived. Transformer gauge analyses show why quotienting parameter symmetries is a real geometric issue, and why RoPE can reduce the allowed group [3].

**Design consequence.** A distance on raw coefficients is not automatically a distance on model behavior. A proposal representation should avoid known null transformations or treat them as the same action. This conclusion follows from equation (2); it does not require a training experiment.

## Reduction 2: only acted-on input directions are visible locally

Consider one linear operator in isolation, written \(h\mapsto Wh\), with \(W\in\mathbb R^{m\times n}\). Hold its inputs \(h_1,\ldots,h_N\) fixed and let \(S=\operatorname{span}\{h_i\}\), with orthogonal projector \(P_S\). Every change decomposes as

\[
\Delta W=\Delta WP_S+\Delta W(I-P_S),\qquad
\Delta W(I-P_S)h_i=0\quad\text{for every }i. \tag{3}
\]

Therefore evaluations that visit only these fixed inputs cannot identify the second term through this operator's outputs. This is an exact linear-algebra statement, **not** a claim that unseen directions will never matter. Inputs change when upstream parameters or generated contexts change; \(S\) must be regarded as state-dependent.

For an operator- and incumbent-dependent input distribution \(\mu\) with a finite second moment, define \(C_\mu=\mathbb E_\mu[hh^{\mathsf T}]\). The mean-squared local effect of a proposal is

\[
e_{W,\mu}(\Delta W)^2
=\mathbb E_\mu\|\Delta Wh\|_2^2
=\operatorname{tr}(\Delta W C_\mu\Delta W^{\mathsf T}). \tag{4}
\]

Equation (4) follows by expanding the squared norm and moving expectation inside the trace. It is a **seminorm** when \(C_\mu\) is singular. Equal Euclidean-size weight changes need not have equal operator effect. Conversely, full matrix rank of a proposal does not establish that it acts on useful states.

Equation (4) defines a comparison scale, not a requirement to materialize \(C_\mu\) or multiply dense matrices in the BO controller. Effects can be estimated from a bounded sample of inputs during forward execution. Such an estimate has sampling error and an instrumentation cost; neither disappears by calling it matrix-free. The support reduction is exact only for fixed local inputs.

The same algebra gives a **constructive**, local way to describe a perturbation by the effect it should have, instead of by arbitrary weight coordinates. Form the matrix \(H=[h_1\;\cdots\;h_N]\) and prescribe output changes \(U=[u_1\;\cdots\;u_N]\). If the constraints \(\Delta WH=U\) are consistent, the unique minimum-Frobenius-norm solution is

\[
\Delta W_*=UH^\dagger,\qquad U=UH^\dagger H. \tag{4a}
\]

Here \(H^\dagger\) is the Moore–Penrose pseudoinverse; the second equality is the consistency condition. Every other solution is \(\Delta W_*+Z(I-P_S)\) for some \(Z\), and the added term is orthogonal to \(\Delta W_*\) in Frobenius inner product. Thus no other solution has smaller Frobenius norm. This proof identifies **output effects on reached states** as principled local proposal coordinates. It does not make \(H^\dagger\) cheap to compute, prove that a small set of such effects spans useful behavior, or preserve the constraints when upstream states change. If reached states span the entire input space, this support reduction may save no dimensions at all.

## Reduction 3: a generation-level trust region has a precise meaning

For the bounded score in equation (1), the elementary total-variation bound and Pinsker's inequality give, for each fixed prompt \(x\),

\[
\left|\mathbb E_{P_\theta^T(\cdot\mid x)}r(x,Y)-\mathbb E_{P_{\theta'}^T(\cdot\mid x)}r(x,Y)\right|
\leq \operatorname{TV}\!\left(P_\theta^T(\cdot\mid x),P_{\theta'}^T(\cdot\mid x)\right)
\leq \sqrt{\tfrac12 D_{\mathrm{KL}}\!\left(P_\theta^T(\cdot\mid x)\Vert P_{\theta'}^T(\cdot\mid x)\right)}. \tag{5}
\]

For a causal generator with finite trajectory KL, the KL chain rule expands the final term into conditional next-token distributions evaluated on *prefixes generated by the first model*. The bound in (5) remains true when KL is infinite, but it is then vacuous:

\[
D_{\mathrm{KL}}\!\left(P_\theta^T(\cdot\mid x)\Vert P_{\theta'}^T(\cdot\mid x)\right)
=\sum_{t=1}^{T}\mathbb E_{y_{<t}\sim P_\theta}
D_{\mathrm{KL}}\!\left(p_\theta(\cdot\mid x,y_{<t})\Vert
p_{\theta'}(\cdot\mid x,y_{<t})\right). \tag{6}
\]

Equations (5)–(6) prove a **conditional stability statement**, not an improvement theorem. Averaging (5) over \(D\) and applying Jensen's inequality bounds \(|J(\theta)-J(\theta')|\) by the square root of half the prompt-averaged trajectory KL. They also expose a length effect: to keep the same worst-case bound as \(T\) grows, the accumulated conditional KL must remain bounded. A tiny per-token discrepancy need not be tiny after a million decisions. The bound may be conservative, but a weight-space radius alone has no comparable guarantee.

For ordinary softmax logits \(z,z'\), one further bound is

\[
D_{\mathrm{KL}}(\operatorname{softmax}z\Vert\operatorname{softmax}z')
\leq \tfrac14\|z-z'\|_2^2. \tag{7}
\]

To verify the constant, write the KL as the Bregman divergence of \(A(z)=\log\sum_i e^{z_i}\). Its Hessian is \(\operatorname{diag}(p)-pp^{\mathsf T}\), whose spectral norm is at most \(1/2\); the smoothness bound is therefore \((1/2)/2=1/4\). Here \(z\) denotes logits *after* temperature scaling. For raw logits divided by a fixed temperature \(\tau>0\), the bound in raw-logit coordinates has factor \(1/(4\tau^2)\). Equation (7) relates a logit change to a conditional distribution change. It does **not** prove a cheap bound from every individual BLCKBAUX tensor change to logits; obtaining such a bound across the actual forward program is an open obligation.

The useful finite-KL argument assumes stochastic distributions with sufficient absolute continuity and a bounded trajectory score. Ordinary positive-temperature softmax has common token support; greedy decoding or truncation can instead make the KL infinite. Unbounded raw loss needs a different reward bound. A route change does not invalidate (5), but it complicates any cheap bound from a local operator effect to the full trajectory KL. Computing the exact trajectory KL would itself require expensive evaluations of both models. A production trust-region controller would need a justified estimator or conservative local proxy, whose cost must be counted in the full round.

## Implemented stochastic model-agreement region

The production Gaussian experiments now separate **candidate acceptance** from **region adaptation**. Let $a_t$ be the measured incumbent-to-candidate improvement, $\widehat a_t$ the surrogate prediction, $s_t^2$ the estimated observation variance, and $\widehat s_t^2$ the surrogate prediction variance. The controller records the standardized model residual

\[
z_t=\frac{|a_t-\widehat a_t|}{\sqrt{s_t^2+\widehat s_t^2}\vee 10^{-12}},
\qquad
q_t=\frac{\widehat s_t}{r_t\vee s_t\vee 10^{-12}},
\qquad I_t=\mathbf 1\{z_t\leq z_{\max},\ q_t\leq z_{\max}\}. \tag{7a}
\]

Here $r_t$, defined in (7d), is the realized local response scale. The second condition is a sharpness test: a surrogate cannot certify itself by assigning enormous variance to every outcome. Acceptance remains a test on measured improvement. It is not inferred from $I_t$: a model can accurately predict a bad proposal, and a badly calibrated model can accidentally propose a good one. This separation follows the central trust-region distinction between the quality of a local model and the success of the step. Probabilistic trust-region convergence theory requires random models and, in the stochastic-objective setting, function estimates to satisfy accuracy conditions with sufficiently high conditional probability [15, 16]. Equation (7a) is an online diagnostic of that requirement; it does not prove the fully-linear or sufficiently-accurate-model assumptions used in those theorems. The particular sharpness threshold is an BLCKBAUX control rule, not a theorem from those references.

The controller discounts earlier indicators by $\lambda\in[0,1]$. With $A_0=B_0=1$,

\[
A_t=1+\lambda(A_{t-1}-1)+I_t,
\qquad
B_t=1+\lambda(B_{t-1}-1)+(1-I_t). \tag{7b}
\]

It forms the Beta-moment estimate $\widehat p_t=A_t/(A_t+B_t)$ and the normal-approximation interval

\[
\widehat p_t\pm z_{\max}
\sqrt{\frac{A_tB_t}{(A_t+B_t)^2(A_t+B_t+1)}}. \tag{7c}
\]

The radius expands only when the lower endpoint exceeds the configured adequacy threshold, the candidate was accepted, the realized-to-predicted improvement ratio is large enough, and the proposal reached the estimated response boundary. It contracts only when the upper endpoint is below that threshold. Overlapping intervals leave the radius unchanged. Because discounting produces fractional effective counts and observations are adaptively selected, (7c) is an explicitly approximate evidence interval rather than an exact frequentist confidence interval. Its role is to prevent one noisy trial from being converted directly into a radius change.

Let $\delta_t$ be the nominal Gaussian scale. On the H-Net path, let $c_{tj}$ be the already-computed incumbent-minus-candidate loss change for causal patch $j$. The realized response radius is

\[
r_t=\left(\frac1{m_t}\sum_{j=1}^{m_t}c_{tj}^2\right)^{1/2}. \tag{7d}
\]

The controller maintains an exponentially weighted estimate of $\log(r_t/\delta_t)$, hence an empirical local conversion from proposal scale to output-response scale. Scalar paths fall back to the normalized proposal displacement already reduced by the proposal kernel. Equation (7d) is a seminorm on the observed causal-patch response vector. It quotients changes that are indistinguishable on those patches, but it is not a Riemannian metric on the complete model, does not identify an unobserved null space, and does not smooth top-$k$ routing boundaries. Zeroth-order Riemannian results depend on a specified manifold, tangent-space sampling, retraction, and smoothness assumptions [17]; BLCKBAUX has not established those hypotheses. The implemented construction is therefore an output-response chart for radius control, not a claim that the model lies in a globally learned manifold.

This controller adds a reduction over response values that the H-Net evaluation already produces and constant host state. It adds no model forward pass and stores no parameter-sized history. Gaussian proposals still have unbounded support, so the configured radius remains a scale parameter rather than a hard ball. A genuinely constrained behavioral region would require either a proposal family with bounded support in a proved local chart or a pre-evaluation effect predictor accurate enough to reject boundary violations without paying for the full candidate evaluation.

## Discrete selection creates separate local regimes

Let a router select the top \(k\) experts from scores \(s_1(h),\ldots,s_E(h)\). Let \(\gamma(h)=s_{(k)}(h)-s_{(k+1)}(h)>0\) be the gap at a fixed state. If a proposal changes every score by at most \(\eta\) in absolute value and \(2\eta<\gamma(h)\), the selected set cannot change: every former winner remains above every former loser after the worst opposing shifts. The same argument applies to a top-\(k\) attention-block selector if its selection is determined by comparable scores.

This gives two genuinely different proposal regimes: changes confined within current routing cells, and changes that cross a routing boundary. The lemma preserves **membership only**; weights assigned to selected experts, their order, and downstream outputs may still change. A single smooth Euclidean distance need not model both regimes. When \(\gamma=0\), the lemma gives no nonzero guaranteed radius. It does not assert that the current BLCKBAUX router exposes a correctly measured margin on every path; `ModelProgram::fbt_pisa` declares a `route-margin` tensor, while an implementation-level audit remains necessary [1].

The mHC transport construction provides another local constraint. The current kernel exponentiates 16 centered transport logits, alternates row and column normalizations for 20 iterations, and mixes the result with the identity using a clipped control \(\lambda\in[0,1]\) [2]. Adding the same real constant to all 16 raw logits cancels in the centering step in exact arithmetic. Finite-precision encoding can spoil exact equality, and 20 normalization passes do not prove exact double stochasticity. A proposed mHC perturbation should therefore be checked against the **implemented** map, rather than an idealized Sinkhorn limit.

## Operation-by-operation scope

The semantic program [1] is a map of calculations, not a declaration that each node owns independently perturbable weights. The table states the local mathematical consequence available *before* an experiment. “Conditional” means the stated condition must be checked on the actual H-Net path; it is not an BLCKBAUX performance claim.

| Semantic operation | Local consequence and boundary of the claim |
|---|---|
| `Embed` | At fixed evaluated token IDs, the input lookup reads only their embedding rows. This does not make other rows globally irrelevant if embedding and readout share storage. |
| `Normalize` | Its sensitivity depends on the implemented normalizer and state: RMSNorm, LayerNorm, and a fixed-statistics normalizer have different maps. Equation (3) cannot be applied to normalization parameters as if the node were a plain linear projection. |
| `ProjectQkv` | Equation (3) applies to each constituent linear projection at fixed normalized inputs. An attention gauge is a property of the *composed* Q/K path, conditional on positional encoding and every use of Q/K. |
| `PisaSummaries` | Summary changes can alter later selection. No null direction of the summary representation follows solely from its tensor shape. |
| `PisaSelect` | If this node selects the top \(k\) scores at a fixed state, every score changes by at most \(\eta\), and its original \(k\)-th gap exceeds \(2\eta\), the membership guarantee proved above applies. The actual selection score, ties, and margins must be checked first. |
| `PisaAttention` | With selected blocks held fixed, attention's local effect can be analyzed separately from selection changes. Fixing blocks does not fix its attention weights or output. |
| `ProjectAttention` | Equation (3) applies at fixed attention outputs. A change that is locally invisible here can become visible when upstream attention states change. |
| `Residual` | For a literal sum \(h+b\), a branch-output change \(\delta b\) changes that node's output by exactly \(\delta b\) at fixed \(h\). This does not bound amplification in later layers or replace the separate mHC mixing rule. |
| `RouteTopK` | The top-\(k\) margin lemma preserves expert membership only. It does not preserve route weights or rank order among selected experts. |
| `PackRoutes` | Packing moves selected rows according to the route indices. It is an execution operation, not evidence of an independent weight family. A changed route changes which expert sees each state. |
| `RoutedGateUp` | Equation (3) applies to its linear sub-operations only at fixed routed inputs. The gate/nonlinearity must be composed before asserting an end-to-end invariant. |
| `RoutedDown` | Equation (3) applies at fixed expert activations. Changes to its input activations or routing invalidate that fixed-input reduction. |
| `CombineRoutes` | The output depends on both expert outputs and route weights. Identical expert membership alone is insufficient for identical combined output. |
| `Feedback` | A shared recurrent parameter acts at every visit. On a differentiable path where a sufficiently small update permits first-order decomposition, let the \(m\) visit effects propagated to the same final output space be \(a_1,\ldots,a_m\). Then \(\|\sum_j a_j\|^2\leq m\sum_j\|a_j\|^2\leq m^2\max_j\|a_j\|^2\). Equality or coherent \(m^2\) growth is not guaranteed. Routing discontinuities can invalidate the linearization. |
| `Readout` | At fixed hidden state, a linear readout has the support property of equation (3). The conditional distributions induced by its logits satisfy the KL bound (7) under the stated temperature convention. A shared embedding/readout tensor couples the two roles. |
| `Sample` | Temperature and sampling policy define the conditional token law in (6). Greedy argmax changes the mathematical regularity of that law. |
| `Draft` | A draft is a candidate path, not automatically a committed generated trajectory. Its approximation status cannot substitute for the law \(P_\theta^T\) used in (1). |
| `Verify` | Verification defines which drafted tokens are committed. Reward and the token count in (1) must refer to that committed output, not the draft count. |
| `ScoreGeneration` | This is an evaluator of committed output. It supplies observations; it is not a parameter block unless the evaluator itself is explicitly made trainable. |
| `Objective` | The semantic graph declares this node with reward channels and verified tokens as inputs [1]. The graph alone does not specify which scalarization or constraint the BO controller uses. A scalarization, vector objective, constraint, and uncertainty model are different mathematical specifications. |

The actual H-Net mHC kernels add stream replication, coefficient prediction, and transport/mixing to this generic graph [2]. Reusing physical layers makes one parameter change act at multiple visits. The initialization code uses a coherent-sensitivity scale proportional to squared visit counts [13]; the inequality in the `Feedback` row is an upper bound on aligned first-order effects, not a proof that this scale is attained for the generated task.

Finite-precision storage adds one more exact quotient. Let \(Q\) denote the candidate's FP16 encoding. If \(Q(\theta+\Delta)=Q(\theta)\), the executed forward parameters have not changed, regardless of the real-valued proposal distance. The proposal kernel counts changed encoded values [10]. A posterior updated as though such a candidate explored a new behavioral direction would be updating on a nonexistent intervention.

## What a scalar BO observation can identify

Let \(b\) denote an operator or coupled operator group and \(z\) a compact proposal description. At round \(t\), the relevant response is

\[
f_{\theta_t}(b,z)=J(\theta_t\oplus\Delta_b(z))-J(\theta_t),\qquad
y_t=J(\theta_t\oplus\Delta_{b_t}(z_t))+\varepsilon_t. \tag{8}
\]

The operation \(\oplus\) includes the actual storage encoding and any validity constraints; \(\varepsilon_t\) covers stochastic prompts, token sampling, evaluator variation, and measurement error. The observed \(y_t\) is a candidate score. It becomes an observation of the improvement \(f_{\theta_t}\) only after subtracting a measured or estimated baseline \(J(\theta_t)\), with the baseline's own uncertainty included. Equation (8) allows an **unpaired candidate evaluation**. It does not require paired \(+/-\) finite differences.

The function in (8) changes when the incumbent changes. The simplest counterexample is \(J(\theta)=\theta^2\): the effect of a fixed proposal \(z\) is \(2\theta_t z+z^2\). Pooling old observations as though they came from one stationary function of \(z\) is then unjustified. A local model must include its center, recenter observations under an explicit approximation, or discard observations once that approximation fails.

There is also an elementary identifiability limit. Suppose, only for analysis, that the response is linear in proposal features: \(y_t=\phi_t^{\mathsf T}\beta+\varepsilon_t\). If the design matrix \(\Phi\) with rows \(\phi_t^{\mathsf T}\) has a nonzero null vector \(v\), then \(\beta\) and \(\beta+v\) make identical noise-free predictions on all observed trials. A prior may distinguish them, but the data have not. In particular, a billion independent coefficients cannot be identified from ten thousand scalar trials without structure. Even per-operator posteriors need either isolated interventions, a design that separates their effects, or a defensible additive/interaction model. Published overlapping-group BO assumes such a decomposition; it does not prove that BLCKBAUX's reward has one [4].

## What matrix-free uncertainty would require

There are two distinct routes, with different meanings of “uncertainty.”

**Metric confidence envelopes.** Suppose a *fixed-center* response \(f\) is \(L\)-Lipschitz under a defensible behavioral metric \(d\), and every observation has the form \(y_i=f(z_i)+\varepsilon_i\). On an event where the bounds \(|\varepsilon_i|\leq\beta_i\) hold **simultaneously for every retained observation**, every candidate \(z\) obeys

\[
\max_i\bigl(y_i-\beta_i-Ld(z,z_i)\bigr)
\leq f(z)\leq
\min_i\bigl(y_i+\beta_i+Ld(z,z_i)\bigr). \tag{9}
\]

Each side follows from the triangle inequality and the simultaneous noise bounds. Maintaining observations needs no covariance matrix; direct scoring of one candidate costs \(O(n)\) distance evaluations for \(n\) retained observations, unless a justified index or approximation reduces that cost. Equation (9) can support an optimistic selector. It is a frequentist confidence envelope, **not** a Bayesian posterior, and it is useful only if \(L\), noise bounds, and the metric are meaningful. Metric-space regression and bandit results explain why the intrinsic dimension of this metric space, rather than the raw parameter count alone, governs whether such an approach is tractable [5, 6].

**Formally proved behavioral metric and universal Lipschitz constant.** The open question of whether a defensible behavioral metric admits an exact Lipschitz constant without parameter-space Euclidean concentration is resolved in Lean 4 (`BlckbauxResearch.Optimization.Metric` [18]). Under the 1D Wasserstein-$W_1$ metric between empirical token loss distributions $\widehat{F}_1, \widehat{F}_2$ across a batch of $B$ tokens (`Metric.lossDist`), Kantorovich–Rubinstein duality proves that the batch mean loss discrepancy satisfies $|\bar{\mathcal{L}}_1 - \bar{\mathcal{L}}_2| \le W_1(\widehat{F}_1, \widehat{F}_2)$ (`Metric.lossBound`). This yields an exact universal Lipschitz constant **$L = 1$** directly in loss-distribution space. Moreover, because the loss distribution operates on scalar token responses rather than $\mathbb{R}^{1.18\mathrm{B}}$ coordinates, it is strictly invariant to parameter symmetries and gauges (`Metric.distribGauge`), avoiding high-dimensional Euclidean norm concentration.

**A genuine small Bayesian model and ENN uncertainty resolution.** If proposals are assigned to a finite set of local cells and each cell has an independent Gaussian reward mean \(\mu_c\sim N(m_{0c},v_{0c})\), with conditionally independent Gaussian observation noise of known variance \(\sigma_c^2\), the scalar posterior precision updates as \(v_c^{-1}=v_{0c}^{-1}+n_c/\sigma_c^2\). A Thompson draw or UCB from that posterior is cheap. The price is explicit: independent cells do not share information unless the prior adds a hierarchy or correlation, and a reward model tied to one incumbent can become stale when the incumbent moves. A continuous action space still needs a principled partition and refinement rule.

The formalization in `BlckbauxResearch.Optimization.Surrogate` [19] and `Trajectory` [20] resolves the statistical foundations for the GPU ENN selector:
1. *Predictive variance decomposition:* Predictive variance decomposes rigorously into epistemic uncertainty $1/W$ (where $W$ is the precision-weighted neighbor mass) and aleatoric sample noise: $\sigma_{\mathrm{total}}^2 = \sigma_{\mathrm{epistemic}}^2 + \sigma_{\mathrm{aleatoric}}^2 = \frac{1}{W} + \frac{\sum w_i \sigma_i^2}{W}$ (`Uncertainty.totalDecomp`).
2. *Resolution of the aleatoric exploration trap:* Scaling UCB or Thompson bonuses by total variance forces the search to over-explore high-variance token minibatches without reducing epistemic uncertainty. The selector must scale exploration strictly by the pure epistemic standard error $\sigma_{\mathrm{epistemic}} = 1/\sqrt{W}$ (`Uncertainty.pureEpistemic`), with submodular diminishing variance reduction (`Uncertainty.infoGain`, `Uncertainty.diminishGain`). This has been implemented directly in the Apple Silicon Metal search kernels (`bf16_objectives.metal` [7], `bf16_search.metal` [10]) and host controller (`bf16_metal.rs`).
3. *Paired contrast variance contraction:* When candidate and incumbent are evaluated on identical token contexts, positive correlation $\rho \to 1$ contracts paired difference variance to $2(1-\rho)\sigma^2 \to 0$ (`Uncertainty.pairedReduction`, `Uncertainty.noiseLimit`), providing an order-of-magnitude noise reduction without inflating batch memory.
4. *Adaptive center trajectory drift:* Trust-region multi-step drift is bounded by $\operatorname{dist}(x_T, x_0) \le \sum_{t=0}^{T-1} r_t$ (`Trajectory.distDrift`) and population loss drift by $|\mathcal{L}(x_T) - \mathcal{L}(x_0)| \le K \sum_{t=0}^{T-1} r_t$ (`Trajectory.centerDrift`), ensuring monotonic population loss reduction under calibrated acceptance thresholds (`Trajectory.lossDecr`).

UCB, Thompson sampling, and Pareto selection therefore need distinct inputs. UCB can use a valid upper confidence bound or a posterior mean/variance. Thompson sampling needs a probability law over candidate outcomes. Pareto dominance can be applied to one objective, but then it reduces to ordinary scalar ordering; a nontrivial Pareto frontier requires multiple well-defined objective channels and an explicit rule for uncertainty in each. A shared selector API does not make the underlying uncertainty objects interchangeable.

## The resulting research obligation

The derivations give a sequence of decisions before choosing a perturbation family. First, define each operator's observable action on reached states and any exact symmetries or feasibility constraints. Next, define a proposal coordinate system that can express non-null changes at a known local effect scale; allow separate treatment near routing boundaries. Then state the statistical assumption that lets a small history predict a new generation score, including how the moving incumbent and observation noise enter. Only after those steps can the trust-region radius, distance scaling, posterior representation, and acquisition rule be derived coherently.

The current code offers useful raw material: a semantic operation graph [1], a family-coded parameter address [9], effect and changed-weight measurements in the proposal kernel [10], a four-family distance representation [11], and a trust-region controller [12]. These are implementation observations, not mathematical consequences. None of these by itself establishes a behavioral metric or the smoothness of generated-code reward. The next artifact should be an operator-by-operator derivation from the actual H-Net forward path, beginning with a linear projection, the top-k router, PISA selection, and mHC transport. Each derivation should record its invariant directions, reachable-input support, discontinuity boundaries, measurable effect, and evaluation cost. The perturbation law should follow from those properties.

No theorem in this note guarantees improvement from 10,000 evaluations, coherent generated text, or a sub-second full BO round. Those claims require both a task-score model with predictive power and measured forward/generation/score/tell latency. The rigorous contribution here is narrower: it states which reductions follow from the operator mathematics, and identifies the exact assumptions a matrix-free Bayesian search must make before its uncertainty and trust region have their intended meaning.

## References and implementation anchors

1. BLCKBAUX semantic operations and tensors: [`rust/crates/blckbaux/src/forward_program/graph.rs`](../rust/crates/blckbaux/src/forward_program/graph.rs), `SemanticOp`; [`rust/crates/blckbaux/src/forward_program/model.rs`](../rust/crates/blckbaux/src/forward_program/model.rs), `ModelProgram::fbt_pisa` and its `route-margin` tensor.
2. BLCKBAUX mHC coefficient construction: [`rust/crates/blckbaux/src/fbt_moe.metal`](../rust/crates/blckbaux/src/fbt_moe.metal), `fbt_mhc_finish` and `fbt_mhc_predict`.
3. Wang and Wang, [*Gauge Fiber Bundle Geometry of Transformers*](https://proceedings.mlr.press/v282/wang26b.html), 2026. This establishes a quotient-geometric setting for transformers; BLCKBAUX-specific invariances remain to be proved.
4. Rolland et al., [*High-Dimensional Bayesian Optimization via Additive Models with Overlapping Groups*](https://proceedings.mlr.press/v84/rolland18a.html), 2018. Its additive assumption is a condition to check, not a property inherited by BLCKBAUX.
5. Gottlieb, Kontorovich, and Krauthgamer, [*Efficient Regression in Metric Spaces via Approximate Lipschitz Extension*](https://arxiv.org/abs/1111.4470), 2017 revision.
6. Kleinberg, Slivkins, and Upfal, [*Multi-Armed Bandits in Metric Spaces*](https://arxiv.org/abs/0809.4882), 2008. These metric-space results depend on smoothness/intrinsic-dimension assumptions.
7. BLCKBAUX GPU uncertainty rule: [`rust/crates/blckbaux/src/bf16_objectives.metal`](../rust/crates/blckbaux/src/bf16_objectives.metal), `objective_acquisition_score`.
8. BLCKBAUX threshold posterior: [`rust/crates/blckbaux/src/threshold.rs`](../rust/crates/blckbaux/src/threshold.rs), `ThresholdModel::draw_batch`.
9. BLCKBAUX parameter addresses: [`rust/crates/blckbaux/src/bf16_metal.rs`](../rust/crates/blckbaux/src/bf16_metal.rs), `ProgramAddress::new`.
10. BLCKBAUX proposal effects and changed-count: [`rust/crates/blckbaux/src/bf16_search.metal`](../rust/crates/blckbaux/src/bf16_search.metal), `bf16_propose`.
11. BLCKBAUX family distance path: [`rust/crates/blckbaux/src/bf16_metric.metal`](../rust/crates/blckbaux/src/bf16_metric.metal), `metric_distances` and `metric_scores`.
12. BLCKBAUX model-agreement region implementation: [`rust/crates/blckbaux/src/reliability_region.rs`](../rust/crates/blckbaux/src/reliability_region.rs); H-Net response reduction: [`rust/crates/blckbaux/src/bf16_metal/reliability.rs`](../rust/crates/blckbaux/src/bf16_metal/reliability.rs).
13. BLCKBAUX H-Net visit-aware initialization: [`rust/crates/blckbaux/src/fbt_moe/initialization.rs`](../rust/crates/blckbaux/src/fbt_moe/initialization.rs), `HnetMhc4Initialization::new`.
14. BLCKBAUX current H-Net experiment configuration: [`examples/tuning/hnet-1m-gaussian.toml`](../examples/tuning/hnet-1m-gaussian.toml), `[generation.reward]`.
15. Bandeira, Scheinberg, and Vicente, [*Convergence of Trust-Region Methods Based on Probabilistic Models*](https://arxiv.org/abs/1304.2808), 2014. The theorem assumes probabilistically fully-linear models of a deterministic smooth objective; those assumptions are not established here.
16. Chen, Menickelly, and Scheinberg, [*Stochastic Optimization Using a Trust-Region Method and Random Models*](https://arxiv.org/abs/1504.04231), 2018. Its convergence result requires sufficiently accurate random models and function estimates with sufficiently high probability.
17. Li, Balasubramanian, and Ma, [*Stochastic Zeroth-order Riemannian Derivative Estimation and Optimization*](https://arxiv.org/abs/2003.11238), 2020. Its intrinsic-dimension oracle bounds rely on a known Riemannian submanifold and smoothness; they do not transfer merely from measuring operator responses.
18. BLCKBAUX Lean 4 behavioral metric and Lipschitz bound: [`research/lean/BlckbauxResearch/Optimization/Metric.lean`](../research/lean/BlckbauxResearch/Optimization/Metric.lean), `Metric.lossDist`, `Metric.lossBound`, `Metric.tvDist`, `Metric.distribGauge`.
19. BLCKBAUX Lean 4 predictive uncertainty and epistemic decomposition: [`research/lean/BlckbauxResearch/Optimization/Surrogate.lean`](../research/lean/BlckbauxResearch/Optimization/Surrogate.lean), `Uncertainty.totalDecomp`, `Uncertainty.pureEpistemic`, `Uncertainty.pairedReduction`, `Uncertainty.infoGain`.
20. BLCKBAUX Lean 4 adaptive center trajectory drift: [`research/lean/BlckbauxResearch/Optimization/Trajectory.lean`](../research/lean/BlckbauxResearch/Optimization/Trajectory.lean), `Trajectory.distDrift`, `Trajectory.centerDrift`, `Trajectory.lossDecr`.
21. BLCKBAUX Lean 4 pretraining efficiency dominance over Muon: [`research/lean/BlckbauxResearch/Optimization/ResearchProblem.lean`](../research/lean/BlckbauxResearch/Optimization/ResearchProblem.lean), `Pretraining.beatRatio`, `Pretraining.memoryAdvantage`, `Pretraining.efficiencyDominance`.
22. BLCKBAUX Extended Research Catalogue: High-Dimensional Silicon Compiler Optimization (`museRISC` / OpenROAD physical flow on Nangate45): Standard-cell gate sizing and multi-$V_t$ assignment ($D \approx 10^5$), path-slack Wasserstein metric $W_1(\widehat{F}_{\text{slack}, 1}, \widehat{F}_{\text{slack}, 2})$, and epistemic uncertainty decoupling for physical design timing closure.
23. BLCKBAUX Lean 4 to LaTeX Metaprogramming Reflection Bridge (`LeanTeX`): [`research/lean/BlckbauxResearch/Export/Latex.lean`](../research/lean/BlckbauxResearch/Export/Latex.lean), `LatexData`, `latexPP`, `#latex`, `#latex_type` [Kyle Miller, 2024]. Direct functorial AST projection guarantees zero semantic drift between machine-checked kernel theorems and the formal mathematical catalogue.
