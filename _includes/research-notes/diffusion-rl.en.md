<p>This blog focuses on diffusion / flow-matching models: using visual generation as our setting, we examine policy optimization for these continuous-time generative models and explore efficient, stable training recipes (we use <strong>Diffusion RL</strong> as an umbrella term for this reward-based optimization). Depending on how reward signals become model updates, common approaches fall into three categories:</p>

<ol><li><strong>Reverse-process RL</strong>, such as <a href="https://arxiv.org/abs/2505.05470">Flow-GRPO</a>: treat denoising as a sequence of actions and use the final output's reward to guide policy updates for those actions.</li><li><strong>Forward-process RL</strong>, such as <a href="https://arxiv.org/abs/2509.16117">DiffusionNFT</a>: score generated outputs, add noise to them, and turn reward feedback into regression training along the forward process.</li><li><strong>Backpropagation through a reward model</strong>, such as <a href="https://arxiv.org/abs/2304.05977">ReFL</a>: propagate reward gradients through generation to update the model directly.</li></ol>

<p>The first two typically need only scores, not a differentiable reward model. The third needs gradients through the reward computation.</p>

<p><strong>We focus on the first two. Before beginning, here is a concise summary of the insights we hope to convey, with the aim of informing future academic research and practical applications:</strong></p>

<ul><li>Building a core mental model for RL in continuous generative processes:<ul><li>Unifying Diffusion RL algorithms conceptually: we describe MSE losses as "forces." Typically, one end of an MSE loss is a policy output and the other is a constructed target. Positive MSE pulls the policy output toward that target (attraction); negative MSE pushes it away (repulsion). Here, force is synonymous with gradient.</li><li>Understanding the relative scales of sample gradients, and gradient aggregation and cancellation;</li><li>Understanding that MSE loss, the loss form underlying RL in continuous-time generative processes, is only a gradient (force) provider: its target is not what we ultimately want to fit.</li><li>Analyzing Diffusion RL training behavior means analyzing the directions and magnitudes of action gradients at individual states, and what happens when they aggregate.</li></ul>

</li><li>Reverse denoising is a special Markov chain with properties we can exploit.<ul><li>These properties help explain why reverse-process RL is usually stable but inefficient, and why Flow-GRPO Fast can be much more efficient than Flow-GRPO by training fewer timesteps.</li></ul>

</li><li>The limitations of forward RL training: a hidden forward–reverse consistency term in the loss affects gradient quality, slows training and introduces instability.<ul><li>The fundamental difference between off-policy and on-policy training in Diffusion RL, and why off-policy strategies can stabilize training;</li><li>Why some forces are small in output space yet produce large aggregate gradients, while others are large in output space but yield small gradients after aggregation.</li></ul>

</li></ul>

<h2 id="en-target" data-section-key="target">1. A core mental model for RL in continuous generative processes</h2>

<p>Both forward and reverse RL can be understood through their underlying MSE push-and-pull terms. An MSE term connects a policy output (typically the denoising velocity field <span class="math">\(v_\theta\)</span>) to a detached target <span class="math">\(v_{\rm tgt}\)</span>. Averaging over D output elements, the loss with a fixed coefficient λ and its force are <span class="math" data-equation="force">\(\ell=\frac{\lambda}{D}\|v_\theta-v_{\rm tgt}\|^2\); \(f=-\nabla_{v_\theta}\ell=\frac{2\lambda}{D}(v_{\rm tgt}-v_\theta)\)</span>. λ &gt; 0 pulls the output toward the target; λ &lt; 0 pushes it away. We call <span class="math">\(f\)</span> the output-space force: this "force" is the loss's <strong>negative output gradient</strong>. Training can be understood as the aggregation of pushes and pulls at intermediate states.</p>

<p><strong>A target provides a gradient, not a destination.</strong> Flow matching already offers an example: multiple forward velocity fields can pass through the same noisy state. The MSE-optimal prediction is their conditional mean (<a href="https://arxiv.org/abs/2210.02747">Flow Matching</a>); it cannot reach every target at once. A post-training target need not be the velocity field we ultimately want either. Its role can simply be to provide a reward-improving "force," or gradient.</p>

> **Insight: useful changes do not require reaching the target.** To understand training, first ask what force the target supplies, rather than treating fitting it as the final task.

<h3 id="en-detail-1">1.1 From forces at individual states to a shared parameter resultant</h3>

<p>There are three layers: <strong>output force → per-sample parameter gradient → gradient aggregation across states</strong>. At a fixed model and fixed training states sᵢ (including prompt, latent, and timestep), let Jᵢ be the output Jacobian with respect to parameters, <span class="math">\(f_i\)</span> the output-space force, and gᵢ the parameter-space gradient:</p>

<div class="math" data-equation="chain">
\[
\begin{aligned}
J_i &= \frac{\partial v_\theta(s_i)}{\partial\theta},\\
f_i\;&\longrightarrow\;g_i=J_i^\top f_i
\;\longrightarrow\;\bar g=\frac{1}{N}\sum_{i=1}^{N}g_i.
\end{aligned}
\]
</div>

<p>N is the sample count; fixed sample weights are included in fᵢ. At the same model and states, aggregation of gradients <span class="math">\(g\)</span> can be understood as a <strong>vector sum. Likewise, we can use vector decomposition to break a force into components and analyze which components cancel and which do not when multiple forces aggregate.</strong> A large output force can map through the Jacobian to a small parameter direction. When sample parameter gradients aggregate, their norms cannot simply be added because their directions differ. Opposing gradients can cancel one another.</p>

<figure class="figure-compact" data-figure="force_overview"><img src="/assets/blog/diffusion-rl/force_overview.png" alt="Figure 1: Gradient mapping, aggregation and relative scales" loading="lazy" width="2400" height="864"><figcaption>Figure 1. Gradient mapping, aggregation and relative scales. Top: output forces map through Jacobians and aggregate across states. Bottom: common scaling preserves the resultant direction; relative weighting changes it. Here s = g₁ + g₂. The relative scales of component forces determine which component the resultant represents.</figcaption></figure>

<p></p>

<h3 id="en-detail-2">1.2 The optimizer does not erase relative scales</h3>

<p><strong>Optimizers and gradient clipping act on the aggregated resultant.</strong> Multiplying every component by the same positive number preserves the resultant's direction; enlarging just one component usually changes it. The commonly used AdamW optimizer is approximately insensitive to overall scale, so multiplying every component loss by the same constant does not significantly affect training. But the relative scales of components before gradient aggregation directly and significantly affect training.</p>

<details class="derivation">
<summary>Approximate scale invariance of Adam / AdamW</summary>

<p>Let hₙ be the aggregated loss gradient at optimizer step n. <a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html">AdamW</a> updates first and second moments, elementwise:</p>

<div class="math" data-equation="adamw">
\[
\begin{aligned}
m_n&=\beta_1m_{n-1}+(1-\beta_1)h_n,\\
q_n&=\beta_2q_{n-1}+(1-\beta_2)h_n^2,\\
\theta_{n+1}&=(1-\eta_n\lambda_w)\theta_n
-\eta_n\frac{\widehat m_n}{\sqrt{\widehat q_n}+\epsilon}.
\end{aligned}
\]
</div>

<p>Here <span class="math">\(\widehat m_n=m_n/(1-\beta_1^n)\)</span> and <span class="math">\(\widehat q_n=q_n/(1-\beta_2^n)\)</span> correct initialization bias; ηₙ is the learning rate and λ_w is decoupled weight decay. Scaling the entire gradient history by the same positive c scales the moments by c and c², so <span class="math">\(c/\sqrt{c^2}\)</span> cancels. This gives an intuitive explanation of approximate overall-scale invariance. But <span class="math">\(h_n\)</span> is already aggregated; AdamW cannot see the relative contributions of its component gradients.</p>

<p>Starting from zero moments, the recurrences give <span class="math">\(m'_n=c m_n\)</span> and <span class="math">\(q'_n=c^2q_n\)</span>; bias correction preserves these relations. The scaled update ratio is therefore <span class="math">\(\widehat m'_n/(\sqrt{\widehat q'_n}+\epsilon)=\widehat m_n/(\sqrt{\widehat q_n}+\epsilon/c)\)</span>. Cancellation is exact without ε and approximate with it.</p>

</details>

Thus, **equal participation in the loss does not mean equal influence on the update**. Opposing gradients cancel. Orthogonal gradients neither directly cancel nor reinforce one another. Many strong, unrelated directions can also dilute a useful direction's share of the resultant.

<blockquote><strong>Insight: not every reinforced action helps reward, and useful directions must survive in the resultant.</strong> To analyze Diffusion RL, use the mental model of vector decomposition and vector sums: ask how actions at each state are pushed or pulled, how these forces map to parameters, and which components dominate or cancel after aggregation. This is our common starting point for understanding Reverse RL and Forward RL.</blockquote>

<h2 id="en-reverse" data-section-key="reverse">2. Reverse RL: why might training fewer actions work better?</h2>

<h3 id="en-detail-3">2.1 Preliminary: Reverse-Process RL</h3>

**From deterministic denoising to actions with computable probabilities.** Throughout this article, t=1 is the noise end and t=0 the data end; denoising proceeds from 1 to 0. Consistent with our earlier convention of pulling toward a generation target, vθ denotes the generation-direction velocity, the negative of the noising-time velocity in <a href="https://arxiv.org/html/2505.05470v1#S4.SS2">Flow-GRPO</a>. Let r=1−t be increasing generation time. The original ODE is deterministic; Flow-GRPO adds stochastic exploration and corrects the drift:

<div class="math" data-equation="reverse-sde">
\[
\begin{aligned}
\mathrm d x_r^{\rm ODE}&=v_\theta(x_r,t,c)\,\mathrm d r,\\
\mathrm d x_r^{\rm SDE}&=\left[b_t v_\theta(x_r,t,c)-\frac{\sigma_t^2}{2t}x_r\right]\mathrm d r
+\sigma_t\,\mathrm d W_r,\\
b_t&=1+\frac{\sigma_t^2(1-t)}{2t}.
\end{aligned}
\]
</div>

<p>Here c is the prompt, Wᵣ is a Wiener process, and σₜ controls exploration. For an exact field, the continuous-time SDE preserves the ODE's marginals. The kth transition has noise time tₖ and step size <span class="math">\(\delta_k=t_k-t_{k+1}>0\)</span>. After Euler–Maruyama discretization, the action at state <span class="math">\(s_{i,k}=(c,t_k,x_{i,k})\)</span>, <span class="math">\(a_{i,k}=x_{i,k+1}\)</span>, has a Gaussian probability density:</p>

<div class="math" data-equation="reverse-transition">
\[
\begin{aligned}
a_{i,k}&=\mu_\theta(s_{i,k})+\sigma_{t_k}\sqrt{\delta_k}\,\xi_{i,k},
\quad \xi_{i,k}\sim\mathcal N(0,I),\\
\mu_\theta(s_{i,k})&=x_{i,k}+\delta_k
\left[b_{t_k}v_\theta(s_{i,k})-\frac{\sigma_{t_k}^2}{2t_k}x_{i,k}\right],\\
\pi_\theta(a_{i,k}\mid s_{i,k})&=\mathcal N\!\left(\mu_\theta(s_{i,k}),\sigma_{t_k}^2\delta_k I\right).
\end{aligned}
\]
</div>

<p>We consider stochastic transitions with <span class="math">\(0<t_k<1\)</span> and σₜ>0, not deterministic endpoints. Holding the sampled state and D-dimensional action fixed gives:</p>

<div class="math" data-equation="reverse-density">
\[
\log\pi_\theta(a_{i,k}\mid s_{i,k})
=-\frac{D}{2}\log(2\pi\sigma_{t_k}^2\delta_k)
-\frac{\|a_{i,k}-\mu_\theta(s_{i,k})\|^2}{2\sigma_{t_k}^2\delta_k}.
\]
</div>

**How does endpoint reward train each action?** Center and standardize endpoint rewards within each G-trajectory prompt group to obtain fixed advantages Aᵢ. G is the trajectory group size, k indexes timesteps, and m is the number of training timesteps. Density ratios give the clipped policy loss below; reference regularization is omitted here:

<div class="math" data-equation="reverse-ppo">
\[
\begin{aligned}
\rho_{i,k}(\theta)&=\frac{\pi_\theta(a_{i,k}\mid s_{i,k})}
{\pi_{\rm old}(a_{i,k}\mid s_{i,k})},\\
L_{\rm policy}&=-\frac{1}{Gm}\sum_{i=1}^{G}\sum_{k\in\mathcal T}
\min\!\left(\rho_{i,k}A_i,\operatorname{clip}(\rho_{i,k},1-\varepsilon,1+\varepsilon)A_i\right).
\end{aligned}
\]
</div>

**MSE exposes the force of each action.** At current=old with inactive clipping, the advantage-weighted negative log density has the same gradient as the policy loss. Substituting the Gaussian mean gives a fixed velocity target and a local MSE gradient provider:

<div class="math" data-equation="reverse-mse">
\[
\begin{aligned}
v^{\rm tgt}_{i,k}&=\frac{a_{i,k}-x_{i,k}
+\delta_k\sigma_{t_k}^2x_{i,k}/(2t_k)}{\delta_k b_{t_k}},\\
\ell^{\rm local}_{i,k}&=A_i w_k\|v_\theta(s_{i,k})-v^{\rm tgt}_{i,k}\|^2,
\quad w_k=\frac{\delta_k b_{t_k}^2}{2\sigma_{t_k}^2},\\
f_{i,k}&=-\nabla_{v_\theta}\ell^{\rm local}_{i,k}
=2A_iw_k(v^{\rm tgt}_{i,k}-v_\theta(s_{i,k})).
\end{aligned}
\]
</div>

<p>Positive advantages pull toward the velocity target corresponding to the sampled action; negative advantages push away. <strong>The relative force scale also includes wₖ, not just advantage.</strong></p>

<p id="en-detail-4" data-review-section><strong>Which actions actually control the update?</strong></p>

<p>The MSE above already offers a clue: even with the same Aᵢ along a trajectory, wₖ, target residuals and Jacobians differ across timesteps. According to the <a href="#eq-en-chain">gradient mapping and aggregation formula</a>, the final resultant depends on these component vectors. We directly measure which directions survive in it.</p>

<p>We measured the action-gradient norm at each training denoising timestep and its cosine with the resultant. In a microbatch probe experiment, the last position had a mean cosine of 0.948 with the resultant; the first was near zero. Late positions largely determined the resultant's direction.</p>

<figure id="en-reverse-gradients" class="figure-compact" data-figure="reverse_gradients"><img src="/assets/blog/diffusion-rl/reverse_gradients_compact.png" alt="Figure 2: Per-position gradient norms and alignment with the resultant" loading="lazy" width="2400" height="810"><figcaption>Figure 2. Per-position gradient norms and alignment with the resultant. Faint lines show 43 probe measurements; bold lines show their mean. These are gradients from the first microbatch, with cosine measured against its resultant. Positions 0 → 8 run from high to low noise.</figcaption></figure>

<p>Throughout this article, denoising proceeds from t=1 to t=0: the closer t is to 1, the higher the noise.</p>

<p>Adjacent positions had a mean cosine of just 0.003, indicating very weak average cooperation; we can approximately treat each pair as orthogonal. This agrees with local exploration in Reverse Process RL: randomness in each action within a trajectory is independent, so both cooperation and opposition among their gradients are weak. More importantly, early directions participated in training, but the resultant barely followed them.</p>

<blockquote><strong>Observation: coverage is not contribution.</strong> Training an action does not guarantee that its direction matters. In these probes, early actions participated in training, but late gradients completely dominated the resultant.</blockquote>

<p><strong>The denominator gives low-noise actions more relative weight.</strong> With Flow-GRPO's <span class="math">\(\sigma_t=\alpha\sqrt{t/(1-t)}\)</span> and fixed exploration coefficient α, the <a href="#eq-en-reverse-mse" data-equation-ref="reverse-mse">MSE weight</a> becomes:</p>

<div class="math" data-equation="reverse-weight">
\[
w_k=\frac{\delta_k(1+\alpha^2/2)^2}{2\alpha^2}\frac{1-t_k}{t_k}.
\]
</div>

For a fixed step size, smaller t means lower noise and larger wₖ. The variance denominator shrinks while its inverse weight grows: low-noise contributions are amplified before aggregation. The measured resultant's strong bias toward late actions is precisely the warning sign of this implicit weighting.

<h3 id="en-detail-5">2.3 Understanding denoising: a special Markov decision process</h3>

An almost-finished image may leave the next action little room to change its quality, yet that action receives the advantage of the entire trajectory.

**The training signal may reward what state an action inherited, rather than what the action contributed.**

<p><strong>What extra signal does group advantage carry?</strong> In a denoising MDP with only terminal reward, let sᵢ,ₖ and aᵢ,ₖ be trajectory i's state and action at step k, and Rᵢ its terminal reward. Using the standard MDP value functions <span class="math">\(V(s)\)</span> and <span class="math">\(Q(s,a)\)</span>, Qₖ(s,a) is the expected reward after taking the action and continuing under the rollout policy. Vₖ(s) averages Q over that policy's actions at the same state. The current action's advantage and the prompt-group signal are:</p>

<div class="math" data-equation="local-advantage">
\[
\begin{aligned}
A^{\rm local}_{i,k}&=Q_k(s_{i,k},a_{i,k})-V_k(s_{i,k}),\\
A^{\rm group}_i&=R_i-b_G(c).
\end{aligned}
\]
</div>

<p>Here <span class="math">\(c\)</span> is the prompt and <span class="math">\(b_G(c)\)</span> is its group's mean terminal reward. Standard RL uses <span class="math">\(A^{\rm local}\)</span>, the action's own advantage; Flow-GRPO instead uses <span class="math">\(A^{\rm group}\)</span>, based on the group mean. We analyze the centered quantity before division by the group STD. The difference decomposes exactly as:</p>

<div class="math" data-equation="credit">
\[
\begin{aligned}
A^{\rm group}_i-A^{\rm local}_{i,k}
&=\underbrace{V_k(s_{i,k})-b_G(c)}_{\text{State-baseline mismatch}}\\
&\quad+\underbrace{R_i-Q_k(s_{i,k},a_{i,k})}_{\text{Monte Carlo noise}}.
\end{aligned}
\]
</div>

<p>Fix the G states under one prompt, <span class="math">\(\mathcal S_k=(s_{1,k},\ldots,s_{G,k})\)</span>, and average over actions and continuations under the same rollout policy. The noise term has zero mean; the expected baseline is <span class="math">\(\overline V_k=G^{-1}\sum_j V_k(s_{j,k})\)</span>:</p>

<div class="math" data-equation="credit-expectation">
\[
\begin{aligned}
\mathbb E[A^{\rm group}_i-A^{\rm local}_{i,k}\mid\mathcal S_k,c]
&=\underbrace{V_k(s_{i,k})-\overline V_k}_{\text{State-value mismatch}}+\underbrace{0}_{\text{Mean Monte Carlo noise}}\\
&=\frac1G\sum_{j=1}^G\bigl[V_k(s_{i,k})-V_k(s_{j,k})\bigr].
\end{aligned}
\]
</div>

What remains is the current state's deviation from the group mean value. These deviations average to zero, but their mean square equals the state-value variance:

<div class="math" data-equation="credit-variance">
\[
\begin{aligned}
\frac1G\sum_{i=1}^G
\left(\mathbb E[A^{\rm group}_i-A^{\rm local}_{i,k}\mid\mathcal S_k,c]\right)^2
&=\frac1G\sum_{i=1}^G\bigl[V_k(s_{i,k})-\overline V_k\bigr]^2\\
&=\operatorname{Var}_{i\sim\mathrm{Unif}(1{:}G)}\bigl[V_k(s_{i,k})\bigr].
\end{aligned}
\]
</div>

**The greater the value differences trajectories already carry, the larger the group baseline's mismatch with individual states.**

**Experiment: fix a state, then see how much actions can change.** At a chosen timestep, retain P prefixes under the same prompt, each fixing a state. Sample and score M branches per prefix. Branches sharing a prefix form one group; branching uses CPS, with ODE elsewhere. Write terminal rewards as Rₚ,ⱼ⁽ᵏ⁾.

- **Within-group spread**: compute each prefix's reward STD, then average, measuring how much can change after fixing a state.
- **Between-group spread**: compute each prefix's mean reward, then take their STD, measuring value differences already carried by states.

<div class="math" data-equation="prefix-statistics">
\[
\begin{aligned}
\widehat V_{p,k}&=\frac1M\sum_{j=1}^M R_{p,j}^{(k)},\\
\widehat s_{\rm within}(k)&=\frac1P\sum_{p=1}^P
\operatorname{STD}_{j}\bigl(R_{p,j}^{(k)}\bigr),\\
\widehat s_{\rm between}(k)&=\operatorname{STD}_{p}\bigl(\widehat V_{p,k}\bigr).
\end{aligned}
\]
</div>

Within / between STD is **0.4543 / 0.3651** early and **0.1709 / 0.5179** late. Their relative sizes reverse.

<figure id="en-shared-noise" class="figure-compact" data-figure="prefix_spread"><img src="/assets/blog/diffusion-rl/prefix_spread_compact.png" alt="Figure 3: Prefix reward spread and learning curves with shared initial noise" loading="lazy" width="2400" height="990"><figcaption>Figure 3. Left: mean within-prefix reward STD versus STD of prefix mean rewards; Early / Late branch at steps 1 / 8. Right: shared versus independent initial noise within each group, training all timesteps with the original SDE and variance denominator. The x-axis is logged step; all archived evaluation points and actual endpoints are retained, without smoothing.</figcaption></figure>

<p>Each prefix's mean reward estimates its V under the probe policy. Squared between-group STD thus estimates the <a href="#eq-en-credit-variance" data-equation-ref="credit-variance">state-value variance</a>, with finite-branch estimation noise. The results support <strong>larger value differences late in denoising, and hence greater mismatch when one group mean replaces each state's V.</strong></p>

**Early, high-noise actions retain greater reward leverage; late, low-noise results depend more on inherited states.** Yet reverse RL reinforces actions. It can therefore attribute inherited state quality to the current action. This experiment measures endpoint differences after branching, not exact per-action Q variance.

<p><strong>What happens when we use a state baseline?</strong> We retained Flow-GRPO's PPO clipped objective, used a critic to predict each state's value, and based advantage estimates on <span class="math">\(R_i-\widehat V_k(s_{i,k})\)</span>. This historical recipe improved both train and eval reward faster early on. At logged step 240, eval reward was <strong>0.664</strong>, compared with <strong>0.527</strong> for the group-advantage control.</p>

<figure class="figure-compact" data-figure="critic_advantage"><img src="/assets/blog/diffusion-rl/critic_advantage.png" alt="Figure 3b: Learning curves with group and critic advantages" loading="lazy" width="2400" height="900"><figcaption>Figure 3b. Group advantage versus PPO-style critic advantage. Both use SD3.5-M, GenEval, K=24, a 10-step SDE rollout and LR=3e-4, with external CFG off and KL=0. The critic recipe disables initial-noise selection and uses centering/STD normalization, a 5-epoch warmup, a 15-epoch ramp and a ranking loss. Left: train. Right: eval. The x-axis is logged step; returned histories retain their actual endpoints without smoothing.</figcaption></figure>

The critic costs extra training and has estimation error. We used a very simple critic without tuning it. These results support a practical direction: **rather than assigning every action the same trajectory-level verdict, estimate the value of the state it inherited, then assess its incremental contribution.**

<p id="en-noise-time-clue" class="structural-clue">LLMs also have crucial decisions and inconsequential actions, but token position alone usually does not identify the crucial ones. Diffusion supplies an additional structural clue: noise time t indicates how far the sample has formed. We can use it to look for training positions more likely to carry strong action credit.</p>

Connecting this to the preceding section exposes the tension: **the actions that dominate updates need not be the actions that can effectively change reward, or may have worse credit assignment.** If late signals carry more inherited state quality, yet implicit loss weighting amplifies them, the resultant may favor less informative directions. That makes high-noise training windows worth testing: they reduce backward computation and give directions with stronger action credit more influence.

> **Insight: high reward is not high action credit.** Prompt-group advantage mixes the quality of the inherited state with the improvement made by the current action. Late in denoising, the former can grow stronger while the latter weakens; Flow-GRPO's low-noise weighting can also let these low-credit actions dominate updates. Conversely, diffusion's noise time supplies a structural clue: a high-noise training window can reduce computation while giving actions with greater reward leverage more influence over the aggregate gradient.

Diffusion RL differs from LLM RL in another way: initial noise supplies different starting-state values before the first action. Our experiments found more pronounced value differences across initial noises in video generation. By the <a href="#eq-en-credit-variance" data-equation-ref="credit-variance">state-value variance formula</a>, this increases advantage estimation error and weakens credit assignment. Sharing one initial noise within a prompt group is one solution. <a href="#en-shared-noise">Figure 3, right</a> shows this comparison: at logged step 240, shared initial noise reaches an eval score of <strong>0.656</strong> versus <strong>0.527</strong> for independent noise, supporting controlled starting points when comparing action contributions.

<h3 id="en-detail-6">2.4 The training location need not be where benefits appear</h3>

We train LoRA only at the first step, but the same parameters can act at other timesteps. At inference, enabling or disabling LoRA at each step controls where the learned function change takes effect. This time, we directly disable the trained position and compare:

- **All off**: the original model, used as the paired baseline.
- **Steps 2–8 only**: the first step is disabled; only untrained positions use LoRA.
- **Step 1 only, step 8 only, and all on**: distinguish the trained position, a single untrained position, and the combined effect of multiple steps.

<figure class="figure-compact" data-figure="gate_transfer"><img src="/assets/blog/diffusion-rl/gate_matrix_ocr_updated.png" alt="Figure 4: Train only step 1 or step 8 and measure OCR gains from seven untrained inference positions" loading="lazy" width="4800" height="1860"><figcaption>Figure 4. LoRA timestep interventions after 120 rollouts. Left: train step 1 only; the seven untrained positions are steps 2–8. Right: train step 8 only; the seven untrained positions are steps 1–7. The y-axis is the OCR gain over disabling LoRA everywhere; error bars show one paired standard error over 64 fixed prompts. Both all-off baselines are 0.064.</figcaption></figure>

After training only step 1, disabling it at inference and enabling only steps 2–8 still improves the OCR score from **0.064 to 0.877**. Enabling only step 1 reaches just **0.296**. The trained position need not participate in inference for the gain to appear: **parameter changes learned at the first step can be realized by untrained later steps.**

> **Insight: selecting training timesteps selects gradient sources, not where gains must appear.** Informative action signals can take effect at other timesteps through shared parameters.

Conversely, training only step 8 and enabling only step 1 gives **0.100**: the gain is small, but not demonstrably absent. With reciprocal single-step gates, training step 1 and enabling step 8 gains **0.024 ± 0.016**, while training step 8 and enabling step 1 gains **0.036 ± 0.020**. The clearest conclusion is therefore **that gains can appear at untrained positions**, not that transfer must be stronger from early to late. Enabling seven later positions versus one early position also cannot establish a directional advantage.

<h3 id="en-detail-7">2.5 Understanding Flow-GRPO Fast: training only high-noise regions can greatly improve efficiency</h3>

The tension is now clear: low-noise actions can inherit reward already carried by their states, yet dominate the update. **Flow-GRPO Fast randomly selects a short contiguous window within a fixed high-noise region, reserving backward computation for actions more likely to change the outcome.** The window can move between iterations; the whole high-noise region need not be trained each time.

<p>A complete trajectory has N transitions. Fix the high-noise candidate region <span class="math">\(\mathcal H_\tau=\{k:\tau\leq t_k&lt;1\}\)</span>, where τ&gt;0 is the noise-time threshold; include only positions with a valid stochastic transition density as above. Choose a window length m and uniformly sample its start j from those that keep the entire window inside this region. Explore stochastically inside the window and generate deterministically outside it. Keep the group advantage and PPO ratio above, training only on the selected short window:</p>

<div class="math" data-equation="fast-window">
\[
\begin{aligned}
\mathcal S_\tau&=\{j:\{j,\ldots,j+m-1\}\subseteq\mathcal H_\tau\},\\
j&\sim\mathrm{Unif}(\mathcal S_\tau),\qquad
\mathcal W_j=\{j,\ldots,j+m-1\},\\
L_{\rm Fast}&=-\frac{1}{Gm}\sum_{i=1}^{G}\sum_{k\in\mathcal W_j}
\min\!\left(\rho_{i,k}A_i,
\operatorname{clip}(\rho_{i,k},1-\varepsilon,1+\varepsilon)A_i\right).
\end{aligned}
\]
</div>

**Pseudocode: the high-noise training core of Flow-GRPO Fast**

```text
Inputs: rollout sampler, noise-time threshold τ, window length m, group size G
Each round:
  Freeze the old policy and identify the fixed high-noise candidate region
  Uniformly sample a valid start j and select m consecutive transitions
  Generate G complete trajectories per prompt with the old policy
    Explore stochastically inside the window; generate deterministically outside
  Score endpoints and compute group advantages
  Cache state, action and old log-prob only for transitions inside the short window
  Draw microbatches from the window; compute current/old PPO ratios
  Backpropagate the clipped PPO loss; update with the chosen accumulation
```

**Timestep selection is hard weighting; timestep reweighting is soft weighting. Both allocate influence over the resultant.** Hard weighting sets contributions outside the window to zero and skips their training forward/backward passes, reducing training samples per trajectory from N to m. The LoRA experiment in Section 2.4 also shows that changes learned at high-noise positions can benefit low-noise predictions: every position need not be trained to benefit.

<p><strong>Within the selected window, we can further adjust timestep weighting.</strong> Reuse the sampling equation above: σₜₖ controls exploration, δₖ is the step size, and ξᵢ,ₖ is standard Gaussian noise. The sampled action's residual from μ_old is therefore <span class="math">\(\sigma_{t_k}\sqrt{\delta_k}\,\xi_{i,k}\)</span>.</p>

<p>Differentiating the MSE gives this residual. Mapping the transition mean back to velocity multiplies it by <span class="math">\(\delta_k b_{t_k}\)</span>. Including advantage Aᵢ and the chosen denominator gives the output force at current=old:</p>

<div class="math" data-equation="score">
\[
\begin{aligned}
f_{i,k}&=A_i\delta_k b_{t_k}\xi_{i,k}\times
\begin{cases}
\sigma_{t_k}\sqrt{\delta_k}, & \text{no denominator},\\
1, & \text{STD denominator},\\
1/(\sigma_{t_k}\sqrt{\delta_k}), & \text{variance denominator}.
\end{cases}
\end{aligned}
\]
</div>

<p>bₜ is the velocity coefficient in the SDE above, constant under our noise schedule. At fixed step size, σₜₖ increases with noise time t. Thus:</p>

- **Remove the denominator: force scales with σₜₖ, emphasizing high noise.**
- **Divide by standard deviation to the first power: remove the noise-scale preference, equalizing training positions on this scale.**
- **Keep the variance denominator: force scales with 1/σₜₖ, emphasizing low noise.**

<figure class="figure-compact" data-figure="reverse_learning"><img src="/assets/blog/diffusion-rl/reverse_learning_compact.png" alt="Figure 5: Denominator intervention and Flow-GRPO Fast versus Naive Flow-GRPO" loading="lazy" width="2400" height="930"><figcaption>Figure 5. Left: training at all timesteps, removing the variance denominator improves reward earlier. Right: Flow-GRPO Fast's early 3-step window versus Naive Flow-GRPO. The x-axis is logged step; the y-axis is GenEval score. Archived evaluation points are shown without smoothing.</figcaption></figure>

> **Insight: give actions with stronger causal influence on reward a larger share of the total gradient.** Flow-GRPO Fast concentrates these useful signals by training only high-noise actions. Shared parameter updates then carry them to untrained states, benefiting low-noise velocity predictions as well.

<blockquote class="training-recipe">
<p><strong>Recipe: Reverse Process RL</strong></p>
<ol>
<li>Randomly select a short contiguous window within a fixed high-noise region; start with a window length of 1 or 2 to reduce dilution by low-action-credit signals.</li>
<li>Choose timestep weighting for the task: emphasize high noise, or equalize training positions on the noise scale.</li>
<li>Share initial noise within each prompt group to reduce initial-condition differences in action credit assignment.</li>
</ol>
</blockquote>

<h2 id="en-forward" data-section-key="forward">3. Understanding Forward Process RL</h2>

<p>Diffusion RL, as the name suggests, combines Diffusion and RL. Reverse RL reinforces the model's own actions at states it explores, aligning closely with the spirit of traditional RL; I call it the "RL school." Forward RL, in my view, is closer in spirit to Diffusion; I call it the "Diffusion school." It scores generated endpoints, re-noises them, and turns feedback into regression at noisy states, <strong>following an idea similar to Diffusion pretraining</strong>. Compared with Reverse RL, Forward RL requires, or enjoys, more elegant mathematical properties. In practice, that elegance often brings greater challenges. This section uses plain language to examine its motivation and essence, the practical limitations I see, and recipes for overcoming them.</p>

<h3 id="en-detail-8">3.1 Preliminary: Forward Process RL</h3>

<p>First, freeze the rollout policy and write its reverse-generated endpoint distribution as <span class="math">\(p_{\rm old}(x_0\mid c)\)</span>, where c is the prompt. Keep our generation-direction velocity convention, with t decreasing from 1 to 0. The straight noising path and its sample target are:</p>

<div class="math" data-equation="forward-path">
\[
x_t=(1-t)x_0+t\epsilon,\qquad u=x_0-\epsilon.
\]
</div>

x₀ is the generated endpoint and ε is resampled noise. These noisy states come from re-noising endpoints, **not from the original reverse rollout**.

<p>Map reward to a fixed goodness function <span class="math">\(r(x_0,c)\in[0,1]\)</span> and <strong>use it to reweight the endpoint distribution</strong>. Write <span class="math">\(\bar r_c=\mathbb E_{p_{\rm old}}[r\mid c]\)</span>. For <span class="math">\(0&lt;\bar r_c&lt;1\)</span>, define:</p>

<div class="math" data-equation="nft-endpoint-tilt">
\[
\begin{aligned}
p_+(x_0\mid c)&\propto r(x_0,c)\,p_{\rm old}(x_0\mid c)
\ \Longrightarrow\ p_+(x_0\mid c)=\frac{r(x_0,c)}{\bar r_c}\,p_{\rm old}(x_0\mid c),\\
p_-(x_0\mid c)&\propto [1-r(x_0,c)]\,p_{\rm old}(x_0\mid c)
\ \Longrightarrow\ p_-(x_0\mid c)=\frac{1-r(x_0,c)}{1-\bar r_c}\,p_{\rm old}(x_0\mid c).
\end{aligned}
\]
</div>

<details class="derivation" id="en-forward-normalization-proof">
<summary>Derivation: why is the denominator the mean goodness?</summary>
<p>A probability density must integrate to 1, so reweighting requires division by the total weight:</p>
<div class="math" data-equation="nft-endpoint-normalization">
\[
\begin{aligned}
\int r(x_0,c)\,p_{\rm old}(x_0\mid c)\,dx_0
&=\mathbb E_{p_{\rm old}}[r\mid c]=\bar r_c,\\
\int [1-r(x_0,c)]\,p_{\rm old}(x_0\mid c)\,dx_0
&=1-\bar r_c.
\end{aligned}
\]
</div>
<p>Dividing by these normalization constants gives the positive and negative distributions above.</p>
</details>

<p>We now have three distributions: positive, negative and original. The first two are defined distributions: they exist, but we do not yet have a policy that generates samples from them. We will address this below. Intuitively, the positive distribution favors high-scoring endpoints and the negative distribution favors low-scoring ones.</p>

<p>As in pretraining, apply Flow Matching's straight-line noising to each of the three endpoint distributions: sample x₀ from <span class="math">\(p_b\)</span>, independently sample timestep t and noise ε, and construct <span class="math">\(x_t=(1-t)x_0+t\epsilon\)</span>, with sample velocity target <span class="math">\(u=x_0-\epsilon\)</span>. Here <span class="math">\(b\in\{+,\mathrm{old},-\}\)</span>; only the source of x₀ differs across the three processes, while all other rules are the same. <strong>The joint distribution of x₀, ε, t and <span class="math">\(x_t\)</span> produced by this process is denoted <span class="math">\(Q_b\)</span></strong>, with prompt c given. Fixing the noisy state <span class="math">\(s=(x_t,t,c)\)</span> still leaves different x₀–ε pairs that could produce it. Flow Matching theory gives the ideal generation-direction field <span class="math" data-equation="nft-ideal-fields">\(v_{\rm fwd}^{b}(s)\)</span> as the conditional mean of these pairs' targets u. Intuitively, it is the field a model would ideally learn through standard Flow Matching pretraining on samples from <span class="math">\(p_b\)</span>. <span class="reader-intuition">By now, you may have a strong intuition: just use MSE regression to pull the model's velocity field toward <span class="math">\(v_{\rm fwd}^{+}\)</span>, moving <span class="math">\(p_{\rm old}\)</span> toward <span class="math">\(p_+\)</span> so that high-reward endpoints become more probable. Then why do we need the negative distribution <span class="math">\(p_-\)</span>? That intuition is right, but the key is that the field we can actually train is the model's reverse-denoising field, <span class="math">\(v_{\rm rvs}^{\rm old}(s)\)</span>. <span class="math">\(v_{\rm rvs}^{\rm old}(s)\)</span> generally differs from the ideal forward-noising field <span class="math">\(v_{\rm fwd}^{\rm old}\)</span>. This may feel confusing; keep that intuition and question in mind as you read on.</span> For now, let us agree that the negative distribution is somehow helpful, and finish building the mental model of Forward Process RL:</p>

<p>Write <span class="math">\(q(s)=\mathbb E_{Q_{\rm old}}[r\mid s]\)</span> and <span class="math">\(g(s)=\operatorname{Cov}_{Q_{\rm old}}(r,u\mid s)\)</span>. When conditional moments exist and <span class="math">\(0&lt;q(s)&lt;1\)</span>, the three ideal fields satisfy:</p>

<div class="math key-equation" data-equation="nft-ideal-collinearity">
\[
\begin{aligned}
q(s)\,[v_{\rm fwd}^{+}(s)-v_{\rm fwd}^{\rm old}(s)]
&=g(s),\\
[1-q(s)]\,[v_{\rm fwd}^{\rm old}(s)-v_{\rm fwd}^{-}(s)]
&=g(s).
\end{aligned}
\]
</div>

<details class="derivation" id="en-forward-collinearity-proof">
<summary>Derivation: why does reward reweighting produce three collinear ideal fields?</summary>
<p>The noising rule and noise distribution are unchanged, so the positive joint distribution is still weighted by <span class="math">\(r/\bar r_c\)</span> relative to the old one. Conditioning on s, Bayes' rule divides by the conditional mean weight <span class="math">\(q(s)/\bar r_c\)</span>, cancelling the prompt-level constant. The negative distribution follows similarly. Below, omit the s arguments of q, g and the fields; all expectations are under the old distribution:</p>
<div class="math" data-equation="nft-conditional-tilt">
\[
\begin{aligned}
Q_+(\cdot\mid s)&=\frac{r/\bar r_c}{q/\bar r_c}\,Q_{\rm old}(\cdot\mid s)
=\frac rq\,Q_{\rm old}(\cdot\mid s),\\
Q_-(\cdot\mid s)&=\frac{(1-r)/(1-\bar r_c)}{(1-q)/(1-\bar r_c)}\,Q_{\rm old}(\cdot\mid s)
=\frac{1-r}{1-q}\,Q_{\rm old}(\cdot\mid s),\\
q v_{\rm fwd}^{+}&=\mathbb E[ru\mid s],\qquad
(1-q)v_{\rm fwd}^{-}=\mathbb E[(1-r)u\mid s],\\
q(v_{\rm fwd}^{+}-v_{\rm fwd}^{\rm old})
&=\mathbb E[ru\mid s]-q\mathbb E[u\mid s]=g,\\
(1-q)(v_{\rm fwd}^{\rm old}-v_{\rm fwd}^{-})
&=(1-q)\mathbb E[u\mid s]-\mathbb E[(1-r)u\mid s]\\
&=\mathbb E[ru\mid s]-q\mathbb E[u\mid s]=g.
\end{aligned}
\]
</div>
<p>This also identifies what probability weighting estimates: the conditional mean of ru is <span class="math">\(qv_{\rm fwd}^{+}\)</span>, not the positive field itself. Recovering the field requires division by q; a ratio of two finite-sample means is not automatically unbiased.</p>
</details>

<p><strong>This is the central formula of Forward RL. Intuitively, at any state reachable by forward noising, the ideal forward-noising fields induced by the three distributions are collinear. From the original ideal field, pulling toward the positive field and pushing away from the negative field follow the same reward-improving direction.</strong> They differ only by positive scale factors, so ideally the two have the same effect (recall that <a href="#en-target">an MSE target is only a gradient provider</a>: the pull and push induce gradients in the same direction, making their training dynamics nearly equivalent apart from scale).</p>

<p>How do we train? The idea is exactly the same as in pretraining: <em>suppose we can sample from both distributions, re-noise the samples to construct an unbiased estimate u of the forward field, and MSE-regress the training object onto that estimate until it converges to the ideal forward field</em> (recall the original Flow Matching pretraining algorithm). The question is how to sample from these distributions. We cannot do so directly: we do not have an expression for the original density; all we can do is call an ODE solver to sample from it. This becomes a Stochastic Simulation problem. Take fitting the positive distribution as an example:</p>

<p>Fix prompt c, but not xₜ. For each endpoint, sample t from the specified training distribution and independently sample <span class="math">\(\epsilon\sim\mathcal N(0,I)\)</span>, constructing <span class="math">\(x_t=(1-t)x_0+t\epsilon\)</span>. Write <span class="math">\(h_\theta(x_0)=\mathbb E_{t,\epsilon}\!\left[\frac12\|v_\theta(x_t,t,c)-(x_0-\epsilon)\|^2\right]\)</span>: the mean MSE after re-noising this endpoint, with fixed c implicit. The problem is:</p>

<div class="math problem-formulation" data-equation="forward-regression-goal">
\[
\begin{aligned}
\text{Goal:}\quad&
L_+(\theta\mid c)=
\mathbb E_{x_0\sim p_+=(r/\bar r_c)p_{\rm old}}
[h_\theta(x_0)],\\
\text{What we can do:}\quad&\text{sample }x_0\sim p_{\rm old}(\cdot\mid c).
\end{aligned}
\]
</div>

<p><strong>Solution: transform L₊ algebraically.</strong> Treat h as one object, substitute the positive-distribution definition, then pull out the normalization constant independent of x₀:</p>

<div class="math" data-equation="forward-weighted-estimator">
\[
\begin{aligned}
L_+(\theta\mid c)
&=\mathbb E_{x_0\sim p_+(\cdot\mid c)}[h_\theta(x_0)]\\
&=\int h_\theta(x_0)\,p_+(x_0\mid c)\,dx_0\\
&=\int h_\theta(x_0)\,\frac{r(x_0,c)}{\bar r_c}
             \,p_{\rm old}(x_0\mid c)\,dx_0\\
&=\frac1{\bar r_c}\int r(x_0,c)\,h_\theta(x_0)
             \,p_{\rm old}(x_0\mid c)\,dx_0\\
&=\frac1{\bar r_c}\mathbb E_{x_0\sim p_{\rm old}(\cdot\mid c)}
        [r(x_0,c)\,h_\theta(x_0)].
\end{aligned}
\]
</div>

<p>Sample x₀ from the old model, resample t and ε, and average r×MSE to obtain an unbiased estimate of the unnormalized expectation above. For a fixed prompt, <span class="math">\(\bar r_c\)</span> depends on the frozen old distribution and fixed r mapping, not θ. Thus r-weighted MSE differs from L₊ by a positive constant: their gradients have the same direction, differing only in scale. (Alternatively, use rejection sampling: treat r as an <strong>acceptance probability</strong>, sampling from the old distribution and accepting with probability r. Accepted samples follow the positive distribution; re-noise them for training. For the negative distribution, replace r with 1−r.)</p>

<p><strong>How does <a href="https://arxiv.org/html/2509.16117v1#S3">DiffusionNFT</a> implement Forward RL?</strong> Each iteration uses a frozen old policy to generate K endpoints per prompt and score them. Instead of using raw rewards directly, it subtracts the group mean, rescales, clips, and maps them to [0,1] as each sample's within-group "goodness": <span class="math">\(r_i=\frac12[1+\operatorname{clip}((R_i-\bar R_c)/Z_c,-1,1)]\)</span>. Here <span class="math">\(\bar R_c\)</span> is the group mean and <span class="math">\(Z_c&gt;0\)</span> controls scale. Above-average samples have r&gt;0.5; below-average samples have r&lt;0.5. The positive and negative branches then receive weights r and 1−r.</p>

<p>For training, re-noise each generated x₀ along a straight path to obtain xₜ and the Flow Matching target <span class="math">\(u=x_0-\epsilon\)</span>. Evaluate the frozen <span class="math">\(v_{\rm old}\)</span> and trainable <span class="math">\(v_\theta\)</span> at the same state. NFT does not train two separate models: it uses the same current output to construct two mirror branches around the old output:</p>

<div class="math" data-equation="mirror-branches">
\[
\begin{aligned}
v_\pm(x_t,t,c)
&=v_{\rm old}(x_t,t,c)\\
&\quad\pm\beta\bigl[v_\theta(x_t,t,c)-v_{\rm old}(x_t,t,c)\bigr].
\end{aligned}
\]
</div>

<p>β&gt;0 is a hyperparameter controlling the magnitude. Here we interpolate <span class="math">\(v_{\rm old}\)</span> and <span class="math">\(v_\theta\)</span>, then regress toward <span class="math">\(u\)</span>. The loss is:</p>

<div class="math" data-equation="nft-objective">
\[
\begin{aligned}
L_{\rm NFT}(\theta\mid c)
&=\mathbb E_{\substack{x_0\sim p_{\rm old}(\cdot\mid c)\\
t\sim\rho,\;\epsilon\sim\mathcal N(0,I)}}\!\Bigl[
r\,\|v_+(x_t,t,c)-(x_0-\epsilon)\|^2\\
&\hspace{5em}+(1-r)\,\|v_-(x_t,t,c)-(x_0-\epsilon)\|^2\Bigr].
\end{aligned}
\]
</div>

<p>After optimizing this batch, NFT updates old through EMA: <span class="math">\(\theta_{\rm old}\leftarrow\eta\theta_{\rm old}+(1-\eta)\theta\)</span>. This policy generates the next batch and supplies the next iteration's anchor. Larger η makes old follow current more slowly.</p>

<p id="en-nft-questions"><strong>DiffusionNFT also reports experimental findings without a mechanistic explanation:</strong></p>
<ul>
<li>Reward collapses if <span class="math">\(v_\theta=v_{\rm old}\)</span>.</li>
<li>Reward also collapses without the negative-branch loss (the second term).</li>
</ul>
<p>Recall the mental model we built in <a href="#eq-en-forward-regression-goal" data-equation-ref="forward-regression-goal">Equation 21</a> and <a href="#eq-en-forward-weighted-estimator" data-equation-ref="forward-weighted-estimator">Equation 22</a>. This construction is confusing and naturally raises questions:</p>
<ol>
<li id="en-nft-question-mirror">What does the roundabout mirror construction ask current to learn? Following our earlier derivation, shouldn't we fit <span class="math">\(v_\theta\)</span> to u? Why fit an interpolation of <span class="math">\(v_\theta\)</span> and <span class="math">\(v_{\rm old}\)</span> to u instead?</li>
<li id="en-nft-question-negative">Why do we also need the negative branch?</li>
<li id="en-nft-question-goodness">Why normalize reward into goodness, when the earlier derivation used raw reward?</li>
</ol>

<h3 id="en-nft-demystifying">3.2 Demystifying Forward Process RL and NFT-style training</h3>

<h4 id="en-nft-objective-equivalence">3.2.1 Equivalent forms of the DiffusionNFT objective</h4>

<p>Let <span class="math">\(a=(2r-1)/\beta\)</span> and define the frozen target <span class="math">\(v_{\rm tgt}(x_0,t,\epsilon,c)=a(x_0-\epsilon)+(1-a)v_{\rm old}(x_t,t,c)\)</span>. Expanding the squares and completing the square gives:</p>

<div class="math key-equation" data-equation="nft-frozen-target">
\[
\begin{aligned}
L_{\rm NFT}(\theta\mid c)
&=\beta^2\,
\mathbb E_{\substack{x_0\sim p_{\rm old}(\cdot\mid c)\\
t\sim\rho,\;\epsilon\sim\mathcal N(0,I)}}\!
\left[\|v_\theta(x_t,t,c)-v_{\rm tgt}(x_0,t,\epsilon,c)\|^2\right]
+\mathrm{const}_1\\
&=\beta^2\,
\mathbb E_{\substack{x_0\sim p_{\rm old}(\cdot\mid c)\\
t\sim\rho,\;\epsilon\sim\mathcal N(0,I)}}\!\Bigl[
a\,\|v_\theta(x_t,t,c)-(x_0-\epsilon)\|^2\\
&\hspace{5em}+(1-a)\,\|v_\theta(x_t,t,c)-v_{\rm old}(x_t,t,c)\|^2\Bigr]
+\mathrm{const}_2.
\end{aligned}
\]
</div>

<details class="derivation" id="en-nft-target-proof">
<summary>Derivation: how do two mirror branches become one target?</summary>
<p>Fix prompt c and one sampled <span class="math">\((x_0,t,\epsilon)\)</span>, where <span class="math">\(x_0\sim p_{\rm old}(\cdot\mid c)\)</span>, <span class="math">\(t\sim\rho\)</span>, and <span class="math">\(\epsilon\sim\mathcal N(0,I)\)</span>, with <span class="math">\(x_t=(1-t)x_0+t\epsilon\)</span>. Write <span class="math">\(d=v_\theta(x_t,t,c)-v_{\rm old}(x_t,t,c)\)</span> and <span class="math">\(e=(x_0-\epsilon)-v_{\rm old}(x_t,t,c)\)</span>. The symbols d and e only shorten the algebra below; r and the old output remain frozen. The branch residuals are βd−e and −βd−e. Expand the squares, then complete the square using <span class="math">\(a=(2r-1)/\beta\)</span>:</p>
<div class="math" data-equation="nft-square-completion">
\[
\begin{aligned}
\ell_{\rm NFT}(\theta;x_0,t,\epsilon,c)
&=r\|\beta d-e\|^2+(1-r)\|-\beta d-e\|^2\\
&=r\bigl(\beta^2\|d\|^2-2\beta\langle d,e\rangle+\|e\|^2\bigr)\\
&\quad+(1-r)\bigl(\beta^2\|d\|^2+2\beta\langle d,e\rangle+\|e\|^2\bigr)\\
&=\beta^2\|d\|^2-2\beta(2r-1)\langle d,e\rangle+\|e\|^2\\
&=\beta^2\bigl(\|d\|^2-2a\langle d,e\rangle+a^2\|e\|^2\bigr)
+(1-\beta^2a^2)\|e\|^2\\
&=\beta^2\|d-ae\|^2
+\underbrace{\bigl[1-(2r-1)^2\bigr]\|e\|^2}_{\text{independent of }\theta},\\
d-ae&=v_\theta(x_t,t,c)-v_{\rm tgt}(x_0,t,\epsilon,c),\\
a\|d-e\|^2+(1-a)\|d\|^2
&=a\bigl(\|d\|^2-2\langle d,e\rangle+\|e\|^2\bigr)
+(1-a)\|d\|^2\\
&=\|d\|^2-2a\langle d,e\rangle+a\|e\|^2\\
&=\|d-ae\|^2+a(1-a)\|e\|^2,\\
\ell_{\rm NFT}
&=\beta^2\bigl[a\|d-e\|^2+(1-a)\|d\|^2\bigr]\\
&\quad+\underbrace{(1-\beta^2a)\|e\|^2}_{\text{independent of }\theta}.
\end{aligned}
\]
</div>
<p>Here <span class="math">\(d-e=v_\theta(x_t,t,c)-(x_0-\epsilon)\)</span>, while d is the current–old difference. Taking expectations over the same x₀, t and ε sampling gives the last line in the main text. The two constants differ, but both are independent of θ: <span class="math">\(\mathrm{const}_2=\mathrm{const}_1-\beta^2\mathbb E[a(1-a)\|e\|^2]\)</span>, with the same sampling distribution.</p>
</details>

<p>The first term matches our <a href="#eq-en-forward-weighted-estimator" data-equation-ref="forward-weighted-estimator">Equation 22</a>, except that <span class="math">\(r\)</span> is replaced by a <span class="math">\(\beta\)</span>-controlled linear transformation of reward, a deliberate design in DiffusionNFT. The equivalent form makes NFT's essence clearer: the pretraining-style regression discussed earlier (the first term) + a second term whose purpose is not yet clear. Under fully on-policy training, current=old, the second term is zero and the whole loss supplies the force <span class="math">\(2\beta(2r-1)[(x_0-\epsilon)-v_{\rm old}(x_t,t,c)]\)</span>, pulling high-scoring samples and pushing low-scoring ones. Section 3.2.3 examines the second term's role. (Answering <a href="#en-nft-question-mirror">Question 1: what does the mirror construction make current learn?</a>)</p>

<h4 id="en-detail-9">3.2.2 The fundamental limitation of Forward RL</h4>

<p>Return to the questions at the <a href="#en-nft-questions">end of Section 3.1</a>. Start with a surprising observation: either branch alone collapses, yet combining them learns. The ideal positive and negative velocity fields point along the same reward-improving direction. Why does practical training need both branches?</p>

<figure class="figure-compact" data-figure="nft_branch_geometry"><img src="/assets/blog/diffusion-rl/nft_branch_geometry.png" alt="Figure 6: Branch ablation and geometric decomposition of forward–reverse consistency forces" loading="lazy" width="2400" height="845"><figcaption>Figure 6. Left: either branch alone fails; joint training improves train GenEval in the displayed window. Right: solid arrows are branch forces; dashed arrows separate reward and consistency components, with only partial cancellation of the latter. Δ in the diagram denotes ideal reward guidance.</figcaption></figure>

<p>Section 3.1 established that positive and negative ideal guidance are collinear relative to the original forward field. <strong>The key is that the actual reverse field <span class="math">\(v_{\rm rvs}^{\rm old}(s)=v_{\rm old}(s)\)</span> need not equal the ideal field induced by straight-line noising its endpoints.</strong> Write their difference as <span class="math">\(\Delta_{\rm fwd}(s)=v_{\rm fwd}^{\rm old}(s)-v_{\rm rvs}^{\rm old}(s)\)</span>. Ideal guidance starts from the forward field, and the theoretical derivations use the ideal forward field induced by the old policy's endpoint distribution. The actual training object, however, is the reverse model's output velocity field, so this gap enters the update.</p>

<p>First analyze the first term in the last line of <a href="#eq-en-nft-frozen-target" data-equation-ref="nft-frozen-target">Equation 25</a>, <span class="math">\(\ell_{\rm fit}=\beta^2a\|v_\theta-u\|^2\)</span>, writing <span class="math">\(f_{\rm fit}=-\nabla_{v_\theta}\ell_{\rm fit}\)</span>. Start with its conditional mean force at current=old (expectations are under Q_old):</p>

<div class="math" data-equation="consistency">
\[
\begin{aligned}
\left.\frac1{2\beta}\mathbb E[f_{\rm fit}\mid s]\right|_{v_\theta=v_{\rm old}}
&=\mathbb E[(2r-1)(u-v_{\rm old})\mid s]\\
&=\underbrace{2g(s)}_{\text{reward covariance}}\\
&\quad+\underbrace{(2q(s)-1)\Delta_{\rm fwd}(s)}_{\text{forward--reverse consistency}}.
\end{aligned}
\]
</div>

<details class="derivation" id="en-nft-consistency-proof">
<summary>Derivation: how does practical training add the consistency term?</summary>
<p>Fix s and differentiate the first term of Equation 25, holding r, a and u frozen. Only after differentiation, substitute current=old and write <span class="math">\(e=u-v_{\rm old}\)</span>. Split <span class="math">\(\mathbb E[re\mid s]\)</span> into covariance and a product of means:</p>
<div class="math" data-equation="nft-consistency-proof">
\[
\begin{aligned}
\nabla_{v_\theta}\ell_{\rm fit}
&=2\beta^2a(v_\theta-u)=2\beta(2r-1)(v_\theta-u),\\
-\left.\frac1{2\beta}\nabla_{v_\theta}\mathbb E[\ell_{\rm fit}\mid s]\right|_{v_\theta=v_{\rm old}}
&=\mathbb E[(2r-1)e\mid s]\\
&=2\mathbb E[re\mid s]-\mathbb E[e\mid s]\\
&=2\bigl(\operatorname{Cov}(r,e\mid s)+q\mathbb E[e\mid s]\bigr)
-\mathbb E[e\mid s]\\
&=2\operatorname{Cov}(r,e\mid s)+(2q-1)\mathbb E[e\mid s],\\
\mathbb E[e\mid s]&=\mathbb E[u\mid s]-v_{\rm old}(s)=\Delta_{\rm fwd}(s),\\
\operatorname{Cov}(r,e\mid s)
&=\operatorname{Cov}(r,u-v_{\rm old}(s)\mid s)=g(s),\\
\mathbb E[(2r-1)e\mid s]&=2g+(2q-1)\Delta_{\rm fwd}.
\end{aligned}
\]
</div>
<p>Subtracting a fixed vector leaves covariance unchanged, so the reward term remains g. The positive branch alone gives <span class="math">\(\mathbb E[re\mid s]=g+q\Delta_{\rm fwd}\)</span>; the negative branch gives <span class="math">\(-\mathbb E[(1-r)e\mid s]=\mathbb E[re\mid s]-\mathbb E[e\mid s]=g-(1-q)\Delta_{\rm fwd}\)</span>. Adding them reinforces the two g terms, but cancels the gap only down to <span class="math">\((2q-1)\Delta_{\rm fwd}\)</span>.</p>
</details>

<p><strong>NFT noises a sample <span class="math">\(x_0\)</span> to obtain s, then substitutes its goodness <span class="math">\(r(x_0)\)</span> and the sampled noise to obtain an estimate of <span class="math">\(\mathbb E[f_{\rm fit}\mid s]\)</span>:</strong></p>

<div class="math" data-equation="nft-sample-force">
\[
\begin{aligned}
f_{\rm fit}=-\nabla_{v_\theta}\ell_{\rm fit}
&=2\beta(2r-1)(u-v_\theta),\\
\left.\frac{f_{\rm fit}}{2\beta}\right|_{v_\theta=v_{\rm old}}
&=\underbrace{(2r-1)[u-v_{\rm fwd}^{\rm old}(s)]}_{\text{sample reward guidance}}\\
&\quad+\underbrace{(2r-1)\Delta_{\rm fwd}(s)}_{\text{sample consistency}}.
\end{aligned}
\]
</div>

<p>The first term estimates <span class="math">\(2g(s)\)</span>; the second is this sample's consistency force. Note that NFT substitutes goodness for r, so the estimator's unbiasedness is compromised to some extent.</p>

<p><strong>Why do we still need the negative branch?</strong> Look at the actual sample. At current=old, omitting the common 2β factor:</p>
<ul>
<li><strong>Positive branch only:</strong> the force is <span class="math">\(r(u-v_{\rm old})=r[u-v_{\rm fwd}^{\rm old}(s)]+r\Delta_{\rm fwd}(s)\)</span>. Every sample pulls toward u, good or bad; low-scoring samples merely pull less. The consistency component also always pulls toward the ideal forward field.</li>
<li><strong>Negative branch only:</strong> the force is <span class="math">\(-(1-r)(u-v_{\rm old})=-(1-r)[u-v_{\rm fwd}^{\rm old}(s)]-(1-r)\Delta_{\rm fwd}(s)\)</span>. Every sample pushes away from u, good or bad; high-scoring samples merely push less. The consistency component always pushes away from the ideal forward field.</li>
</ul>
<p>Combining them yields <span class="math">\((2r-1)(u-v_{\rm old})\)</span>: samples above the group mean pull; samples below it push. The consistency coefficient also changes from the one-sided r or −(1−r) to the signed 2r−1, allowing extra components from different samples to cancel during global gradient aggregation across states. This answers both <a href="#en-nft-question-negative">Question 2: why do we need positive and negative branches?</a> and <a href="#en-nft-question-goodness">Question 3: why replace r with goodness obtained by normalizing reward?</a>, sacrificing unbiasedness: if r is goodness, then <span class="math">\(\sum (2r-1)=0\)</span>, balancing the coefficients of consistency forces across states. But the gap varies across states, so balanced coefficients do not guarantee cancellation of the weighted vectors (<span class="math">\(\sum (2r-1)=0\)</span> does not imply <span class="math">\(\sum \mathrm{grad}\,(2r-1)f=0\)</span>). Any uncancelled residual still slows training or makes reward training collapse outright (Figure 6, right).</p>

<blockquote><strong>Insight: the regression signal in Forward RL contains a hidden consistency term.</strong> This term does not help improve reward and can directly cause training to collapse. Joint positive/negative-branch training and reward normalization are indispensable because they can partially cancel this term's gradients globally, but cannot eliminate them completely.</blockquote>

<p><strong>Matching endpoint distributions does not imply matching intermediate distribution evolution.</strong> Fine-tuning changes the actual reverse denoising dynamics. Reversing this evolution from t=1 to 0 need not match the evolution induced by straight-line re-noising of its endpoints; the two may differ substantially. In this sense, Reverse Process RL respects the current model's reverse process rather than assuming that straight-line noising inverts it.</p>

<h4 id="en-nft-restoration-role">3.2.3 The role of restoration</h4>

<p>Section 3.2.1 decomposed NFT's objective into two terms. The previous subsection examined the first term's problems; the second compensates for them. Let d be current minus reference and e be target minus reference. The output-space force is:</p>

<div class="math" data-equation="mirror-force">
\[
\begin{aligned}
d&=v_\theta-v_{\rm old},\qquad e=u-v_{\rm old},\\
f&=\underbrace{2\beta(2r-1)e}_{\text{feedback guidance}}
-\underbrace{2\beta^2d}_{\text{restoration}}.
\end{aligned}
\]
</div>

<details class="derivation" id="en-nft-restoration-proof">
<summary>Derivation: how does restoration emerge from the NFT loss?</summary>
<p>Use the square expansion in Section 3.2.1. With r, u and old outputs frozen, only d changes when differentiating with respect to the current output:</p>
<p>Specifically, <span class="math">\(\nabla_{v_\theta}\|d\|^2=2d\)</span>, <span class="math">\(\nabla_{v_\theta}\langle d,e\rangle=e\)</span>, and <span class="math">\(\nabla_{v_\theta}\|e\|^2=0\)</span>. Differentiate each term and negate the result to obtain the force below.</p>
<div class="math" data-equation="nft-restoration-proof">
\[
\begin{aligned}
\ell_{\rm NFT}
&=\beta^2\|d\|^2
-2\beta(2r-1)\langle d,e\rangle+\|e\|^2,\\
\nabla_{v_\theta}\ell_{\rm NFT}
&=2\beta^2d-2\beta(2r-1)e,\\
f=-\nabla_{v_\theta}\ell_{\rm NFT}
&=\underbrace{2\beta(2r-1)e}_{\text{feedback guidance}}
-\underbrace{2\beta^2d}_{\text{restoration}}.
\end{aligned}
\]
</div>
<p>Restoration comes from the quadratic loss term, not an extra regularizer added by EMA. At current=old, d=0 and it vanishes. Once current moves against a frozen reference, it points back toward old. Taking expectations and elementwise means preserves this decomposition.</p>
</details>

<p>The first term contains both kinds of guidance in Section 3.2.2; the second pulls current back toward old. Restoration limits displacement from the old output, but can also compete with the direction that improves reward.</p>

<p>We ran two informative experiments:</p>
<ol>
<li>Retain EMA rollouts but remove the old-policy anchor from the training gradient: replace <span class="math">\(v_{\rm tgt}(x_0,t,\epsilon,c)=a(x_0-\epsilon)+(1-a)v_{\rm old}(x_t,t,c)\)</span> in <a href="#eq-en-nft-frozen-target" data-equation-ref="nft-frozen-target">Equation 25</a> with an interpolation of <span class="math">\(u=x_0-\epsilon\)</span> and <span class="math">\(v_\theta\)</span>, while keeping <span class="math">\(p_{\rm old}\)</span> unchanged (forward noising still uses endpoints from old-policy rollouts). Reward still collapses, much like the fully on-policy result (Figure 7b).</li>
<li>Without EMA, change the number of minibatches per rollout from one to two: instead of updating once on all samples, randomly split them into equal halves and update once on each. One minibatch improves reward quickly before collapsing. Two minibatches improve about 15.2 times more slowly (first reaching OCR=0.9 at 320 rather than 21 steps), but have not collapsed in the observed window (Figure 7a).</li>
</ol>

<figure class="figure-compact" data-figure="nft_arxiv_restoration"><img src="/assets/blog/diffusion-rl/nft_arxiv_restoration.png" alt="Figure 7: Restoration ablations, gradient spike and force-scale reversal" loading="lazy" width="2400" height="1048"><figcaption>Figure 7. Left: β=1, no EMA. One update versus two disjoint updates first reach OCR=0.9 at 21 versus 320 optimizer steps, a 15.2-fold slowdown. The gray dashed curve is the one-update run's pre-clipping gradient norm before collapse. Middle: retaining EMA rollouts but removing the training anchor still collapses train GenEval. Right: at second updates over the first 18 rollouts, restoration has 9.4 times smaller output-force RMS but 4.9 times larger parameter-gradient norm; both are ratios of means.</figcaption></figure>

<p><strong>Experiment 1 separates old data from old outputs.</strong> Rollouts still come from the EMA policy, yet removing the old-policy anchor from the loss makes training collapse again (Figure 7b). In this ablation, stabilization comes from the anchor's restoring force, not from off-policy rollouts alone.</p>

<p><strong>Experiment 2 shows that a second update supplies restoration even without EMA.</strong> The first update starts at current=old, where restoration is zero. After parameters move, the second update's <span class="math">\(v_\theta^{(1)}\)</span> differs from the frozen <span class="math">\(v_{\rm old}\)</span>. Its force splits into guidance and restoration toward old (the <a href="#eq-en-mirror-force" data-equation-ref="mirror-force">restoration decomposition</a> and Figure 8). The second update uses the other half of the samples; they need not be the same samples as in the first update. Shared parameters have moved, so these states also experience displacement.</p>

<p><strong>Counterintuitively, restoration is smaller in output space but stronger in parameter space (<a href="#en-detail-11">Section 3.3</a> examines this systematically).</strong> Decompose the second update's force and measure both components at the same current parameters and states. Over the first 18 rollouts, restoration's mean output-force RMS is about 9.4 times smaller, yet its aggregate parameter-gradient norm is about 4.9 times larger (Figure 7c). It limits drift but also strongly dominates the update, explaining why training becomes more stable yet learns much more slowly.</p>

<blockquote id="en-detail-10"><p><strong>Insight: the fundamental difference between on-policy and off-policy Diffusion RL.</strong> In this NFT-style regression, the key is not how old the data are, but whether current has moved away from the old anchor in the loss when evaluating the gradient. Through the <a href="#eq-en-chain" data-equation-ref="chain">force and gradient decomposition in Section 1</a>, this displacement systematically supplies a restoring force toward old. EMA keeps old lagging; two updates keep the same frozen old after the first update moves current. Both are off-policy updates here, and empirically we find that this restoring force converts efficiently into an aggregate parameter gradient. Pure on-policy current=old has no such force. This may be a key distinction between Diffusion RL and LLM RL, or a property of continuous-time, continuous-space RL built on MSE losses.</p></blockquote>

<p id="en-repulsion-discussion"><strong>Discussion: does a second Reverse RL update supply the same restoration?</strong> It does not automatically supply NFT's uniformly restoring term. Return to the <a href="#eq-en-reverse-mse" data-equation-ref="reverse-mse">local Reverse RL MSE</a>: positive advantage uses positive MSE to pull toward a target; negative advantage uses negative MSE to push away. NFT's negative branch instead uses positive MSE to pull toward a mirrored target. The two ways to repel can give the same gradient at old, yet differ after leaving it.</p>

<p>Fix the same state, old output and target u, keeping <span class="math">\(e=u-v_{\rm old}\)</span> and <span class="math">\(d=v_\theta-v_{\rm old}\)</span>. Set β=1 and omit a shared positive weight:</p>

<div class="math">
\[
\begin{aligned}
\ell_{\rm repel}&=-\|v_\theta-u\|^2,
&f_{\rm repel}&=-2e+\underbrace{2d}_{\text{away from old}},\\
\ell_{\rm mirror}&=\|v_\theta-(2v_{\rm old}-u)\|^2,
&f_{\rm mirror}&=-2e-\underbrace{2d}_{\text{toward old}}.
\end{aligned}
\]
</div>

<div class="figure-pair geometry-comparison">
<figure class="figure-compact figure-schematic" data-figure="nft_second_update_geometry"><img src="/assets/blog/diffusion-rl/nft_second_update_geometry.png" alt="Figure 8: Two MSE descent directions and the second-update guidance/restoration decomposition" loading="lazy" width="2400" height="2871"><figcaption>Figure 8. Blue denotes the first MSE descent direction; solid orange denotes the second after the model moves. Dashed orange decomposes it into the original guidance and restoration toward old. This fixed-state, fixed-target illustration does not require reusing the same training samples.</figcaption></figure>
<figure class="figure-compact figure-schematic" data-figure="repulsion_geometry"><img src="/assets/blog/diffusion-rl/repulsion_geometry.png" alt="Figure 8b: Negative-MSE descent directions point away from v tgt and the displacement component repels old" loading="lazy" width="2400" height="2871"><figcaption>Figure 8b. Negative MSE: both descent directions point away from v_tgt, and the displacement component points away from old. Figure 8's layout and style are retained, with force directions reversed. Arrows are translatable output-force vectors.</figcaption></figure>
</div>

<p>At <span class="math">\(d=0\)</span>, both forces equal <span class="math">\(-2e\)</span>. After moving, negative MSE amplifies displacement, while mirrored positive MSE pulls it back. NFT's branches differ in guidance but share the sign of their quadratic terms, so aggregation across states retains <a href="#eq-en-mirror-force">uniform restoration</a>. In Reverse RL, the displacement term changes sign with advantage, so the restoration coefficients across states sum to zero. But zero group-mean advantage alone does not establish cancellation: displacement, weights and Jacobians differ across states. This comparison fixes the MSE coefficients; actual <a href="https://arxiv.org/html/2505.05470v1#S4.SS1">Flow-GRPO</a> also multiplies by a probability ratio and is affected by PPO clipping.</p>

<blockquote><strong>Insight: the same first-step gradient can hide opposite subsequent dynamics. If the algorithm does not use purely on-policy updates, designing a gradient provider</strong> requires considering not only where it pushes the model now, but also whether it amplifies or restores the displacement after the model moves.</blockquote>

But why can a seemingly small restoration term have such a large effect?

<h3 id="en-detail-11">3.3 Understanding which forces become strong</h3>

<h4 id="en-force-comparison">3.3.1 Four forces: what survives the three layers?</h4>

<p>Return to the <a href="#eq-en-chain" data-equation-ref="chain">three-layer model</a>. To analyze a force's training behavior, we follow its three layers (<em>three quantities and two rates</em>): 1. the force acting on each state; 2. how that force becomes a parameter gradient at that state; 3. how these gradients aggregate into a total gradient. We measure <strong>output-force magnitude F → per-state parameter-gradient magnitude P → cross-state aggregate-gradient magnitude A</strong>. The arrows define two <em>rates</em>: <strong>per-state response rate = P/F</strong>, how much parameter gradient a unit output force produces; and <strong>aggregation retention = A/P</strong>, how much survives when the states' parameter gradients are added.</p>

<details class="derivation" id="en-force-conversion-proof">
<summary>Recall: how do the three quantities and two rates connect?</summary>
<p>Using the output force <span class="math">\(f_i\)</span> and per-state parameter direction <span class="math">\(g_i=J_i^\top f_i\)</span> defined earlier, the three quantities and two rates are:</p>
<div class="math" data-equation="conversion">
\[
\begin{aligned}
F&=\frac1N\sum_i\|f_i\|,\qquad
P=\frac1N\sum_i\|g_i\|,\qquad
A=\left\|\frac1N\sum_i g_i\right\|,\\
\underbrace{\frac{P}{F}}_{\text{Per-state response rate}}
&=\frac{\sum_i\|g_i\|}{\sum_i\|f_i\|},\qquad
\underbrace{\frac{A}{P}}_{\text{Aggregation retention}}
=\frac{\|\sum_i g_i\|}{\sum_i\|g_i\|},\\
\frac{A}{F}&=\frac{P}{F}\times\frac{A}{P}.
\end{aligned}
\]
</div>
<p>With consistent norms and reductions and nonzero denominators, these identities are exact. P/F is the output-force-norm-weighted per-state response rate.</p>
</details>

<p>In Diffusion RL, we typically encounter four kinds of forces:</p>

<ul>
<li><strong>Reverse RL policy force (e.g., Flow-GRPO):</strong> attraction or repulsion along the random action residual at the old-policy anchor; advantage sets its sign and scale.</li>
<li><strong>Forward RL guidance (e.g., DiffusionNFT):</strong> an endpoint-based regression request. Removing current–old restoration still leaves reward covariance and <a href="#en-detail-9">forward–reverse consistency</a>.</li>
<li><strong>Restoration from off-policy training:</strong> undoing current's departure from old. We distinguish a lagging EMA old policy from the old policy frozen for a second update.</li>
<li><strong>KL / reference force:</strong> limiting current's departure from the base model.</li>
</ul>

<p>Within one algorithm pipeline, the <a href="#en-detail-1">earlier "vector-sum" model</a> lets us decompose forces and analyze each one's three quantities and two rates separately. The table below gives our experimental results directly:</p>

<figure id="en-three-layer-probe" class="figure-compact force-summary" data-figure="force_measured_overview">
<div class="force-table-scroll" tabindex="0" role="region" aria-label="Three quantities and two rates for four forces">
<table class="force-table">
<thead><tr><th scope="col">Force</th><th scope="col">Measurement source</th><th scope="col">Output force<br>F</th><th scope="col">Per-state gradient<br>P</th><th scope="col">Aggregate gradient<br>A</th><th scope="col">Per-state response rate<br>P/F</th><th scope="col">Aggregation retention<br>A/P</th></tr></thead>
<tbody>
<tr data-run="9mau8hwi" data-component="guidance"><th class="force-kind" scope="row">Flow-GRPO<br>guidance</th><td>Flow-GRPO</td><td>0.863</td><td>3.29</td><td>0.978</td><td>3.8</td><td>29.7%</td></tr>
<tr data-run="26rilzmp" data-component="guidance"><th class="force-kind" scope="row">DiffusionNFT<br>guidance</th><td>NFT</td><td>4.08</td><td>169</td><td>41.7</td><td>41.4</td><td>24.7%</td></tr>
<tr data-run="26rilzmp" data-component="restore"><th class="force-kind" scope="row">EMA restoration</th><td>NFT</td><td>0.765</td><td>107</td><td>39.8</td><td>139.7</td><td>37.2%</td></tr>
<tr data-run="adnxitig" data-component="restore"><th class="force-kind" scope="row">Second-update<br>restoration</th><td>Two-update NFT</td><td>0.567</td><td>78.2</td><td>32.7</td><td>137.9</td><td>41.8%</td></tr>
<tr data-run="9mau8hwi" data-component="reference_mse"><th class="force-kind" scope="row">KL loss</th><td>Flow-GRPO</td><td>0.435</td><td>63.4</td><td>25.8</td><td>145.6</td><td>40.6%</td></tr>
</tbody>
</table>
</div>
<figcaption>Table 1. Three quantities and two rates for five force measurements. F, P and A are in units of 10⁻³; the rates are P/F and A/P. Parameter gradients are measured before Adam/clipping.</figcaption>
</figure>

<blockquote class="observation">
<p><strong>Observation:</strong></p>
<ul>
<li>Flow-GRPO and NFT EMA guidance have per-state response rates of about <strong>3.8 and 41</strong>, with aggregation retention of roughly <strong>30% and 25%</strong>. Differences in per-state response rate are key to their differing learning efficiency and stability, while aggregation retention shows little difference.</li>
<li>Both NFT restoration mechanisms have markedly higher per-state response rates and aggregation retention than NFT guidance.</li>
<li>KL loss has a high per-state response rate and aggregation retention.</li>
</ul>
</blockquote>

Note: the NFT guidance measurements in this table cover only the early, stable phase. Its per-state response rate can spike (<a href="#en-force-stability">Figure 10</a>), which this table does not show.

<h5 id="en-nft-art-dynamics">1. Understanding Forward Process RL instability through three quantities and two rates</h5>

Run pure on-policy NFT longer, tracking guidance's output force, per-state response rate, retention and OCR reward separately:

<figure id="en-force-stability" class="figure-compact" data-figure="force_stability"><img src="/assets/blog/diffusion-rl/force_stability_nft.png" alt="Figure 10: Longitudinal NFT output force, per-state response rate, retention and OCR reward" loading="lazy"><figcaption>Figure 10. Pure on-policy NFT instability accompanies spikes in per-state response rate, not a comparable surge in aggregation retention. The x-axis counts rollouts; response rates average valid states equally.</figcaption></figure>

NFT's per-state response rate rises from about **44** initially to **1239/2784** at rollouts 44/45, alongside a reward collapse; retention does not show a comparable surge. **The reward collapse comes not from a sudden increase in NFT's output force, nor from more gradients becoming aligned, but from a unit output force mapping to a raw parameter gradient orders of magnitude larger, causing a spike in the gradient used for updates.**

This connects to the <a href="#en-detail-9">fundamental limitation of Forward RL</a>: NFT's actual force contains not only ideal reward-improving guidance but also a self-consistency force. **An overly sensitive consistency residual is a concrete candidate mechanism for the per-state response rate spike.** As training proceeds, the reverse denoising velocity field may drift further from the ideal straight-line forward-noising field induced by the same endpoints, making the consistency force harder to control. This is also consistent with the trend in panel (b).

<h5 id="en-restoration-rescue">2. How can the two restoration mechanisms help?</h5>

Table 1 gives NFT EMA and second-update restoration per-state response rates of about **140 and 138**, with retention of **37% and 42%**. Within their respective paired windows, restoration has only about **19% and 12%** of guidance's output force, yet about **95% and 63%** of its parameter resultant. Small in output space does not mean weak as a constraint.

Why are restoration requests jointly realizable? Locally, a parameter movement Δθ changes each state's output by dᵢ ≈ JᵢΔθ. **The same parameter direction −Δθ can undo all these changes: Jᵢ(−Δθ) ≈ −dᵢ.** This is joint realizability; it does not require parallel output directions.

For uniformly weighted, unnormalized output MSE, their aggregation takes the following form, omitting a common positive coefficient:

<div class="math">
\[
\begin{aligned}
g_i^{\rm restore}&=-J_i^\top J_i\Delta\theta,\\
\left\langle g_i^{\rm restore},-\Delta\theta\right\rangle
&=\|J_i\Delta\theta\|^2\ge0,\\
\left\langle \frac1N\sum_i g_i^{\rm restore},-\Delta\theta\right\rangle
&=\frac1N\sum_i\|J_i\Delta\theta\|^2.
\end{aligned}
\]
</div>

Each state's restoration contribution has a component pointing back: **contributions along the shared reversal direction add rather than cancel**; perpendicular components may still cancel. Actual loss weights and normalization must remain inside each term. This gives a local explanation for restoration coherence, not a guarantee of higher per-state response rate.

> **Insight: joint realizability matters more than identical output directions.** Restoration tries to undo an already realized function change. This offers an intuition for efficient conversion without requiring parallel output forces or guaranteeing restoration dominance.

Track output force, per-state response rate and aggregation retention separately for the two restoration mechanisms:

<figure id="en-restoration-three-metrics" class="figure-compact" data-figure="restoration_three_metrics"><img src="/assets/blog/diffusion-rl/restoration_three_metrics.png" alt="Figure 11: Output force, per-state response rate and aggregation retention of NFT EMA and second-update restoration" loading="lazy"><figcaption>Figure 11. Three diagnostics for the two restoration mechanisms. The x-axis counts rollouts; response rate is per-window P/F and retention is A/P. Available data cover only the first 20 rollouts.</figcaption></figure>

These short-run curves show how restoration converts to parameter gradients; they cannot yet tell us whether it keeps pace with, and constrains, long-term guidance spikes.

EMA and second updates, however, intervene on different schedules:
- **Second update:** the first starts at current=old, without restoration. The second has restoration, but old is synchronized to current at the end of the rollout; the next rollout starts from zero again.
- **EMA:** old lags across rollouts, so restoration does not reset each round. Rollouts and targets also follow this slowly changing old policy.

<h5 id="en-flow-force-mapping">3. Flow-GRPO: why is even the per-state response rate so low?</h5>

Cross-state cancellation cannot explain all of Flow-GRPO's low conversion. In the full Flow-GRPO time course, per-state response rate is already in single digits, while retention is close to NFT:

<figure id="en-flow-mapping-course" class="figure-compact" data-figure="flow_mapping_course"><img src="/assets/blog/diffusion-rl/flow_mapping_course.png" alt="Figure 12: Full Flow-GRPO per-state response rate and aggregation retention over training" loading="lazy"><figcaption>Figure 12. Flow-GRPO's low response rate appears before cross-state aggregation. The two update windows are plotted separately; response rate is P/F and retention is A/P.</figcaption></figure>

<p>At the old-policy anchor, Flow-GRPO attracts or repels along the random action-residual axis. Advantage sets its sign and scale. Crucially, <strong>one state's parameter gradient sums contributions from every output element: <span class="math">\(g_i=J_i^\top f_i=\sum_{d=1}^{D}f_{i,d}\nabla_\theta v_{\theta,d}(s_i)\)</span>, where d indexes output coordinates. From the optimizer's perspective, each element of Flow-GRPO's force is independent (an isotropic, independent Gaussian), yet all are reinforced by the same advantage-dependent scale, so their gradient contributions cancel substantially</strong>. This differs from cancellation across states in the final layer; the low per-state response rate of random directions in Figure 11 offers a clue consistent with this observation. In forward RL, force elements are highly correlated, carrying information from an endpoint distribution. Their per-state response rate is therefore much higher, which also helps explain forward RL's greater efficiency than Flow-GRPO.</p>

<h4 id="en-force-competition-role">3.3.2 When does force magnitude matter? When two forces compete.</h4>

Adam may largely attenuate a common scaling of one force. Changing the ratio between two forces instead changes the direction entering the optimizer. Here we focus on two situations in which two forces compete.

<p><strong>Case Study 1: off-policy stabilization in forward RL</strong></p>

<p>One experiment makes this visible: pure on-policy NFT (without EMA), at two <span class="math">\(\beta\)</span> values, 1 and 0.1. In the <a href="#eq-en-mirror-force" data-equation-ref="mirror-force">earlier decomposition</a>, guidance magnitude is proportional to beta; here guidance includes both reward-improving guidance and the self-consistency force. At each beta, we compare one and two updates per rollout. As <a href="#en-nft-restoration-role">discussed earlier</a>, a second update introduces restoration.</p>

<figure class="figure-compact" data-figure="mirror_dynamics"><img src="/assets/blog/diffusion-rl/mirror_dynamics_compact.png" alt="Figure 13: One-update and two-disjoint-update OCR curves at beta 1 and 0.1" loading="lazy" width="2400" height="930"><figcaption>Figure 13. OCR curves for one update and two disjoint updates. Left: β=1; right: β=0.1. The x-axis counts optimizer steps. Restoration in the second update changes the competition between forces.</figcaption></figure>

> **Observation:**
> - With one update, reward rises at similar speeds for β=0.1 and β=1 (compare the teal curves).
> - With two updates per rollout, the difference is substantial: β=0.1 learns much faster, but is clearly less stable later.

<p>Divide the <a href="#eq-en-mirror-force" data-equation-ref="mirror-force">existing NFT force decomposition</a> by the common positive coefficient <span class="math">\(2\beta^2\)</span> to reveal the competition:</p>

<div class="math" data-equation="mirror-relative-scale">
\[
\frac{f}{2\beta^2}
=\underbrace{\frac{2r-1}{\beta}(u-v_{\rm old})}_{\text{guidance}}
-\underbrace{(v_\theta-v_{\rm old})}_{\text{restoration}}.
\]
</div>

At the first update, current=old and restoration is zero; β primarily changes guidance's global scale. Given <a href="#en-detail-2">modern optimizers' approximate invariance to global scaling</a>, parameter displacement and output-space movement should be similar. At the second, current has moved while old remains frozen. After extracting the common coefficient, similar displacement gives a similar restoration term, and β changes guidance's relative share.

<figure id="en-force-competition" class="figure-compact" data-figure="force_competition"><img src="/assets/blog/diffusion-rl/force_competition_compact.png" alt="Figure 14: Initial movement and second-window restoration versus guidance strength and direction" loading="lazy" width="2400" height="960"><figcaption>Figure 14. Similar initial movement, different subsequent competition. Left: output-change MSE at the first update. Middle and right: restoration/guidance gradient-norm ratio and cosine in second windows.</figcaption></figure>

The first update's output-change MSE is **0.002318/0.002311**, with parameter-displacement norms both about **0.888**. Yet the ratio of mean restoration/guidance norms in the second windows falls from **4.85 to 0.78**. Nearly equal initial movement leads to different learning speeds. **A force's scale is global without a competing force (here, restoration), but becomes relative once restoration appears.**

**Case Study 2: how should we tune KL in continuous-time, continuous-space RL?**

Both KL / reference regularization and restoration undo function changes the model has already realized. They differ in where they pull back: restoration returns toward a frozen or EMA old policy, while reference regularization returns toward a fixed base. The earlier intuition of joint realizability also applies here: undoing an already realized change differs from making new reward-driven requests. A small reference residual need not imply a weak parameter constraint. Here we use model-output MSE regularization, not a literal distributional KL.

<p>Tuning <span class="math">\(\lambda\)</span> means tuning the final contribution <span class="math">\(\lambda\bar g_{\rm ref}\)</span>, not merely the ratio of loss values or output forces. Recall the three layers: <strong>output-force magnitude → per-state gradient response → cross-state aggregate gradient.</strong> In continuous space, different directions have different per-state response rates and aggregation retention. After both arrows, similar output forces can have very different parameter effects.</p>

For example, Flow-GRPO's raw reference output force is only **0.49 times** guidance's, yet its aggregate gradient is **25.9 times** as large. In Fast, these ratios are **1.80 and 184.7**. Comparing only the first layer can substantially underestimate the constraint.

<p>Matched-window diagnostics give an equal-norm coefficient: <span class="math">\(\lambda_{\rm match}=A_{\rm guidance}/A_{\rm ref}\)</span>. Here A is the mean of per-window aggregate-gradient norms; NFT guidance excludes old-policy restoration:</p>

<div class="force-table-scroll" tabindex="0" role="region" aria-label="Reference equal-norm coefficients by algorithm">
<table class="force-table">
<thead><tr><th scope="col">Paradigm / algorithm</th><th scope="col">Guidance aggregate gradient<br>A</th><th scope="col">Raw reference aggregate gradient<br>A</th><th scope="col">Equal-norm coefficient<br>λ</th></tr></thead>
<tbody>
<tr><th scope="row">Forward / NFT + EMA</th><td>0.0417</td><td>0.00539</td><td><strong>7.73</strong></td></tr>
<tr><th scope="row">Reverse / Flow-GRPO</th><td>0.000996</td><td>0.0258</td><td><strong>0.0387</strong></td></tr>
<tr><th scope="row">Reverse / Flow-GRPO Fast</th><td>0.000377</td><td>0.0696</td><td><strong>0.00541</strong></td></tr>
</tbody>
</table>
</div>

At these measured states, NFT needs a coefficient of about **7.7** for reference to match guidance's aggregate-gradient norm; Flow-GRPO needs only about **0.039**, and Fast about **0.0054**. The applied training coefficient was **zero** in all these runs: this is offline scale calibration from raw probes, not training with those coefficients or an optimal-coefficient estimate. Guidance, the current–base gap and both conversion rates contribute to the difference; these are not fixed recommendations for forward versus reverse RL.

> **Insight: choose KL coefficients in relation to how the policy force converts.** Calibrate parameter influence at the third layer, not just loss values or output forces at the first; then use the forces' cosine and actual drift to decide how strong the constraint should be.

> **Recipe: constraints should not suppress ordinary learning, yet must keep up when guidance becomes dangerous.** Track both forces' output magnitudes, per-state response rates, aggregation retention and weighted parameter resultants. Second-update restoration and KL tuning are both instances of this competition.

<h2 id="en-recipes" data-section-key="recipes">4. What would we measure before choosing a recipe?</h2>

<p>When tuning the ratio between forces, <strong>tune the relative strengths of the aggregated parameter gradients, rather than the force magnitudes (i.e., loss values): in continuous-time, continuous-space diffusion models, different force directions can have very different effects (<a href="#en-force-comparison">see the three quantities and two rates in Section 3.3.1</a>).</strong></p>

**Reverse: how do actions mix?** Measure gradient norms by noise position and cosine with the resultant. Ask whether influential actions can change reward. Try informative short windows, controlled initial noise or better baselines, comparing learning progress separately from reduced backward computation.

**Forward: how do forces mix?** At matched points, measure guidance/restoration output forces and parameter gradients, their cosine and resultant direction. A small loss component can dominate. Competing directions need not be exact opposites.

**Stability: how does the function actually move?** Measure current-reference gaps and actual output changes at fixed states. Adaptive restoration is worth exploring: allow useful movement and intervene on dangerous drift. Current curves motivate that experiment; they do not validate a particular rule.

**Open experiment: can restoration intervene only when useful?** Compare fixed and adaptive restoration on reward, fixed-state drift and failure rates at matched sampling cost. Specify the rule in advance and use multiple seeds rather than retrospectively selecting the best coefficient.

The common question is not "how much did loss fall?" It is: **which directions reached the shared update, and what did they achieve?** Reverse hides that question in action credit; forward hides it in the conversion of regression forces. Making the intermediate quantities visible is often the quickest way to understand an unusual curve.
