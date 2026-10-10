<p>这篇 blog 关注 diffusion / flow-matching models：以视觉生成为场景，理解这类连续时间生成模型的策略优化，探索高效、稳定的训练配方 （下文把这类基于奖励的优化统称为 <strong>Diffusion RL）</strong>。从奖励信号怎样变成模型更新出发，常见方法可以分成三类：</p>

<ol><li><strong>基于反向过程的 RL</strong>，如 <a href="https://arxiv.org/abs/2505.05470">Flow-GRPO</a>：把去噪过程看成一串 action，用生成结果的 reward 来指导这些 action 的策略更新。</li><li><strong>基于前向过程的 RL</strong>，如 <a href="https://arxiv.org/abs/2509.16117">DiffusionNFT</a>：给生成结果打分，再对它们加噪，把奖励反馈转成前向过程上的回归训练。</li><li><strong>通过奖励模型反传梯度</strong>，如 <a href="https://arxiv.org/abs/2304.05977">ReFL</a>：让奖励梯度穿过生成过程，直接指导模型更新。</li></ol>

<p>前两类通常只需要奖励模型给出分数，不要求它可微；第三类需要沿奖励计算路径反传梯度。</p>

<p><strong>本文聚焦前两类问题展开， 在全文开始之前， 先精炼总结一下本文旨在传达的洞见， 希望能够启发未来的学术研究或者业界落地应用：</strong></p>

<ul><li>连续生成过程RL的核心心智模型搭建：<ul><li>概念层面统一Diffusion RL算法：MSE loss 称为 “力”， 通常MSE loss的一头是我们的policy的产物， 另一头是构造出来的一个目标， 正MSE loss 把policy 产物拉向这个MSE目标（称为拉力）， 负MSE loss 把policy 产物推离这个MSE 目标（称为推力）。这里的力和梯度同义。</li><li>理解样本梯度相对尺度，以及梯度聚合和抵消现象；</li><li>理解连续时间生成过程的RL的loss载体-MSE loss 只是一个梯度（力）提供器， 其MSE 目标而非我们真正希望拟合上去的目标。</li><li>分析Diffusion RL的训练行为， 本质是分析各个state上的action梯度方向和大小，以及它们在聚合时发生的事情。</li></ul>

</li><li>反向去噪过程是一个特殊的马尔可夫链， 具备一些可以被我们利用的性质。<ul><li>以此理解为什么反向过程RL通常稳定但是有效率低下的问题， 且为什么Flow-GRPO-Fast 通过训练更少的timestep可以效率远高于Flow-GRPO。</li></ul>

</li><li>前向RL训练的局限性：loss里面有一个隐藏的前向-反向自一致性项，影响梯度质量， 减慢训练并且带来不稳定性。<ul><li>Off-Policy 训练在Diffusion RL里面和on-policy 最本质的区别， 以及理解为什么Off-policy 策略能够稳定训练。</li><li>理解为什么有些力在输出空间不大，但是能产生很大的聚合梯度， 有些力在输出空间力很大， 但是聚合后梯度很小。</li></ul>

</li></ul>

<h2 id="zh-target" data-section-key="target">1. 连续生成过程 RL 的核心心智模型</h2>

<p>前向与反向 RL，都可以从底层的 MSE 推拉项来理解。一个 MSE 项的一头是 policy 的输出（通常是去噪速度场 <span class="math">\(v_\theta\)</span>），另一头是停止梯度的 target <span class="math">\(v_{\rm tgt}\)</span>。对 D 个输出元素取平均，带固定系数 λ 的 loss 与它提供的力为 <span class="math" data-equation="force">\(\ell=\frac{\lambda}{D}\|v_\theta-v_{\rm tgt}\|^2\); \(f=-\nabla_{v_\theta}\ell=\frac{2\lambda}{D}(v_{\rm tgt}-v_\theta)\)</span>。λ &gt; 0 把输出拉向 target，λ &lt; 0 把输出推离 target。我们称这个 <span class="math">\(f\)</span> 为输出空间的力，这里的“力”就是 loss 对输出的<strong>负梯度</strong>；训练可以理解为各个中间 state 上推力与拉力的聚合作用。</p>

<p><strong>Target 是梯度提供器，不是目的地。</strong> Flow matching 已经给过我们一个例子：同一个含噪 state 对应多个经过它的前向速度场，MSE 的最优预测是它们的条件均值（<a href="https://arxiv.org/abs/2210.02747">Flow Matching</a>），不可能同时到达每个 target。Post-training 构造的 target 也不必是我们最终想要的速度场；它可以只负责提供改善 reward 的“力”， 即梯度。</p>

> **Insight：不必到达 target，也能学到有用的变化。** 分析训练时，先看 target 提供了什么力，而不是把拟合它当成最终任务。

<h3 id="zh-detail-1">1.1 从各个 state 的力，到共享参数的合力</h3>

<p>中间有三层：<strong>输出力 → 单样本参数梯度 → 跨 state 梯度聚合</strong>。固定模型与训练状态 sᵢ（包括 prompt、latent 和 timestep），记 Jᵢ 为输出对参数的 Jacobian，<span class="math">\(f_i\)</span> 为输出空间的力，gᵢ 为参数空间的梯度：</p>

<div class="math" data-equation="chain">
\[
\begin{aligned}
J_i &= \frac{\partial v_\theta(s_i)}{\partial\theta},\\
f_i\;&\longrightarrow\;g_i=J_i^\top f_i
\;\longrightarrow\;\bar g=\frac{1}{N}\sum_{i=1}^{N}g_i.
\end{aligned}
\]
</div>

<p>N 是样本数，固定样本权重已包含在 fᵢ 中。在同一模型和同一批状态上，梯度 <span class="math">\(g\)</span> 的聚合可以被理解为 <strong>向量和， 同理，也可以使用向量分解的心智模型，把一个力分解成多个component， 来分析多个力聚合的时候，有哪些成分（component）会抵消， 哪些成分不会相互抵消</strong>。很大的输出力，经过 Jacobian 可能只留下很小的参数方向；各样本的参数梯度聚合的时候， 模长不能直接相加， 因为它们有不同的方向。相反方向的梯度可能相互抵消。</p>

<figure class="figure-compact" data-figure="force_overview"><img src="/assets/blog/diffusion-rl/force_overview.png" alt="图1：梯度映射、聚合与相对尺度" loading="lazy" width="2400" height="864"><figcaption>图1：梯度映射、聚合与相对尺度。上：输出力经 Jacobian 映射，再跨 state 聚合。下：整体缩放保持合力方向，相对加权改变方向；s = g₁ + g₂。分项力的相对尺度决定合力代表谁。</figcaption></figure>

<p></p>

<h3 id="zh-detail-2">1.2 相对尺度不会被优化器抹去</h3>

<p><strong>优化器和梯度裁剪处理的是聚合后的合力</strong>。合力的所有分项一起乘同一个正数，合力方向不变；只放大其中一项，通常会改变合力方向。常用的 AdamW 优化器对整体尺度近似不敏感， 所以如果给每一个分项loss乘以一个常数， 不会显著影响训练，但是梯度聚合前所有分项的相对尺度会直接显著影响训练。</p>

<details class="derivation">
<summary>Adam / AdamW 的整体尺度近似不变性</summary>

<p>记第 n 次更新的聚合 loss 梯度为 hₙ。<a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html">AdamW</a> 用它的一阶、二阶移动平均构造更新（所有乘除与平方都逐元素进行）：</p>

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

<p>其中 <span class="math">\(\widehat m_n=m_n/(1-\beta_1^n)\)</span>、<span class="math">\(\widehat q_n=q_n/(1-\beta_2^n)\)</span> 是偏差修正，ηₙ 是学习率，λ_w 是独立的 weight decay。若整段梯度历史都乘同一个正数 c，一阶矩乘 c，二阶矩乘 c²，更新中的 <span class="math">\(c/\sqrt{c^2}\)</span> 就抵消了。这直觉上解释了整体尺度的近似不敏感；但 <span class="math">\(h_n\)</span> 是聚合后的梯度，而内部的梯度成分的占比是 AdamW 无法看到的。</p>

<p>具体地，从零矩状态出发，递推得到 <span class="math">\(m'_n=c m_n\)</span>、<span class="math">\(q'_n=c^2q_n\)</span>；偏差修正也保留这个关系。于是缩放后的更新比值为 <span class="math">\(\widehat m'_n/(\sqrt{\widehat q'_n}+\epsilon)=\widehat m_n/(\sqrt{\widehat q_n}+\epsilon/c)\)</span>。忽略 ε 时严格抵消；保留 ε 时近似成立。</p>

</details>

<p>因此，<strong>同样参与 loss，不等于同样影响更新</strong>。反向的梯度会抵消；正交的梯度不直接抵消，也不相互增强。许多强而不相关的方向，还会稀释一个有用方向在合力中的占比。</p>

<blockquote><strong>Insight：并不是每个被强化的action都对reward有用，而有用方向必须在合力中活下来。</strong> 分析 Diffusion RL，可以使用“向量分解与向量和”的心智模型，要看各个 state 上的 action 被怎样推拉、这些力怎样映射到参数，以及聚合之后谁主导、什么力成分被抵消。这是后文理解 Reverse RL 和 Forward RL 的共同起点。</blockquote>

<h2 id="zh-reverse" data-section-key="reverse">2. Reverse RL：为什么训练更少的 action，反而可能学得更好？</h2>

<h3 id="zh-detail-3">2.1 Preliminary：Reverse-Process RL</h3>

**从确定的去噪，到可计算概率的 action。** 全文规定 t=1 是噪声端，t=0 是数据端，去噪从 1 走向 0。为与前文“拉向生成目标”的方向一致，这里 vθ 表示生成方向的速度；它与 <a href="https://arxiv.org/html/2505.05470v1#S4.SS2">Flow-GRPO 原文</a>沿加噪时间定义的速度差一个负号。令 r=1−t 为向前增长的生成时间。原来的 ODE 是确定的；Flow-GRPO 加入随机探索，同时修正漂移：

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

<p>c 是 prompt，Wᵣ 是 Wiener 过程，σₜ 控制探索强度；精确速度场下，这个连续时间 SDE 与 ODE 的边缘分布一致。第 k 个转移的噪声时间为 tₖ，步长 <span class="math">\(\delta_k=t_k-t_{k+1}>0\)</span>。Euler–Maruyama 离散化后，state <span class="math">\(s_{i,k}=(c,t_k,x_{i,k})\)</span> 的 action <span class="math">\(a_{i,k}=x_{i,k+1}\)</span> 就有高斯概率密度：</p>

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

<p>以下只讨论 <span class="math">\(0<t_k<1\)</span>、σₜ>0 的随机转移；确定性端点不能直接代入这个密度。固定采样出来的 state 和 action，对 D 维 action，log density 是：</p>

<div class="math" data-equation="reverse-density">
\[
\log\pi_\theta(a_{i,k}\mid s_{i,k})
=-\frac{D}{2}\log(2\pi\sigma_{t_k}^2\delta_k)
-\frac{\|a_{i,k}-\mu_\theta(s_{i,k})\|^2}{2\sigma_{t_k}^2\delta_k}.
\]
</div>

**终点 reward 怎样训练每个 action？** 同一 prompt 采样 G 条轨迹。终点 reward 在组内减均值、除标准差，得到固定的 advantage Aᵢ，再用新旧策略的密度比构造 PPO 裁剪 loss。G 是轨迹组大小，k 是时步编号，m 是训练时步数。这里先不展开参考策略正则项：

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

**写成 MSE，才能看清每个 action 提供的力。** 在 current=old、裁剪未生效的锚点，负的 advantage×log density 与 PPO loss 的梯度相同。把高斯均值代入，定义固定的 velocity target，便得到这个局部 MSE 梯度提供器：

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

<p>正 advantage 拉向采样 action 对应的 velocity target，负 advantage 推离它；<strong>力的相对尺度还乘着 wₖ，并不只是 advantage。</strong></p>



<p id="zh-detail-4" data-review-section><strong>哪些 action 真正在控制更新？</strong></p>

<p>上面的 MSE 已经给出了线索：即使同一条轨迹的 Aᵢ 相同，不同 timestep 仍有不同的 wₖ、target 残差和 Jacobian。根据 <a href="#eq-zh-chain">梯度映射与聚合公式</a>，最终的合力取决于这些分项向量。我们直接测量谁在合力中留下了方向。</p>

<p>我们测了各训练denoising timestep的action的梯度模长，以及它与合力的 cosine。在一个 microbatch 的探针实验中，最后一个位置与合力的平均 cosine 为 0.948，第一个接近零。后期位置基本决定了合力方向。</p>

<figure id="zh-reverse-gradients" class="figure-compact" data-figure="reverse_gradients"><img src="/assets/blog/diffusion-rl/reverse_gradients_compact.png" alt="图2：逐位置梯度模长与合力方向" loading="lazy" width="2400" height="810"><figcaption>图2：逐位置梯度模长与合力方向。浅线为 43 次探针测量，粗线为均值；测量对象是第一个 microbatch 的梯度，cosine 相对于该 microbatch 的合力。位置 0 → 8 对应从高噪声到低噪声。</figcaption></figure>

<p>全文规定去噪从 t=1 走向 t=0，越靠近 1，越代表高噪声。</p>

<p>相邻位置的平均 cosine 只有 0.003，说明平均协同很弱，可以近似认为每一对都正交， 这和Reverse Process RL的local探索重合， 即一个轨迹里面每一个action 的随机性都是独立的， 所以它们的梯度协同性和拮抗性同时很弱。但更重要的现象是，早期方向虽然也被训练，合力却几乎没有沿着它们走。</p>

<blockquote><strong>Observation：覆盖不等于贡献。</strong> 训练一个 action，不保证它的方向有影响。在这组探针里，早期 action 也参与了训练，但后期梯度完全主导了合力。</blockquote>

<p><strong>分母给低噪声 action 更大的相对权重。</strong> Flow-GRPO 原文使用 <span class="math">\(\sigma_t=\alpha\sqrt{t/(1-t)}\)</span>，α 是固定探索系数。代入 <a href="#eq-zh-reverse-mse" data-equation-ref="reverse-mse">MSE 权重</a>：</p>

<div class="math" data-equation="reverse-weight">
\[
w_k=\frac{\delta_k(1+\alpha^2/2)^2}{2\alpha^2}\frac{1-t_k}{t_k}.
\]
</div>

固定步长时，t 越小、噪声越低，wₖ 越大。这里变小的是方差分母，变大的是它的倒数权重：低噪声部分在聚合前就被放大。实测合力严重倾向后期 action，正是这种隐式加权值得警惕的表现。

<h3 id="zh-detail-5">2.3 理解去噪过程：一个特殊的马尔可夫决策过程（MDP）</h3>

一张几乎生成完的图，下一步可能很难改变其质量，却仍然收到整条轨迹的 advantage。

**这时，训练信号可能主要在奖励“这个 action 接手了什么状态”，而不是“这个 action 做出了什么贡献”。**

<p><strong>Group advantage 多带了什么？</strong> 对只有终点 reward 的去噪 MDP，记第 i 条轨迹在第 k 步的 state、action 为 sᵢ,ₖ、aᵢ,ₖ，终点 reward 为 Rᵢ。采用标准强化学习中 MDP 对应的价值函数概念 <span class="math">\(V(s)\)</span> 和 <span class="math">\(Q(s,a)\)</span>：Qₖ(s,a) 是采取这个 action、再按 rollout policy 继续生成后的期望 reward；Vₖ(s) 是在同一 state 下，对该 policy 的不同 action 取平均。真正属于当前 action 的 advantage，与 prompt group 给出的信号分别是：</p>

<div class="math" data-equation="local-advantage">
\[
\begin{aligned}
A^{\rm local}_{i,k}&=Q_k(s_{i,k},a_{i,k})-V_k(s_{i,k}),\\
A^{\rm group}_i&=R_i-b_G(c).
\end{aligned}
\]
</div>

<p><span class="math">\(c\)</span> 是 prompt，<span class="math">\(b_G(c)\)</span> 是该 prompt group 的平均终点 reward。标准 RL 优化的应该是 <span class="math">\(A^{\rm local}\)</span>，即该动作真实的优势函数，但 Flow-GRPO 方法实际优化的是 <span class="math">\(A^{\rm group}\)</span>，即基于 group mean 的优势函数。下面分析除以 group STD 前的中心化量。两种 advantage 的差可以精确分解为：</p>

<div class="math" data-equation="credit">
\[
\begin{aligned}
A^{\rm group}_i-A^{\rm local}_{i,k}
&=\underbrace{V_k(s_{i,k})-b_G(c)}_{\text{State-baseline mismatch}}\\
&\quad+\underbrace{R_i-Q_k(s_{i,k},a_{i,k})}_{\text{Monte Carlo noise}}.
\end{aligned}
\]
</div>

<p>固定同一 prompt 下的 G 个 state：<span class="math">\(\mathcal S_k=(s_{1,k},\ldots,s_{G,k})\)</span>，按同一 rollout policy 对 action 和后续生成求期望。噪声项均值为零，baseline 的期望为 <span class="math">\(\overline V_k=G^{-1}\sum_j V_k(s_{j,k})\)</span>：</p>

<div class="math" data-equation="credit-expectation">
\[
\begin{aligned}
\mathbb E[A^{\rm group}_i-A^{\rm local}_{i,k}\mid\mathcal S_k,c]
&=\underbrace{V_k(s_{i,k})-\overline V_k}_{\text{State-value mismatch}}+\underbrace{0}_{\text{Mean Monte Carlo noise}}\\
&=\frac1G\sum_{j=1}^G\bigl[V_k(s_{i,k})-V_k(s_{j,k})\bigr].
\end{aligned}
\]
</div>

剩下的是当前 state 相对组内平均价值的偏差。其组内均值为零，平方均值却等于 state-value 方差：

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

<p><strong>不同轨迹已经携带的价值差异越大，group baseline 对各个 state 的错配就越大。</strong></p>

**实验：固定 state，再看动作还能改变多少。** 在选定时步，保留同一 prompt 下的 P 个 prefix（各固定一个 state），每个分叉 M 次并打分。同 prefix 的分支构成一组；分叉位置用 CPS，其余位置用 ODE。终点 reward 记为 Rₚ,ⱼ⁽ᵏ⁾。

- **组内差异**：各 prefix 的 reward STD，再取平均，衡量固定 state 后的变化空间。
- **组间差异**：各 prefix 的平均 reward，再算 STD，衡量已有 state 的价值差异。

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

早期的组内 / 组间 STD 为 **0.4543 / 0.3651**；晚期变为 **0.1709 / 0.5179**，相对大小反转。

<figure id="zh-shared-noise" class="figure-compact" data-figure="prefix_spread"><img src="/assets/blog/diffusion-rl/prefix_spread_compact.png" alt="图3：Prefix reward spread 与共享 initial noise 的学习曲线" loading="lazy" width="2400" height="990"><figcaption>图3：左，同 prefix 内的 reward STD 均值，与不同 prefix 平均 reward 的 STD；Early / Late 对应第 1 / 8 步分叉。右，组内共享 vs 独立 initial noise，全部时步训练，使用原 SDE 与方差分母。横轴为日志 step；保留全部归档 eval 点和实际终点，不平滑。</figcaption></figure>

<p>各 prefix 的平均 reward 估计其在 probe 策略下的 V，因此组间 STD 的平方对应 <a href="#eq-zh-credit-variance" data-equation-ref="credit-variance">state-value 方差</a>，但含有限分支的估计噪声。结果支持：<strong>晚期 state 价值差异更大，用一个 group mean 替代各自的 V，错配也更重。</strong></p>

**高噪声早期，动作仍有较大的 reward 杠杆；低噪声晚期，结果更多由继承的 state 决定。** Reverse RL 强化的却是 action。这就容易把已有的 state 质量，当成当前动作的贡献。实验测的是分叉后的终点差异，并非逐动作的精确 Q 方差。

<p><strong>换成 state baseline，会发生什么？</strong> 我们保留 Flow-GRPO 的 PPO 裁剪目标，用 critic 预测各个 state 的价值，将 advantage 改为基于 <span class="math">\(R_i-\widehat V_k(s_{i,k})\)</span> 的估计。这个历史版本的早期 train、eval reward 都升得更快：logged step 240，eval 从 group-advantage 对照的 <strong>0.527</strong> 提高到 <strong>0.664</strong>。</p>

<figure class="figure-compact" data-figure="critic_advantage"><img src="/assets/blog/diffusion-rl/critic_advantage.png" alt="图3b：Group advantage 与 critic advantage 的学习曲线" loading="lazy" width="2400" height="900"><figcaption>图3b：Group advantage 与 PPO-style critic advantage。两者均为 SD3.5-M、GenEval、K=24、10-step SDE rollout、LR=3e-4，无外部 CFG、KL=0；critic 版本关闭 initial-noise selection，使用 centering/STD normalization、5 轮 warmup、15 轮 ramp 和 ranking loss。左为 train，右为 eval；横轴为日志 step，保留取回曲线的实际终点，不平滑。</figcaption></figure>

<p>Critic 需要额外训练，也有估计误差。我们只用了一个很简单的 critic，并且没有进行调优。这组结果支持一个实用方向：<strong>与其让每个 action 接收相同的轨迹评价，不如估计它接手的 state 已经值多少，再评价它的增量贡献。</strong></p>

<p id="zh-noise-time-clue" class="structural-clue">LLM 也有关键决策和无关紧要的动作，但 token 位置通常不足以识别关键决策。Diffusion 多了一个结构线索：噪声时间 t 告诉我们样本形成到了哪个阶段，可以据此寻找更可能具有强 action credit 的训练位置。</p>

<p>把这点接回上一节，矛盾就清楚了：<strong>实际主导更新的 action，未必是能有效改变 reward 的 action，或者说 credit assignment 更差。</strong> 如果后期信号更多混入继承的 state 质量，却又被 loss 的隐式 weighting 放大，合力就可能偏向信息较弱的方向。这是高噪声训练窗口值得尝试的原因：既少做 backward，也让更有 action credit 的方向获得影响力。</p>

<blockquote><strong>Insight：</strong>高 reward 不等于高 action credit。 Prompt-group advantage 混合了“接手的 state 有多好”与“当前 action 改善了多少”。去噪后期，前者可能更强、后者却更弱；Flow-GRPO 的低噪声加权还可能让这些 credit 较弱的动作主导更新。反过来，Diffusion 的噪声时间提供了一个结构线索：选择高噪声训练窗口，可能同时减少计算，并让更有 reward 杠杆的动作在聚合梯度中获得更大影响力。</blockquote>

<p>Diffusion RL 和 LLM RL 还有一个区别是，在第一个 action 之前，initial noise 就已带来不同的起点价值。我们的实验发现，不同 initial noise 的价值差异在视频里面更为明显。根据<a href="#eq-zh-credit-variance" data-equation-ref="credit-variance">状态价值方差公式</a>，这会增加 advantage 估计误差，使得 credit assignment 变弱。让一个 prompt group 共享一个 initial noise 是一个解决方案。<a href="#zh-shared-noise">图3右</a>展示了这一对照：在 logged step 240，共享 initial noise 的 eval 为 <strong>0.656</strong>，独立 noise 对照为 <strong>0.527</strong>，支持先控制起点差异，再比较 action 的贡献。</p>

<h3 id="zh-detail-6">2.4 训练的位置，不一定是收益出现的位置</h3>

<p>我们做一个很tricky的实验： 利用LoRA training的特性来分离“哪些timestep被送进去训练”和“训练影响到哪些timestep”。 <em>只在某一步训练 LoRA</em>，但根据全局参数共享，同一套参数也能作用于其他时步。<em>推理时，通过逐时步开启或关闭 LoRA，控制这次训练得到的函数变化在哪里生效</em>。具体地， 我们做两组实验， 分别只训练第一个timestep（即非常高噪声的区域）和第八个timestep（非常低噪声的区域），对于两组训练结果， 在eval推理的时候分别比较：</p>

<p></p>

<ul><li><strong>全部timestep都关闭LoRA</strong>：相当于没有训练。</li><li><strong>只开第 2–8 步</strong>：第一步关闭，只让未训练位置使用 LoRA。</li><li><strong>只开第 1 步或只开第 8 步：</strong>如果只让被训练的timestep影响到那一个timestep，会如何？</li><li><strong>全部开启</strong>：探究被训练timestep对全部timestep的共同作用效果。</li></ul>

<figure class="figure-compact" data-figure="gate_transfer"><img src="/assets/blog/diffusion-rl/gate_matrix_ocr_updated.png" alt="图4：只训练第一步或第八步，比较七个未训练位置启用 LoRA 的 OCR 收益" loading="lazy" width="4800" height="1860"><figcaption>图4：120 rollout 后的 LoRA 时步干预。左只训练第 1 步，七个未训练位置为第 2–8 步；右只训练第 8 步，七个未训练位置为第 1–7 步。纵轴为相对全部关闭 LoRA 的 OCR 增量，误差线为64条固定 prompts 上配对差值的一个标准误。两组 all-off 基线均为0.064。</figcaption></figure>

<p>有如下几个有信息量的观察：</p>

<ul><li>只训练第一个timestep的时候：<ul><li>推理时关闭第一个timestep的LoRA，开启2-8 timestep的LoRA， 分数仍然大幅上涨， 逼近于默认全部timestep LoRA都开启的情况。</li><li>推理时只开启第一个timestep的LoRA，关闭2-8 timestep的LoRA， 分数仍然上涨但是幅度很小，效果不太好。</li></ul>

</li><li>只训练第八个timestep的时候：<ul><li>推理时只开启第八个或第一个timestep的LoRA，分数基本没有上涨。</li><li>推理时关闭第八个timestep的LoRA，开启1-7 timestep的LoRA，分数显著上涨。但是涨幅仍远远不如 训练Step 1，推理开启step 2-8的LoRA 实验。</li></ul>

</li><li>两组实验里面推理开启全部LoRA都是效果最好的。</li></ul>

<p>能得到的结论有：</p>

<ul><li>如果两组实验都只让被训练的timestep影响自身timestep， 隔离对其他timestep的影响： 两组的上涨都很少，但Step 1 组上涨远超 Step 8组。说明 Step 8 组训练的action基本和reward 没什么正相关， 相比于Step 1 组。</li><li>如果两组实验都只让被训练的timestep影响1-8中间的其他timestep，但不影响自身timestep： Step 1 组的reward上涨逼近最高（默认所有timestep都被影响的情况）， 但Step8组有上涨但涨幅明显更低。 说明第一步的action训练大大增益了后面timestep选择更能通往高reward的action，而第八步的action和reward因果性不大， 无法很好地增益其他timestep。</li></ul>

> **Insight：选择训练时步，是选择梯度来源，不是限定收益出现的位置。** 有效的动作信号可以通过共享参数，在其他时步生效。

<p></p>

<h3 id="zh-detail-7">2.5 理解 Flow-GRPO-Fast：更高效的Reverse RL算法， 缓解信用分配问题</h3>

前面的矛盾是：低噪声 action 容易接收到 state 已经携带的 reward，却可能主导更新。**Flow-GRPO Fast 在固定高噪声区域内随机选一个连续短窗口，把有限的 backward 留给更可能改变结果的 action。** 每轮窗口的位置可以不同，不必训练整个高噪声区域。

<p>设完整轨迹有 N 个转移，固定高噪声候选区域 <span class="math">\(\mathcal H_\tau=\{k:\tau\leq t_k&lt;1\}\)</span>，其中 τ&gt;0 是噪声时间阈值；只考虑前文可计算随机转移密度的位置。指定窗口长度 m，在所有能完整落入该区域的起点中均匀采样 j。窗口内随机探索，窗口外确定性生成；保留前面的 group advantage 与 PPO ratio，只在选中的短窗口上训练：</p>

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

**伪代码：Flow-GRPO Fast 的高噪声训练核心**

```text
输入：rollout sampler、噪声时间阈值 τ、窗口长度 m、group size G
每轮：
  冻结 old policy，确定固定的高噪声候选区域
  从合法起点中均匀采样 j，选择连续 m 步窗口
  用 old policy 为每个 prompt 生成 G 条完整轨迹
    窗口内随机探索，窗口外确定性生成
  给最终结果打分，计算各轨迹的 group advantage
  只缓存短窗口内转移的 state、action 和 old log-prob
  从窗口内转移中取 microbatch，计算 current/old PPO ratio
  对 PPO clipped loss 做 backward，按既定累积方式更新参数
```

<p><strong>选时步是硬加权，调整 timestep weighting 是软加权：两者都在分配哪些 action 能影响合力。</strong> 硬加权把窗口外的权重置零，还能跳过对应的训练 forward/backward；每条轨迹的训练样本从 N 个减到 m 个。2.4 的 LoRA 实验也说明，高噪声位置学到的函数变化可以在低噪声位置产生收益，不必逐步训练才能逐步受益。</p>

<p><strong>在选中的窗口内，还可以进一步做 timestep weighting。</strong> 沿用前面的采样公式：σₜₖ 控制探索强度，δₖ 是步长，ξᵢ,ₖ 是标准高斯噪声。因此 action 与均值 μ_old 的差，就是 <span class="math">\(\sigma_{t_k}\sqrt{\delta_k}\,\xi_{i,k}\)</span>。</p>

<p>MSE 求导给出这个残差；从转移均值换到 velocity，再乘 <span class="math">\(\delta_k b_{t_k}\)</span>。把 advantage Aᵢ 和选定的分母乘进去，就得到 current=old 时的输出力：</p>

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

<p>bₜ 是前面 SDE 中 velocity 的系数，在本文的噪声日程下为常数。固定步长时，σₜₖ 随噪声时间 t 增大，因而：</p>

- **去掉分母：力随 σₜₖ 增大，强调高噪声。**
- **分母用标准差的一次方：消除噪声尺度的偏重，让训练时步在这一尺度上平权。**
- **保留方差分母：力随 1/σₜₖ 增大，强调低噪声。**

<figure class="figure-compact" data-figure="reverse_learning"><img src="/assets/blog/diffusion-rl/reverse_learning_compact.png" alt="图5：分母对照与 Flow-GRPO Fast 对比 Naive Flow-GRPO" loading="lazy" width="2400" height="930"><figcaption>图5：左，全部时步训练，去掉方差分母更早取得 reward 提升；右，Flow-GRPO Fast 的早期 3 步窗口与 Naive Flow-GRPO 对比。横轴为日志 step，纵轴为 GenEval score；展示归档原始 eval 点，不平滑。</figcaption></figure>

<blockquote><strong>Insight：让对 reward 更有因果贡献的 action，在总梯度里占更大比重。</strong> Flow-GRPO-Fast 通过只训练高噪声 action，集中保留这些有效信号， 系统性地改善信用分配问题；共享参数更新再把它们带到未训练的 state，让低噪声速度场也受益。</blockquote>

<p>注意，它能有效的本质是得益于Diffusion去燥过程这个MDP的特殊（<a href="#zh-noise-time-clue">结构线索，见2.3</a>），让我们有了一些可利用的先验信息， 缓解信用分配问题。但仍然不是从根源解决Reverse-Process RL信用分配的根本局限。并且， 也不能在所有场景下一味无脑地只训练高噪声，还需要尊重reward model的性质：因为我们优化的是<span class="math">\(R(x_0)\)</span>， 而timestep t 能告诉我们的是<span class="math">\(x_0\)</span>。所以需要分析<span class="math">\(R(x_0)\)</span>会响应<span class="math">\(x_0\)</span>的哪些成分。我们的经验是：高噪声区域很大程度上决定了<span class="math">\(x_0\)</span>欧氏空间的大体位置， 这个信息几乎一定是会引起reward model的高度响应的，所以Flow-GRPO-Fast思路几乎可以快速提升reward到一定水平。 但部分Reward Model 对<span class="math">\(x_0\)</span> 欧氏空间位置的细小变化仍然敏感，所以在这种情况下适当加入中低噪声的action训练， 对提高reward上限有帮助。</p>

<h2 id="zh-forward" data-section-key="forward">3. 理解 Forward Process RL</h2>
<p>Diffusion RL，顾名思义，由两部分组成：Diffusion 和 RL。Reverse RL 在模型自己探索到的 state 上强化自己产生的 action，和传统 RL 的精神比较重合，笔者称之为“RL 派”；而笔者认为 Forward RL 从精神上更贴近 Diffusion 本质，笔者称之为“Diffusion 派”。它给生成终点打分，再重新加噪，<strong>用类似于 Diffusion 预训练的思想</strong>把反馈转成加噪点上的回归训练。Forward RL 相比于 Reverse RL，需要或者说拥有更美妙的数学性质支撑，但是美妙的数学性质在实际训练中往往也意味着更大的挑战。这个章节尽量用通俗易懂的语言，详细剖析 Forward RL 的动机和本质、笔者认为它在实际场景中的局限，以及克服这些局限的配方。</p>
<h3 id="zh-detail-8">3.1 Preliminary：Forward Process RL</h3>

<p>首先约定，冻结 rollout policy，记它通过反向去噪生成的终点分布为 <span class="math">\(p_{\rm old}(x_0\mid c)\)</span>，c 是 prompt。沿用生成方向的速度约定，t 从 1 降到 0。直线加噪路径及其样本 target 为：</p>

<div class="math" data-equation="forward-path">
\[
x_t=(1-t)x_0+t\epsilon,\qquad u=x_0-\epsilon.
\]
</div>

x₀ 是生成终点，ε 是重新采样的 noise。这里的含噪 state 来自 endpoint 的重新加噪，**不是原来 reverse rollout 经过的 state**。

<p>把 reward 映射成固定的 goodness <span class="math">\(r(x_0,c)\in[0,1]\)</span>，<strong>用它重加权终点分布</strong>。记 <span class="math">\(\bar r_c=\mathbb E_{p_{\rm old}}[r\mid c]\)</span>，在 <span class="math">\(0&lt;\bar r_c&lt;1\)</span> 时定义：</p>

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

<details class="derivation" id="zh-forward-normalization-proof">
<summary>展开推导：分母为什么是平均 goodness？</summary>
<p>概率密度的积分必须等于 1，所以重加权之后要除以总权重：</p>
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
<p>分别除以这两个归一化常数，就得到上面的正负分布。</p>
</details>

<p>现在我们有了三个分布：正分布， 负分布和原分布。注意， 前两个分布是定义出来的， 因为它们一定存在， 但我们暂时没有一个policy可以生成符合这些分布的点， 后文会讲怎么解决这个问题；直觉上， 正分布强调偏向高分终点，负分布偏向低分终点。 </p>

<p>像预训练一样，我们对三个 endpoint 分布分别采用 Flow Matching 式的直线加噪：从 <span class="math">\(p_b\)</span> 中采样 x₀，再独立采样 timestep t 和噪声 ε，构造 <span class="math">\(x_t=(1-t)x_0+t\epsilon\)</span>，样本速度 target 为 <span class="math">\(u=x_0-\epsilon\)</span>。其中 <span class="math">\(b\in\{+,\mathrm{old},-\}\)</span>，三个采样过程只有 x₀ 的来源不同，其他规则相同。<strong>记这个过程产生的 x₀、ε、t、<span class="math">\(x_t\)</span> 的联合分布为 <span class="math">\(Q_b\)</span></strong>，prompt c 作为给定条件。固定含噪 state <span class="math">\(s=(x_t,t,c)\)</span>，仍然可能有不同的 x₀ 和 ε 配对产生它。根据 Flow Matching 理论，理想生成方向速度场 <span class="math" data-equation="nft-ideal-fields">\(v_{\rm fwd}^{b}(s)\)</span>，就是这些配对所对应的 u 的条件平均。直观上，它是用分布 <span class="math">\(p_b\)</span> 的样本进行标准 Flow Matching 预训练后，模型理想情况下学到的速度场。<span class="reader-intuition">到这里读者可能已经有了很强的直觉：我们只需要通过MSE regression把我们模型输出的速度场往<span class="math">\(v_{\rm fwd}^{+}\)</span>拉就可以了，因为这样就能把<span class="math">\(p_{\rm old}\)</span>往<span class="math">\(p_+\)</span>微调，这样的话高奖励的endpoint会获得更大概率。并且基于此，读者可能会思考疑惑，为什么还需要负分布<span class="math">\(p_-\)</span>？这个直觉是正确的，但这里的核心就在于我们真正能训练的速度场是模型输出的反向去燥速度场，即<span class="math">\(v_{\rm rvs}^{\rm old}(s)\)</span>。<span class="math">\(v_{\rm rvs}^{\rm old}(s)\)</span>通常不等于理想前向加燥速度场<span class="math">\(v_{\rm fwd}^{\rm old}\)</span>。这里可能会比较confusing，请带着这个直觉和疑惑继续往下读。</span>我们暂且先假设并同意负分布也是 somehow 有帮助的，然后先完成 Forward Process RL 的剩余心智模型搭建：</p>
<p>记 <span class="math">\(q(s)=\mathbb E_{Q_{\rm old}}[r\mid s]\)</span>、<span class="math">\(g(s)=\operatorname{Cov}_{Q_{\rm old}}(r,u\mid s)\)</span>。在条件矩存在且 <span class="math">\(0&lt;q(s)&lt;1\)</span> 时，我们能推出三个理想场满足：</p>

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

<details class="derivation" id="zh-forward-collinearity-proof">
<summary>展开推导：reward 重加权为什么得到共线的三个理想场？</summary>
<p>加噪规则与 noise 分布不变，所以正分布的联合权重相对旧分布仍是 <span class="math">\(r/\bar r_c\)</span>。固定 s 后，Bayes 公式再除以该权重的条件均值 <span class="math">\(q(s)/\bar r_c\)</span>，prompt 级常数便消去；负分布同理。下面省略 q、g 和速度场的 s 参数，期望均在旧分布下：</p>
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
<p>这也说明概率加权估计的对象：ru 的条件均值是 <span class="math">\(qv_{\rm fwd}^{+}\)</span>，不是直接等于正场。要得到正场还需除以 q；有限样本中，用两个样本均值之比估计它，并不自动无偏。</p>
</details>

<p><strong>这也是Forward RL最核心的公式。用直觉语言解释这个公式: 对于任何一个前向加燥可能的state，三个分布在此state上诱导的理想前向加燥速度场共线。从原始理想场拉向正场，与推离负场，沿着同一条 reward 改善方向。</strong>两者只差正的尺度因子。所以往正场拉和推离负场的理想情况下效果是一样的（recall <a href="#zh-target">§1</a> MSE target 只是梯度提供器，所以正负场的推拉力反映到梯度上方向是一样的，所以对训练dynamics来说近乎等价（方向一致，只有尺度系数的差异））</p>

<p>那么如何训练？和预训练一模一样的思路，<em>假设我们可以从正负两个分布里面采样，然后从样本加燥构造前向速度场的无偏估计u，用训练对象来MSE regress这个无偏估计，最后训练对象会收敛理想前向速度场</em> (recall Flow-Matching 原论文的预训练算法）。关键于这个接下来如何从正负分布采样？事实上我们没办法直接采样，因为我们拿不到原分布的表达式，我们唯一能做的事情是调用ODE solver给原分布采样。于是现在问题变成了一个Stochastic Simulation问题, 以拟合正分布为例：</p>

<p>固定 prompt c，但不固定 xₜ。对每个 endpoint，按既定训练分布采样 t，并独立采样 <span class="math">\(\epsilon\sim\mathcal N(0,I)\)</span>，构造 <span class="math">\(x_t=(1-t)x_0+t\epsilon\)</span>。记 <span class="math">\(h_\theta(x_0)=\mathbb E_{t,\epsilon}\!\left[\frac12\|v_\theta(x_t,t,c)-(x_0-\epsilon)\|^2\right]\)</span>：它是这个 endpoint 重新加噪后的平均 MSE，隐含固定的 c。问题可以写成：</p>

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

<p><strong>解法：对 L₊ 进行代数变换。</strong>将 h 当作整体，代入正分布定义，再提出与 x₀ 无关的归一化常数：</p>

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

<p>从旧模型采样 x₀，重新采样 t、ε，计算 r×MSE 并取平均，即可无偏估计上述未归一化期望。在固定 prompt 下，<span class="math">\(\bar r_c\)</span> 由冻结旧分布和固定的 r 映射决定，与 θ 无关，所以 r 加权 MSE 与 L₊ 只差一个正的常数。如果算梯度， 梯度方向一致， 只有前面系数的差别。(注： 另一种做法是Rejection Sampling: 把 r 当作<strong>接受样本的概率</strong>：从旧分布采样，以概率 r 接受，接受后的样本就来自正分布，再重新加噪训练。负分布同理，只需把 r 换成 1−r。)</p>

<p><strong><a href="https://arxiv.org/html/2509.16117v1#S3">DiffusionNFT</a> 如何实现Forward RL？</strong>每轮先用冻结的 old policy，为每个 prompt 生成 K 个 endpoint 并打分。它不直接使用 raw reward，而是先减去 group 平均 reward，再缩放、截断，映射到 [0,1] 作为一个样本的组内&quot;goodness&quot; （好坏度）：<span class="math">\(r_i=\frac12[1+\operatorname{clip}((R_i-\bar R_c)/Z_c,-1,1)]\)</span>。这里 <span class="math">\(\bar R_c\)</span> 是 group 均值，<span class="math">\(Z_c&gt;0\)</span> 控制尺度。高于均值的样本 r&gt;0.5，低于均值的样本 r&lt;0.5；接下来，用 r 和 1−r 分别给正、负分支加权。</p>

<p>训练时，把生成的 x₀ 重新直线加噪，得到 xₜ 和 Flow Matching target <span class="math">\(u=x_0-\epsilon\)</span>。在同一个 state 上，分别计算冻结的 <span class="math">\(v_{\rm old}\)</span> 和待训练的 <span class="math">\(v_\theta\)</span>。NFT 并不分别训练两个模型，而是围绕 old 输出，用同一个 current 输出构造两个镜像分支：</p>

<div class="math" data-equation="mirror-branches">
\[
\begin{aligned}
v_\pm(x_t,t,c)
&=v_{\rm old}(x_t,t,c)\\
&\quad\pm\beta\bigl[v_\theta(x_t,t,c)-v_{\rm old}(x_t,t,c)\bigr].
\end{aligned}
\]
</div>

<p>β&gt;0 是控制幅度的超参数。这里是在用<span class="math">\(v_{\rm old}\)</span> 和 <span class="math">\(v_\theta\)</span>进行插值，然后去回归<span class="math">\(u\)</span>。loss 为：</p>

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

<p>优化完这批样本后，NFT 用 EMA 更新 old：<span class="math">\(\theta_{\rm old}\leftarrow\eta\theta_{\rm old}+(1-\eta)\theta\)</span>，再用它生成下一批样本、构造下一轮的 anchor。η 越大，old 跟随 current 越慢。</p>

<p id="zh-nft-questions"><strong>DiffusionNFT 还抛出了一些实验结论，但未给出机理上的解释：</strong></p>
<ul>
<li>如果 <span class="math">\(v_\theta=v_{\rm old}\)</span>，那么 reward 会崩掉。</li>
<li>如果没有负分支 loss（即后面一项），reward 也会崩掉。</li>
</ul>
<p>回忆我们之前搭建的心智模型（<a href="#eq-zh-forward-regression-goal">公式 21</a> 和 <a href="#eq-zh-forward-weighted-estimator">公式 22</a>），这里会比较 confusing，自然会引出一些问题：</p>
<ol>
<li id="zh-nft-question-mirror">这个看起来绕了一圈的 mirror 构造，究竟让 current 学什么？根据我们之前的推导，这里不应该是用 <span class="math">\(v_\theta\)</span> 去拟合 u 吗？为什么是用 <span class="math">\(v_\theta\)</span> 和 <span class="math">\(v_{\rm old}\)</span> 的插值去拟合 u？</li>
<li id="zh-nft-question-negative">为什么还需要负分支？</li>
<li id="zh-nft-question-goodness">为什么 reward 还需要归一化成 goodness，之前的推导不是使用 raw reward 吗？</li>
</ol>

<h3 id="zh-nft-demystifying">3.2 Demystifying Forward Process RL and NFT-style training</h3>

<h4 id="zh-nft-objective-equivalence">3.2.1 DiffusionNFT objective 的等价转化</h4>

<p>令 <span class="math">\(a=(2r-1)/\beta\)</span>，冻结 target 为 <span class="math">\(v_{\rm tgt}(x_0,t,\epsilon,c)=a(x_0-\epsilon)+(1-a)v_{\rm old}(x_t,t,c)\)</span>。展开两个平方、再配方，可以推出：</p>

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

<details class="derivation" id="zh-nft-target-proof">
<summary>展开推导：两个镜像分支如何合成一个 target？</summary>
<p>固定 prompt c 和一次采样 <span class="math">\((x_0,t,\epsilon)\)</span>，其中 <span class="math">\(x_0\sim p_{\rm old}(\cdot\mid c)\)</span>、<span class="math">\(t\sim\rho\)</span>、<span class="math">\(\epsilon\sim\mathcal N(0,I)\)</span>，并令 <span class="math">\(x_t=(1-t)x_0+t\epsilon\)</span>。记 <span class="math">\(d=v_\theta(x_t,t,c)-v_{\rm old}(x_t,t,c)\)</span>、<span class="math">\(e=(x_0-\epsilon)-v_{\rm old}(x_t,t,c)\)</span>。下面只为展开代数而用 d、e 简写；r 和 old 输出保持冻结。两支残差分别是 βd−e 与 −βd−e，先展开平方，再用 <span class="math">\(a=(2r-1)/\beta\)</span> 配方：</p>
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
<p>这里 <span class="math">\(d-e=v_\theta(x_t,t,c)-(x_0-\epsilon)\)</span>，d 则是 current 与 old 的差。对相同的 x₀、t、ε 采样取期望，就得到正文最后一行。两次变换的常数不同，但都与 θ 无关：<span class="math">\(\mathrm{const}_2=\mathrm{const}_1-\beta^2\mathbb E[a(1-a)\|e\|^2]\)</span>，其中期望沿用相同的采样分布。</p>
</details>

<p>这里的前一项就和我们的<a href="#eq-zh-forward-weighted-estimator">公式 22</a>吻合了，只不过把 <span class="math">\(r\)</span> 换成了 reward 的一个 <span class="math">\(\beta\)</span> 控制的线性变换，这是 DiffusionNFT 的刻意设计。等价变换后可以更清楚地看到 NFT 的本质：前文讨论过预训练式回归训练（第一项） + 一个暂时不明意图的第二项。在完全 on-policy 即 current=old 时，第二项为 0，整个 loss 给 current 的力是 <span class="math">\(2\beta(2r-1)[(x_0-\epsilon)-v_{\rm old}(x_t,t,c)]\)</span>，正好对应高分拉、低分推。3.2.3会剖析第二项的作用。（回应<a href="#zh-nft-question-mirror">问题 1：mirror 构造究竟让 current 学什么？</a>）</p>



<h4 id="zh-detail-9">3.2.2 Forward RL 的根本局限</h4>

<p>回到<a href="#zh-nft-questions">3.1 末尾</a>我们留下的问题，先看最直接的反常现象：正支和负支单独训练都会崩溃，放在一起却能学起来。理论上的正负理想速度场沿着同一个 reward 改善方向，为什么实际训练需要两条分支？</p>

<figure class="figure-compact" data-figure="nft_branch_geometry"><img src="/assets/blog/diffusion-rl/nft_branch_geometry.png" alt="图6：正负分支消融与前向反向自一致性力的几何分解" loading="lazy" width="2400" height="845"><figcaption>图6. 左：单独训练正支或负支失败，联合训练在图示窗口内提高 train GenEval。右：实线为正负分支的力，虚线拆出 reward 与自一致性分量；后者只能部分抵消。图中的 Δ 是理想 reward guidance 方向。</figcaption></figure>

<p>3.1 已证明：正负理想场相对原始 forward 场的 guidance 共线。<strong>但重点在于：实际 reverse 场 <span class="math">\(v_{\rm rvs}^{\rm old}(s)=v_{\rm old}(s)\)</span>，不一定等于其终点分布重新直线加噪诱导的理想场。</strong>记两者之差为 <span class="math">\(\Delta_{\rm fwd}(s)=v_{\rm fwd}^{\rm old}(s)-v_{\rm rvs}^{\rm old}(s)\)</span>。理想 guidance 从 forward 场出发， 所有理论推导也是基于old policy 产生endpoint 分布诱导出来的理想forward场，但实际训练对象确是 reverse process 模型真实输出的速度场，这个 gap 因而进入更新。</p>

<p>先分析<a href="#eq-zh-nft-frozen-target" data-equation-ref="nft-frozen-target">公式 25</a>最后一行的第一项 <span class="math">\(\ell_{\rm fit}=\beta^2a\|v_\theta-u\|^2\)</span>，记 <span class="math">\(f_{\rm fit}=-\nabla_{v_\theta}\ell_{\rm fit}\)</span>。先看它在 current=old 时的条件平均力（期望在 Q_old 下）：</p>

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

<details class="derivation" id="zh-nft-consistency-proof">
<summary>展开推导：实际力如何多出自一致性项？</summary>
<p>固定 s，从公式 25 的第一项求导；r、a、u 均冻结。求导后再代入 current=old，并记 <span class="math">\(e=u-v_{\rm old}\)</span>，把 <span class="math">\(\mathbb E[re\mid s]\)</span> 拆成 covariance 与均值乘积：</p>
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
<p>减去一个固定向量不改变 covariance，所以 reward 项仍是 g。正支单独给出 <span class="math">\(\mathbb E[re\mid s]=g+q\Delta_{\rm fwd}\)</span>；负支给出 <span class="math">\(-\mathbb E[(1-r)e\mid s]=\mathbb E[re\mid s]-\mathbb E[e\mid s]=g-(1-q)\Delta_{\rm fwd}\)</span>。相加后，两份 g 同向，gap 只抵消到 <span class="math">\((2q-1)\Delta_{\rm fwd}\)</span>。</p>
</details>

<p><strong>NFT使用一个样本<span class="math">\(x_0\)</span>加燥得到s, 然后把其goodness <span class="math">\(r(x_0)\)</span>和加的噪声代入则得到 <span class="math">\(\mathbb E[f_{\rm fit}\mid s]\)</span>的一个估计：</strong></p>

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

<p>第一项是 <span class="math">\(2g(s)\)</span> 的估计，第二项是这个样本携带的自一致性力。注意NFT使用的goodness来代替r， 所以估计的无偏性会被一定程度破坏。</p>

<p><strong>为什么还需要负分支？</strong>看实际样本：在 current=old 时，去掉共同的 2β 因子：</p>
<ul>
<li><strong>只有正分支：</strong>力为 <span class="math">\(r(u-v_{\rm old})=r[u-v_{\rm fwd}^{\rm old}(s)]+r\Delta_{\rm fwd}(s)\)</span>。无论样本好坏，都往 u 拉；低分样本只是拉得弱。自一致性分量也始终往理想 forward 场拉近。</li>
<li><strong>只有负分支：</strong>力为 <span class="math">\(-(1-r)(u-v_{\rm old})=-(1-r)[u-v_{\rm fwd}^{\rm old}(s)]-(1-r)\Delta_{\rm fwd}(s)\)</span>。无论样本好坏，都推离 u；高分样本只是推得弱。自一致性分量始终把模型推离理想 forward 场。</li>
</ul>
<p>两支合用，才得到 <span class="math">\((2r-1)(u-v_{\rm old})\)</span>：高于组均值的样本拉，低于组均值的样本推。自一致性系数也从单向的 r 或 −(1−r)，变成可正可负的 2r−1，使不同样本的额外分量有机会在全局跨state 梯度聚合的时候相抵。这同时解释了<a href="#zh-nft-question-negative">问题 2：为什么需要正负分支</a>， 和<a href="#zh-nft-question-goodness">问题 3：为什么需要用normalize reward之后的goodness分数取代这里的r</a>， 舍弃掉无偏性： 因为如果r是goodness，则满足 <span class="math">\(\sum (2r-1)=0\)</span> 也就是不同state上的自一致性力的系数平衡。但 gap 随 state 变化，系数平衡不保证加权向量抵消（<span class="math">\(\sum (2r-1)=0\)</span> 不意味着<span class="math">\(\sum \mathrm{grad}\,(2r-1)f=0\)</span>）。没有抵消干净的残差仍然会减慢训练， 或者reward训练直接崩溃。（如图6 右所示）</p>





<blockquote><strong>Insight：Forward RL 的回归信号有一个隐藏的自一致性项。</strong> 这一项对提升reward没有帮助， 并且会直接打崩训练。 正负分支联合训练和reward归一化必不可少， 因为它们可以一定程度上让该项的梯度在全局抵消， 但是无法完全抵消。</blockquote>

<p><strong>终点分布相同，不代表中间的分布演化相同。</strong> 微调改变了真实的反向去噪 dynamics；把这段从 t=1 到 0 的分布演化反过来看，并不保证与终点重新直线加噪得到的演化一致，两者甚至可能相差很远。在这个意义上，Reverse Process RL尊重了当前模型的反向过程，而不是把直线加噪默认当作它的逆过程。</p>

<h4 id="zh-nft-restoration-role">3.2.3 Restoration 的作用</h4>

<p>3.2.1 将NFT的目标分解为了两项， 上一个小章节研究了第一项里面的问题， 而第二项是对它的弥补。 令 d 为 current−reference，e 为 target−reference，输出空间的力为：</p>

<div class="math" data-equation="mirror-force">
\[
\begin{aligned}
d&=v_\theta-v_{\rm old},\qquad e=u-v_{\rm old},\\
f&=\underbrace{2\beta(2r-1)e}_{\text{feedback guidance}}
-\underbrace{2\beta^2d}_{\text{restoration}}.
\end{aligned}
\]
</div>

<details class="derivation" id="zh-nft-restoration-proof">
<summary>展开推导：恢复力如何从 NFT loss 中出现？</summary>
<p>沿用 3.2.1 的平方展开。r、u 与 old 输出均冻结，所以对 current 输出求导时，只有 d 随之变化：</p>
<p>具体地，<span class="math">\(\nabla_{v_\theta}\|d\|^2=2d\)</span>、<span class="math">\(\nabla_{v_\theta}\langle d,e\rangle=e\)</span>，而 <span class="math">\(\nabla_{v_\theta}\|e\|^2=0\)</span>。逐项求导，再取负号，就是下面的力。</p>
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
<p>恢复力来自 loss 中的二次项，不是 EMA 额外添加的正则项。在 current=old 时 d=0，它为零；current 移动而参考冻结时，它就指向 old。取期望和逐元素平均会保留这个分解。</p>
</details>

<p>第一项包含 3.2.2 的两种 guidance；第二项把 current 拉回 old。恢复力限制偏离旧输出的幅度，但也可能与改善 reward 的方向竞争。</p>

<p>我们做了两组非常有信息量的实验：</p>

<ol><li>保留 EMA rollout、移除训练梯度中的 old-policy anchor：通过将<a href="#eq-zh-nft-frozen-target" data-equation-ref="nft-frozen-target">公式 25</a> 里面的<span class="math">\(v_{\rm tgt}(x_0,t,\epsilon,c)=a(x_0-\epsilon)+(1-a)v_{\rm old}(x_t,t,c)\)</span> 变成 <span class="math">\(u=x_0-\epsilon\)</span> 和 <span class="math">\(v_\theta\)</span>的插值，同时保持<span class="math">\(p_{\rm old}\)</span> 不变（仍然由old policy rollout得到的分布来前向加燥）。 结果是仍然崩溃，和纯on-policy的结果差不多 （图7 b）；</li><li>关掉 EMA 时，把每个rollout的batch 里面的mini-batch数 从1 变成2， 即把更新方式为： 所有样本放在一起做一次更新， 变成把所有样本随机分成两等份， 然后依次做两次更新。 结果是minibatch 数为1 的reward快速上涨然后崩溃， 而2个minibatch的实验 比1个minibatch的实验reward上涨慢约15.2倍（首次达到OCR=0.9需要320步，而非21步）， 但暂时没有发生崩溃 （图7 a)。</li></ol>

<figure class="figure-compact" data-figure="nft_arxiv_restoration"><img src="/assets/blog/diffusion-rl/nft_arxiv_restoration.png" alt="图7：恢复消融、梯度峰值与力的尺度反转" loading="lazy" width="2400" height="1048"><figcaption>图7. 左：β=1、无 EMA，一次更新与两次 disjoint 更新首次达到 OCR=0.9 分别需要21与320个 optimizer steps，后者约慢15.2倍；灰色虚线为一次更新崩溃前的 pre-clipping 参数梯度模长。中：保留 EMA rollout、移除训练 anchor，train GenEval 仍崩溃。右：前18个 rollout 的第二次更新中，恢复输出力 RMS 小9.4倍，参数梯度模长大4.9倍，均为均值之比。</figcaption></figure>



<p><strong>实验 1 区分了“旧数据”和“loss里面的old policy”的作用。</strong>Rollout 仍来自 EMA policy，去掉 loss 中的 old-policy anchor 后，训练却再次崩溃（图7b）。在这组消融里，稳定训练的是 anchor 带来的恢复力，而不是 off-policy rollout 本身。</p>

<p><strong>实验 2 说明：不使用 EMA，第二次更新也会产生恢复力。</strong>第一次更新从 current=old 出发，恢复项为零；参数移动后，第二次更新的 <span class="math">\(v_\theta^{(1)}\)</span> 已脱离冻结的 <span class="math">\(v_{\rm old}\)</span>，于是它的力可拆成 guidance 与返回 old 的 restoration（<a href="#eq-zh-mirror-force">恢复力分解</a>、图8）。第二次用的是另一半样本，不需要是和第一批同样的样本；共享参数已经移动，所以这些 state 同样会感受到偏移。</p>

<p><strong>反直觉的是，恢复力在输出空间更小，在参数空间却更强（<a href="#zh-detail-11">3.3节</a>来系统性探究这个现象）。</strong>把第二次更新的力分解，在相同 current 参数与 state 上测量：前18个 rollout 中，恢复输出力的平均 RMS 小约9.4倍，聚合参数梯度模长却大约4.9倍（图7c）。它限制了漂移，但也严重主导了更新，解释了为什么训练更稳定，却学得慢很多。</p>

<blockquote id="zh-detail-10"><p><strong>Insight：On-policy 和 Off-policy 在 Diffusion RL 里面最本质的区别。</strong>对于这里的 NFT 式回归，关键不是数据有多旧，而是求梯度时 current 是否已经脱离 loss 中的 old anchor。结合<a href="#eq-zh-chain">第1节的力与梯度分解</a>，这种偏移会系统性地产生一股返回 old 的恢复力。EMA 让 old 持续滞后；两次更新则在第一步移动后，继续对同一个冻结 old 更新。两者都是这里的 off-policy 更新， 且我们经验性的发现这个恢复力转化为参数空间的梯度的聚合转化率很高。纯 on-policy 的 current=old 则没有这股力。这或许是Diffusion RL 区别于 LLM RL 的一个非常关键的点，或者说是连续时间和连续空间以MSE loss为底层形式的RL的特性。</p></blockquote>

<p id="zh-repulsion-discussion"><strong>Discussion：Reverse RL 的第二次更新，也会产生这种恢复力吗？</strong>不会自动产生 NFT 那样统一返回 old 的恢复项。回到<a href="#eq-zh-reverse-mse" data-equation-ref="reverse-mse">Reverse RL 的局部 MSE</a>：正 advantage 用正 MSE 拉近 target，负 advantage 用负 MSE 推远 target。NFT 的负支却是用正 MSE 拉向镜像 target。两种“推远”在 old 处可以给出相同梯度，离开 old 后却不同。</p>

<p>固定同一个 state、old 输出与 target u，沿用 <span class="math">\(e=u-v_{\rm old}\)</span>、<span class="math">\(d=v_\theta-v_{\rm old}\)</span>。取 β=1、略去共同的正权重，比较两种实现：</p>

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
<figure class="figure-compact figure-schematic" data-figure="nft_second_update_geometry"><img src="/assets/blog/diffusion-rl/nft_second_update_geometry.png" alt="图8：两次MSE下降方向及第二次更新的guidance与restoration分解" loading="lazy" width="2400" height="2871"><figcaption>图8. 蓝色是第一步的 MSE 下降方向；橙色实线是模型移动后的第二步方向，虚线拆成原 guidance 与返回 old 的 restoration。示意固定一个 state 和 target，不要求实际训练重复同一批样本。</figcaption></figure>
<figure class="figure-compact figure-schematic" data-figure="repulsion_geometry"><img src="/assets/blog/diffusion-rl/repulsion_geometry.png" alt="图8b：负MSE的两次下降方向均远离v tgt，偏移分力推离old" loading="lazy" width="2400" height="2871"><figcaption>图8b. 负 MSE：两次下降方向都远离 v_tgt，偏移分力也推离 old。沿用图8的布局与样式，只反转梯度方向。箭头是可平移的输出力向量。</figcaption></figure>
</div>

<p>在 <span class="math">\(d=0\)</span> 时，两者的力都是 <span class="math">\(-2e\)</span>；移动之后，负 MSE 继续放大偏移，镜像正 MSE 则把偏移拉回。NFT 的正负支虽然 guidance 不同，二次项却同号，所以不同state聚合起来仍是<a href="#eq-zh-mirror-force">统一的恢复</a> 。Reverse RL 的偏移项则随 advantage 改变符号，所以不同state的恢复力的系数和为0，直觉上理解，它会跨state大幅抵消。</p>

<blockquote><strong>Insight：相同的第一次更新梯度，可以隐藏相反的后续动力学。如果当前算法不是纯on-policy更新，那么设计梯度提供器时</strong>，不只要看它现在把模型推向哪里，还要看模型移动后，它会是放大这次偏移，还是恢复这次偏移。</blockquote>

但为什么看起来很小的恢复项，会有这么大的影响？

<h3 id="zh-detail-11">3.3 理解哪种力会很强</h3>

<h4 id="zh-force-comparison">3.3.1 四类力，经过三层后还剩多少？</h4>

<p>回到<a href="#eq-zh-chain">三层心智模型</a>，分析一股力的训练行为， 我们需要分析它的三层转化（<em>即三个量 + 两个率</em>）： 1. 作用到每个state上面的力的大小， 2. 每个state上这个力转化为该state上的梯度的转化率 3. 各个state的梯度聚合在一起， 形成总梯度。我们可以通过测三个量来分析它们：<strong>输出力大小 F → 单样本参数梯度大小 P → 跨 state 聚合梯度大小 A</strong>。两个箭头分别定义两个<em>率</em>：<strong>单样本响应率 = P/F</strong>：单位输出力能产生多大的参数梯度；以及<strong>聚合保留率 = A/P</strong>：各 state 的参数梯度相加后保留了多少。</p>

<details class="derivation" id="zh-force-conversion-proof">
<summary>回忆：三量两率如何连起来？</summary>
<p>沿用前文的输出力 <span class="math">\(f_i\)</span> 与单 state 参数方向 <span class="math">\(g_i=J_i^\top f_i\)</span>，三个量与两个率统一写为：</p>
<div class="math" data-equation="conversion">
\[
\begin{aligned}
F&=\frac1N\sum_i\|f_i\|,\qquad
P=\frac1N\sum_i\|g_i\|,\qquad
A=\left\|\frac1N\sum_i g_i\right\|,\\
\underbrace{\frac{P}{F}}_{\text{单样本响应率}}
&=\frac{\sum_i\|g_i\|}{\sum_i\|f_i\|},\qquad
\underbrace{\frac{A}{P}}_{\text{聚合保留率}}
=\frac{\|\sum_i g_i\|}{\sum_i\|g_i\|},\\
\frac{A}{F}&=\frac{P}{F}\times\frac{A}{P}.
\end{aligned}
\]
</div>
<p>统一模长与归约、分母非零时，上式严格成立。P/F 是按输出力模长加权的单样本响应率。</p>
</details>

<p>在Diffusion RL的实践中， 我们大概会和四种力打交道：</p>

<ul><li><strong>Reverse RL （如FlowGRPO）的 policy 力：</strong>在 old 锚点沿随机 action 残差推拉；advantage 决定正负与尺度。</li><li><strong>Forward RL （如DiffusionNFT）的 guidance 力：</strong>来自 endpoint 的回归请求。排除 current–old 恢复项后，仍包含 reward covariance 和<a href="#zh-detail-9">前向–反向 consistency</a>。</li><li><strong>Off Policy策略产生的恢复力：</strong>撤销 current 相对 old 的变化。这里区分 EMA 的滞后 old，以及第二次更新时冻结的 old。</li><li><strong>KL／reference 力：</strong>限制 current 偏离 base model 太远的力。</li></ul>


<p>在同一个算法pipeline 里面， 我们可以根据<a href="#zh-detail-1">前文的&quot;矢量和&quot;心智模型</a> 来分解不同的力， 然后独立分析每份力的“三个量和两个率”。下表直接给出我们的实验结果：</p>


<figure id="zh-three-layer-probe" class="figure-compact force-summary" data-figure="force_measured_overview">
<div class="force-table-scroll" tabindex="0" role="region" aria-label="四类力的三量两率实验结果">
<table class="force-table">
<thead><tr><th scope="col">力</th><th scope="col">测量来源</th><th scope="col">输出力<br>F</th><th scope="col">单样本梯度<br>P</th><th scope="col">聚合梯度<br>A</th><th scope="col">单样本响应率<br>P/F</th><th scope="col">聚合保留率<br>A/P</th></tr></thead>
<tbody>
<tr data-run="9mau8hwi" data-component="guidance"><th class="force-kind" scope="row">Flow-GRPO<br>guidance</th><td>Flow-GRPO</td><td>0.863</td><td>3.29</td><td>0.978</td><td>3.8</td><td>29.7%</td></tr>
<tr data-run="26rilzmp" data-component="guidance"><th class="force-kind" scope="row">DiffusionNFT<br>guidance</th><td>NFT</td><td>4.08</td><td>169</td><td>41.7</td><td>41.4</td><td>24.7%</td></tr>
<tr data-run="26rilzmp" data-component="restore"><th class="force-kind" scope="row">EMA 恢复力</th><td>NFT</td><td>0.765</td><td>107</td><td>39.8</td><td>139.7</td><td>37.2%</td></tr>
<tr data-run="adnxitig" data-component="restore"><th class="force-kind" scope="row">第二次更新<br>恢复力</th><td>NFT 双更新</td><td>0.567</td><td>78.2</td><td>32.7</td><td>137.9</td><td>41.8%</td></tr>
<tr data-run="9mau8hwi" data-component="reference_mse"><th class="force-kind" scope="row">KL loss</th><td>Flow-GRPO</td><td>0.435</td><td>63.4</td><td>25.8</td><td>145.6</td><td>40.6%</td></tr>
</tbody>
</table>
</div>
<figcaption>表1：五项力的三量两率。F、P、A 均以 10⁻³ 为单位；两率分别为 P/F 与 A/P。参数梯度在 Adam／裁剪前测量。</figcaption>
</figure>

<blockquote class="observation">
<p><strong>Observation：</strong></p>
<ul>
<li>Flow-GRPO 与 NFT EMA guidance 的单样本响应率约为 <strong>3.8、41</strong>，聚合保留率约为 <strong>30%、25%</strong>。不同算法的单样本响应率构成它们算法行为差异（效率和稳定性差异）的关键，而聚合保留率几乎没有明显差异。</li>
<li>NFT的两种恢复力的单样本响应率显著高于NFT guidance的单样本响应率，以及聚合保留率也更高。</li>
<li>KL Loss 的单样本响应率和聚合保留率很高。</li>
</ul>
</blockquote>

注意：这张表里面NFT guidance的数据仅仅只是前期平稳期的数据，我们实际上发现NFT guidance的单样本响应率会发生spike（<a href="#zh-force-stability">图10</a>），但是这张表里面没有体现。
<h5 id="zh-nft-art-dynamics">1. 从“三量两率”视角再深入剖析Forward Process RL的失稳来源</h5>

把纯 on-policy NFT 跑长一些，分别追踪 guidance 的输出力、单样本响应率、保留率与 OCR reward：

<figure id="zh-force-stability" class="figure-compact" data-figure="force_stability"><img src="/assets/blog/diffusion-rl/force_stability_nft.png" alt="图10：NFT 的输出力、单样本响应率、保留率和 OCR reward 长期曲线" loading="lazy"><figcaption>图10：纯 on-policy NFT 的失稳主要伴随单样本响应率尖峰，而非聚合保留率暴涨。横轴为 rollout；响应率对有效 state 等权平均。</figcaption></figure>

<p>NFT 的单样本响应率 从初始约 <strong>44</strong> 升至 rollout 44/45 的 <strong>1239/2784</strong>，伴随 reward 崩溃；保留率并没有出现同量级的暴涨。Reward Collapse的原因<strong>不是NFT的力突然增大，也不是“更多梯度终于一致了”，而是单位输出力映射出的原始参数梯度突然变大几个量级， 导致最后用于更新的梯度发生spike。</strong></p>

<p>这照应了前文讲到的<a href="#zh-detail-9">Forward RL 的根本局限</a>：NFT的实际力不止包含理想的reward-improving guidance ，也带着self-consistency force。<strong>这部分残差是否变得过度敏感、造成 单样本响应率 spike，是一个具体的机制嫌疑。</strong>从机理上来讲， 随着训练进行， 反向去燥的速度场可能会越来越偏离相同endpoint诱导的直线前向加燥的理想速度场，或者说这个consistency force 越来越不可控， 这也和图(b)的趋势吻合。</p>

<h5 id="zh-restoration-rescue">2. 两种恢复力怎样救援？</h5>

<p>表1已经显示：NFT EMA 与第二次更新的恢复 单样本响应率和聚合保留率都很高， 单样本响应率为约为 <strong>140、138</strong>，保留率约为 <strong>37%、42%</strong>。所以输出空间看起来小，但是转化为梯度的约束很强， 会明显挤占提升reward的力的梯度。</p>

为什么恢复请求更容易被共同实现？在局部线性近似下，一次参数移动 Δθ 在各 state 造成的变化为 dᵢ ≈ JᵢΔθ。**取同一个参数方向 −Δθ，就能同时撤销所有 state 的变化：Jᵢ(−Δθ) ≈ −dᵢ。**这就是“共同可实现”，不要求各 state 的输出方向相同。

用均匀加权、未归一化的输出 MSE 看它如何聚合，省略共同正系数：

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

每个 state 的恢复贡献都带着一部分“往回走”的分量：**沿共同撤回方向的贡献只会累加，不会正负抵消**；垂直于它的部分仍可能抵消。实际 loss 的权重和归一化须保留在各项内部。这提供了恢复一致性的局部解释，但不能单凭它推出单样本响应率必然更高。

> **Insight：共同可实现，比输出方向相同更重要。** 恢复试图撤销一组已经实现过的函数变化。这为高转化率提供了机制直觉，不要求各 state 的输出力同向，也不保证恢复一定主导。

把两种恢复力的输出力、单样本响应率与聚合保留率分别展开：

<figure id="zh-restoration-three-metrics" class="figure-compact" data-figure="restoration_three_metrics"><img src="/assets/blog/diffusion-rl/restoration_three_metrics.png" alt="图11：NFT EMA 与第二次更新恢复力的输出力、单样本响应率和聚合保留率" loading="lazy"><figcaption>图11：两种恢复力的三项诊断。横轴为 rollout；响应率为逐窗口 P/F，保留率为 A/P。现有数据只覆盖前20轮。</figcaption></figure>

这些短程曲线显示了恢复力如何转化，尚不能回答 guidance 出现长期 spike 时，它能否跟上并约束更新。

不过，EMA 与第二次更新不是同一种救援时序：
- **第二次更新：**第一次从 current=old 出发，没有恢复；第二次有了恢复，但本轮结束就把 old 同步到 current，下一轮又从零开始。
- **EMA：**old 跨 rollout 滞后，恢复不会每轮清零；rollout 和 target 也跟随这个缓慢变化的 old。

<p></p>

<h5 id="zh-flow-force-mapping">3. Flow-GRPO：为什么单 state 的 单样本响应率 就这么低？</h5>

不能把 Flow-GRPO 的低转化全部归因于跨 state 抵消。完整 Flow-GRPO 的时间曲线里，单样本响应率 本身就在个位数，保留率却与 NFT 接近：

<figure id="zh-flow-mapping-course" class="figure-compact" data-figure="flow_mapping_course"><img src="/assets/blog/diffusion-rl/flow_mapping_course.png" alt="图12：完整 Flow-GRPO 的单样本响应率 与聚合保留率时间曲线" loading="lazy"><figcaption>图12：Flow-GRPO 的低响应率在跨 state 聚合之前就出现。分别绘制两个更新窗口；响应率为 P/F，保留率为 A/P。</figcaption></figure>

<p>在 old 锚点，Flow-GRPO 沿随机 action 残差的轴推拉。Advantage 决定这条轴的正负与大小。核心是，<strong>一个 state 的参数梯度，是输出向量各 element 梯度贡献的和：<span class="math">\(g_i=J_i^\top f_i=\sum_{d=1}^{D}f_{i,d}\nabla_\theta v_{\theta,d}(s_i)\)</span>，其中 d 是输出坐标。以优化器的视角来看，FlowGRPO的力的每一个element是独立的（各向同性独立高斯分布）， 但都被同一个advantage所决定的尺度来强化，所以各个element的梯度会发生大幅抵消</strong>；这与最后一层跨 state 抵消是两件事，且图11的随机方向低 单样本响应率 提供了线索和这个观察align。另一方面，Forward-RL范式下力的不同element是高度相关的（被某种endpoint分布的信息决定）， 所以单样本响应率大得多， 这也是Forward-RL比Flow-GRPO更高效的原因。</p>

<p></p>

<h4 id="zh-force-competition-role">3.3.2 什么时候需要考虑力的大小？存在两股力竞争的时候。</h4>

<p>一股力被整体缩放，Adam 可能大幅淡化尺度变化；两股力的比例改变，却会改变输入优化器的方向。这里主要讨论两种情况下我们会遇到“两股力”的竞争： 1. 使用Off Policy技巧稳定训练（对应Case Study 1），以及 2. 使用KL loss来防止微调导致漂移过远 （对应Case Study 2）。 </p>

<p><strong>Case Study 1：不同Guidance Strength什么时候才会产生影响？</strong></p>

<p>一个实验就能让这个现象一目了然：纯on-policy NFT （无EMA），两组<span class="math">\(\beta\)</span>值， 分别为1和0.1。 根据<a href="#eq-zh-mirror-force" data-equation-ref="mirror-force">前文的分解</a>， beta和guidance力的大小成正比（此处guidance混入了reward-improving guidance 和 self-consistency force)。对两组实验分别设置每次rollout进行一次更新和两次更新，如<a href="#zh-nft-restoration-role">前文</a>讨论， 如果是两次更新，那么第二次更新会产生恢复力。</p>

<figure class="figure-compact" data-figure="mirror_dynamics"><img src="/assets/blog/diffusion-rl/mirror_dynamics_compact.png" alt="图13：β=1和0.1的一次更新与两次 disjoint 更新 OCR 曲线" loading="lazy" width="2400" height="930"><figcaption>图13：一次更新与两次 disjoint 更新的 OCR 曲线。左 β=1，右 β=0.1；横轴为 optimizer step。第二次更新引入恢复，改变了力的竞争。</figcaption></figure>

> **Observation：**
> - 如果只有一次更新，那么 β=0.1 和 β=1 的 reward 上升速度没有明显差异（对比青色线）。
> - 如果每次 rollout 两次更新，那么 β=0.1 和 β=1 的 reward 上升速度差异巨大：β=0.1 要快很多，但后期明显更不稳定。

<p>把<a href="#eq-zh-mirror-force" data-equation-ref="mirror-force">已有的 NFT 力分解</a>同除以共同的正系数 <span class="math">\(2\beta^2\)</span>，便能看清竞争：</p>

<div class="math" data-equation="mirror-relative-scale">
\[
\frac{f}{2\beta^2}
=\underbrace{\frac{2r-1}{\beta}(u-v_{\rm old})}_{\text{guidance}}
-\underbrace{(v_\theta-v_{\rm old})}_{\text{restoration}}.
\]
</div>

<p>第一次 current=old，恢复项为零；β 主要改变 guidance 的整体尺度， 根据<a href="#zh-detail-2">现代优化器的整体尺度近似不变性</a>， 第一次更新的参数位移和输出空间位移应该差不多。第二次 current 已移动、old 仍冻结，相似的位移对应提取共同系数后相似的恢复项，而改变 β 就改变了 guidance 的相对份额。</p>

<figure id="zh-force-competition" class="figure-compact" data-figure="force_competition"><img src="/assets/blog/diffusion-rl/force_competition_compact.png" alt="图14：首次更新位移及第二窗口恢复与 guidance 的强度和方向" loading="lazy" width="2400" height="960"><figcaption>图14：初次移动相近，后续竞争不同。左为首个 update 的输出变化 MSE；中、右为第二窗口恢复／guidance 梯度的模长比与 cosine。</figcaption></figure>

<p>首个更新的输出变化 MSE 为 <strong>0.002318／0.002311</strong>，参数位移模长均约 <strong>0.888</strong>。但第二窗口的恢复/guidance 平均模长之比从 <strong>4.85 降至0.78</strong>。起步几乎同样远，随后学得快慢却不同：<strong>同一个力的尺度，在没有竞争力（这里是恢复力）时是整体尺度，有了恢复力后就成为相对尺度。所以$beta=1$ 的两次更新实验，restoration在第二次更新的时候严重压过了guidance，导致严重减速，但也让训练更稳定一些。</strong></p>

**Case Study 2：连续时间、连续空间 RL 中，KL 应该怎么调？**

KL／reference 与恢复力都在撤销模型已经实现过的函数变化，区别在于拉回哪里：恢复力拉回本轮冻结或 EMA 的 old，reference 正则拉回固定的 base。前文“共同可实现”的直觉也适用于这里：撤销已经实现过的变化，与提出一组新的 reward 请求不同。小的 reference 残差，不一定对应弱的参数约束。下面使用 model-output MSE 正则，不是严格的分布 KL。

<p>调系数 <span class="math">\(\lambda\)</span>，本质是在调最终进入合力的 <span class="math">\(\lambda\bar g_{\rm ref}\)</span>，而不只是调 loss 或输出力的比例。Recall 三层心智模型：输出力大小 → 单样本梯度响应 → 跨 state 聚合梯度。连续空间中，不同方向的单样本响应率不同，跨 state 的保留率也不同。两个箭头走完，相似的输出力可以变成完全不同的参数影响。例如，Flow-GRPO 的 raw reference 输出力只有 guidance 的 <strong>0.49 倍</strong>，聚合梯度却是 <strong>25.9 倍</strong>；只在第一层比较大小，就会严重低估约束。</p>

<p>我们可以用同窗口诊断算一个等模长系数：<span class="math">\(\lambda_{\rm match}=A_{\rm guidance}/A_{\rm ref}\)</span>。这里 A 是逐窗口聚合梯度模长的均值，NFT guidance 已剥离 old-policy 恢复项：</p>

<div class="force-table-scroll" tabindex="0" role="region" aria-label="不同算法的 reference 等模长系数">
<table class="force-table">
<thead><tr><th scope="col">训练范式／算法</th><th scope="col">Guidance 聚合梯度<br>A</th><th scope="col">Raw reference 聚合梯度<br>A</th><th scope="col">等模长系数<br>λ</th></tr></thead>
<tbody>
<tr><th scope="row">Forward／NFT + EMA</th><td>0.0417</td><td>0.00539</td><td><strong>7.73</strong></td></tr>
<tr><th scope="row">Reverse／Flow-GRPO</th><td>0.000996</td><td>0.0258</td><td><strong>0.0387</strong></td></tr>
<tr><th scope="row">Reverse／Flow-GRPO Fast</th><td>0.000377</td><td>0.0696</td><td><strong>0.00541</strong></td></tr>
</tbody>
</table>
</div>

<p>在这些已测状态上，NFT 需要约 <strong>7.7</strong> 的系数，reference 才与 guidance 的聚合梯度等模长；Flow-GRPO 只需要约 <strong>0.039</strong>，Fast 约 <strong>0.0054</strong>。这些实验的训练系数实际均为 <strong>0</strong>：表中的数据是少数几次rollout的平均值，不对不同算法KL最优系数构成建议，仅旨在直觉上帮助理解这个现象。</p>

<blockquote><strong>Insight：KL 系数不能脱离 policy 力的转化来选。</strong> 应对齐跨state的聚合梯度，而不是只对齐第一层的 loss／输出力；根据聚合梯度的比例来调整KL系数，以决定KL力该有多强。</blockquote>

<blockquote><strong>Insight：在设计恢复力这种保护机制时，平时不能让保护机制压掉reward学习，危险时也不能让它跟不上 guidance。</strong> 同时监控两股力的输出大小、单样本响应率、聚合保留率，以及加权后的参数合力；第二次更新和 KL 调参，都是这个竞争问题。</blockquote>

<h2 id="zh-recipes" data-section-key="recipes">4. 选配方前，我们会先测什么？</h2>

<p>在调控力的比例的时候，<strong>应调控聚合后梯度的相对强弱，而不是力的相对大小（即loss值），因为Diffusion 这种连续时间连续空间模型，连续空间中不同力的方向的收益大不一样（<a href="#zh-force-comparison">见3.3.1的三量两率分析</a>）。</strong></p>

**Reverse：看 action 如何混合。** 测逐噪声位置梯度模长和与合力的 cosine，问有影响的 action 能否改变 reward。试有信息的短窗口、受控 initial noise 或更好的 baseline，分别比较学习进展和减少 backward 的收益。

**Forward：看各股力如何混合。** 在同点同时测 guidance/restoration 输出力和参数梯度，再看二者 cosine 与合力方向。小 loss 分项也可能主导；对抗的方向也不一定精确反向。

**稳定性：看函数实际怎么动。** 固定 state 测 current–reference gap 与真实输出变化。Adaptive restoration 很值得探索：允许有用移动，在危险漂移时介入。当前曲线提出这个实验方向，还没有证明某个规则有效。

**留白实验：恢复能否只在有用的时候介入？** 固定恢复与 adaptive 恢复，对比 reward、fixed-state drift、失败率和相同采样成本。预先规定规则，多 seed 验证，不事后挑最好看的系数。

统一的问题不是“loss 降了多少”，而是：**哪些方向到达了共享更新，它们实现了什么？** Reverse 把问题藏在 action credit 里，Forward 把问题藏在回归力的转化里。让中间量可见，往往是理解异常曲线最快的方式。

<h2 id="zh-algebra" data-section-key="algebra">附录：想看代数的读者</h2>


<h3 id="zh-detail-14">为什么出现 covariance？</h3>

固定重新加噪规则，把旧 endpoint 分布倾斜为“旧分布 × exp(倾斜强度 × reward)”，然后归一化。在倾斜强度为零处，条件 target 均值对强度的导数就是 reward 与 target 的条件 covariance。这是局部导数关系，不代表有限倾斜等于一步 covariance 更新。NFT feedback 与 raw reward 的 covariance 只在固定正 affine 变换且无 clipping 等条件下成比例。

<div class="math" data-equation="reward-tilt">
\[
\begin{aligned}
p_\lambda(x_0\mid c)&\propto p_{\rm old}(x_0\mid c)e^{\lambda R(x_0,c)},\\
\left.\frac{\partial}{\partial\lambda}\mathbb E_\lambda[u\mid s]\right|_{\lambda=0}
&=\operatorname{Cov}(R,u\mid s).
\end{aligned}
\]
</div>

c 是 prompt，R 是 endpoint reward，λ 是倾斜强度。这个等式还要求相关矩有限，并允许在积分号内求导。

<h3 id="zh-detail-15">Baseline 改变噪声，不一定改变期望方向</h3>

固定 prompt baseline 时，给定 state：

<div class="math" data-equation="baseline">
\[
\begin{aligned}
\mathbb E[(R-b(c))^2\mid s]
&=\operatorname{Var}(R\mid s)\\
&\quad +(V_k(s)-b(c))^2.
\end{aligned}
\]
</div>

b(c) 是固定的 prompt baseline，R 是从该 state 继续生成后的终点 reward。

后一项是 prompt baseline 没去掉的继承质量。标准 policy-gradient 条件下，state-only baseline 的期望 score contribution 为零，但能改变有限样本噪声。这个恒等式不是有限 group、STD-normalized GRPO 的梯度方差定理。

<h3 id="zh-detail-16">Transfer 与恢复共享 Jacobian</h3>

小步 plain gradient descent 下，记学习率为 η，state j 的输出变化近似为：

<div class="math" data-equation="transfer">
\[
\Delta v_j\approx\frac{\eta}{N}\sum_iJ_jJ_i^\top f_i.
\]
</div>

这不是 Adam 的等式。对于共享参数已经实现的位移 Δθ，局部有 dᵢ ≈ JᵢΔθ；平方参考惩罚的参数下降方向为：

<div class="math" data-equation="restoration">
\[
g_{\rm restore}\approx-\frac{1}{N}\sum_iJ_i^\top J_i\Delta\theta.
\]
</div>
