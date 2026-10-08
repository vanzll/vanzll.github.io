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

LLM 也有关键决策和无关紧要的动作，但 token 位置通常不足以识别关键决策。Diffusion 多了一个结构线索：噪声时间 t 告诉我们样本形成到了哪个阶段，可以据此寻找更可能具有强 action credit 的训练位置。

<p>把这点接回上一节，矛盾就清楚了：<strong>实际主导更新的 action，未必是能有效改变 reward 的 action，或者说 credit assignment 更差。</strong> 如果后期信号更多混入继承的 state 质量，却又被 loss 的隐式 weighting 放大，合力就可能偏向信息较弱的方向。这是高噪声训练窗口值得尝试的原因：既少做 backward，也让更有 action credit 的方向获得影响力。</p>

<blockquote><strong>Insight：</strong>高 reward 不等于高 action credit。 Prompt-group advantage 混合了“接手的 state 有多好”与“当前 action 改善了多少”。去噪后期，前者可能更强、后者却更弱；Flow-GRPO 的低噪声加权还可能让这些 credit 较弱的动作主导更新。反过来，Diffusion 的噪声时间提供了一个结构线索：选择高噪声训练窗口，可能同时减少计算，并让更有 reward 杠杆的动作在聚合梯度中获得更大影响力。</blockquote>

<p>Diffusion RL 和 LLM RL 还有一个区别是，在第一个 action 之前，initial noise 就已带来不同的起点价值。我们的实验发现，不同 initial noise 的价值差异在视频里面更为明显。根据<a href="#eq-zh-credit-variance" data-equation-ref="credit-variance">状态价值方差公式</a>，这会增加 advantage 估计误差，使得 credit assignment 变弱。让一个 prompt group 共享一个 initial noise 是一个解决方案。<a href="#zh-shared-noise">图3右</a>展示了这一对照：在 logged step 240，共享 initial noise 的 eval 为 <strong>0.656</strong>，独立 noise 对照为 <strong>0.527</strong>，支持先控制起点差异，再比较 action 的贡献。</p>

<h3 id="zh-detail-6">2.4 训练的位置，不一定是收益出现的位置</h3>

我们只在第一步训练 LoRA，但同一套参数也能作用于其他时步。推理时，通过逐时步开启或关闭 LoRA，控制这次训练得到的函数变化在哪里生效。对同一个 checkpoint，比较：

- **只开第一步**：后续步使用原始模型。
- **全部开启**：第一步不变，后续步也使用训练后的 LoRA。

<figure class="figure-compact" data-figure="gate_transfer"><img src="/assets/blog/diffusion-rl/gate_transfer_compact.png" alt="图4：第一步训练的 LoRA，在后续步开启也能贡献收益" loading="lazy" width="2400" height="930"><figcaption>图4：第一步训练，推理只开第一步 vs 全部 10 步开启。左为 reward，右为配对差值；使用相同 checkpoint、64 条固定 prompt 与相同种子，横轴为日志 step。收益随 checkpoint 变化，后期缩小并偶尔转负。</figcaption></figure>

在 logged step 60，关闭后续步的 LoRA，使 reward 从 **0.703 降至 0.512**。这说明：**第一步训练得到的参数变化，也能在未训练的后续步贡献收益。** 后续步没有额外接受训练，而是共享参数把第一步学到的变化带到了那里。

> **Insight：选择训练时步，是选择梯度来源，不是限定收益出现的位置。** 有效的动作信号可以通过共享参数，在其他时步生效。

**待补实验：这种迁移是否非对称？** 关闭第一步，只开第八步或第二至八步；再反过来，训练第八步、只开第一步。现有实验两种条件都保留第一步 LoRA，尚不能回答这一问题。

<h3 id="zh-detail-7">2.5 理解 Flow-GRPO Fast：只训练高噪声区域，可以显著提效</h3>

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

<figure class="figure-compact" data-figure="reverse_learning"><img src="/assets/blog/diffusion-rl/reverse_learning_compact.png" alt="图5：分母对照与 Flow-GRPO Fast 对比 Naive Flow-GRPO" loading="lazy" width="2400" height="930"><figcaption>图5：左，全部时步训练，去掉方差分母更早取得 reward 提升；右，Flow-GRPO Fast 的早期 3 步窗口与 Naive Flow-GRPO 对比。横轴为日志 step，纵轴为 GenEval score；展示归档原始 eval 点，不平滑。各 run 的采样与打分协议见数据说明。</figcaption></figure>

> **Insight：让对 reward 更有因果贡献的 action，在总梯度里占更大比重。** Flow-GRPO Fast 通过只训练高噪声 action，集中保留这些有效信号；共享参数更新再把它们带到未训练的 state，让低噪声速度场也受益。

<blockquote class="training-recipe">
<p><strong>Recipe：Reverse Process RL</strong></p>
<ol>
<li>在固定高噪声区域内随机选连续短窗口；建议窗口长度直接取 1 或 2，减少低 action-credit 信号的稀释。</li>
<li>根据任务选择 timestep weighting：强调高噪声，或让训练时步的噪声尺度平权。</li>
<li>同一 prompt group 固定 initial noise，减少初始条件差异对 action credit 的干扰。</li>
</ol>
</blockquote>

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

<p>在 <span class="math">\(d=0\)</span> 时，两者的力都是 <span class="math">\(-2e\)</span>；移动之后，负 MSE 继续放大偏移，镜像正 MSE 则把偏移拉回。NFT 的正负支虽然 guidance 不同，二次项却同号，所以不同state聚合起来仍是<a href="#eq-zh-mirror-force">统一的恢复</a> 。Reverse RL 的偏移项则随 advantage 改变符号，所以不同state的恢复力的系数和为0，但也不能仅凭组内 advantage 均值为零就断言它抵消：各 state 的偏移、权重和 Jacobian 都不同。这是固定系数 MSE 的对照；实际 <a href="https://arxiv.org/html/2505.05470v1#S4.SS1">Flow-GRPO</a> 还乘概率比，并受 PPO clip影响。</p>

<blockquote><strong>Insight：相同的第一步梯度，可以隐藏相反的后续动力学。如果当前算法不是纯on-policy更新，那么设计梯度提供器时</strong>，不只要看它现在把模型推向哪里，还要看模型移动后，它会是放大这次偏移，还是恢复这次偏移。</blockquote>

但为什么看起来很小的恢复项，会有这么大的影响？

<h3 id="zh-detail-11">3.3 理解哪种力会很强</h3>

<h4 id="zh-force-comparison">3.3.1 多种常见力的分析</h4>

<p>回到第 1 节的<a href="#eq-zh-chain" data-equation-ref="chain">三层心智模型</a>：输出力 → 各 state 的 Jacobian 映射 → 跨 state 聚合。<strong>一股力能推动模型多少，取决于它经过这三层后还剩多少，而不只是起点有多大。</strong></p>

<p><strong>先比较这五类力。</strong> 前两层不容易直接校准，却可以分别测量；不能把最终强弱全部归因于第三层的抵消。</p>

<ul>
<li><strong>Reverse RL 的推拉：</strong>轨迹级 advantage 混合了 state 价值与 action 贡献，不同 action 的请求未必协同；独立探索噪声本身不能证明参数梯度抵消更多。</li>
<li><strong>Forward RL／NFT 的 guidance：</strong>从高奖励 endpoint 构造的请求有机会更协同。但 NFT guidance 还包含前向–反向自一致性项（见<a href="#zh-detail-9">3.2.2 节</a>），需要分项测量才能判断抵消来自哪里。</li>
<li><strong>DiffusionART 的 covariance guidance：</strong>把奖励相关的 covariance 与自一致性项分开，有望留下更协同的指导；是否进一步提高聚合保留率，仍待匹配的三层对照。</li>
<li><strong>恢复力：</strong>撤销已经实现过的变化；分项实验发现它可以在输出空间很小，却在参数空间很强。</li>
<li><strong>KL（reference-MSE）正则力：</strong>限制偏离 reference；下面测其未加权梯度，不把它等同于已施加的 KL 更新。</li>
</ul>

**恢复力给出了一个反直觉的例子。** 在同一参数、同一批状态上分解第二次更新，恢复项的输出力小约 **9.4 倍**，参数梯度反而大约 **4.9 倍**：

<figure class="figure-compact" data-figure="mirror_reversal"><img src="/assets/blog/diffusion-rl/mirror_reversal_compact.png" alt="图9：同点输出力与参数梯度的尺度反转" loading="lazy" width="2400" height="840"><figcaption>图9：小输出力，大参数梯度。圆点与连线为 18 组配对测量，菱形为均值。β=1，rollout 0–17 的第二窗口；两项在相同参数与状态上、optimizer 之前测量，纵轴为对数尺度。比值为均值之比，不是 Adam 更新比例。</figcaption></figure>

> **Observation：恢复的优势出现在转化过程中。** 它的输出力本身更小，却在映射和聚合之后更强。接下来要区分：是 Jacobian 放大更多，还是跨 state 抵消更少？

<details class="derivation" id="zh-force-conversion-proof">
<summary>如何把转化率拆成“映射增益 × 聚合保留率”？</summary>
<p>沿用第1节的 <span class="math">\(f_i\)</span>（输出力）、<span class="math">\(g_i=J_i^\top f_i\)</span>（单个 state 的参数方向）与 <span class="math">\(\bar g=N^{-1}\sum_i g_i\)</span>（平均合力）。统一模长与归约、分母非零时：</p>
<div class="math" data-equation="conversion">
\[
\begin{aligned}
\kappa&=\frac{\|\bar g\|}{N^{-1}\sum_i\|f_i\|}
=\gamma\rho,\\
\gamma&=\frac{\sum_i\|g_i\|}{\sum_i\|f_i\|},\qquad
\rho=\frac{\|\sum_i g_i\|}{\sum_i\|g_i\|}.
\end{aligned}
\]
</div>
<p>γ 衡量输出力映射后放大多少；ρ 衡量参数梯度相加后保留多少，位于 0 与 1 之间。两者相乘，才是整体转化率 κ。上面的 temporal probe 只测跨 timestep 的保留率，不能直接代入逐样本分解，也不能用输出 RMS 替代这里的平均模长。</p>

<div markdown="1">
先区分两件事：一股输出力经过网络后会被放大多少，以及不同 state 的参数梯度相加时还能保留多少。**相邻梯度更同向，不一定意味着总梯度保留更多。**

<p><a href="#zh-reverse-gradients">图2</a>测的是一个 microbatch 内各 timestep 的梯度，不是逐样本输出力。它展示了晚期分项主导合力；相邻 timestep 的 cosine 均值约为 0.003，接近正交，不能据此断言存在强烈的反向抵消。我们重新查了 Flow-GRPO 与 NFT 的 temporal probe，已有量如下：</p>

- **Flow-GRPO：**逐 timestep 梯度模长之和，平均为 **0.00285**；先把梯度向量相加再取模长，平均为 **0.00190**；逐次聚合保留率的均值为 **0.669**。
- **NFT：**对应值为 **3.426、1.993、0.552**，相邻 timestep 的 cosine 均值约为 **0.275**。相邻方向更一致，聚合保留率却没有更高：前者只看相邻夹角，后者还取决于全部方向与相对模长。

这两组是历史配置，不用原始模长横比算法强弱；它们也没有同时记录完整的“输出力 → 逐样本参数梯度 → 完整 update 合力”链条。**下面的 NFT 分项测量则直接连接了输出力与聚合参数梯度。**
</div>
</details>

**新增诊断：把三层分别测出来。** 我们在同一诊断窗口内，分别测输出力模长均值、逐 state 参数梯度模长均值，以及梯度向量平均后的模长，再拆成映射增益 γ 与聚合保留率 ρ：

<figure id="zh-three-layer-probe" class="figure-compact" data-figure="three_layer_force_probe"><img src="/assets/blog/diffusion-rl/three_layer_force_probe.png" alt="图12：恢复力与 reference-MSE 的映射增益和聚合保留率" loading="lazy" width="4800" height="1980"><figcaption>图12：高转化率发生在哪一层？左为映射增益（对数轴），右为聚合保留率。每对分项在相同参数与诊断状态上测量；数值为跨窗口平均模长之比，不是逐窗口比例的均值，也不是 Adam 更新占比。NFT 双更新只取第二窗口。Fast 为未完成实验的部分结果；reference-MSE 未加权、仅用于诊断。采样数与覆盖范围见数据说明。</figcaption></figure>

**恢复力的优势不只来自抵消少。** EMA 与双更新 NFT 中，恢复力的映射增益分别约为 guidance 的 **3.4 倍、3.0 倍**；在已读取的所有非零恢复窗口中，这个增益都更高。聚合保留率的优势却没有这么稳定：EMA 约 **78%** 的有效窗口更高，双更新只有 **55%**，后者逐窗口保留率之比的中位数约 **1.08**。因此，不能把恢复力的全部优势归因于跨 state 协同。

> **Observation：这次最稳定的反转发生在映射层。** 小输出力经 Jacobian 映射后可以获得更高增益；更少抵消则不是每个窗口都成立。高转化率也不等于必然主导：本批 EMA 与双更新 NFT 的恢复/guidance 聚合模长之比分别约为 0.95、0.63，未重现图9那样的平均主导。

Fast guidance 的保留率约为 **28%**，NFT guidance 约为 **25%**，没有出现“Reverse 一定抵消更多”的排序。不过两者的诊断 state 数与噪声位置不同，不能据此反向给算法排名；DiffusionART 的完整三层对照仍待补。

为什么恢复可能更容易实现？它要求模型撤销共享参数**刚刚共同实现过的变化**，而 reward guidance 提出的新请求未必能由同一次参数更新共同满足。

<p>这个直觉有一个局部支点。若参数已经移动了 <span class="math">\(\Delta\theta\)</span>，则各 state 的偏移约为 <span class="math">\(d_i\approx J_i\Delta\theta\)</span>，恢复的参数下降方向约为 <span class="math">\(-N^{-1}\sum_iJ_i^\top J_i\Delta\theta\)</span>，省略共同系数。它们都在响应同一个已发生的参数变化；不要求各 state 的输出力平行。</p>

> **Insight：共同可实现，比输出方向相同更重要。** 恢复试图撤销一组已经实现过的函数变化，而不是提出一组全新的请求。这为它的高转化率提供了机制直觉；它不要求各 state 的输出力同向，也不证明恢复一定主导。

这个直觉也可以延伸到映射层：已发生的输出位移来自 Jacobian 对参数变化的作用，不是任意的输出方向。它可能更集中在网络容易响应的方向上；经过 Jacobian 的转置映射时，这些方向还能被进一步放大。但“可实现”不保证“高增益”，具体还取决于位移与 Jacobian 高增益方向的对齐。实验确认的是映射增益更高，这个谱方向解释仍是机制猜想。

<h4 id="zh-force-competition-role">3.3.2 什么时候需要考虑力的大小</h4>

当两股力竞争，相对尺度才决定谁留下。整体缩放一股力，可能被现代优化器大幅淡化；改变两股力的比例，却会改变聚合方向。

**Case Study 1：Diffusion RL 里的 Off-policy 训练。** 图10比较一次更新与两次 disjoint 更新：

<figure class="figure-compact" data-figure="mirror_dynamics"><img src="/assets/blog/diffusion-rl/mirror_dynamics_compact.png" alt="图10：一次更新与两次 disjoint 更新的稳定性和速度" loading="lazy" width="2400" height="930"><figcaption>图10：一次更新与两次 disjoint 更新；左 β=1，右 β=0.1。完整 online train OCR 曲线，不平滑，横轴为记录 reward 时的 optimizer step。每个 one/disjoint pair 使用同一训练 commit；跨 β 来自不同版本，具体配方见数据说明。</figcaption></figure>

两条青色线都能快速学起来；粉色线却对 β 更敏感：β=1 学得很慢，β=0.1 更快，但后期仍会崩溃与恢复。关键不是“某个绝对尺度最好”，而是**第二次更新多了一股与 guidance 竞争的力**。

<p>对前面未归一化的镜像 MSE 核心，令 <span class="math">\(d=v_\theta-v_{\rm old}\)</span>，提取共同系数后：</p>

<div class="math" data-equation="mirror-relative-scale">
\[
\frac{f}{2\beta^2}
=\underbrace{\frac{2r-1}{\beta}(u-v_{\rm old})}_{\text{guidance}}
-\underbrace{(v_\theta-v_{\rm old})}_{\text{restoration}}.
\]
</div>

第一次更新 current=old，恢复项为零。整体缩放 guidance，可能被 Adam 大幅淡化；第二次更新 current 已经移动，old 仍冻结，缩放 guidance 就是在改变它相对于恢复项的份额。**同一个尺度，在第一次更新中是整体尺度，在第二次更新中成为相对尺度。** 这里的“恢复尺度相同”指提取共同系数后、相同位移下的核心项，不是原始恢复系数与 β 无关。

W&B 的真实测量补上了两个关键环节：

<figure id="zh-force-competition" class="figure-compact" data-figure="force_competition"><img src="/assets/blog/diffusion-rl/force_competition_compact.png" alt="图11：初次更新位移、第二窗口分项梯度比例与方向" loading="lazy" width="2400" height="960"><figcaption>图11：两次 disjoint 更新的 β=1 / 0.1 配方。左：首个 optimizer update 在诊断状态上的 velocity 变化 MSE。中、右：rollout 0–17 第二窗口的恢复/guidance 参数梯度模长比与 cosine，保留全部有效点、不平滑。虚线分别为等模长与正交；跨 β 不是严格单变量对照。</figcaption></figure>

首个更新的输出变化 MSE 为 **0.002318 / 0.002311**，参数位移模长均约 **0.888**：起步移动确实几乎相同。第二窗口中，恢复/guidance 的平均模长之比却从 **4.85 降至 0.78**。这比单看 reward 更直接地支持“恢复份额改变了”的解释；cosine 也说明，恢复主导不等于每次都与 guidance 精确反向。后续模型走上不同轨迹，不能把起步相同推广成全程相同。

这些 run 使用 normalized-X0 与 current-STD，而非上式的裸 MSE；图11测的是实际 loss 的分项梯度。因此，核心公式解释相对尺度竞争，实际强度则以测量为准。


**Case Study 2：KL divergence。** 一旦 current 偏离 reference，reward guidance 与正则力就同时参与更新。需要调的是它们在参数空间的相对模长与方向；KL 系数小，不保证它的梯度份额小。

**Reference-MSE 也会发生更大的尺度反转。** 在 Fast 的相同非零诊断窗口里，它的输出力约为 guidance 的 **1.8 倍**，聚合参数梯度却约为 **185 倍**。这不是 KL 已经压住了训练：本批正则系数为 0，reference-MSE 只用于测量。真正加入正则后，要比较系数加权后的参数梯度及其方向，而不是把 loss 数值当成力。

**下一步：验证机制，而不是重复测总模长。** 新 probe 已拆开 NFT 与 Fast 的三层；还需要匹配 state 数与噪声区域，补上 DiffusionART 和独立 consistency 分项，并检验恢复方向是否更对齐 Jacobian 的高增益方向。

<h2 id="zh-recipes" data-section-key="recipes">4. 选配方前，我们会先测什么？</h2>

**Reverse：看 action 如何混合。** 测逐噪声位置梯度模长和与合力的 cosine，问有影响的 action 能否改变 reward。试有信息的短窗口、受控 initial noise 或更好的 baseline，分别比较学习进展和减少 backward 的收益。

**Forward：看各股力如何混合。** 在同点同时测 guidance/restoration 输出力和参数梯度，再看二者 cosine 与合力方向。小 loss 分项也可能主导；对抗的方向也不一定精确反向。

**稳定性：看函数实际怎么动。** 固定 state 测 current–reference gap 与真实输出变化。Adaptive restoration 很值得探索：允许有用移动，在危险漂移时介入。当前曲线提出这个实验方向，还没有证明某个规则有效。

**留白实验：恢复能否只在有用的时候介入？** 固定恢复与 adaptive 恢复，对比 reward、fixed-state drift、失败率和相同采样成本。预先规定规则，多 seed 验证，不事后挑最好看的系数。

统一的问题不是“loss 降了多少”，而是：**哪些方向到达了共享更新，它们实现了什么？** Reverse 把问题藏在 action credit 里，Forward 把问题藏在回归力的转化里。让中间量可见，往往是理解异常曲线最快的方式。

<h2 id="zh-algebra" data-section-key="algebra">附录：想看代数的读者</h2>

<h3 id="zh-detail-13">镜像目标的恢复项</h3>

固定反馈、target、reference，对未归一化核心展开：

<div class="math" data-equation="mirror-loss">
\[
\begin{aligned}
L&=r\|\beta d-e\|^2+(1-r)\|-\beta d-e\|^2\\
&=\beta^2\|d\|^2
-2\beta(2r-1)\langle d,e\rangle+\|e\|^2.
\end{aligned}
\]
</div>

最后一项对 current 是常数。求负梯度得到前面的 guidance−restoration 分解。条件下的“r 与 target 的乘积均值 = covariance + 两个均值的乘积”，得到 consistency 残差。逐元素平均另带共同归约因子；实际 normalization 必须单独核对。

<div class="math" data-equation="covariance-identity">
\[
\begin{aligned}
\mathbb E[ru\mid s]
&=\operatorname{Cov}(r,u\mid s)\\
&\quad+\mathbb E[r\mid s]\mathbb E[u\mid s].
\end{aligned}
\]
</div>

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

<h2 id="zh-evidence" data-section-key="evidence">数据说明</h2>

图来自历史 SD3.5-M LoRA 实验，不作跨论文性能比较。Reverse 为归档 GenEval 相关 eval aggregate，Forward 为 online OCR。单 seed 原始曲线，不伪造置信区间；checkpoint 不是独立 seed。Run、metric、数据和 hash 放图的 manifest，不在正文里用 run ID 代替图。

- Reverse intervention：K=24，LR=3e-4，10 rollout positions，40 eval steps，seed42，无外部 CFG、KL=0。Temporal probe 训练零起算位置0–8、累积8个microbatch；图2保留这一编号。独立的 Prefix 图保留历史记录的位置1/8标签。图5右的 Fast run 使用早期 3 步 CPS 窗口及相应的无方差分母打分；Naive 使用原 SDE 打分。这是完整配置的对比，窗口与分母的贡献需用匹配采样器的消融拆分。分母公式比较 old 锚点的输出力；修改 log-prob 分母也会改变后续 PPO ratio/clipping。
- 图3右为跨历史版本的 initial-noise 对照：模型、reward、K、训练步数、LR、batch 与累积设置相同；shared run 每10轮 eval，对照每30轮 eval，保存频率也不同。比较相同 logged step 的 eval，保留各自的实际测量频率。
- Forward 公式分析未归一化 NFT mirror-MSE 核心；normalized-X0 实现的 normalization 与归约会改变具体系数，图用实现自身的分项 probe。
- 图6显示历史窗口0–100；原始说明称该分支实验去掉EMA，但现有历史run名称含tf0.99，实际开关仍需核对，不据名称断言使用EMA或将其作为严格on-policy对照。图7使用2026-10-02更新的原图，左侧延伸到500个optimizer steps，以首次达到OCR=0.9比较21与320步；两次更新是disjoint，而非旧AdvBridge replay实验。中间保留EMA-anchor消融，它的原始统计轴为fresh rollout iteration，图中标成Training steps；左右两类轴不用于横向效率比较。图8是固定state与target的几何示意，不是实验测量。每张图的来源快照和资产hash见manifest，源文件未修改。
- 图9–11：normalized-X0 Mirror、current-STD，无 rollout EMA/KL。One/disjoint pair 同训练 commit，但每次 update 的 batch 与参考同步节奏不同；beta0.1 跨版本，不称严格 beta-only 对照。图10 reward event 用实际 optimizer-step 计数，仅有初始 held-out eval，不作 wall-time 或最终泛化结论。
- 图9平均 guidance/restoration 输出 RMS 为 2.633e-5 / 2.803e-6，参数梯度 norm 为 0.00812 / 0.03939；单位不同，比值均为均值之比。Probe 为数值重建，最大 normalized reconstruction error 约 0.0041，测量训练参数子空间；完整 runtime AMP/clipping 口径仍待核对。β=0.1 版本前18次 probe 的 restoration/guidance 参数梯度比约0.78，β=1 为4.85；OCR 0.8 首次到达 update48 / 232，不能据此隔离恢复系数的因果作用。
- 图11于2026-10-07从 W&B 定向取回：位移来自 opt/update/v_delta_to_prev_current_mean，代码对更新前后相同诊断状态的输出差逐元素平方后取均值，不是 RMS 或整条轨迹位移；参数位移为实际更新前后的 trainable-parameter L2 norm。梯度分项取 probe/mirror_components，先检查 valid，再保留前18轮的全部第二窗口测量。数据、CSV、统计和训练 commit 保存于 force_conversion_wandb.json 与 force_competition_compact.json；初次位移与第二窗口的梯度测量不是同一个时点。
- 第3.3节 temporal 数值来自 aggregation_review_wandb.json：Flow-GRPO / NFT 分别有43 / 42次 probe、9 / 10个 timestep，均为 microbatch 0 的 LoRA 梯度；不是完整累积 update 或逐样本梯度。每次先求模长和、向量和模长及二者之比，再分别跨 probe 取均值。Flow-GRPO 合力模长由 logged coherence × 逐时步模长和重建；NFT 与直接记录的 gradient_sum_norm 一致到浮点误差。两者 rollout 分别为 SDE / deterministic，梯度累积为8 / 16，不能据此给算法排序。已有 output-force RMS 不等于 MSE，且对应另一组 NFT 分项 probe；这些历史 run 未测完整三层；图12来自新增的三层 probe。
- 图12来自 2026-10-09 定向查询的 [NFT EMA](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/26rilzmp)、[NFT 双更新](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/adnxitig) 与 [Fast](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/75pf0fok)。SD3.5-M LoRA、OCR、seed42、LR=3e-4，无外部 CFG，正则系数为0。NFT 为10步确定性 rollout，Fast 为10步 rollout 中的3步 CPS 窗口；NFT 每窗口80个诊断 state，Fast 24个，均不是完整 optimizer update 的所有样本。图中分别使用18个 EMA 非零恢复配对窗口、20个第二更新配对窗口和32个 Fast 非零 reference 配对窗口。云端原始覆盖为19/40/33个窗口，远端报告本地为20/40/34；Fast 因存储配额退出于17/20轮，未补齐或外推。
- 图12的 γ/ρ 是跨所选窗口平均模长之比；78%/55% 与1.08则来自逐窗口配对比例。重建误差最大约0.66%/0.70%/1.06%，不是无误差的数值分解。NFT guidance 尚包含 consistency，Fast signed displacement 不是统一正恢复力。W&B 记录运行 commit 为2a451c0…，不同于交付的dfc49cd…；修复 diff 尚未核实。三层原始模长及来源 hash 见 [CSV](/assets/blog/diffusion-rl/three_layer_force_probe.csv) 与 [测量说明](/assets/blog/diffusion-rl/three_layer_force_probe.json)。
- 待补实验包括 action controllability、完整 gate matrix、Fast compute accounting、匹配三层对照与机制干预、adaptive restoration。
