---
layout: research-post
title: 'Understanding Policy Optimization in Continuous-Time Generative Processes:
  Phenomena, Insights, and Recipes'
title_zh: 理解连续时间生成过程中的策略优化：现象、洞察与配方
title_en: 'Understanding Policy Optimization in Continuous-Time Generative Processes:
  Phenomena, Insights, and Recipes'
description: 'From output-space forces to shared-parameter updates: understanding
  the efficiency and stability of diffusion RL.'
permalink: "/blog/diffusion-rl/"
lang: en
bilingual: true
draft: false
math: true
authors:
- name: Zhenglin Wan
  url: "/"
toc:
- id: target
  title: 核心心智模型
  title_en: A shared mental model
- id: reverse
  title: 理解 Reverse RL
  title_en: Understanding reverse RL
- id: forward
  title: 理解 Forward RL
  title_en: Understanding forward RL
- id: recipes
  title: 先测什么？
  title_en: What would we measure first?
unlisted: true
sitemap: false
date: '2026-10-07 12:00:00 +0800'
description_en: 'From output-space forces to shared-parameter updates: understanding
  the efficiency and stability of diffusion RL.'
image: "/assets/blog/diffusion-rl/force_overview.png"
bibtex: |
  @misc{wan2026continuousgenerativepolicy,
    author = {Zhenglin Wan},
    title = {Understanding Policy Optimization in Continuous-Time Generative Processes: Phenomena, Insights, and Recipes},
    year = {2026},
    howpublished = {Research note},
    url = {https://vanzll.github.io/blog/diffusion-rl/}
  }
---
<div data-language-panel="zh" lang="zh-CN" hidden>
{% capture chinese %}{% include research-notes/diffusion-rl.zh.md %}{% endcapture %}
{{ chinese | markdownify }}
</div>
<div data-language-panel="en" lang="en">
{% capture english %}{% include research-notes/diffusion-rl.en.md %}{% endcapture %}
{{ english | markdownify }}
</div>
