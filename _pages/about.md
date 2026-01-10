---
permalink: /
title: ""
excerpt: ""
author_profile: true
lang: en
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}





<span class='anchor' id='about-me'></span>
# 😊 About me

***Resume***: [Resume](https://drive.google.com/file/d/1fYYlj0MhJGsPPDdWUMB9LHMTqUo3bL1s/view?usp=sharing) (Updated on 6 Jan, 2026)

***Contact***: **vanzl3386 [at] gmail.com (preferred)**, vanzl [at] u.nus.edu, zhenglin.wan [at] ntu.edu.sg, 121090525 [at] link.cuhk.edu.cn

---
<style>
/* Slightly reduce spacing between top-level headings on the home page */
.page__content > h1 { margin-top: 1.7em !important; margin-bottom: 0.45em; }
.page__content > h1:first-of-type { margin-top: 0.45em !important; }
@media (max-width: 640px) { .page__content > h1 { margin-top: 1.4em !important; } }
/* Use Times New Roman for body content and slightly smaller size */
.page__content { font-family: "Times New Roman", Times, serif; font-size: 1.00em; }
.page__content p, .page__content li, .page__content div { font-family: "Times New Roman", Times, serif; }
/* Headings in Times New Roman and emphasized styling */
.page__content h1, .page__content h2, .page__content h3 { font-family: "Times New Roman", Times, serif; font-weight: 700; color: #222; text-rendering: optimizeLegibility; }
.page__content h1 { border-bottom: 2px solid #e5e5e5; padding-bottom: 4px; }
.page__content h2, .page__content h3 { border-left: 4px solid #2a72d4; padding-left: 10px; background: linear-gradient(to right, rgba(42,114,212,0.06), rgba(42,114,212,0)); border-radius: 4px; }

/* Collapsible section styles */
.section-toggle-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  border: none;
  border-radius: 20px;
  padding: 6px 16px;
  font-family: "Times New Roman", Times, serif;
  font-size: 0.9em;
  font-weight: 600;
  cursor: pointer;
  margin: 8px 0 12px 0;
  box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
  transition: all 0.3s ease;
}
.section-toggle-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}
.section-toggle-btn:active {
  transform: translateY(0);
}
.section-toggle-btn .toggle-icon {
  display: inline-block;
  transition: transform 0.3s ease;
  font-size: 0.85em;
}
.section-toggle-btn.expanded .toggle-icon {
  transform: rotate(180deg);
}
.collapsible-content {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.5s ease-out, opacity 0.3s ease;
  opacity: 0;
}
.collapsible-content.expanded {
  max-height: 5000px;
  opacity: 1;
  transition: max-height 0.7s ease-in, opacity 0.3s ease;
}
/* Dark mode support for toggle button */
@media (prefers-color-scheme: dark) {
  .section-toggle-btn { background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%); box-shadow: 0 3px 10px rgba(90, 103, 216, 0.4); }
}
html.dark .section-toggle-btn, body.dark .section-toggle-btn, html[data-theme="dark"] .section-toggle-btn { 
  background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important; 
  box-shadow: 0 3px 10px rgba(90, 103, 216, 0.4) !important; 
}
</style>

<!-- Dark mode is handled globally in layout/head; local button/styles removed -->


Hi! I am Zhenglin Wan (万政霖), a first-year CS Ph.D. student at [HPC-AI Lab](https://ai.comp.nus.edu.sg/) at National University of Singapore (NUS), advised by [Prof. Yang You](https://www.comp.nus.edu.sg/~youy/) and supported by *NUS Research Scholarship*. Previously, I served as an Research Staff in Nanyang Technological University (NTU), working with [Prof. Bo An](https://personal.ntu.edu.sg/boan/). I received my B.Sc from Chinese University of Hong Kong (CUHK) on 2025 Fall (completed undergrad in Shenzhen campus). I also have spent a long time working with [Prof. Ivor Tsang](https://www.a-star.edu.sg/cfar/about-cfar/management/prof-ivor-tsang) at [Centre for Frontier AI Research (CFAR)](https://www.a-star.edu.sg/cfar), [IHPC](https://www.a-star.edu.sg/ihpc/), [A\*STAR](https://www.a-star.edu.sg/) in Singapore, and have interned at [Hong Kong Generative AI Research & Development Center (HKGAI)](https://www.hkgai.info/), led by [Prof. Yike Guo](https://cse.hkust.edu.hk/admin/people/faculty/profile/yikeguo) in HKUST.

---

I am a RL (Reinforcement Learning) believer. Previously I mainly studied improving RL or inverse RL from algorithmic side: such as improving the policy diversity (like [EBC](https://arxiv.org/abs/2410.06151)) and policy expressiveness (like [GoRL](https://arxiv.org/abs/2512.02581)). Recently, I am focusing on RL and LLM Agent: RL to empower LLM agents with long-horizon human-like reasoning, planning and decision-making skills.

---
I am grateful to have received help and guidance for my career from so many people—such as Flint, Song, David, Ivor, Bo, Xingrui, and many others. I also understand many truly talented students may not have the opportunities to reach their full potential. So, if you are an undergrad or master student and believe I could offer advice, information, or opportunities that might help with your career, feel free to reach out to me by email. Meanwhile, I am also a student mentor in [CFAR](https://www.a-star.edu.sg/cfar), [IHPC](https://www.a-star.edu.sg/ihpc/), [A*STAR](https://www.a-star.edu.sg/) at Singapore. If you are interested in visiting or intern opportunities, you are welcomed to drop me an email for a chat.

<style>
.mentees-list {
  margin-top: 0.5em;
  margin-bottom: 1em;
  padding-left: 20px;
}
.mentees-list li {
  margin-bottom: 2px;
  padding-bottom: 0;
}
</style>
**Mentees (research-based, majority of them are co-mentored with scientists at CFAR, IHPC, A*STAR):**
<ul class="mentees-list">
  <li>Yaxin Zhou (Master's student @ CMU, America)</li>
  <li>Jingxuan Wu (Master's student @ UNC, America)</li>
  <li>Xu Pan (Master’s student @ Wuhan University)</li>
  <li>Chubin Zhang (Master’s student @ Beijing University of Posts and Telecommunications)</li>
  <li>Dongchu Xie (Undergrad @ CUHK(SZ))</li>
  <li>Xin Yan (Undergrad @ Beijing Normal University)</li>
  <li>He Ma (Undergrad @ CUHK(SZ))</li>
  
</ul>


<mark style="background-color: yellow; color: black;">Besides, I am ALWAYS open to collaborations, networking and intern opportunities, feel free to contact me by email! (vanzl3386 [at] gmail.com)</mark>

<span class='anchor' id='news'></span>
# 🔥 News
<style>
/* News section scroll window styles (scoped) */
.news .scroll-window { max-height: 640px; overflow-y: auto; padding: 8px 6px; border: 1px solid #eaeaea; border-radius: 12px; background: #fff; box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 6px 14px rgba(0,0,0,0.04); }
.news .scroll-window::-webkit-scrollbar { width: 8px; }
.news .scroll-window::-webkit-scrollbar-thumb { background: #ddd; border-radius: 4px; }
</style>

<div class="news" markdown="1">
<div class="scroll-window" markdown="1">
- *2026.01*  &nbsp;🎉 We released our new work **CaveAgent** (a new product-inspired function calling paradigm for LLM Agent) ([[Paper]](https://arxiv.org/abs/2601.01569v1), [[Source Code]](https://github.com/acodercat/cave-agent))!
- *2025.12*  &nbsp;🎉 We released our new work **GoRL** for online Reinforcement Learning with generative policies. ([[Paper]](https://arxiv.org/abs/2512.02581), [[Code]](https://github.com/bennidict23/GoRL))
- *2025.10*  &nbsp;🎉 Our paper **FM-IRL** (Flow Matching for inverse RL) is on Arxiv. ([[Paper]](https://arxiv.org/abs/2510.09222), [[Code]](https://github.com/vanzll/FM_IRL)).
- *2025.10*  &nbsp;🎉 **OSCAR** (training free technique for diverse image generation) is on Arxiv. ([[Paper]](https://arxiv.org/abs/2510.09060), [[Code]](https://github.com/Johnny221B/OSCAR)).
- *2025.08*  &nbsp;🎉🎉 Received my B.Sc from [The Chinese University of Hong Kong](https://www.cuhk.edu.hk/english/index.html) with 1-st class honor, looking forward to new journey!
- *2025.05*  &nbsp;🎉🎉 [**EBC**](https://arxiv.org/abs/2410.06151) is accepted by ICML 2025 ([Code](https://github.com/vanzll/EBC) available).
- *2025.03*  &nbsp;🎉🎉 One paper accepted by ICLR 2025 (generative models for robot learning workshop).
- *2024.12*  &nbsp;🎉🎉 One paper accepted by AAMAS 2025 (oral).
- *2024.12*  &nbsp;🎉🎉 One paper accepted by AAAI 2025 (oral).
- *2024.09*  &nbsp;🎉🎉 One invention patent is published.
- *2024.09*  &nbsp;🎉🎉 Awarded *Academic Performance Scholarship* (for top 5% students) for consecutive two years.
- *2024.05*  &nbsp;🎉🎉 One invention patent is officially granted. 
- *2023.12*  &nbsp;🎉🎉 As tech co-founder, I co-founded enterprise "Metasequoia Intelligence" based in Shenzhen, China.
- *2021.09*  &nbsp;🎉🎉 Awarded *Zhejiang Guolong Inspirational* and *Diligentia Bowen* Scholarship to support my undergraduate studies in CUHK(SZ).
- *2019.09*  &nbsp;🎉🎉 Lucky to win the 1-st prize in Provincial Chinese Mathematics Olympiad (CMO). Thanks for this intellectually-rewarding experience.

</div>
</div>

<script>
(function() {
  function setNewsScrollWindowHeight() {
    var container = document.querySelector('.news .scroll-window');
    if (!container) return;
    var firstLi = container.querySelector('li');
    if (!firstLi) return; // fallback to CSS default max-height
    var liRect = firstLi.getBoundingClientRect();
    var liStyle = window.getComputedStyle(firstLi);
    var marginTop = parseFloat(liStyle.marginTop) || 0;
    var marginBottom = parseFloat(liStyle.marginBottom) || 0;
    var perItem = liRect.height + (marginTop + marginBottom);
    var target = perItem * 9; // show ~9 items
    container.style.maxHeight = target + 'px';
  }
  function onReady(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn, { once: true });
    } else { fn(); }
  }
  onReady(setNewsScrollWindowHeight);
  var resizeTimeout;
  window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(setNewsScrollWindowHeight, 150);
  });
})();
</script>

<span class='anchor' id='Publication-List'></span>
# 📝 Selected Publications

<button class="section-toggle-btn" onclick="toggleSection('publications-section', this)">
  <span class="toggle-icon">▼</span> Click to expand
</button>

<div id="publications-section" class="collapsible-content" markdown="1">

## Conference papers and Preprints
Please scroll down to view all publications.
\* denotes joint-first-author and equal contribution.
<style>
/* Publications section styles (scoped) */
.publications { font-family: "Times New Roman", Times, serif; font-size: 0.96em; }
.publications .bibliography { list-style: none; margin: 0; padding: 0; }
.publications .bibliography li { margin: 10px 0; }
.publications .pub-row {
  display: flex;
  gap: 12px;
  align-items: center;
  background: #fff;
  border: 1px solid #eee;
  border-radius: 12px;
  padding: 14px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
  max-width: 900px;
  margin-left: auto;
  margin-right: auto;
}
.publications .pub-row .abbr.pub-thumb {
  position: relative;
  flex: 0 0 320px;
  max-width: 320px;
  padding: 0 15px; /* keep original padding intent */
}
.publications .pub-row .abbr.pub-thumb img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 10px 16px rgba(0,0,0,0.12);
}
.publications .pub-row .abbr.pub-thumb .badge {
  position: absolute;
  top: -8px;
  left: -8px;
  background: #e74d3c;
  color: #fff;
  padding: 4px 8px;
  border-radius: 6px;
  font-weight: 700;
}
.publications .pub-content { flex: 1 1 auto; padding-left: 4px; }
.publications .title { font-size: 1.14rem; font-weight: 700; line-height: 1.35; }
.publications .author { font-size: 0.98rem; }
.publications .periodical { font-size: 0.96rem; }
.publications .honor { font-size: 0.96rem; color: #e74d3c; font-weight: 700; margin: 4px 0; }
.publications .intro { font-size: 0.90rem; color: #555; font-weight: 600; font-style: italic; margin: 4px 0; }
.publications .links a { font-size: 12px !important; }
.publications .links { margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap; }
.publications .pub-button {
  font-family: "Times New Roman", Times, serif;
  background: #fff;
  color: #333;
  border: 1px solid #ddd;
  border-radius: 0;
  padding: 8px 14px;
  font-size: 1rem;
  text-decoration: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.12);
  transition: transform .05s ease, box-shadow .2s ease, border-color .2s ease;
}
.publications .pub-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 18px rgba(0,0,0,0.16);
  border-color: #bbb;
  text-decoration: none;
}
.publications .title a { color: #2a72d4; text-decoration: none; }
.publications .title a:hover { text-decoration: underline; color: #1e5bb8; }
/* Scroll window to show only a few items initially */
.publications .scroll-window { max-height: 640px; overflow-y: auto; padding: 8px 6px; border: 1px solid #eaeaea; border-radius: 12px; background: #fff; box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 6px 14px rgba(0,0,0,0.04); }
.publications .scroll-window::-webkit-scrollbar { width: 8px; }
.publications .scroll-window::-webkit-scrollbar-thumb { background: #ddd; border-radius: 4px; }
@media (max-width: 640px) {
  .publications .pub-row { flex-direction: column; }
  .publications .pub-row .abbr.pub-thumb { max-width: 100%; flex-basis: auto; }
  .publications .scroll-window { max-height: 420px; }
}
</style>

<div class="publications" markdown="1">
<div class="scroll-window">
<ul class="bibliography">

{% for link in site.data.publications.main %}

<li>
<div class="pub-row">
  <div class="col-sm-3 abbr pub-thumb" style="position: relative;padding-right: 15px;padding-left: 15px;">
    {% if link.image %} 
    <img src="{{ link.image }}" class="teaser img-fluid z-depth-1" style="width: 100%; height: auto;">
    {% endif %}
    {% if link.conference_short %} 
    <abbr class="badge">{{ link.conference_short }}</abbr>
    {% endif %}
  </div>
  <div class="col-sm-9 pub-content" style="position: relative;padding-right: 15px;padding-left: 20px;">
      <div class="title"><a href="{{ link.paper | default: '/404.html' }}">{{ link.title }}</a></div>
      <div class="author">{{ link.authors }}</div>
      <div class="periodical"><em>{{ link.conference }}</em>
      </div>
      {% if link.honor %}
      <div class="honor">{{ link.honor }}</div>
      {% endif %}
      {% if link.intro %}
      <div class="intro">{{ link.intro }}</div>
      {% endif %}
    <div class="links">
      <a href="{{ link.paper | default: '/404.html' }}" class="pub-button paper" role="button" target="_blank">Paper</a>
      <a href="{{ link.code | default: '/404.html' }}" class="pub-button code" role="button" target="_blank">Code</a>
      <a href="{{ link.website | default: '/404.html' }}" class="pub-button website" role="button" target="_blank">Website</a>
      {% if link.github_folks %} 
      <a target="_blank" href ="https://github.com/{{ link.github_stars }}"><img alt="GitHub forks" align="right" src="https://img.shields.io/github/forks/{{ link.github_folks }}?style=social"></a>
      {% endif %}
      {% if link.github_stars %} 
      <a target="_blank" href ="https://github.com/{{ link.github_stars }}"><img alt="GitHub stars" align="right" src="https://img.shields.io/github/stars/{{ link.github_stars }}?style=social"></a>
      {% endif %}
    </div>
  </div>
</div>
</li>



{% endfor %}

</ul>
</div>
</div>

<script>
(function() {
  function setScrollWindowHeight() {
    var container = document.querySelector('.publications .scroll-window');
    if (!container) return;
    var firstLi = container.querySelector('.bibliography > li');
    var firstRow = container.querySelector('.pub-row');
    if (!firstRow || !firstLi) return;
    var rowRect = firstRow.getBoundingClientRect();
    var liStyle = window.getComputedStyle(firstLi);
    var marginTop = parseFloat(liStyle.marginTop) || 0;
    var marginBottom = parseFloat(liStyle.marginBottom) || 0;
    var perItem = rowRect.height + (marginTop + marginBottom);
  var target = perItem * 3.5; // show ~3.5 items
    container.style.maxHeight = target + 'px';
  }
  function onReady(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn, { once: true });
    } else { fn(); }
  }
  onReady(setScrollWindowHeight);
  var resizeTimeout;
  window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(setScrollWindowHeight, 150);
  });
})();
</script>

## Invention Patents
As these works are patented in China, all these names are directly translated from Chinese.
- **A Method, System, Terminal Device, and Storage Medium for Air Quality Spatial Inference** (Granted)
  
   Inventor: Jun Song, Yibo Xu, Yiwen Pan, Maohao Ran, **<u>Zhenglin Wan</u>**, Xiaoyun Yan, Yike Guo
- **A Single-UAV Atmospheric Pollutant Source Tracing Method Based on Gradient Ascent and Physical Kinematics** (Public)

   Inventor: **<u>Zhenglin Wan</u>**, Jun Song, Yibo Xu, Maohao Ran, Yike Guo

## White Paper
As these works are presented in China, these names are directly translated from Chinese.

- **White Paper on Cross-Border Economic Large Language Model**

</div>

<span class='anchor' id='educations'></span>
# 📖 Educations

<button class="section-toggle-btn" onclick="toggleSection('educations-section', this)">
  <span class="toggle-icon">▼</span> Click to expand
</button>

<div id="educations-section" class="collapsible-content" markdown="1">

<style>
.edu-list { list-style: none; margin: 0; padding: 0; }
.edu-item { 
  background: #fff; border: 1px solid #eee; border-radius: 12px; 
  box-shadow: 0 4px 12px rgba(0,0,0,0.06); padding: 14px; 
  display: flex; gap: 16px; align-items: center; max-width: 900px; margin: 10px auto;
}
.edu-text { flex: 1 1 auto; font-family: "Times New Roman", Times, serif; }
.edu-title { font-weight: 700; margin: 0 0 6px; }
.edu-sub { color: #555; margin: 0 0 6px; }
.edu-time { color: #777; font-size: 0.95em; }
.edu-img { flex: 0 0 200px; max-width: 200px; }
.edu-img img { width: 100%; height: auto; border-radius: 8px; box-shadow: 0 6px 14px rgba(0,0,0,0.10); }
@media (max-width: 640px) { .edu-item { flex-direction: column; } .edu-img { max-width: 100%; flex-basis: auto; } }
</style>

<ul class="edu-list">
  <li class="edu-item">
    <div class="edu-text">
      <p class="edu-title">Doctor of Philosophy (Ph.D)</p>
      <p class="edu-sub">National University of Singapore (NUS)</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>Affiliation: Department of Computer Science, School of Computing</li>
        <li>Advisor: Prof. Yang You</li>
      </ul>
      <p class="edu-time">2026.01 -</p>
    </div>
    <div class="edu-img"><img src="assets/institutions/nus.png" alt="NUS"></div>
  </li>
  <li class="edu-item">
    <div class="edu-text">
      <p class="edu-title">Bachelor of Science (B.Sc)</p>
      <p class="edu-sub">The Chinese University of Hong Kong (CUHK)</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>1-st class honor</li>
        <li>Major: Statistics & Data Science, GPA: 3.85/4.0, Rank: 7%</li>
        <li>Finished my undergraduate in CUHK-Shenzhen campus, while the degree is offered by CUHK.</li>
      </ul>
      <p class="edu-time">2021.09 - 2025.08</p>
    </div>
    <div class="edu-img"><img src="assets/institutions/cuhk.png" alt="CUHK(SZ)"></div>
  </li>
</ul>

</div>

---
<span class='anchor' id='internships'></span>
# 💻 Internships and Work Experiences

<button class="section-toggle-btn" onclick="toggleSection('internships-section', this)">
  <span class="toggle-icon">▼</span> Click to expand
</button>

<div id="internships-section" class="collapsible-content" markdown="1">

<style>
.exp-list { list-style: none; margin: 0; padding: 0; }
.exp-item { 
  background: #fff; border: 1px solid #eee; border-radius: 12px; 
  box-shadow: 0 4px 12px rgba(0,0,0,0.06); padding: 14px; 
  display: flex; gap: 16px; align-items: center; max-width: 900px; margin: 10px auto;
}
.exp-text { flex: 1 1 auto; font-family: "Times New Roman", Times, serif; }
.exp-title { font-weight: 700; margin: 0 0 6px; }
.exp-sub { color: #555; margin: 0 0 6px; }
.exp-time { color: #777; font-size: 0.95em; }
.exp-img { flex: 0 0 200px; max-width: 200px; }
.exp-img img { width: 100%; height: auto; border-radius: 8px; box-shadow: 0 6px 14px rgba(0,0,0,0.10); background: #fff; }
@media (max-width: 640px) { .exp-item { flex-direction: column; } .exp-img { max-width: 100%; flex-basis: auto; } }
</style>

<style>
/* Experiences section scroll window styles (scoped) */
.experiences .scroll-window { max-height: 640px; overflow-y: auto; padding: 8px 6px; border: 1px solid #eaeaea; border-radius: 12px; background: transparent; box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 6px 14px rgba(0,0,0,0.04); }
.experiences .scroll-window::-webkit-scrollbar { width: 8px; }
.experiences .scroll-window::-webkit-scrollbar-thumb { background: #ddd; border-radius: 4px; }
</style>

<style>
/* Dark mode overrides for Internships section */
@media (prefers-color-scheme: dark) {
  .experiences .scroll-window { background: #111; border-color: #333; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 6px 14px rgba(0,0,0,0.6); }
  .exp-item { background: #111; border-color: #333; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
}
</style>

<style>
/* Dark mode overrides (class/data-attribute toggles) for Internships section */
html.dark .experiences .scroll-window,
body.dark .experiences .scroll-window,
html[data-theme="dark"] .experiences .scroll-window,
:root[data-theme="dark"] .experiences .scroll-window,
[data-scheme="dark"] .experiences .scroll-window { background: #111 !important; border-color: #333 !important; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 6px 14px rgba(0,0,0,0.6) !important; }

html.dark .exp-item,
body.dark .exp-item,
html[data-theme="dark"] .exp-item,
:root[data-theme="dark"] .exp-item,
[data-scheme="dark"] .exp-item { background: #111 !important; border-color: #333 !important; box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important; }

html.dark .exp-img img,
body.dark .exp-img img,
html[data-theme="dark"] .exp-img img,
:root[data-theme="dark"] .exp-img img,
[data-scheme="dark"] .exp-img img { background: transparent !important; }
</style>

<div class="experiences">
<div class="scroll-window">
<ul class="exp-list">
  <li class="exp-item">
    <div class="exp-text">
      <p class="exp-title">Full-time Research Staff</p>
      <p class="exp-sub">Nanyang Technological University, Singapore</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>Affiliation: College of Computing and Data Science (CCDS)</li>
        <li>Advisor: Prof. Bo An</li>
      </ul>
      <p class="exp-time">2025.09 - 2026.01</p>
    </div>
    <div class="exp-img"><img src="assets/institutions/ntu.png" alt="NTU"></div>
  </li>


  <li class="exp-item">
    <div class="exp-text">
      <p class="exp-title">Intern (Remote)</p>
      <p class="exp-sub">Hong Kong Generative AI Research & Development Center (HKGAI), HKUST</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>Affiliation: HKGAI</li>
        <li>Director: Prof. Yike Guo</li>
        <li>Proposed a new product-inspired Agentic Function Calling paradigm (<a href="https://arxiv.org/abs/2601.01569v1">[[CaveAgent]]</a>).</li>
      </ul>
      <p class="exp-time">2025.05 - 2025.09</p>
    </div>
    <div class="exp-img"><img src="assets/institutions/hkgai.png" alt="HKGAI"></div>
  </li>


  <li class="exp-item">
    <div class="exp-text">
      <p class="exp-title">Intern Researcher</p>
      <p class="exp-sub">Agency for Science, Technology and Research (A*STAR), Singapore</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>Affiliation: Centre for Frontier AI Research (CFAR), Institute of High Performance Computing (IHPC)</li>
        <li>Advisor: Prof. Ivor Tsang, Dr. Xingrui Yu</li>
      </ul>
      <p class="exp-time">2024.07 - 2025.09  (3 months full-time + 11 months part-time)</p>
    </div>
    <div class="exp-img"><img src="assets/institutions/astar.png" alt="A*STAR"></div>
  </li>


  <li class="exp-item">
    <div class="exp-text">
      <p class="exp-title">Research Assistant</p>
      <p class="exp-sub">The Chinese University of Hong Kong, Shenzhen</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>Affiliation: School of Data Science</li>
        <li>Advisor: Prof. Jianfeng Mao</li>
      </ul>
      <p class="exp-time">2023.06 - 2024.06</p>
    </div>
    <div class="exp-img"><img src="assets/institutions/cuhksz.png" alt="CUHK(SZ)"></div>
  </li>

  <li class="exp-item">
    <div class="exp-text">
      <p class="exp-title">Machine Learning Algorithm Engineer Intern</p>
      <p class="exp-sub">HUIYINTONG, Shenzhen, China</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>Mentor: Dr. Jun Song</li>
      </ul>
      <p class="exp-time">2023.07 - 2024.01</p>
    </div>

  </li>



 </ul>
</div>
</div>
<span class='anchor' id='person'></span>

<script>
(function() {
  function setExpScrollWindowHeight() {
    var container = document.querySelector('.experiences .scroll-window');
    if (!container) return;
    var firstLi = container.querySelector('.exp-list > li');
    if (!firstLi) return; // fallback to CSS default max-height
    var rowRect = firstLi.getBoundingClientRect();
    var liStyle = window.getComputedStyle(firstLi);
    var marginTop = parseFloat(liStyle.marginTop) || 0;
    var marginBottom = parseFloat(liStyle.marginBottom) || 0;
    var perItem = rowRect.height + (marginTop + marginBottom);
    var target = perItem * 3; // show ~3 items
    container.style.maxHeight = target + 'px';
  }
  function onReady(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn, { once: true });
    } else { fn(); }
  }
  onReady(setExpScrollWindowHeight);
  var resizeTimeout;
  window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(setExpScrollWindowHeight, 150);
  });
})();
</script>

</div>

<span class='anchor' id='services'></span>
# 🎈 Services

<button class="section-toggle-btn" onclick="toggleSection('services-section', this)">
  <span class="toggle-icon">▼</span> Click to expand
</button>

<div id="services-section" class="collapsible-content" markdown="1">

- Reviewer of AAAI, ICLR, ICML, NeurIPS

</div>

<span class='anchor' id='honors-and-awards'></span>
# 🎖 Honors, Awards and Scholarships

<button class="section-toggle-btn" onclick="toggleSection('honors-section', this)">
  <span class="toggle-icon">▼</span> Click to expand
</button>

<div id="honors-section" class="collapsible-content" markdown="1">

- **NUS Research Scholarship** (approx. ￥250000/year plus full tuition fee subsidy, for entire Ph.D studies in NUS)
  
- **1-st class honor undergraduate student** awarded by The Chinese University of Hong Kong

- **Yearly Academic Scholarship: B Class** (for GPA Top 3%, ￥40000)  

- **Yearly Academic Scholarship: C Class** (for GPA Top 5%, ￥20000)  

- **Yearly Dean List Award** (Outstanding 1-st class Performance, for 3 years)  

- **Diligentia Bowen Scholarship** (￥120000, Undergraduate Admission Scholarship for 1-st prize in Provincial CMO)

- **Zhejiang Guolong Inspirational Scholarship** (￥120000, Undergraduate Admission Scholarship for top 0.5% students in Chinese College Entrance Exam)  

- **1st-Prize in Chinese Mathematics Olympiad (CMO)-Chongqing Province**

</div>
  
<span class='anchor' id='press-media'></span>

# 💬 Press/Media

<button class="section-toggle-btn" onclick="toggleSection('press-section', this)">
  <span class="toggle-icon">▼</span> Click to expand
</button>

<div id="press-section" class="collapsible-content" markdown="1">

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">White Paper</div><img src='personal_page_sources/white_paper.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">


- The co-author of the first White Paper on Cross-Border Economic Large Language Model in Shenzhen, China.
[深圳卫视：深圳发布首个跨境经济大模型白皮书](https://www.sohu.com/a/786227332_121123831)

</div>
</div>

</div>

<span class='anchor' id='person'></span>
# Miscellaneous

<button class="section-toggle-btn" onclick="toggleSection('misc-section', this)">
  <span class="toggle-icon">▼</span> Click to expand
</button>

<div id="misc-section" class="collapsible-content" markdown="1">

- In my spare time, I'm an **music enthusiast**. I’ve been playing guitar for more than 10 years and began teaching myself the piano when I was 15. During my undergraduate, I played music in two bands: "Minor Blue" and "Major Pink." See our photos: 

<div style="display: flex; justify-content: space-between;">
  <img src='images/Major pink.jpg' alt="sym" width="33%" style="display: inline-block;">
  <img src='images/Minor Blue.jpg' alt="sym" width="33%" style="display: inline-block;">
  <img src='images/Minor Blue_1.jpg' alt="sym" width="33%" style="display: inline-block;">
</div>







- I am also a **15-years chess player**, with the honor of "National Level-3 Athlete". I love the process of comprehensive planning, logical-thinking and reasoning. Visit my [Lichess profile](https://lichess.org/@/Carlos1333860).

- I love play **basketball** 🏀. Sports makes me energetic.

- I play video games like **League of Legends**, where I achieved the "diamond" level as my historically highest honor. I also play 3A games like Elden Ring, Dark Souls, Nier Automata, and elder scrolls.

- I have a deep interest in **philosophy of mind**, particularly Buddhism and Taoism, as paths to explore the fundamental nature of human existence. I am also intrigued by the potential integration of these philosophical insights with modern artificial intelligence.

</div>

<script>
function toggleSection(sectionId, btn) {
  var content = document.getElementById(sectionId);
  var isExpanded = content.classList.contains('expanded');
  
  if (isExpanded) {
    content.classList.remove('expanded');
    btn.classList.remove('expanded');
    btn.innerHTML = '<span class="toggle-icon">▼</span> Click to expand';
  } else {
    content.classList.add('expanded');
    btn.classList.add('expanded');
    btn.innerHTML = '<span class="toggle-icon">▼</span> Click to collapse';
  }
}
</script>



