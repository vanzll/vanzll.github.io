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
***Contact***: **vanzl3386 [at] gmail.com (main)**, vanzl [at] u.nus.edu, 121090525 [at] link.cuhk.edu.cn

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

.news .scroll-window,
.publications .scroll-window,
.experiences .scroll-window {
  max-height: 640px;
  overflow-y: auto;
  padding: 8px 6px;
  border: 1px solid #eaeaea;
  border-radius: 8px;
  background: #fff;
}
.scroll-window::-webkit-scrollbar { width: 8px; }
.scroll-window::-webkit-scrollbar-thumb { background: #ddd; border-radius: 4px; }
.scroll-window:focus-visible { outline: 2px solid #2a72d4; outline-offset: 2px; }
.dark-mode .experiences .scroll-window { background: #0f141b; border-color: #2a2f3a; }
@media (max-width: 640px) { .publications .scroll-window { max-height: 420px; } }
</style>

<!-- Dark mode is handled globally in layout/head; local button/styles removed -->


I am Zhenglin Wan (万政霖), a CS Ph.D. student at [HPC-AI Lab](https://ai.comp.nus.edu.sg/) at National University of Singapore (NUS), advised by [Prof. Yang You](https://www.comp.nus.edu.sg/~youy/). Previously, I worked at Nanyang Technological University (NTU) with [Prof. Bo An](https://personal.ntu.edu.sg/boan/), [Centre for Frontier AI Research (CFAR)](https://www.a-star.edu.sg/cfar), [IHPC](https://www.a-star.edu.sg/ihpc/), [A\*STAR](https://www.a-star.edu.sg/) with [Prof. Ivor Tsang](https://www.a-star.edu.sg/cfar/about-cfar/management/prof-ivor-tsang) , [Hong Kong Generative AI Research & Development Center (HKGAI)](https://www.hkgai.info/) with the team led by HKUST [Prof. Yike Guo](https://cse.hkust.edu.hk/admin/people/faculty/profile/yikeguo). I received my B.Sc (with 1-st class honor) in Statistics and Data Science from The Chinese University of Hong Kong (CUHK).

My research is driven by the curiosity of genuinely understanding (both theoretically and intuitively) the rationale, spirit and failure modes of ML algorithms and methods. Based on this principle, I am recently working on fundamental side of Multi-modal generation and RL Post-training. Particularly:

- RL post-training on continuous-space generative models (such as diffusion model)
- Unified AR-based understanding and Diffusion-based Generation and the RL-post-training paradigm of such systems
- Long-horizon agents, such as coding agent, agentic VLA and agentic Video Generation, and the RL-post-training paradigm for both un-unified and unified paradigm of generation and understanding.

I love intellectual games in my spare time: I am a 17-years chess player as *National Chess Athlete* in China (Level 3), and have won the 1-st prize in Chinese Mathematics Olympiad (CMO)-1st round by 1.5-year's part-time self-training. For academic discussion and collaboration, please don't hesitate to reach out to me via email!
 


<span class='anchor' id='news'></span>
# 🔥 News


<div class="news" markdown="1">
<div class="scroll-window" markdown="1" tabindex="0" role="region" aria-label="News" data-visible-items="9">
- *2026.05*  &nbsp;🎉🎉 **GoRL** is accepted to ICML 2026, providing a new perspective for online RL training with diffusion/flow policy for robot learning. ([[Paper]](https://arxiv.org/abs/2512.02581), [[Code]](https://github.com/bennidict23/GoRL))
- *2026.05*  &nbsp;🎉🎉 **OSCAR** (training free method for diverse rollout of diffusion model) is accepted to ICML 2026. ([[Paper]](https://arxiv.org/abs/2510.09060), [[Code]](https://github.com/Johnny221B/OSCAR)).
- *2026.04*  &nbsp;🎉🎉 I was invited to give a talk by Meta MRS (Stable online RL for alignment of diffusion-based foundation models).
- *2026.01*  &nbsp;🎉🎉 [Cave-Agent](https://arxiv.org/abs/2601.01569) is open-sourced: an object-oriented coding agentic framework with superior token-efficiency, empowering the agentic AI system being built by the Hong Kong Government.
- *2025.05*  &nbsp;🎉🎉 [**EBC**](https://arxiv.org/abs/2410.06151) (Evolutionary Strategy for robot learning to learn diverse policies) is accepted by ICML 2025.

</div>
</div>


<span class='anchor' id='blog'></span>
# <i class="fas fa-pen-nib" aria-hidden="true"></i> Blog

[Research notes →]({{ '/blog/' | relative_url }})

{% assign research_posts = site.posts | where: 'layout', 'research-post' | where_exp: 'post', 'post.draft != true' %}
{% for post in research_posts %}
- [{{ post.title_en | default: post.title }}]({{ post.url | relative_url }})
{% else %}
New posts coming soon.
{% endfor %}

<span class='anchor' id='invited-talks'></span>
# 🎤 Invited Talks

<div id="invited-talks-section" markdown="1">

- **Object-Oriented Agent Infrastructure** — Invited by *Qingke AI Community*
- **Stable Online Alignment of Diffusion-based Foundation Model** — Invited by *Meta (MRS)*

</div>

<span class='anchor' id='Publication-List'></span>
# 📝 Selected Publications

<div id="publications-section" markdown="1">

## Conference papers and Preprints
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
  border-radius: 8px;
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
.publications .honor p { margin: 0; }
.publications .honor a { color: inherit; text-decoration: underline; }
.publications .honor a:hover { color: inherit; text-decoration: none; }
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
@media (max-width: 640px) {
  .publications .pub-row { flex-direction: column; }
  .publications .pub-row .abbr.pub-thumb { max-width: 100%; flex-basis: auto; }
}
</style>

<div class="publications" markdown="1">
<div class="scroll-window" tabindex="0" role="region" aria-label="Selected publications" data-visible-items="3.5">
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
      <div class="honor">{{ link.honor | markdownify }}</div>
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


</div>

<span class='anchor' id='internships'></span>
# 💻 Internships and Work Experiences

<div id="internships-section" markdown="1">

<style>
.exp-list { list-style: none; margin: 0; padding: 0; }
.exp-item { 
  background: #fff; border: 1px solid #eee; border-radius: 8px;
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
/* Dark mode overrides for Internships section */
@media (prefers-color-scheme: dark) {
  .exp-item { background: #111; border-color: #333; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
}
</style>

<style>
/* Dark mode overrides (class/data-attribute toggles) for Internships section */

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
<div class="scroll-window" tabindex="0" role="region" aria-label="Internships and work experiences" data-visible-items="3">
<ul class="exp-list">
  <li class="exp-item">
    <div class="exp-text">
      <p class="exp-title">Full-time Research Staff</p>
      <p class="exp-sub">Nanyang Technological University, Singapore</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>Affiliation: College of Computing and Data Science (CCDS)</li>
        <li>Advisor: Prof. Bo An</li>
      </ul>
      <p class="exp-time"></p>
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
      <p class="exp-time"></p>
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
      <p class="exp-time"></p>
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
      <p class="exp-time"></p>
    </div>
    <div class="exp-img"><img src="assets/institutions/cuhksz.png" alt="CUHK(SZ)"></div>
  </li>


 </ul>
</div>
</div>


</div>

<span class='anchor' id='honors-and-awards'></span>
# 🎖 Honors, Awards and Scholarships

<div id="honors-section" markdown="1">

- **NUS Research Scholarship** (Ph.D stipend and tuition fee subsidy)
  
- **1-st class honor undergraduate student** awarded by The Chinese University of Hong Kong

- **Yearly Academic Scholarship: B Class** (for GPA Top 3%, ￥40000)  

- **Yearly Academic Scholarship: C Class** (for GPA Top 5%, ￥20000)  

- **Yearly Dean List Award** (Outstanding 1-st class Performance, for 3 years)  

- **Diligentia Bowen Scholarship** (￥120000, Undergraduate Admission Scholarship for 1-st prize in Provincial CMO)

- **Zhejiang Guolong Inspirational Scholarship** (￥120000, Undergraduate Admission Scholarship for top 0.5% students in Chinese College Entrance Exam)  

- **1st-Prize in Chinese Mathematics Olympiad (CMO)-1st round**

</div>
  
<span class='anchor' id='person'></span>
# Miscellaneous

<div id="misc-section" markdown="1">

- In my spare time, I'm an **music enthusiast**. I’ve been playing guitar for more than 10 years and began teaching myself the piano when I was 15. During my undergraduate, I played music in two bands: "Minor Blue" and "Major Pink." See our photos: 

<div style="display: flex; justify-content: space-between;">
  <img src='images/Major pink.jpg' alt="sym" width="33%" style="display: inline-block;">
  <img src='images/Minor Blue.jpg' alt="sym" width="33%" style="display: inline-block;">
  <img src='images/Minor Blue_1.jpg' alt="sym" width="33%" style="display: inline-block;">
</div>


- I am also a **17-years chess player**, with the honor of "National Level 3 Chess Athlete". I love the process of comprehensive planning, logical-thinking and reasoning. Visit my [Lichess profile](https://lichess.org/@/Carlos1333860).

- I play video games like **League of Legends**, where I achieved the "diamond" level as my historically highest honor. I also play 3A games like Elden Ring, Dark Souls, Nier Automata, and elder scrolls.

- I have a deep interest in **philosophy of mind**, particularly Buddhism and Taoism, as paths to explore the fundamental nature of human existence. I am also intrigued by the potential integration of these philosophical insights with modern artificial intelligence.

</div>

<script src="{{ '/assets/js/home-scroll.js' | relative_url }}" defer></script>
