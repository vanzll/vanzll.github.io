---
permalink: /
title: ""
excerpt: ""
author_profile: true
mmv_globe_d: uc46Usqtqu46611K0F3RqKvrq3w6VrhNaTbD2FUfoGo
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
- *2025.10*     **FM-IRL** is available on [arxiv](https://arxiv.org/abs/2510.09222) and [github](https://github.com/vanzll/FM_IRL).
- *2025.10*     **OSCAR** is available on [arxiv](https://arxiv.org/abs/2510.09060) and [github](https://github.com/Johnny221B/OSCAR).
- *2025.09*  &nbsp;🎉🎉 Awarded *NUS Research Scholarship* to support my Ph.D studies in [National University of Singapore](https://nus.edu.sg/) beginning from Jan, 2026.
- *2025.09*  &nbsp;🎉🎉 Joined [Nanyang Technological University](https://www.ntu.edu.sg/) as a research staff.
- *2025.08*  &nbsp;🎉🎉 Received my B.Sc from [The Chinese University of Hong Kong](https://www.cuhk.edu.hk/english/index.html) with 1-st class honor, looking forward to new jouney!
- *2025.05*  &nbsp;🎉🎉 [**EBC**](https://arxiv.org/abs/2410.06151) is accepted by ICML 2025 ([Code](https://github.com/vanzll/EBC) available).
- *2025.03*  &nbsp;🎉🎉 One paper accepted by ICLR 2025 (generative models for robot learning workshop).
- *2024.12*  &nbsp;🎉🎉 One paper accepted by AAMAS 2025 (oral).
- *2024.12*  &nbsp;🎉🎉 One paper accepted by AAAI 2025 (oral).
- *2024.09*  &nbsp;🎉🎉 One invention patent is published.
- *2024.09*  &nbsp;🎉🎉 Awarded *Academic Performance Scholarship* (for top 5% students) for consecutive two years.
- *2024.05*  &nbsp;🎉🎉 One invention patent is officially granted. 
- *2023.12*  &nbsp;🎉🎉 As tech co-founder, I co-founded enterprise "Metasequoia Intelligence" based in Shenzhen, China.
- *2021.09*  &nbsp;🎉🎉 Awarded *Zhejiang Guolong Inspirational* and *Diligentia Bowen* Scholarship to support my undergraduate studies in CUHK(SZ).
- *2019.09*  &nbsp;🎉🎉 Lucky to win the 1-st prize in CMO 1-st round (Chongqing Province). Thanks for this intellectually-rewarding experience.

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



<span class='anchor' id='about-me'></span>
# 😊 About me

***Resume***: [Resume](https://drive.google.com/file/d/1fYYlj0MhJGsPPDdWUMB9LHMTqUo3bL1s/view?usp=sharing) (Updated on 8 Oct, 2025)

***Contact***: vanzl3386 [at] gmail.com (preferred), carlos [at] metaseq.ai, 121090525 [at] link.cuhk.edu.cn

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
</style>

<style>
/* Widen reading width on desktop without affecting mobile */
@media (min-width: 1200px) {
  .page { max-width: 1400px !important; }
  .page__inner-wrap { max-width: 1120px; margin-left: auto; margin-right: auto; }
  .page__content { max-width: 1120px !important; }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .page__inner-wrap { max-width: 1040px; margin-left: auto; margin-right: auto; }
  .page__content { max-width: 1040px !important; }
}
</style>

Zhenglin Wan (万政霖 in Chinese) received his Bachelor of Science (B.Sc) from [The Chinese University of Hong Kong](https://www.cuhk.edu.hk/english/index.html) on 2025 Fall. He is currently a research staff in [Nanyang Technological University](https://www.ntu.edu.sg/), working with [Prof. Bo An](https://personal.ntu.edu.sg/boan/), and also an incoming Ph.D. student in [School of Computing (SoC)](https://www.comp.nus.edu.sg/), [National University of Singapore (NUS)](https://nus.edu.sg/). Previously, he has been an intern researcher at [Centre for Frontier AI Research](https://www.a-star.edu.sg/cfar), [(A*STAR)](https://www.a-star.edu.sg/) at Singapore, under the supervision of [Prof. Ivor Tsang](https://www.a-star.edu.sg/cfar/about-cfar/management/prof-ivor-tsang) and collaborated with [Prof. Ong Yew Soon](https://www3.ntu.edu.sg/home/asysong/home.html). In CUHK(SZ) campus, he was advised by [Prof. Jianfeng Mao](https://sds.cuhk.edu.cn/en/teacher/268) and [Prof. Ming Yan](https://mingyan08.github.io/). Besides, he also closely worked with [Prof. Pingfu Chao](https://scst.suda.edu.cn/10/47/c11250a528455/page.htm) and [Dr. Jun Song](https://scholars.hkbu.edu.hk/en/persons/JUNSONG) both academically and industrially. 

---

My research focus lies in 1)**Agentic AI**: How to embed the human's learning and decision-making patterns into (both virtual and embodied) agentic paradigm? How to enable intelligent agent to perform high-level, long-horizen planning and strategic thinking? 2)**Decision-Making in Machine Learning**: the algorithmic side of reinforcement learning (RL), as well as its intersection with modern generative models.


<mark style="background-color: yellow; color: black;">I am always open to collaborations, networking and potential intern opportunities, feel free to drop me an email! (vanzl3386 [at] gmail.com)</mark>



<span class='anchor' id='Publication-List'></span>
# 📝 Selected Publications

## Conference papers and Preprints
Please scroll down to view all publications.
<style>
/* Publications section styles (scoped) */
.publications { font-family: "Times New Roman", Times, serif; font-size: 0.96em; }
.publications .bibliography { list-style: none; margin: 0; padding: 0; }
.publications .bibliography li { margin: 10px 0; }
.publications .pub-row {
  display: flex;
  gap: 12px;
  align-items: flex-start;
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



<span class='anchor' id='educations'></span>
# 📖 Educations

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
      <p class="edu-sub">National University of Singapore (NUS), School of Computing</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>Advisor: TBD</li>
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
        <li>Major: Data Science, GPA: 3.85/4.0, Rank: 7%</li>
        <li>Finished my undergraduate in School of Data Science, CUHK-Shenzhen campus, while the degree is offered by CUHK.</li>
      </ul>
      <p class="edu-time">2021.09 - 2025.08</p>
    </div>
    <div class="edu-img"><img src="assets/institutions/cuhk.png" alt="CUHK(SZ)"></div>
  </li>
</ul>



---
<span class='anchor' id='internships'></span>
# 💻 Internships and Work Experiences
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

<ul class="exp-list">
  <li class="exp-item">
    <div class="exp-text">
      <p class="exp-title">Full-time Research Staff</p>
      <p class="exp-sub">Nanyang Technological University, Singapore</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>Affliation: College of Computing and Data Science (CCDS)</li>
        <li>Advisor: Prof. Bo An</li>
      </ul>
      <p class="exp-time">2025.09 - 2026.01</p>
    </div>
    <div class="exp-img"><img src="assets/institutions/ntu.png" alt="NTU"></div>
  </li>
  <li class="exp-item">
    <div class="exp-text">
      <p class="exp-title">Intern Researcher</p>
      <p class="exp-sub">Agency for Science, Technology and Research (A*STAR), Singapore</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>Affliation: Centre for Frontier AI Research (CFAR), Institute of High Performance Computing (IHPC)</li>
        <li>Advisor: Prof. Ivor Tsang, Dr. Xingrui Yu</li>
      </ul>
      <p class="exp-time">2024.07 - 2025.09</p>
      <ul style="margin: 6px 0 6px 18px; padding: 0; font-size: 0.96em; color: #555;">
        <li>2024.07 - 2024.10: Full time (onsite)</li>
        <li>2024.10 - 2025.09: Part time (remote)</li>
      </ul>
    </div>
    <div class="exp-img"><img src="assets/institutions/astar.png" alt="A*STAR"></div>
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
<span class='anchor' id='person'></span>



<span class='anchor' id='services'></span>
# 🎈 Services
- Reviewer of AAAI, ICLR, ICML, NeurIPS

<span class='anchor' id='honors-and-awards'></span>
# 🎖 Honors, Awards and Scholarships
- **NUS Research Scholarship** (approx. ￥250000/year plus full tuition fee subsidy, for entire Ph.D studies in NUS)
  
  *Janurary 2026-*
  
- **1-st class honor undergraduate student** awarded by The Chinese University of Hong Kong
  
  *August 2025*

- **Yearly Academic Scholarship: B Class** (for GPA Top 3%, ￥40000)  
  *September 2024*

- **Yearly Academic Scholarship: C Class** (for GPA Top 5%, ￥20000)  
  *September 2023*

- **Yearly Dean List Award** (Outstanding 1-st class Performance)  
  *Consecutive three years: September 2022 - September 2025*

- **Diligentia Bowen Scholarship** (Undergraduate Admission Scholarship, ￥120000)
  
  *September 2021*

- **Zhejiang Guolong Inspirational Scholarship** (Undergraduate Admission Scholarship, ￥120000)  
  *September 2021*

- **1st-Prize in Chinese Mathematics Olympics (CMO)-1st Round (Chongqing Province)**
  
  *September 2019*

  
<span class='anchor' id='press-media'></span>


# 💬 Press/Media
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">White Paper</div><img src='personal_page_sources/white_paper.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">


- The co-author of the first White Paper on Cross-Border Economic Large Language Model in Shenzhen, China.
[深圳卫视：深圳发布首个跨境经济大模型白皮书](https://www.sztv.com.cn/ysz/dsdb/szws/ssxw/80141762.shtml)

</div>
</div>

<span class='anchor' id='person'></span>

# Miscellaneous
In my spare time, I’m an **music enthusiast**. I’ve been playing guitar for more than 10 years and began teaching myself the piano and electronic piano at the age of 15. Music has always been a significant part of my life. During my undergraduate, I had the incredible opportunity to play in two bands: "Minor Blue" and "Major Pink." These moments remain some of my most cherished memories. See our photos: 

<div style="display: flex; justify-content: space-between;">
  <img src='images/Major pink.jpg' alt="sym" width="33%" style="display: inline-block;">
  <img src='images/Minor Blue.jpg' alt="sym" width="33%" style="display: inline-block;">
  <img src='images/Minor Blue_1.jpg' alt="sym" width="33%" style="display: inline-block;">
</div>



---



I am also a **15-years chess player**, with the honor of "National Level-3 Athlete". I love the process of comprehensive planning, logical-thinking and inferring. Visit my [Lichess profile](https://lichess.org/@/Carlos1333860).

---

I love play **basketball** 🏀. Sports makes me energetic.

---

I play video games like **League of Legends**, where I achieved the "diamond" level as my historically highest honor. I also play 3A games like Elden Ring, Dark Souls, Nier Automata, and elder scrolls.

---

I have a deep interest in **philosophy of mind**, particularly Buddhism and Taoism, as paths to explore the fundamental nature of human existence. I am also intrigued by the potential integration of these philosophical insights with modern artificial intelligence.





