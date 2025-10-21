---
permalink: /
title: ""
excerpt: ""
author_profile: true
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
- *2025.09*  &nbsp;🎉🎉 Awarded *NUS Research Scholarship* to support my Ph.D studies in [National University of Singapore](https://nus.edu.sg/) beginning from Jan, 2026.
- *2025.09*  &nbsp;🎉🎉 Joined [Nanyang Technological University](https://www.ntu.edu.sg/) as a research staff.
- *2025.08*  &nbsp;🎉🎉 Received my B.Sc from [The Chinese University of Hong Kong](https://www.cuhk.edu.hk/english/index.html) with 1-st class honor, looking forward to new jouney!
- *2025.05*  &nbsp;🎉🎉 One paper accepted by ICML 2025.
- *2025.03*  &nbsp;🎉🎉 One paper accepted by ICLR 2025 (generative models for robot learning workshop).
- *2024.12*  &nbsp;🎉🎉 One paper accepted by AAMAS 2025 (oral).
- *2024.12*  &nbsp;🎉🎉 One paper accepted by AAAI 2025 (oral).
- *2024.09*  &nbsp;🎉🎉 One invention patent is published.
- *2024.09*  &nbsp;🎉🎉 Awarded *Academic Performance Scholarship* (for top 5% students) for consecutive two years.
- *2024.05*  &nbsp;🎉🎉 One invention patent is officially granted. 
- *2023.12*  &nbsp;🎉🎉 As tech co-founder, I co-founded enterprise "Metasequoia Intelligence" based in Shenzhen, China.
- *2021.09*  &nbsp;🎉🎉 Awarded *Zhejiang Guolong Inspirational* and *Diligentia Bowen* Scholarship to support my undergraduate studies in CUHK(SZ).


<span class='anchor' id='about-me'></span>
# 😊 About me

## **TL; DR**:

***Resume***: [Resume](https://drive.google.com/file/d/1fYYlj0MhJGsPPDdWUMB9LHMTqUo3bL1s/view?usp=sharing) (Updated on 8 Oct, 2025)

***Contact***: vanzl3386 [at] gmail.com (preferred), carlos [at] metaseq.ai, 121090525 [at] link.cuhk.edu.cn

---

Zhenglin Wan (万政霖 in Chinese) received his Bachelor of Science (B.Sc) from [The Chinese University of Hong Kong](https://www.cuhk.edu.hk/english/index.html) on 2025 Fall. He is currently a research staff in [Nanyang Technological University](https://www.ntu.edu.sg/), working with [Prof. Bo An](https://personal.ntu.edu.sg/boan/), and also an incoming Ph.D. student in [School of Computing (SoC)](https://www.comp.nus.edu.sg/), [National University of Singapore (NUS)](https://nus.edu.sg/). Previously, he have been an intern researcher at [Centre for Frontier AI Research](https://www.a-star.edu.sg/cfar), [(A*STAR)](https://www.a-star.edu.sg/) at Singapore, under the supervision of [Prof. Ivor Tsang](https://www.a-star.edu.sg/cfar/about-cfar/management/prof-ivor-tsang) and collaborated with [Prof. Ong Yew Soon](https://www3.ntu.edu.sg/home/asysong/home.html). In CUHK(SZ) campus, he was advised by [Prof. Jianfeng Mao](https://sds.cuhk.edu.cn/en/teacher/268) and [Prof. Ming Yan](https://mingyan08.github.io/). Besides, he also closely worked with [Prof. Pingfu Chao](https://scst.suda.edu.cn/10/47/c11250a528455/page.htm) and [Dr. Jun Song](https://scholars.hkbu.edu.hk/en/persons/JUNSONG) both academically and industrially. 

---

My research focus lies in 1)**Agentic AI**: How to embed the human's learning and decision-making patterns into (both virtual and embodied) agentic paradigm? How to enable intelligent agent to perform high-level, long-horizen planning and strategic thinking? 2)**Decision-Making in Machine Learning**: the algorithmic side of reinforcement learning (RL), as well as its intersection with modern generative models.


<mark style="background-color: yellow; color: black;">I am always open to collaborations, networking and potential intern opportunities, feel free to drop me an email! (vanzl3386@gmail.com)</mark>


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
  










  


  
<span class='anchor' id='Publication-List'></span>
# 📝 Selected Publications
<h1 id="publications"></h1>

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
@media (max-width: 640px) {
  .publications .pub-row { flex-direction: column; }
  .publications .pub-row .abbr.pub-thumb { max-width: 100%; flex-basis: auto; }
}
</style>

<div class="publications" markdown="1">
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

## Invention Patents
As these works are patented in China, all these names are directly translated from Chinese.
- **A Method, System, Terminal Device, and Storage Medium for Air Quality Spatial Inference** (Granted)
  
   Inventor: Jun Song, Yibo Xu, Yiwen Pan, Maohao Ran, **<u>Zhenglin Wan</u>**, Xiaoyun Yan, Yike Guo
- **A Single-UAV Atmospheric Pollutant Source Tracing Method Based on Gradient Ascent and Physical Kinematics** (Public)

   Inventor: **<u>Zhenglin Wan</u>**, Jun Song, Yibo Xu, Maohao Ran, Yike Guo

## White Paper
As these works are presented in China, these names are directly translated from Chinese.

- **White Paper on Cross-Border Economic Large Language Model**


<!--
<span class='anchor' id='academic-results'></span>
# 📝 Selected Academic Results

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">AAMAS 2025 Accepted (oral presentation)</div><img src='personal_page_sources/QD-IL/show.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Imitation From Diverse Behaviors](https://arxiv.org/abs/2411.06965)

Author: Xingrui Yu <sup>1</sup>, **Zhenglin Wan <sup>1</sup>**, David Bossens, Yueming Lyu, Qing Guo, Ivor Tsang (1:Co-first author &amp; Equal Contribution)

**Supervisor:** [Xingrui Yu](https://www.a-star.edu.sg/cfar/about-cfar/our-team/yu-xingrui), [David Bossens](https://www.a-star.edu.sg/cfar/about-cfar/our-team/david-bossens), [Ivor Tsang](https://www.a-star.edu.sg/cfar/about-cfar/management/prof-ivor-tsang).

**Host:** [Centre for Frontier AI Research](https://www.a-star.edu.sg/cfar), [A*STAR](https://www.a-star.edu.sg/), Singapore  

**Duration:** Jun. 2024 - Oct. 2024  


- Imitation learning (IL) approaches succeed in fast adaptation for dozens of applications. However, traditional IL assumes the demonstrations comes from one specific expert policy, thus containing only one behavior pattern. This way may result in the lack of robustness of agent to deal with various and unpredicatable real-world situations.
- To address above issue, this work pioneers a new paradigm- learning from diverse behaviors. We proposed a novel quality-diversity imitation learning model to enable the agent to learn a broad set of skills from limited but diverse demonstrations. With multiple viable and high-performing skills equipped, the agent can robustly deal with unseen and stochastic environment.

</div>
</div>

---

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">AAAI 2025 Accepted(oral presentation)</div><img src='personal_page_sources/POI/show.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

Next-POI Recommendation via Multi-Objective Adversarial Imitation Learning

**Supervisor:** [Pingfu Chao](https://www.researchgate.net/profile/Pingfu-Chao), [Xingrui Yu](https://www.a-star.edu.sg/cfar/about-cfar/our-team/yu-xingrui)  

**Duration:** Feb. 2024 - Aug. 2024  

**Author:** **Zhenglin Wan**, Anjun Gao, Xingrui Yu, Pingfu Chao*, Jun Song, Maohao Ran

- We proposed a multi-objective imitation learning architecture with variational inference to address the notorious data sparsity and data noise issue in POI data, and we utilize a novel adaptive graphs to capture spatial-temporal dependencies of POI sequences and user patterns.
- Demonstrated that the proposed architecture outperformed the current state-of-the-art by 8%, and by 31% in extremely sparse data scenarios.

</div>
</div>

---

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Journal Under Review</div><img src='personal_page_sources/GRAND/show.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

Graph-based Reinforcement learning Approach for influential Node Detection in airport delay networks

**Supervisor:** [Jianfeng Mao](https://sds.cuhk.edu.cn/en/teacher/268)  
**Host:** School of Data Science, CUHK (Shenzhen)  
**Duration:** Jun. 2023 - Aug. 2024  
**Author:** Chi Li, **Zhenglin Wan**, Kaize Wang, Yuxuan Huang, Chengxi Li, Jianfeng Mao*

- We developed a data-driven method combining graph representation learning with value-based reinforcement learning to identify key airports in delay networks, addressing the issue of low inference efficiency and cost-unaware management.
- Empirically, we achieved a 32% performance increase on the US airport delay dataset. Submitted a paper to *Transportation Research Part E: Logistics and Transportation Review*.

</div>
</div>

---

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Project</div><img src='personal_page_sources/Deep View/show2.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[DEEP VIEW: Mobile Inspection and Environmental Digital AI Supervision Service Project]()

**Project Consultant:** [Yike Guo](https://cse.hkust.edu.hk/admin/people/faculty/profile/yikeguo), Provost of HKUST  
**Supervisor:** [Jun Song](https://scholars.hkbu.edu.hk/en/persons/JUNSONG), HKBU  
**Duration:** Nov. 2023 - May. 2024  

- We developed an automated environmental monitoring system using LLM agents and drones for comprehensive air quality monitoring.
- We integrated multi-source data fusion, drone scheduling, and deep learning-based analysis into a unified system for real-time environmental oversight.

</div>
</div>

---

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Journal Under Review</div><img src='personal_page_sources/AlphaAir/show.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[LLM-Based Air Quality Analysis and Decision-Making Agent]()

**Supervisor:** [Jun Song](https://scholars.hkbu.edu.hk/en/persons/JUNSONG), HKBU  
**Duration:** Nov. 2023 - Jan. 2024  
**Author:** Jun Song*, Chendong Ma, Maohao Ran, **Zhenglin Wan**

- We integrated air quality assessment into tools for an LLM-based agent using LangChain, enabling real-time interaction for efficient data analysis.
- Submitted to *Science Advances*.

</div>
</div>

---

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Journal of Geophysical Research - Atmospheres</div><img src='personal_page_sources/NetGBM/show.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Machine Learning-Driven Spatiotemporal Analysis of Ozone Exposure and Health Risks in China](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024JD041593)

**Institution:** Metasequoia Intelligence  
**Duration:** Feb. 2023 - May. 2023  

</div>
</div>

---

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Australian Database Conference (Best Paper Runner-Up)</div><img src='personal_page_sources/ADC/figure.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Hierarchical Spatial-Temporal Graph-Enhanced Model for Map-Matching](https://link.springer.com/chapter/10.1007/978-981-96-1242-0_4)

Author: Anjun Gao*, Zhenglin Wan*, Pingfu Chao*, Shunyu Yao (*: co-first author &amp; equal contribution)

**Institution:** Soochow University, Metasequoia Intelligence 

**Duration:** Jun. 2024 - Oct. 2024



</div>
</div>
-->
---


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
.edu-img { flex: 0 0 240px; max-width: 240px; }
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
        <li>Major GPA: 3.85/4.0, Rank: 7%</li>
        <li>Finished my undergraduate in School of Data Science, CUHK-Shenzhen campus.</li>
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
        <li>Advisor: Prof. Ivor Tsang</li>
      </ul>
      <p class="exp-time">2024.07 - 2024.10</p>
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


<span class='anchor' id='press-media'></span>


# 💬 Press/Media
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">White Paper</div><img src='personal_page_sources/white_paper.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">


- The co-author of the first White Paper on Cross-Border Economic Large Language Model in Shenzhen, China.
[深圳卫视：深圳发布首个跨境经济大模型白皮书](https://www.sztv.com.cn/ysz/dsdb/szws/ssxw/80141762.shtml)

</div>
</div>


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
<!--
<div>Several years ago, I was once diagnosised with Obsessive-Compulsive Disorder **(OCD, 强迫症 in Chinese)**, specifically Obsessive-Thinking, a mental illness as painful as depression. My mind would get stuck on meaningless details, both academic and personal, generating unrelentless ideas or questions and I couldn’t control it. Realizing that medicine doesn't work for me, I have to rescue myself through the wisdom in mind-philosophy books and exploring Buddhist and Taoist philosophies. The most inspiring part of this journey was learning to practice mindfulness ("正念" in Chinese）—observing what was happening inside my mind **without judging it** and finding inner peace, allowing my brain to become my tool rather than being controlled by it. This experience became a valuable treasure, shaping my attitude towards life and challenges afterwards. I'd like to recommend you the book *The Power of Now*, along with Buddhist classics like The Diamond Sutra and Taoist texts like Tao Te Ching and Zhuangzi, which are beneficial to all of us, especially who are facing similar difficulties.
</div>
<p></p>
-->




