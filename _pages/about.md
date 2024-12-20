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

<span class='anchor' id='about-me'></span>
# 😊 About me

Greetings! I am Zhenglin Wan, an undergraduate student from CUHK-Shenzhen, majoring in Data Science. You can find my CV at [CV](https://drive.google.com/file/d/1fYYlj0MhJGsPPDdWUMB9LHMTqUo3bL1s/view?usp=sharing). My office/working station in campus: 320 A, ChengDao building (also known as Shenzhen key Laboratory of loT Intelligent Systems and Wireless Network Technology).

During high school, I was fortunate to win the first prize in the Chinese High School Mathematics Olympics, during which I developed personalized paradigm of learning and the habit of independent study. This enables me to achieve nearly all A grades (4.0/4.0) in my major-related courses (statistics, mathematics, and computer science) in a self-directed way, which allowed me to save time and engage in challenging and interesting research projects. My primary research interests lie in: 1) **Decision-Making in Machine Learning**: How to embed the human's learning and decision-making patterns into machine's learning paradigm, such as Reinforcement Learning, Imitation Learning, Multi-agent systems? I am also interested in adopting statistical view to understand these problems.  2) **The application of decision-making support systems in urban context**: such as smart transportation systems, environment monitoring, etc.

---

I have been an intern researcher at [Centre for Frontier AI Research](https://www.a-star.edu.sg/cfar), Agency for Science, Technology and Research [(A*STAR)](https://www.a-star.edu.sg/) at Singapore, working on developing robust imitation learning algorithm that is capable to deal with unpredicatable emergencies and cater to diverse real-world needs. During my undergraduate, I am lucky to work with [Dr. Jun Song](https://scholars.hkbu.edu.hk/en/persons/JUNSONG) and co-found one company, as technical collaborator, named "Metasequoia Intelligence" with him. Our aim is to utilizing the power of LLM agents to integrate data querying, data process, data visualization and intelligent decision making, thus making AI contribute to the urban well-being. We have published serveral invention patents and academic papers together. Prior to that, I was the key member of the research group of Prof. [Jianfeng Mao](https://sds.cuhk.edu.cn/en/teacher/268), the Assistant Dean of School of Data Science of CUHK-Shenzhen, playing a key role in a project regarding decision-making systems in transportation network.

I am actively seeking interesting research opportunities and Ph.D. positions starting in Fall 2025. If you are looking for a dedicated and self-motivated candidate, please DO contact me at vanzl3386@gmail.com or carlos@metaseq.ai !

<span class='anchor' id='news'></span>
# 🔥 News
- *2024.12*  &nbsp;🎉🎉 My co-first-author paper *Imitation From Diverse Behaviors: Wasserstein Quality Diversity Imitation Learning with Single-Step Archive Exploration* is accepted (oral presentation) by *The 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)*.
- *2024.12*  &nbsp;🎉🎉 My first-author paper *Next-POI Recommendation via Multi-Objective Adversarial Imitation Learning* is accepted by *The 39th Annual AAAI Conference on Artificial Intelligence (AAAI 2025)*.
- *2024.10*  &nbsp;🎉🎉 My co-first-author paper *Hierarchical Spatial-Temporal Graph-Enhanced Model for Map-Matching* is accepted by *Australasian Database Conference 2025*, to be hold at Tokyo on 16 Dec (oral presentation ＆ best paper runner-up), and published at [*Springer Nature*](https://link.springer.com/chapter/10.1007/978-981-96-1242-0_4).
- *2024.10*  &nbsp;🎉🎉 I have ended my memorable attachment at [Centre for Frontier AI Research](https://www.a-star.edu.sg/cfar), Agency for Science, Technology and Research [(A*STAR)](https://www.a-star.edu.sg/) at Singapore, serving as an intern researcher.
- *2024.09*  &nbsp;🎉🎉 Our paper [*Machine Learning-Driven Spatiotemporal Analysis of Ozone Exposure and Health Risks in China*](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024JD041593) is accepted by *Journal of Geophysical Research - Atmospheres*.
- *2024.09*  &nbsp;🎉🎉 My invention patent *A Single-UAV Atmospheric Pollutant Source Tracing Method Based on Gradient Ascent and Physical Kinematics* is published.
- *2024.05*  &nbsp;🎉🎉 My invention patent *A Method, System, Terminal Device, and Storage Medium for Air Quality Spatial Inference* is officially granted! 
- *2023.12*  &nbsp;🎉🎉 I co-founded one enterprise named "Metasequoia Intelligence" (in Chinese 水杉智算) located at FuTian, Shenzhen, China, with [Dr. Jun Song](https://scholars.hkbu.edu.hk/en/persons/JUNSONG) from HKBU, as technical collaborator.

<span class='anchor' id='academic-results'></span>
# 📝 Selected Academic Results

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Conference Under Review</div><img src='personal_page_sources/QD-IL/show.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Quality Diversity Imitation Learning](https://arxiv.org/abs/2410.06151)
Author: **Zhenglin Wan**, Xingrui Yu*, David Bossens, Yueming Lyu, Qing Guo, Flint Xiaofeng Fan, Ivor Tsang

**Supervisor:** [Xingrui Yu](https://www.a-star.edu.sg/cfar/about-cfar/our-team/yu-xingrui), [David Bossens](https://www.a-star.edu.sg/cfar/about-cfar/our-team/david-bossens), [Ivor Tsang](https://www.a-star.edu.sg/cfar/about-cfar/management/prof-ivor-tsang).
**Host:** [Centre for Frontier AI Research](https://www.a-star.edu.sg/cfar), [A*STAR](https://www.a-star.edu.sg/), Singapore  
**Duration:** Jun. 2024 - Oct. 2024  


- Imitation learning (IL) approaches succeed in fast adaptation for dozens of applications. However, traditional IL assumes the demonstrations comes from one specific expert policy, thus containing only one behavior pattern. This way may result in the lack of robustness of agent to deal with various and stochastic situations.
- To address above issue, this work pioneers a new paradigm- learning from diverse behaviors. We proposed a novel quality-diversity imitation learning model to enable the agent to learn a broad set of skills from limited but diverse demonstrations. With multiple skills equipped, the agent can robustly deal with unseen and stochastic environment.

</div>
</div>

---

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">AAAI 2025 Accepted</div><img src='personal_page_sources/POI/show.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

Next-POI Recommendation via Multi-Objective Adversarial Imitation Learning

**Supervisor:** [Pingfu Chao](https://www.researchgate.net/profile/Pingfu-Chao), [Xingrui Yu](https://www.a-star.edu.sg/cfar/about-cfar/our-team/yu-xingrui)  
**Host:** [Centre for Frontier AI Research](https://www.a-star.edu.sg/cfar), A*STAR, Singapore  
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

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Patent Pending</div><img src='' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Unmanned Aerial Vehicle Pollutant Tracking using Optimization Methods]()

**Institution:** Metasequoia Intelligence  
**Duration:** Nov. 2023 - Feb. 2024  

- Developed optimization algorithms for UAV-based pollutant tracking, demonstrating notable improvements in accuracy and efficiency.
- Conducted simulations validating the algorithms' effectiveness in real-world tracking, resulting in two invention patents.

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

Author: Anjun Gao†, Zhenglin Wan†, Pingfu Chao*, Shunyu Yao (†: co-first author)

**Institution:** Soochow University, Metasequoia Intelligence 

**Duration:** Jun. 2024 - Oct. 2024



</div>
</div>

---

<span class='anchor' id='honors-and-awards'></span>
# 🎖 Honors and Awards
- **Zhejiang Guolong Inspirational & Diligentia Bowen II Scholarship** (Outstanding Academic Performance)  
  *September 2021*

- **Yearly Dean List Award** (Outstanding Academic Performance)  
  *Consecutive three years: September 2022 - September 2024*

- **Yearly Academic Scholarship: C Class** (Year GPA Top 5%)  
  *Second Academic Year: September 2023*

- **Yearly Academic Scholarship: B Class** (Year GPA Top 3%)  
  *Third Academic Year: September 2024*

<span class='anchor' id='educations'></span>
# 📖 Educations
- *2021.09 - 2025.06 (now)*, The Chinese University of Hong Kong, Shenzhen 

<span class='anchor' id='press-media'></span>
# 💬 Press/Media
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">White Paper</div><img src='personal_page_sources/white_paper.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">


- The co-author of the first White Paper on Cross-Border Economic Large Language Model in Shenzhen, China.
[深圳卫视：深圳发布首个跨境经济大模型白皮书](https://www.sztv.com.cn/ysz/dsdb/szws/ssxw/80141762.shtml)

</div>
</div>

---
<span class='anchor' id='internships'></span>
# 💻 Internships
- *2023.07 - 2024.01*, Machine Learning Algorithm Engineerer, [HUIYINTONG]([https://github.com/](http://www.hytii.cn/index.html)), China.  
- *2024.07 - 2024.10*, Intern Researcher, Agency for Science, Technology and Research [A*STAR](https://www.a-star.edu.sg/), Singapore

<span class='anchor' id='person'></span>
# Personal Life
For the time I am not doing research, I am a **music** enthusiast: I have been playing guitar for ten years and began learning to play the piano and electronic piano independently when I was 15 years old. During my undergraduate years, I established a band named "Minor Blue" and performed multiple times inside and outside campus. I plan to learn playing drums during my spare time at Ph.D.

I am also a ten-year chess player, with national third-rank honor. I love the process of comprehensive planning, logical-thinking and inferring.




