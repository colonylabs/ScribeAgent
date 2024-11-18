<div align="center">
  <h1>ScribeAgent: Towards Specialized Web Agents Using <br>Production-Scale Workflow Data</h1>
  <a href="https://arxiv.org/">
    <img src="https://img.shields.io/badge/arXiv-0000.xxxxx-b31b1b.svg" alt="arXiv">
  </a>
</div>

## Abstract

Large Language Model (LLM) agents are rapidly improving to handle increasingly complex web-based tasks. Most of these agents rely on general-purpose, proprietary models like GPT-4 and focus on designing better prompts to improve their planning abilities. However, general-purpose LLMs are not specifically trained to understand specialized web contexts such as HTML, and they often struggle with long-horizon planning. 
We explore an alternative approach that fine-tunes open-source LLMs using production-scale workflow data collected from over  250 domains
corresponding to 6 billion tokens. This simple yet effective approach shows substantial gains over prompting-based  agents on existing benchmarks - ScribeAgent achieves state-of-the-art performance on Mind2Web and substantially improves the baseline task success rate  from 37.2\% to 51.3\%   on WebArena.
We further  perform detailed ablation studies on various fine-tuning design choices and provide  insights into LLM selection, training recipes, context window  optimization, and effect of dataset sizes.
