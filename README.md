<div align="center">
	<h1>UrBench: A Comprehensive Benchmark for Evaluating Large Multimodal Models in Multi-View Urban Scenarios</h1>
	<a href="https://arxiv.org/abs/2408.01812"><img src='https://img.shields.io/badge/arXiv-2408.01812-red?logo=arXiv' alt='arXiv'></a>
	<a href="https://longhz140516.github.io/SkyDiffusion.github.io/"><img src='https://img.shields.io/badge/Project-SkyDiffusion-green' alt='Project'></a>
	<a href=""><img src='https://img.shields.io/badge/python-3.9-blue.svg' alt='Python'></a>
	<a href=""><img src='https://img.shields.io/badge/License-Apache%202.0-yellow' alt='Python'></a>
</div>


<img src="./static/images/fig2_overview_2.jpg" alt="">

## Abstract
Recent evaluations of Large Multimodal Models (LMMs) have explored their capabilities in various domains, with only few benchmarks specifically focusing on urban environments. 
Moreover, existing urban benchmarks have been limited to evaluating LMMs with basic region-level urban tasks under singular views, leading to incomplete evaluations of LMMs' abilities in urban environments. 
To address these issues, we present UrBench, a comprehensive benchmark designed for evaluating LMMs in complex multi-view urban scenarios. UrBench contains 11.6K meticulously curated questions at both region-level and role-level that cover 4 task dimensions: Geo-Localization, Scene Reasoning, Scene Understanding, and Object Understanding, totaling 14 task types. 
In constructing UrBench, we utilize data from existing datasets and additionally collect data from 11 cities, creating new annotations using a cross-view detection-matching method. With these images and annotations, we then integrate LMM-based, rule-based, and human-based methods to construct large-scale high-quality questions. 
Our evaluations on 21 LMMs show that current LMMs struggle in the urban environments in several aspects. 
Even the best performing GPT-4o lags behind humans in most tasks, ranging from simple tasks such as counting to complex tasks such as orientation, localization and object attribute recognition, with an average performance gap of 17.4%. 
Our benchmark also reveals that LMMs exhibit inconsistent behaviors with different urban views, especially with respect to understanding cross-view relations. UrBench datasets and benchmark results will be publicly available.