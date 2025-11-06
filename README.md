# Airbnb NYC Data Analysis: AI-Assisted Workflow

---

## Context

Product teams often use insights from product analytics to inform product strategy.  This project explores how AI can be applied to improve the quality of insights generated from standard data analysis workflows. The goal is to evaluate how different AI-assisted approaches improve the data analysis performed on an existing dataset. We aim to identify how AI can uncover overlooked patterns, deepen analytical rigor, and produce more actionable insights with minimal human context.

You can find more details on this project [here](https://www.priankr.com/post/airbnb-nyc-data-analysis).

### Overview

This project evaluates three distinct AI-assisted methodologies for improving data analysis insights under low-context conditions—scenarios where the AI receives minimal guidance about analytical goals or methods. The objective is to identify reproducible workflows that data analysts and stakeholders can leverage to extract deeper, more actionable insights from similar datasets.

### Core Question

**How can AI improve the quality of insights generated from standard data analysis workflows?**

The project compares three approaches:

- **Enhanced Insights**: AI reviews existing analysis to uncover overlooked patterns
- **Expanded Analysis**: AI builds upon existing work with advanced techniques
- **Independent Analysis**: AI conducts fresh analysis without prior influence

---

## Dataset Information

**Source**: [Kaggle - New York City Airbnb Open Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)

**Description**: The dataset captures Airbnb listing activity and metrics in New York City for 2019, providing comprehensive information about hosts, geographical distribution, pricing, availability, and review patterns.

---

## Project Files

### Input Files

| File | Description |
| --- | --- |
| `AB_NYC_2019.csv` | Raw dataset containing Airbnb listing activity and metrics for NYC in 2019, including host information, geographical data, pricing, availability, and review metrics |
| `airbnb_nyc_data_analysis.ipynb` | Original Jupyter Notebook with baseline Python-based data analysis, including methodology, visualizations, and initial insights |
| `airbnb_analysis_overview.md` | Summary of the baseline data analysis results and key findings |

### Analysis Prompts

| File | Description |
| --- | --- |
| `airbnb_data_analysis_prompts_v1.txt` | Initial and optimized prompts for **Approach 1** (Enhanced Insights), including standardization prompt |
| `airbnb_data_analysis_prompts_v2.txt` | Initial and optimized prompts for **Approach 2** (Expanded Analysis), including standardization prompt |
| `airbnb_data_analysis_prompts_v3.txt` | Initial and optimized prompts for **Approach 3** (Independent Analysis), including standardization prompt |
| `airbnb_data_analysis_comparison_prompt.txt` | Initial and optimized prompts for comparing outputs across all three approaches |

### AI-Generated Data Analysis

| File | Description |
| --- | --- |
| `airbnb_nyc_data_analysis_v1.ipynb` | **Approach 1**: Enhanced Insights - Jupyter Notebook with AI-assisted analysis that reviews and extends the original analysis |
| `airbnb_nyc_data_analysis_v2.ipynb` | **Approach 2**: Expanded Analysis - Jupyter Notebook with AI-generated advanced analytical techniques building upon existing work |
| `airbnb_nyc_data_analysis_v3.ipynb` | **Approach 3**: Independent Analysis - Jupyter Notebook with AI-conducted fresh analysis from scratch |

### Generated Outputs

| File | Description |
| --- | --- |
| `airbnb_analysis_overview_v1.md` | **Approach 1 output**: Enhanced insights derived from reviewing the original analysis |
| `airbnb_analysis_overview_v2.md` | **Approach 2 output**: Expanded analysis incorporating advanced analytical techniques |
| `airbnb_analysis_overview_v3.md` | **Approach 3 output**: Independent analysis conducted from scratch |
| `airbnb_data_analysis_comparison.md` | Comparative analysis evaluating the quality, depth, and uniqueness of insights across all three approaches |

---

## Workflow

You can find a detailed version of this workflow [here](https://www.priankr.com/post/airbnb-nyc-data-analysis). This is just an overview of the process.
This project implements a systematic workflow to evaluate three AI-assisted data analysis approaches. Each approach produces unique insights that are then standardized and compared to identify optimal use cases.

1. **Enhanced Insights Analysis:** Generate new insights by having AI review the existing analysis without modifying the underlying methodology.
2. **Expanded Analysis:** Build upon the existing analysis using advanced analytical techniques suggested by AI. 
3. **Independent Analysis:** Conduct fresh analysis from scratch without influence from prior work.
4. **Comparative Analysis:** Evaluate the quality, depth, and uniqueness of insights across all three approaches.
5. **Identify Optimal Use Cases**: Determine which approach works best for different analytical scenarios

### Key Findings

**Approach 1 (Enhanced Insights)** translates descriptive findings into actionable product recommendations, making it ideal for stakeholder communication and operational planning.

**Approach 2 (Expanded Analysis)** delivers the strongest balance of statistical rigor and business relevance through advanced methods like regression and clustering, making it optimal for validation and experiment design.

**Approach 3 (Independent Analysis)** uncovers the most original insights—particularly around host concentration and professional host behavior—making it valuable for exploratory discovery, though it benefits from more explicit analytical direction.

---

## Related Resources

- [Airbnb NYC Listings Data Analysis](https://www.kaggle.com/priankravichandar/airbnb-nyc-listings-data-analysis) — Kaggle Notebook
- [Airbnb NYC Data Analysis](https://www.priankr.com/post/airbnb-nyc-data-analysis) — Creating an enhanced AI-assisted workflow to conduct data analysis on Airbnbs in NYC.
- [Meta Prompting Guide](https://www.priankr.com/post/how-to-optimize-prompts-guide-to-meta-prompting) — Learn how to optimize prompts using meta-prompting techniques

---
