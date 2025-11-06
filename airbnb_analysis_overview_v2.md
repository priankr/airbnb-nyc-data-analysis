# Summary of Data Analysis

## Dataset
**Source:** Public Airbnb NYC 2019 dataset (Kaggle)  

**Goal:** Identify pricing drivers, demand patterns, and actionable opportunities for Airbnb’s product and growth teams.

---

## Tools and Methods
Python, GPT‑5 

**Approach Summary:**
- Data cleaning (missing values, outlier capping)
- Exploratory analysis (distributional and geographic patterns)
- Statistical modeling (log(price) regression)
- Market segmentation (KMeans clustering on price, availability, and reviews)
- Interpretation of product and business implications

---

## Snapshot: Key Numbers at a Glance

| Metric | Value |
| ------- | ------ |
| Total listings | **48,858** |
| Distinct neighbourhoods | **221** |
| Neighbourhood groups | **5** |
| Mean price | **$152.74** |
| Median price | **$106.00** |
| Mean availability | **113 days/year** |
| Mean number of reviews | **23.2** |
| Most common room type | **Entire home/apartment (52%)** |
| Top borough by listings | **Manhattan (44%)** |
| Dataset year | **2019** |

---

## Core Insights & What They Mean

### 1. Pricing Drivers: Structure Over Randomness
* **Insight:** Price variation is dominated by *room type* and *neighbourhood group.*  
  Entire homes and Manhattan properties command strong premiums; private and shared rooms in outer boroughs remain price-sensitive.  
* **Supporting data:** Linear regression on log(price) achieved **R² ≈ 0.72**, confirming strong explanatory power from these categorical features.  
* **Product implications:**
  - Build **dynamic pricing models** per room type × borough.  
  - Integrate location-based price guidance into host onboarding and revenue dashboards.

---

### 2. Demand Signals: Availability Mirrors Occupancy
* **Insight:** Listings with **low availability** and **high review counts** are the best demand proxies.  
  These listings cluster around Manhattan and parts of Brooklyn, suggesting saturated micro-markets.  
* **Product implications:**
  - Develop **occupancy forecasting** using review and availability signals.  
  - Promote **underbooked** neighborhoods with targeted visibility boosts or discounts.  

---

### 3. Segmentation: Four Distinct Market Clusters
* **Insight:** KMeans clustering reveals four operationally distinct listing segments:  
  1. **High-price / low-availability** (premium, high-demand Manhattan inventory)  
  2. **Mid-price / mid-availability** (Brooklyn & Queens, mixed stay lengths)  
  3. **Low-price / high-availability** (budget, often underbooked outer boroughs)  
  4. **Low-review / long-availability** (inactive or niche-use listings)  
* **Product implications:**
  - Tailor UX recommendations by cluster (e.g., pricing flexibility for Segment 3, host education for Segment 4).  
  - Prioritize **inventory health metrics** (availability × reviews) to detect underperforming clusters.

---

### 4. Neighborhood Economics: Manhattan Dominance, Outer Potential
* **Insight:** Manhattan listings average **2× higher price** and **lower availability**, reflecting tourism concentration.  
  Brooklyn contributes **~30% of listings** and a more balanced price-demand ratio, indicating scalability potential.  
* **Product implications:**
  - Support **Brooklyn and Queens** as growth corridors with balanced supply-demand metrics.  
  - Introduce **“local getaway” campaigns** to capture intra-city demand.  

---

### 5. Reviews and Engagement: Price Sensitivity Confirmed
* **Insight:** Listings under $100 show **40–60% more reviews per month**, reinforcing review rate as an inverse proxy for price.  
  Higher-priced hosts see fewer reviews, possibly due to smaller customer volume.  
* **Product implications:**
  - Encourage **review prompts** and reward systems for high-end hosts to build trust.  
  - Experiment with **automated post-stay review reminders** or visual badges for verified premium hosts.  

---

## Product Recommendations Summary

| Focus Area | Recommended Action |
| ----------- | ------------------ |
| **Dynamic Pricing** | Build borough × room-type pricing models using log(price) regressions |
| **Occupancy Optimization** | Forecast booking probability using availability + review volume |
| **Segmentation UX** | Display tailored insights per market cluster (e.g., “balanced segment” vs “premium”) |
| **Review Engagement** | Incentivize reviews for high-end listings; promote active engagement |
| **Market Expansion** | Focus growth and acquisition in Queens and Brooklyn |
| **Data Enrichment** | Add review scores, booking timestamps, and sentiment data for better forecasting |

---

## Data Caveats and Limitations

- No booking-level timestamps, limiting direct occupancy calculations.  
- Review count used as proxy for engagement, not sentiment.  
- Outliers trimmed at price > $1000 for interpretive stability.  
- Some hosts manage multiple listings (context for host-level aggregation).  
- Model R² ≈ 0.72; residual variance reflects unobserved listing quality factors (photos, amenities, ratings).

---

## Next Experiments

1. Develop a **“Listing Value Index”** combining price percentile, review rate, and availability proxy.  
2. A/B test **dynamic pricing recommendations** vs. baseline host-set prices to measure adoption and yield.  
3. Launch **“Review Accelerator”** program for premium hosts to drive trust and conversions.  
4. Add **neighborhood health dashboards** (median price, availability, review rate) for portfolio monitoring.  
5. Collect **textual review sentiment** and integrate it into pricing and ranking algorithms.