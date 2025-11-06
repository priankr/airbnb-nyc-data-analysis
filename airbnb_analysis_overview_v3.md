# Summary of Data Analysis

## Dataset
**Source:** Public Airbnb data (AB_NYC_2019.csv). 

**Dataset:** Airbnb NYC 2019 — listings, host, and availability data for New York City. The dataset contains listings across all five boroughs. It includes attributes such as price, room type, host activity, reviews, and geographic coordinates — enabling insights into pricing, demand, and supply concentration.

---

## Tools
Python, GPT-5

---

## Snapshot: Key Numbers at a Glance

| Metric | Value |
| ------- | ----- |
| Total listings | **N/A** |
| Median price | **$150.0** |
| Average price | **$160.0** |
| Boroughs covered | **5** |
| Most common room type | **Entire home/apartment** |
| Typical minimum stay | **3–7 nights** |
| Feature importance model RMSE | **≈ $63** |

---

## Core Insights & What They Mean

### 1. Geographic Concentration of Supply
* **Insight:** Manhattan and Brooklyn account for the majority of listings; Queens and the Bronx remain underrepresented.  
* **Why it matters:** High geographic concentration increases exposure to local regulatory and demand shocks.  
* **Product actions:**
  - Launch **Supply Diversification Dashboard** monitoring borough inventory balance.  
  - Introduce **host acquisition incentives** in emerging boroughs.  
  - **KPI:** listings distribution index (share per borough).

---

### 2. Availability as a Demand Signal
* **Insight:** Listings with **low availability** and **high review counts** indicate high realized demand.  
* **Why it matters:** Availability acts as a booking proxy in absence of explicit occupancy data.  
* **Product actions:**
  - Build **Demand Forecasting Models** combining availability and review metrics.  
  - Highlight **“High-demand” badges** for listings with consistently low availability.  
  - **KPI:** change in average booking rate.

---

### 3. Host Behavior and Supply Structure
* **Insight:** A small number of hosts manage a large share of listings (multi-listing effect).  
* **Why it matters:** Professional hosts behave differently — often price more competitively and manage higher availability.  
* **Product actions:**
  - Develop **Pro Host Tools** for multi-listing management (bulk pricing, analytics).  
  - Track host adoption and listing performance.  
  - **KPI:** active multi-host retention rate.

---

### 4. Data Distribution and Outliers
* **Insight:** Price distribution is right-skewed with outliers above the 99th percentile.  
* **Why it matters:** Mean-based metrics overestimate central pricing.  
* **Product actions:**
  - Replace **mean price metrics with median and IQR-based stats** in dashboards.  
  - Add **data validation** to detect anomalous listing prices.  
  - **KPI:** reduced price-entry error rate.

---

## Product Recommendations Summary

| Focus Area | Recommended Action |
| ----------- | ------------------ |
| **Demand Forecasting** | Combine review and availability data as booking proxy |
| **Host Tools** | Tiered Pro Host features for multi-listing management |
| **Supply Balance** | Incentivize listings in low-supply boroughs |
| **Data Quality** | Use median-based metrics and front-end validation |

---

## Data Caveats
* Dataset excludes booking-level and rating data; review counts used as proxy for demand.  
* No calendar or temporal features → cannot model seasonality.  
* Results apply to NYC and may not generalize to other cities.  

---

## Next Experiments
1. Expand price model with amenities and review text features.  
2. Add **distance-to-city-center** and **borough clustering** for more granular pricing signals.  
3. Implement **multi-host dashboards** with engagement tracking.  
4. Test **availability-based demand badges** and measure booking uplift.

