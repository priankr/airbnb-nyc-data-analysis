# Summary of Data Analysis

## Dataset
**Analysis of a public Airbnb dataset from Kaggle**: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

The dataset describes the Airbnb listing activity and metrics in NYC, NY for 2019. It includes all the needed information to find out more about hosts, geographical availability, and necessary metrics to make predictions and draw conclusions.


## Tools
Python, GPT-5

## Snapshot: Key Numbers at a Glance

| Metric                            | Value                                                 |
| --------------------------------- | ----------------------------------------------------- |
| Total listings                    | **48,858**                                            |
| Average price                     | **$152.74**                                           |
| Average availability              | **113 days/year**                                     |
| Average stay length               | **7 days**                                            |
| Top neighbourhood                 | **Williamsburg (3,917 listings)**                     |
| Neighborhoods covered             | **221**                                               |
| Listings in Manhattan             | **21,643 (highest)**                                  |
| Room type mix                     | **Majority Entire home/apartments and Private rooms** |
| Top 10 neighborhoods              | Represent **47.9%** of all listings                   |
| Avg. distance to major attraction | **3.1 miles**                                         |

---

## Core Insights & What They Mean

### 1. Occupancy and Seasonality

* **Insight:** Availability patterns indicate occupancy differences. Low availability = high occupancy.
  Formula: `occupancy_est = (365 - availability) / 365`.
* **Pattern:** Highest activity in **June**, lowest in **February** → strong seasonal travel cycle.
* **Why it matters:** Informs marketing timing, dynamic pricing, and minimum-stay adjustments.
* **Product actions:**

  * Launch **off-season campaigns** (February) with lower minimum stays.
  * Use **season-based pricing multipliers** to smooth demand curves.

---

### 2. Location and Proximity Insights

* **Insight:** Average Airbnb is **3.1 miles** from a major attraction.

  * **Bronx** listings are *closest* to attractions but few in number → opportunity to expand supply.
  * **Queens** listings have *low availability* but strong location access → optimize listing visibility.
  * **Manhattan** listings are both *numerous* and *centrally located*, reinforcing tourism dominance.
* **Product actions:**

  * Factor **distance-to-attraction** into search ranking and pricing algorithms.
  * Pilot **“Near Major Attractions”** tag to promote listings within 2–3 miles.

---

### 3. Manhattan’s Split Market

* **Insight:** Manhattan’s price spread is largest across all boroughs — a **multimodal market**.

  * Includes both high-end entire apartments and budget shared rooms.
* **Why it matters:** A single pricing model will misrepresent real value.
* **Product actions:**

  * Create **micro-market pricing models** (by neighborhood + room type).
  * Incorporate **percentile-based** pricing indicators (P25–P75).
  * Highlight **Entire home/apts** as premium inventory for dynamic pricing.

---

### 4. Reviews Reflect Price Sensitivity

* **Insight:** Lower-priced listings receive more reviews, implying higher occupancy or engagement.

  * **Staten Island:** few listings but *most reviews per listing*.
  * **Manhattan:** many listings but *fewer reviews per listing* due to higher price points.
* **Product actions:**

  * Introduce **review boost incentives** for high-end hosts (timed post-stay messages or loyalty points).
  * Encourage **photo-rich reviews** to build trust where review count is low.

---

### 5. Stay Duration and Listing Type

* **Insight:**

  * Average stay = **7 days** across all listings.
  * **Spuyten Duyvil** shows unusually long average stays (**48 days**) → potential long-term rentals.
  * **Manhattan** averages **9-day stays**, aligning with tourism.
* **Why it matters:** Two distinct stay types require differentiated UX and policies.
* **Product actions:**

  * Add **“Monthly Stay” mode** in search and host setup flow.
  * Adjust **fees and cancellation rules** for long-term bookings.

---

### 6. Supply Concentration and Market Risk

* **Insight:** 48% of all listings come from just 10 neighborhoods → geographic overdependence.
* **Why it matters:** High exposure to localized regulation or demand shocks.
* **Product actions:**

  * Build a **Supply Diversification Dashboard** tracking inventory distribution.
  * Flag neighborhoods crossing **50% share** for monitoring and growth balancing.

---

## Product Recommendations Summary

| Focus Area          | Recommended Action                                  |
| ------------------- | --------------------------------------------------- |
| **Pricing**         | Neighborhood + room-type models using percentiles   |
| **Reviews**         | Incentive program for premium listings              |
| **Inventory**       | Different UX for short vs. long stays               |
| **Geographic Risk** | Diversification KPI and alerts                      |
| **Seasonality**     | Off-season discounting and flexible stays           |
| **Proximity**       | “Near Attractions” tag and map ranking factor       |
| **Data Expansion**  | Booking-level and host response data for validation |

---

## Data Caveats

* Missing booking-level and review-score data means reliance on proxies (availability, reviews).
* Small neighborhoods can skew averages — use medians and IQR for insight stability.
* Some long-stay listings may represent leases; verify using `minimum_nights` field.

<b>Additional Data necessary</b><br>
The data only tells us if a review was left or not for any given listing. It would be beneficial to know what score each listing received when they were reviewed. We can only go off the number of reviews listings receive and assume listings (and by extension neighbourhoods and neighbourhood groups) with more reviews are preferable.

---

## Next Experiments

1. Correlate **price percentile × occupancy proxy × review rate** to create a “Value Index.”
2. A/B test **review incentives** for high-priced listings and measure review volume lift.
3. Build **neighborhood health dashboard** (median price, occupancy proxy, review rate, availability).
4. Once booking data is available, train **host churn prediction models**.

---

