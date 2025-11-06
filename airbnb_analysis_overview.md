## Summary of Data Analysis

### Dataset
**Analysis of a public Airbnb dataset from Kaggle**: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data


The dataset describes the Airbnb listing activity and metrics in NYC, NY for 2019. It includes all the needed information to find out more about hosts, geographical availability, and necessary metrics to make predictions and draw conclusions.


## Tools
Python


## Insights

### Availability
- There are 48858 listings in total.
- The majority of the listings are Entire homes/apts or Private rooms.
- On average, any given listing is available 113 days a year.
    - Listings in Staten Island have the greatest availability and receive the most reviews per listing.
    - However, Staten Island also receives the least number of reviews overall.
    - Listings in Manhattan have the least availability and receive the least reviews per listing.
    - However, Manhattan receives the second-highest number of reviews overall.
- The average duration of stay for all listings is 7 days.
    - Listings in the Spuyten Duyvil neighborhood offer the longest average duration of stay at approximately 48 days.
    - Listings in the Manhattan neighborhood group offer the longest average duration of stay at approximately 9 days.

### Location
- There are 221 neighbourhoods with Williamsburg having the most listings (3917).
- The top 10 neighborhoods represent about 47.95% of all listings.
- There are 5 neighbourhood groups with Manhattan having the most listings (21643).
- On average, any given listing is 3.1 miles from the closest major attraction.
    - Listings in the Bronx are the closest major attraction in the city.
    - However, the Bronx also has the second lowest number of listings.
    - Listings in Queens have the least availability and receive the least reviews per listing.
    - Manhattan has the greatest number of listings and they are second closest to a major attraction in the city.

### Price
- Average Price Across all listings: 152.74
- There are 55 neighborhoods with average listing prices above the average for all listings.
- There are 166 neighborhoods with average listing prices below the average for all listings.
- The largest standard deviation in price is in Manhattan.
- The spread of prices is greatest in Manhattan.
- As expected, listings with Entire home/apt are the most expensive.

### Number of Reviews
- Across all categories (Room Type, Neighbourhood, etc.), less expensive Listings receive more reviews.
- The majority of reviews are left in June which indicates that the majority of customers used a rental in June. Meanwhile, the least reviews are left in February, which indicates that the fewest customers used a rental in February.
- Staten Island has the largest number of reviews as compared to the actual number of listings, which indicates that reviews were left more frequently for stays in listings that were within the Staten Island neighborhood group.
- Manhattan has the second largest number of listings but has the least number of reviews compared to the actual number of listings, which indicates that reviews are left less frequently for stays in the Manhattan neighborhood group. The possible reasons for this are as follows:
    - The average listing price is also the highest of all neighborhood groups.
    - Manhattan’s average listing price is also above the average for all listings.


## Additional Data Necessary
The data only tells us if a review was left or not for any given listing. It would be beneficial to know what score each listing received when they were reviewed. We can only go off the number of reviews listings receive and assume listings (and by extension neighborhoods and neighborhood groups) with more reviews are preferable.