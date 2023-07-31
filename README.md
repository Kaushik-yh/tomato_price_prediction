# SURVEY ON LSTM MODELS : CASE STUDY ON AGRI / STOCK DATA

## Introduction

  RNNs, a type of deep neural network, are specialized for processing sequential data and are highly suited for applications such as TS prediction, NLP, and speech recognition due to their unique structure. One key advantage of RNNs is the ability to share parameters between time steps, allowing for the capture of long-term dependencies in data. Unlike feed-forward networks, RNNs maintain an internal state that can capture dependencies over time. The LSTM network, the most popular type of RNN, was developed to address the issue of vanishing gradients that can occur in RNNs. Other RNN variations include the GRU (Gated Recurrent Unit), which has a simpler structure and requires less training time than LSTM, and the BiLSTM (Bidirectional LSTM).
  
## Data Collection and Exploratory Data Analysis
  This research project focuses on enhancing agricultural product trading through the use of LSTM models, with a specific focus on daily price data of agricultural commodities sold in the Indian market. The Government of India has launched a publicly accessible data portal called Data.gov.in, which serves as a central repository for storing all analytical and statistical data related to the country, including agriculture. The agricultural data available on this portal is used for this research project to analyze trends and patterns in the Indian agricultural sector. The data sets available on Data.gov.in cover various aspects of the agricultural sector, including crop production, land use, livestock and fisheries, agricultural inputs, and rainfall statistics, among others. 
  
  Most of the agricultural data sets available on Data.gov.in are time-series data, which cover multiple years, allowing users to analyze trends and patterns over time. The agricultural data sets cover different states and regions of India, providing users with information on the agricultural situation in various parts of the country. The data sets available on Data.gov.in are sourced from various government agencies and are subject to quality checks to ensure accuracy and consistency. They are also available in multiple formats, including CSV, XLS, and PDF, making them easy to download and use for analysis. This research project aims to study the impact of various factors, such as climate change, irrigation, and fertilizer use, on crop production and yield, and forecast the crop prices. The insights generated from this research can help farmers and agribusinesses make informed decisions about which crops to cultivate and how to optimize their yields.

  The Agricultural Product Market Committee (APMC) publishes information about the sales and volume of agricultural products traded each day to ensure fair pricing and prevent illegal activity. The government runs websites that provide citizens with up-to-date information on agriculture, including news, trends, and subsidies. Each state government has its own portal, which is monitored for reliability by government agencies. The information from all states is compiled on the national website Data.gov.in. We used a portal called "Krishi Marata Vahini" maintained by the Government of Karnataka, which provides daily and historical prices for major agricultural products in Karnataka and other states, as well as information on imports and exports. We used the quantity of goods sold/bought as an additional feature for our dataset, which helped us develop a multivariate time series forecasting dataset.

## Choosing the Commodity
  The above mentioned sources had a variety of dataset to choose from, allowing us to choose between the dataset to suite to our needs. To perform this study, we consider dataset related to 3 different crops based on several criteria, which are listed bellow:
  
  **Crop availability**: To conduct proper analysis, it is important to choose crops that are locally grown and widely available across the country. India produces crops such as Banana, Onion, Tomato, Wheat, Soy, and more, which can be analyzed to benefit the locals. It is best to avoid analyzing crops that are completely imported from other countries, as this involves non-quantifiable variables such as global trade policies and logistics that can disrupt prices, and there is no local reference to compare the prices.
  
  **Crop Seasonality**: India has three main farming seasons, which are called Kharif, Rabi, and Zaid. Kharif season is from June to October and includes crops that are grown during the monsoon season, such as rice and lentils. Rabi season is from November to April or May and includes crops like barley, wheat, and mustard. Zaid season is during the summer months, starting in March and ending in June, and includes crops such as pumpkin, cucumber, and bitter gourd. Any crop that is grown during these seasons or is grown throughout the year can be selected for analysis. 
  
  **Shelf life of the Crop**: Since some of the crops are seasonal in nature, and can be cultivated only once a year, it becomes very important for traders and whole-sellers to store the produce and make it available to the customers during the off seasons. Such storage will influence the price of the crop, since there is imbalance in demand and supply during the off seasons, the crops include wheat, rice, cereals, pulses and many more which have the shelf life of a complete year. whereas the fruits like Mango and watermelon are summer crops, and are perishable in short amount of time, such crops will again have an impact on the supply and demand, thus making the predictions more challenging. Depending on the above listed features, I have identified crops which depict unique characteristics in terms of Crop availability, seasonality and shelf life of the Crop.
  
**Tomato**: The other dataset chosen for analysis is the "Tomato" crop. India is know to be the 2 largest producer of tomato in the world, and the season prevails throughout the year with production peaking at the beginning and end of every year. This is the third most consumed crop in India falling behind Potato and Onion. Some of the major tomato producing states in India are Tamil Nadu, Andhra Pradesh, Karnataka, Madhya Pradesh, Gujarat, Odisha, West Bengal, Bihar, Telangana, Uttar Pradesh, Maharashtra, Chattisgarh, Haryana, and Himachal Pradesh with production combining to roughly 193.97 Lakh Tons in year 2018-19 and is estimated to grow year-on-year. Tomatoes require a warm environment to develop, they are planted in month of March and April and harvested in late summer with a period of 70 days to grow. Because of the hot and humid climatic conditions in southern states of India, tomato has become a year round crop with major producing states are Tamil Nadu, Andhra Pradesh and Karnataka. For our study, we have considered the trade transactions that took place in the Kolar, a largest tomato market in Karnataka. The trade details are considered for a period of 4 years starting from January 2020 to March 2023. The dataset includes "Min Price", the "Max Price" the "Modal price" and the "Quantity sold" in Quintal. Restricting the dataset to a single local market will help us predict the market trends in a particular region that would give incites to the local framers if it’s profitable to produce this commodity and when would it be beneficial to sow the seeds to time the harvestright, and maximise the profit.

## Data Visualization and Prepossessing
The dataset for tomato was collected from Data.gov.in for the year 2020 to 2023, the data collected had frequency of one day with the trade details of all the APMC markets in India. The dataset contained features "state", "district", "market", "commodity", "variety", "arrival date", "min price", "max price" and "modal price". Since the focus was on to identifying the price trend of the tomato for a region of Kolar, we must filter the dataset accordingly. The State is Karnataka, District is Kolar, the Market is Kolar, the Commodity is Tomato (since we are considering only 1 crop), the variety is also Tomato, because the source from which the dataset was retrieved combined all the verity into single class. The Quantity was picked from the "Krishi Marata Vahini" website hosted by the Karnataka state Agricultural Ministry. Bellow is the representation Min price, Max price and Modal Price of daily data.
![tomato_max_min_mode](https://github.com/Kaushik-yh/tomato_price_prediction/assets/138836652/39dc7d34-3133-44e2-940c-0144abcd4c35)
Figure: Distribution of Min, Max and Modal Price in Tomato Dataset

## Models Comparison and Results
  After a through analysis, we observe the the LSTM models have a great potential on predicting the near future values based on the behaviour of dataset observed in the near past and and distant past. We have narrowed down the search to 2 best models that not just provide good results for the immediate prediction, but was also capable of producing better predictions for distant futures also.

**Bi-Directional LSTM - Multivariate input - Single day input**
| Error Function | Stacked Bi-Directional LSTM  | After Hyperparameter tuning  |
| -------------- | ---------------------------- | ---------------------------- |
|  MAPE for t+1  |         0.107763             |            0.107543          |
|  MAPE for t+2  |         0.14615              |            0.144895          |
|  MAPE for t+3  |         0.174743             |            0.16947           |
|  MAPE for t+4  |         0.189884             |            0.178648          |
|  MAPE for t+5  |         0.202764             |            0.18765           |

The prediction vs Actual can be analysed from the bellow graph

![tomato_predictions](https://github.com/Kaushik-yh/tomato_price_prediction/assets/138836652/547dbf97-c687-440a-b963-c3f905b89a5b)
Figure: Predicted vs Actual for time t+1

![Tomato_7a_Pred_t1_To_t5](https://github.com/Kaushik-yh/tomato_price_prediction/assets/138836652/89227d81-df5e-4efd-8616-056056d9de31)
Figure: Predicted vs Actual last test value t+1 to t+5

**Bi-Directional LSTM - Multivariate input - 30 days lookback input**
| Error Function | Stacked Bi-Directional LSTM  | After Hyperparameter tuning  |
| -------------- | ---------------------------- | ---------------------------- |
|  MAPE for t+1  |         0.133212             |            0.143644          |
|  MAPE for t+2  |         0.170854             |            0.167463          |
|  MAPE for t+3  |         0.193271             |            0.187054          |
|  MAPE for t+4  |         0.215227             |            0.200886          |
|  MAPE for t+5  |         0.228227             |            0.21111           |

![Tomato_7c_PredVsAct](https://github.com/Kaushik-yh/tomato_price_prediction/assets/138836652/8b43f6c9-bc37-47a7-8581-2763e7b94618)
Figure: Predicted vs Actual for time t+1

![Tomato_7c_Pred_t1_To_t5](https://github.com/Kaushik-yh/tomato_price_prediction/assets/138836652/1e858b50-e235-4102-95d4-c2ced38a4740)
Figure : Predicted vs Actual for last test value t+1 to t+5
