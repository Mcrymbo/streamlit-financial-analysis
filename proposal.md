# PREDICTING TRADE VOLUMES IN SUB-SAHARAN AFRICA: A MACHINE LEARNING AND SIMULATION-BASED APPROACH

# Abstract

This study will explore the macroeconomic effects on trade volume for select countries in Sub-Saharan Africa (SSA) by using machine learning techniques. With past study, traditional machine learning techniques such as regression and time-series were used, which failed to capture the non-linear and dynamic relationship between variables, hence the need for machine learning. We will utilise secondary data collected from the World Bank and the International Monetary Fund websites for the period 1970-2024. The study will utilise a CRISP-DM methodology. For this study, Random Forests, XGBoost, and Long Short-Term Memory (LSTM) networks will be used to predict trade volumes and identify feature variables that affect trade in the given countries. Aside from predicting trade volumes, clustering algorithms such as K-Means and DBScan will be used to group the countries into their different economic categories. Monte Carlo simulations will assess how shocks in the macroeconomic variables influence trade volumes and cause structural shifts among economic clusters. Lastly, explainable tools, SHAP and LIME will be used to enhance the interpretability of the models that makes it easy for policymakers to understand the results of the macroeconomic drivers of trade in the region. The expected results are predicted future trade volumes, and clustered countries into their different economic status. Countries that are vulnerable to macroeconomic and external shocks will also be identified. The findings will be made in an interactive web based dashboard that illustrates the effects of these variables on trade and the effects of shocks. Overall, this study will use machine learning in international economics to provide a novel framework and data driven insights for economic policymaking.

---

# Chapter 1: Introduction

## 1.1 Background

International trade is the exchange of goods, services and capital across international borders. The exchanges take place in three ways, exports, imports and entrepot trade. Exports are when products are exchanged outside the country. Import trade is when goods and services are exchanged into the country, while entrepot trade involves importing goods into a territory then exporting them later without any local distribution (42).

The differences in trade types occur due to the imbalance of availability of resources, raw materials, and or technology among the countries. According to (46), merchandise exports in 2024 were $24.55 trillion, and merchandise imports were $24.83 trillion.

Trade is a major contributor of economic growth across multiple economies. For instance, countries like the United States and China are some of the key global players of trade in the world. A world bank report shows that exports account for about 36% of China’s total gross domestic product, while for the United States, exports are about 26% of the GDP (47). Other key players in global trade are the European Union (EU), Japan, and Germany. There are different trade blocks across the world that are affected by different rules and policies. However, international trade is often affected by various external and internal factors. For instance, the Russian-Ukrainian war has vastly affected trade across many economic blocks (5). According to the World Trade Organisation ((WTO), trade growth has experienced a fall from 4.7% to 3.4% since the beginning of the RussianUkrainian war in February of 2022. With Russia and Ukraine being major commodity exporter, this war has had a negative impact on its ability to participate in global trade (52). This war also has negative implications on countries that are directly involved with the two at war. The economic impact of the war has rippled across many global channels, affecting the trade volumes and other sectors in the countries (38).

Another recent shock that has greatly affected trade volumes is the COVID-19 pandemic. When the pandemic first hit the world, highly globalized countries such as the United Kingdom, Italy, the USA, and France were among the first countries to be hit fast and hard (34). While globalization has positively impacted a lot in the world economies, exposure to such shocks could cause a ripple effect due to supply chain risks and exposure to external factors. When the pandemic hit, a lot of countries closed borders, leading to a globalised retreat. Countries need to develop international trade resilience, where they are able to resist any disruptions to international trade, and recover in case of one (32).

Global shocks have both long term and short-term effects on international trade. The figure below shows how global trade volumes have evolved over the past 10 years. There is a clear drop in trade volume in 2020, which can be attributed to the global health pandemic, COVID-19.

**Figure 1: Trade Volume Trends Over the Years (22)**

In the African context, imports stifle growth and exports boost economic growth (39). In the intra-continental trade, Africa accounts for about 12% of it. During the pandemic, the World Trade Organization, 2020, estimated the volume of imports to decrease by 16% and the volume of exports to decrease by 8%. Africa is a major exporter of raw materials and agricultural products, while it majorly imports automobiles and other technological products.

With these shocks, there is a need for a model that is able to predict shocks and how the different economies react to shock. once the simulations are done, there is a need for comparing how the different countries shift within their economic clusters. Alternatives, there is a need to explore how an economic shock in one economic block affects the other and vice versa. These data analysis and visual dashboards are critical for policy makers, governments and financial advisors to use the data driven insights in making future decisions.

---

## 1.1.2 Global Trends in Trade Volumes

The volumes of trade worldwide have increased significantly over the last one hundred years because of globalization, technology, and the liberalized policy that eased the restrictions on the flow of goods and services. Multilateral trade growth came about after the years following World War II, when tariffs were reduced by means of the General Agreement on Tariffs and Trade (GATT) and the formation of the World Trade Organization (WTO) in 1995. Between 1950 and 2020, the volume of global merchandise trade grew almost forty times due to technological innovation and enhanced logistics (53). The contemporary use of container shipping, e-commerce, and digital platforms has now accelerated the pace and efficiency of cross-border trade. Nonetheless, since 2019, the world trade has been impacted by disruptions caused by COVID-19, supply chain disruptions, geopolitics, and fluctuations in energy prices.

In 2023, the WTO recorded that the world merchandise trade showed a reduction of approximately 2 percent but a recovery of 4% in 2024, of an estimated value of 31.5 trillion (53). The positive long-term trend is volatile and regionally dispersed: the developed economies have reached the stage of trade saturation, and the developing economies, and in particular in Asia, are increasing market share. A second significant advancement is the emergence of the services trade, which had contributed approximately 27% of the world trade in 2024 and had been increasing more rapidly compared to the goods trade (53). Such sectors as financial services, IT outsourcing, and tourism are the leading contributors to this change. Concurrently, global commerce is also being redefined by environmentalist factors, digitalization, and re-shoring of supply chains (3).

Figure 2 below illustrates the overall upward trajectory in trade volumes since 1950, with visible dips during global recessions and the COVID-19 crisis.

**Figure 2: Evolution of Trade (55)**

---

## 1.1.3 Macroeconomic Effects on Trade Volumes

The macroeconomic factors that have a very strong effect on trade volumes are GDP growth, inflation, exchange rates, and monetary or fiscal policies. With increased global production, the demand for imports and exports increases in line with it. Still, in economic downturns, trade reduces at a higher rate than GDP since the goods that are traded are more susceptible to business shocks (54). Direct effects on the competitiveness of trade: when the domestic currency is weak, exports are more competitive, as the commodity is cheaper in foreign markets, whereas when the currency is strong, exports may fall, but imports may be cheaper. Inflation is also significant; when domestic inflation is high, economic competitiveness in exports will be lost and the cost of importing will be elevated.

There are equal influences of trade policy and uncertainty. Protectionist measures, tariffs, and quotas distort the global supply chains and disincentivize investment in international production. According to IMF research (27), this is because tariff increases and uncertainty in trade policies lower the world investment and trade globally. Equally, the European Central Bank (16) also noted that trade policy shocks, particularly the implementation of tariffs, lead to long-term deteriorations in industrial production and exports. Another macro factor is financial conditions. The difficulty in accessing trade finance when the economy is in decline means that firms will be unable to trade internationally. In general, the sensitivity of the global trade industry to macroeconomic fluctuations highlights its volatility: trade increases more rapidly during booms but decreases more rapidly during crises, as seen in the cases of 2008 and 2020.

---

## 1.2 Problem statement

In an ideal financial world, international economists and financial advisors use machine learning techniques to predict variables that affect trade volumes and how their countries react to different shock scenarios. Machine learning techniques are not only good for predicting trade volumes, but also provide interpretable results for the factors that affect trade volumes in Sub-Saharan Africa countries. Over the years, traditional econometric methods like regression and time-series models have been used to establish the relationship between the various macroeconomic variables and trade volume. These methods, however, have limited assumptions of linearity, stationarity, and normality. The restrictions hinder the ability of the models to capture the dynamic, multivariate, and nonlinear interactions that affect the macroeconomic outcomes in Sub-Saharan Africa. Trade economics is affected by various shocks, some that are controllable and some are not. These shocks are often external and affect different countries across different economic zones. As a growing continent, Africa is affected by these challenges, thus affecting its trade, since Africa is one of the major exporters of goods in the world. Some of these global shocks include the global pandemic, tariff changes, geopolitical conflicts, and climate change, among others. Therefore, there is a need to form a predictive model that can estimate how different world economies behave when exposed to different shock scenarios. This makes it easier for economic and financial advisors to make decisions and policies for their countries and organizations at large. Therefore, this research aims to bridge the gap by predicting trade volumes across multiple Sub-Saharan Africa countries, and clustering these countries according to their different economic groups. Simulations will be made to evaluate how trade volumes react to shocks, which represent the global shocks. An analysis of how the countries fluctuate under the shock scenarios will also be done to check the economic stability of the African countries.

---

## 1.3 Research objectives

**General objective:** The research will be aimed at analysing the drivers of trade volumes, trends in trade volumes and using machine learning techniques to predict future values and simulate shocks and how it impacts the economies.

**Specific objectives:**

i. To analyse trends in trade volumes across different Sub-Saharan Africa economic blocs  
ii. To predict trade volumes using different machine learning models and perform feature engineering.  
iii. To classify the sub-Saharan countries into different clusters based on their macroeconomic variables and trade volumes patterns.  
iv. To model different shock scenarios of the variables using Monte Carlo simulation, to check how it affects trade volumes and economic stability.

---

## 1.4 Research questions

i. How does trade volume in Sub-Saharan Africa behave over time, and what are the factors for it?  
ii. Which machine learning algorithms give the best and most accurate forecasts of trade volumes given the available variables?  
iii. What economic clusters emerge from Sub-Saharan Africa countries using trade volume and the select macroeconomic factors?  
iv. How do the Sub-Saharan Africa countries’ trade volumes behave under different shock scenarios?

---

## 1.5 Research scope

This research bridges the gap between machine learning techniques and international macroeconomics. It seeks to improve the understanding of underlying trade patterns and what influences the different economies in Sub Saharan Africa. This study will employ machine learning to understand the factors that directly or indirectly influence the trade dynamics within the countries. The study will use of predictive modeling tools to predict trade based on inflation and GDP, clustering the countries based in their economic performance and positions, and simulating potential shocks in inflation and GDP to check how trade volume behaves and the vulnerability of these countries to socks. Using machine learning interpretable tools such as SHAP and LIME, this study will provide a framework of study for insights and policymakers to use in the future when making economic and financial decisions for their countries, or Africa as a whole. More importantly, this study is aimed at providing reliable, accurate and data driven approaches for the current economic problems facing Africa, given that it is one of the leading exporters of goods and services in the world.

---

## 1.6 Research limitations

One of the most significant limitations to this study is the unavailability of data. While the data will be used directly from the World bank website, there is a possibility of having inconsistent and incomplete data since some of it is collected in years and some is in months. There are different variations of data reporting for different counties, making it hard for the study.

Machine learning models might be quite complex for the regular economists and financial analysts to understand and interpret. With factors such as overfitting data into models, such an approach needs high data science skills to implement and model.

The study is mainly focused on understanding the effects of macroeconomic features on trade, while not putting other microeconomic factors into consideration.

---

## 1.7 Research Justification

The work undertaken in this study will contribute to narrowing the existing gap between machine learning and international economics, particularly in the predictive modelling of macroeconomic trends. Most existing studies utilize machine learning (ML) to predict stock market projections or microeconomic behaviour; however, few studies examine the potential of ML to explain large-scale economic phenomena, such as inflation and trade. This research aims to improve both aspects of the model to enhance prediction accuracy and make it more interpretable by utilizing state-of-the-art techniques in feature selection, model tuning, and validation. The results may be significant for policymakers, banks, and trade organizations as a data-driven input to inform policy actions during inflationary periods.


# CHAPTER 2: LITERATURE REVIEW

## 2.1 Overview

There are various macroeconomic variables that affect trade volumes, such as inflation, GDP, currency exchange rates, and economic tariffs and policies. With the development of machine learning techniques, international economics can use these techniques to analyse, predict, and cluster different trade volumes. This chapter explores the different studies that have been done in the aim to bridge this gap.

---

## 2.2 Theoretical Literature Review

### 2.2.1 Gravity Model of Trade

The gravity model of trade was first used to explain international trade flows in 1962 by Jan Tinbergen. The gravity model of international trade states that trade volumes between countries is directly proportional to the economic mass of the countries and their relative trade frictions (8). This model provides an explanation for the data in real world trade. The gravity model provides a baseline for analysing trade flows, suggesting that trade between two countries is positively associated with their economic size and negatively related to trade costs such as distance and tariffs (Ravi Kumar et al., 2024).

In log-linear form:

\[
\ln(T_{ij}) = \alpha \ln(GDP_i) + \beta \ln(GDP_j) - \gamma \ln(Dist_{ij}) + (controls), (1)
\]

where \(T_{ij}\) denotes trade flows.

This model has been a success over time as it is used to accurately predict trade volumes. It helps to bridge economic theory with empirical results to model trade activity.

In a study, (21) analyses the factors affecting trade in Sino-Africa, using the gravity model. The results of the study show that gross domestic product affects exports from Africa to China. Furthermore, there is a positive correlation between real exchange rates and African exports, while it affects imports negatively. The study also found out that a recession has a negative impact on both imports from China and exports to China (21).

However, (43) found some limitations to the gravity trade model. The study highlights a gap between theoretical and empirical studies. In the SSA context, the model highlights GDP as a critical driver of trade growth, reinforcing the expectation that rising domestic and partner GDP will boost trade volumes (23).

---

### 2.2.2 Purchasing Power Parity (PPP)

Invented in 1918 by Swedish economist Gustav Cassel, the purchasing power parity is the value of goods that a single unit of a currency in one country can buy from another country (29). The concept of PPP is based on the ‘law of one price’, where similar goods with similar prices but in different markets are expressed in the same currency. While the gravity model emphasizes GDP, the Purchasing Power Parity (PPP) theory links inflation to trade competitiveness through relative prices and exchange rates (25).

Relative PPP posits that exchange rate movements should reflect inflation differentials:

\[
\Delta \ln S_{i/j} \approx \pi_i - \pi_j, (2)
\]

where \(S_{i/j}\) is the bilateral exchange rate, and \( \pi_i \) and \( \pi_j \) are inflation rates.

In short, this model relates domestic and foreign commodity prices and exchange rates. Purchasing power parity is important since it makes it possible to compare the outputs of different countries despite the price differences and market exchange rate differences (29).

In SSA, inflationary episodes often weaken currencies, raising import costs and reducing export competitiveness (25). PPP thus offers a key mechanism by which inflation affects trade flows.

Purchasing power parity is often preferred for use in trading analyses by policymakers and researchers. The PPP theory does not show major fluctuations in the short run, but indicates which direction exchange rate moves in the long run.

(7) conducted a literature analysis study where they used literature from the Scopus database on purchasing power parity. The study conducted an analysis using RStudio, VOSviewewr, and advanced excel to compare these articles. The results were that exchange rates play a huge role on the purchasing power of different countries, thus validating the PPP theory.

---

### 2.2.3 Heckscher-Ohlin Theory

This theory was developed in the early 20th century by Eli Heckscher and Bertil Ohlin. It states that countries mainly export goods that use their abundant and cheap production factors such as land and labour, and import goods that require scarce factors. This model explains the different trade patterns across different countries, as they are based on national resource endowments.

**Figure 3: Heckscher-Ohlin Theory (45)**

(6) conducted a study to analyse the Heckscher-Ohlin theory using Bangladesh-US data. The study used ordinary least square (OLS) techniques on data from NBER International Trade and Geography Data and the UN Comtrade Database between the years 2018 and 2008. The results showed that Bangladesh and the United States follow the theory since there is plenty of labour in Bangladesh, and this has led it to retain its trading patterns since 2008 (6).

In this study, this model can be used to explain some trends and patterns that will be identified in the analysis stage. Through this and identifying the different resources and plenty of availability of labour, it is easier to explain and understand why the different African trading blocs have certain trends.

---

## 2.3 Empirical Literature Review

### 2.3.1 Predicting Trade Volumes

To predict trade volumes in the agricultural department, (19) used trained data from the United Nations using supervised models and neural networks. The aim was to predict agricultural trade patterns and identifying any underlying factors that affect trade volumes. The effect of tariffs and other economic variables was evaluated. While seven out of 12 pairs predicted was predicted closely to the actual values, grouping, categorizing and hyper-tuning the variables gave better results.

It is important to accurately predict trade volumes as this will help in economic forecasts, especially in small economies. (14) aims to forecast world trade using big data and machine learning. With a dataset of 11017 series on key economic indicators, the study uses techniques such as lasso regression, random forests regression, and linear ensembles to do the predictive modelling. The study finds that there are no statistical differences in the accuracy of the forecasted values, whether it is with respect to technique, or the dataset used. The dataset can be either big or small.

To evaluate how machine learning algorithms can be used to predict international bilateral trade flows on imports and exports in Croatia, (30), did a study using the gravity model. This study used a dataset on Croatian bilateral trade with about 180 countries between 2001 and 2019. The algorithm was used to forecast imports and export volumes for the next one year (2020). Other machine learning algorithms used are gaussian processes, linear regression, and multilayer perception. The algorithms performed well in predicting the Croatian bilateral trade volumes, with neural network multilayer perceptron as the best performing model.

Trading activities affect the stock markets in various countries. (56) analyses the effectiveness of machine learning in predicting trade volume. The study used data collected from Wind Information Co. that includes trading volumes of China. Two nonlinear autoregressive (NAR) neural network models were considered. The results were that a simple model using ten hidden neurons and thirty lags led to a stable and accurate prediction as shown by the root mean square error measurement (56).

Zhu, (2025) (58) did a study to forecast the export-import volume of China’s economic trade. The study analysed key contributors of trade such as gross domestic product (GDP) and producer price index (PPI). Seven key indicators are used as variables using correlation coefficient. Support vector machines was deployed on data from 2003 to 2022 and the results showed that ISSA-SVM model is a reliable method of predicting trade volumes.

---

### 2.3.2 Global trends in trade volumes

A study by Batarseh et al., (2019) (9) analysed the applications of machine learning in forecasting international trade trends. The study explored data collected from USDA’s Foreign Agricultural Services’ Global Agricultural Trade System (FAS - GATS) published by the United States Department for Agriculture. Both supervised and unsupervised machine learning models were used, namely Linear regression, K-means clustering, Pearson correlations, boosting and Autoregressive Integrated Moving Average (ARIMA). The study aimed at predicting future exports and imports of a specific commodity. LightGBM had an R-squared of 88% and XGBoost has 69%, meaning machine learning models can be explored to predict international trade.

Predicting trade patterns is important for decision-making in both the public and private sectors. Gopinath et al., (2020) (18) focused on seven major agricultural commodities and used data driven approaches to decipher any patterns in the trade. This study used both supervised and unsupervised machine learning techniques on a training dataset between 2010 and 2020. The results of the study showed machine learning models are great for predicting trade patterns. Supervised machine learning techniques identified the major economic factors that affect trade flows.

De-globalization trends affect trade networks in many ways. They can either improve or reduce economic growth. In a study, Silva et al., (2024) (44) used section-level trade data between 2010 and 2022 from countries exceeding 200, to identify the shifts in trade networks due to deglobalization trends. This study identified that key players in the global markets such as the United States, China and Germany maintain market dominance. Using non-linear models like random forests, gradient boosting, and light gradient boosting machine, the study proved that machine learning models outperform traditional linear econometric models in making predictions.

In an aim to study the different patterns in trade economics, Jeong et al., (2024) (28) used machine learning techniques as an alternative to older conventional methods of exploring and understanding trends. The study used monthly and HS-6 digit data to forecast the trends. The study did a comparison between Poisson Pseudo Maximum Likelihood (MPML) estimator and machine learning methods to do this analysis. The results showed that while PPML has rich research and applications in the trend prediction, ML methods performed better and had better accuracy margins.

---

### 2.3.3 Shock simulations for trade volumes

International trade is often affected by various shocks such as the pandemic, international wars, and climate change among others. All these are factors, and how the countries react to them affect global trading. To study the effects of COVID-19 crisis on exports, Dueñas et al., (2021) (15) did a study using a control group that is not affected by the shock, and a group that is exposed to the pervasive nature of the shock. the study explored the effectiveness of using causal machine learning techniques in predicting the firm’s trade. The study found that the probability of a firm surviving the export market under the COVID-19 shock decreased by about 20 percent in April 2020.

Due to this, a country needs to develop a strategic shock parametrization mechanism. (24) studied the use of the Joint Operations Area Resilience Model, a system that integrates big data and machine learning into open data for managing the shock parameters.

Another shock that affects trade volumes is global tariff fluctuations. These volatile shifts affect global market shares, incomes, and supply chain stability, thus affecting the overall trading capacity of the country. (2), argues that traditional econometric models are great for trend analysis and does not account for causal relationships, hence the need for causal machine learning (CML) techniques. This study forecast the impacts of global tariff shocks on US agricultural trade. Findings show that global market tariffs have a huge impact on trade volumes.

---

### 2.3.4 Monte Carlo Shock simulations

Monte Carlo methods have been widely used to model uncertainty in African macroeconomics. Nchofoung (2022) (36) employed Monte Carlo-generated impulse responses to analyze trade shocks in Sub Saharan Africa. Poldena et al., (2023) (40), conducted a study that used a novel agent-based model (ABM) to forecast different macroeconomic variables under different shock scenarios. The model is useful for stress-testing and predicting the effects of different monetary and fiscal macroeconomic policies. Ahmed et al., (2023) (4), examined the relationship between inflation and different macroeconomic variables such as GDP and trade balance. Using data from the UK from 2010 to 2022, a VAR model was utilised which showed that inflation shocks cause a decrease in GDP and trade balance. Abdel-Latif et al. (2025) (1) applied stochastic shocks to stress-test South Africa’s financial stability. Beyond SSA, Celestin et al. (2025) (11) used Monte Carlo simulations to model financial volatility during global crises, showing the adaptability of the method. Despite these applications, no study has directly simulated inflation and GDP shocks on SSA trade volumes, underscoring the novelty of this research.

---

## 2.4 Research gap

A lot of research has been done on international trade modelling, but a lot of these studies use traditional econometric models like regression and time series to show the relationship between the different macroeconomic variables and trade performance (6); (21); (23).

However, these methods of study assume linearity, stationarity, and normality, thus they do not capture the nonlinear and dynamic relationship in trade patterns, especially with the vast macroeconomic systems in Sub-Saharan Africa (43).

With recent studies using machine learning techniques to predict future economic trends, a lot of them are focused on developed or global economies such as the United States and China (58); (56); (14); (28). There is little research on simulating macroeconomic shocks in the Sub-Saharan Africa context, despite the vulnerabilities and economic differences of these countries with the industrialized nations (2); (36).

In addition to these, there are very few studies that combine machine learning techniques and simulation such as Monte Carlo, which is great for evaluating how the different economies behave under external shocks such as inflation, the pandemic, and external geopolitical conflicts ((40); (1)). These studies do not explore interpretable machine learning approaches for use by policymakers to provide transparent, data-driven insights on trade behaviour and shock response (15); (24).

---

## 2.5 Literature Review Summary

Below are the findings of the literature review in a table format:

| Research Objective / Question | Subtopic | Key Authors & Studies | Summary of Findings | Identified Gap / Limitation | Relevance to Current Study |
|---|---|---|---|---|---|
| 1. To analyze trends in trade volumes across SSA economic blocs | Trade volume patterns and determinants in SSA | Guan (2020); Herman (2023); Akther et al. (2022); Joseph & Ibrahim (2022) | GDP, exchange rate, and inflation significantly influence trade flows; regional blocs (EAC, ECOWAS, SADC) show differing trade trends. | Most studies use static econometric models that fail to capture dynamic or nonlinear changes. | Establishes baseline trade behavior and supports data-driven modeling across SSA blocs. |
|  | Regional trade integration and policy effects | Adegoke et al. (2025); Nchofoung (2022); UNECA (2024) | Policy integration (e.g., AfCFTA) boosts intra-African trade; data inconsistency limits cross-country comparisons. | Limited real-time data and lack of computational modeling for forecasting integration outcomes. | Supports harmonized data and ML evaluation of policy-driven trade growth. |
| 2. To predict trade volumes using ML models | Predictive modeling of trade and macroeconomics | Xu et al. (2023); Jeong et al. (2024); Dubovik et al. (2022) | ML models like Random Forest and XGBoost outperform linear models in trade prediction accuracy. | Prior work focuses on global or developed economies; limited SSA applications. | Provides methodological basis for applying ML to SSA trade data. |
|  | Feature importance and explainable AI in forecasting | Dueñas et al. (2021); Hodicky et al. (2025) | Introduced SHAP and LIME for interpreting model decisions; improved transparency for policymakers. | Few studies use explainable AI within trade forecasting contexts. | Integrates SHAP/LIME to reveal key macroeconomic influences in SSA trade. |
| 3. To classify SSA countries into clusters | Trade-based economic clustering and heterogeneity | Ravi Kumar et al. (2024); Herman (2023); Akther et al. (2022) | Found heterogeneity in trade drivers across income and policy structures. | Clustering often focuses on income or growth—not trade-macro combinations. | Clustering allows discovery of structural trade groupings using ML. |
|  | Regional economic structures and typologies | IMF (2024); World Bank (2023); Adegoke et al. (2025) | Identified disparities in infrastructure, industrial capacity, and policy coordination. | Lacks computational grouping to validate structural patterns. | Supports use of K-Means or Hierarchical clustering for SSA typologies. |
| 4. To model shock scenarios using Monte Carlo simulation | Economic shocks and trade volatility | Poldena et al. (2023); Abdel-Latif et al. (2025); Dueñas et al. (2021) | External shocks (pandemics, conflicts, inflation) severely disrupt trade flows. | Minimal simulation-based studies assessing resilience in SSA. | Introduces stochastic simulation to quantify SSA trade stability. |
|  | Monte Carlo simulation in macroeconomic analysis | Nchofoung (2022); Hodicky et al. (2025) | Monte Carlo effectively models uncertainty in policy outcomes. | Lacks integration with predictive ML frameworks. | Combines ML with simulation to analyze trade resilience under uncertainty. |


# CHAPTER 3: METHODOLOGY

The project will be aimed at identifying the feature importance of the main economic variables that affect trade volume. These features will be used for predictive modelling to predict future values of trade volumes. The project will then use machine learning clustering models to group the countries into their different economies. Lastly, Monte Carlo simulations will be explored to identify how different macroeconomic shocks affect trade volumes, and how the different countries and economic blocks shift across different scenarios. In this chapter, a step-by-step details and explanation of this analysis will be done to show how the project will arrive at the projected outputs.

## 3.1 Research Design

For this study, we will emulate the CRISP-DM (CRoss Industry Standard Process for Data Mining) methodology. CRISP-DM is a de facto method of processing huge amounts of data which uses an independent based model of data processing. It consists of 6 main parts, from business understanding up to deployment. This method will be great to provide a structured framework for the study.

The first phase of the model is business understanding. In this phase, we will identify the key stakeholders for the project, understand the main objectives of the study and what is required from a business perspective. This stage identifies the main goals of the data mining project. By understanding this, the data mining types and success criteria are identified and explained, before a comprehensive project plan is created.

The next step is data understanding, which involves collecting data from the source, exploring and describing it. In this step, data quality is checked to ensure that it meets the project requirements and it will be useful for data mining. An understanding of the data attributes and variables is important for the data processing. This step is important for developing the hypothesis of the study.

Once the data is great, the next step is data preparation. This step involves data cleaning where missing values are imputed and duplicates are removed. Data is prepared from initial data depending on the model to be used. New attributes can be constructed and data transformed to fit the model.

In the modelling phase, a modelling technique is chosen. This step involves building a test case and applying the model to it. The type of model chosen depends on the data used and business problem. Specific parameters are set for model building. An evaluation of the model is done using appropriate evaluation criteria and the best model is chosen.

Next is the evaluation phase, where results are checked and compared to the project objectives. This evaluation ensures that the model is built and trained correctly and it chieves the business objectives that are set. The results are further interpreted and future actionable insights drawn.

Lastly, the deployment stage, which uses the knowledge from the study in an organized and presentable manner. This can be done in either a report, a user guide or a software component. In this stage, the guide shows the planning, monitoring, and maintenance of the analysis.

The flow chart below is a visual representation of the methodology steps that will be undertaken in this study:

**Figure 5: Research Methodology**

---

## 3.2 Population and sampling

For this study, we will explore select Sub-Saharan Africa (SSA) countries that have consistent data available. The SSA is divided into three main economic blocs, namely the East African Community (EAC), Economic Community of West African States (ECOWAS), and the Southern African Development Community (SADC). Therefore, for this study, we will explore about three countries from each economic bloc; Kenya, Uganda, Tanzania, Nigeria, Ghana, Senegal, South Africa, Zambia, and Zimbabwe.

---

## 3.3 Data Understanding

The data is collected by the World Bank using official government sources such as central banks and statistical databases. For instance, inflation data is collected from the IMF’s world economic outlook, international financial statistics, ILOSTAT, and UNdata. The World Bank collects trade data from the Observatory of Economic Complexity (MIT). The World Bank aggregates this data into country level and classification of goods before publishing it on their website.

The tariffs data will be collected from the World Bank website. Tariffs data is collected yearly and the data is available for all years. For this study, the data collected will be filtered to cover only data from 1970 to 2024. The inflation dataset will be collected from the World Bank website. The World Bank gets this data from the International Financial Statistics database and the International Monetary Fund (IMF) (49). This data is collected annually, therefore we will filter the dataset in the website to get data between 1970 and 2024. Trade dataset will be collected from the world bank website, which collects the data from country official statistics and the National Statistical Organizations or Central Banks. The yearly data will be collected from 1970 to 2024. The GDP data will also be collected from the World Bank website for data from 1970 to 2024 (46). Exchange rates data is collected monthly, quarterly, and annually. For this project, we will use the annual exchange rate data from the International Monetary Fund website (27). Annual Foreign Direct Investment data is collected from the world Bank website. This data is available for the years 1970 to 2024.

---

## 3.4 Data Preprocessing

In the data pre-processing stage, we ensure that all data is ready for machine learning. The study will have different datasets, therefore they will need to be merged into a single data frame. In order to fit into this format, the data will be transformed into a long format. A long format dataset will be easier for merging, visualization, and statistical modelling. A dataset that is transformed to a long format is more efficient, scalable, and maintainable for data science operations. It will be necessary to handle time periods consistently since the dataset for this study will be time sensitive when merging the three datasets. The dataset will be merged using the country name and year columns.

Next will be checking for missing values. To address the missing values problem, all rows with missing values will be imputed from the dataset. In the case of a column having more than 30% of missing values, the column will be removed completely from the dataset.

After merging datasets and checking for missing values, the next step will be checking for outliers. Outliers are values that are either extremely low or high compared to the rest of the data.

Once the data is cleaned, the selected Sub-Saharan Africa countries will be chosen and filtered from the dataset. The columns will then be converted and all missing values imputed using the median or mean. A check for duplicated values will be done and all duplicates deleted from the dataset.

---

## 3.5 Exploratory Data Analysis

Exploratory data analysis (EDA) is the process of doing a dataset analysis to understand the different variables, visualizations and correlations. EDA is useful for identifying patterns and anomalies within the dataset. It summarizes the characteristics of the dataset and gives a visual understanding of the variables.

In this study, we will explore the descriptive statistics of the dataset to give a better understanding of the data. The descriptive statistics will give an analysis of central tendency analyses, variability, and distribution of data through values like mean, mode, and median of the variables. The descriptive statistics will be used to simplify the data as a whole. The mean is the average of numerical values within a variable. Median is the measure of of the midpoint of data, regardless of if it is in ascending or descending order. The mode is the value that appears most frequent within the dataset. Standard deviation measures how spread the dataset it. A large value indicates that the data is largely spread.

Next step will be data visualizations. In this step, complex and large datasets are set in graphical and visual formats that are easier to understand and draw trends and patterns. For the study, a boxplot will be plotted to show inflation rates. This will be important to show how the inflation rates differ among the different SSA countries. It will show a summary of the distribution, different quartiles, median, minimum and maximum values of the inflation rates. Box plots are useful for comparing the distributions of values of inflation and the other macroeconoemic variables in the different countries and showing which countries have high inflation rates and those with lower rates. This visualization will also be important to see if the skewness of the inflation rates.

A line plot of trade volume will be plotted to show the trends in trade volume among the different countries. Since the data will be continuous data over the years, a line plot will aid in visualizing how the trends behave over the years. Lastly, a histplot will be drawn to visualize the distributions of the three variables.

A scatterplot will be used to visualize if there are linear relationships between the variables. A scatterplot is a graph used to show the relationship between any two continuous variables in a dataset. A dot in a scatterplot represents a single observation’s two values. In a scatterplot, the dot patterns show patterns, trends and correlations between the variables, thus showing how strong and correlated the variables are. For instance, there could be positive, negative and zero correlations.

A different correlation analysis will be done using a heat map. A correlation analysis identifies the relationship between two variables, how strong the relationship is, and the direction of the relationship. These values range between -1 and 1, with a -1 being a strong negative correlation, 0 being no correlation between the variables, and 1 is a strong positive correlation. In a correlation heatmap, correlation does not mean causation. This means that if two or more variables are correlated, it does not mean they cause each other.

---

## 3.6 Feature Engineering

Feature engineering is the process of transforming raw data into relevant information used by machine learning models. Feature engineering uses both domain knowledge and statistical techniques to improve the quality and predictive power of machine learning models. In this study, feature engineering will be used to check which features play a strong role in predicting the trade volumes in the select SSA countries. Feature engineering is important as it reduces overfitting of data, improves model interpretability, and enhances overall performance.

In this study, there are many variables collected from different website databases. According to the literature above, all these variables play a part in trade volumes. This calls for a need for feature engineering to identify the top variables that affect trade volumes. Therefore, we will use normalization to set the parameters into equal features that are suitable for machine learning techniques. We will use normalization since it is not sensitive to outliers and it scales the vectors to a unit length.

\[
X_{scaled} = \frac{x_i}{\|x\|}
\]

(3)

We will also use log transformations to stabilize variance and reduce any skewness in the dataset.

---

## 3.7 Machine Learning Models

Machine learning is a branch of artificial intelligence that uses statistical algorithms to learn from data and draw predictions and decisions. For this study, we will explore various machine learning algorithmns to predict and classify countries into their different economic categories. The study will also use Monte Carlo shock simulations to check how the countries are performing variate under different shocks on the economic variables identified under feature engineering.

### 3.7.1 Random Forests

Random forests is a machine learning algorithm that uses decision trees and ensemble methods to predict and classify variables. Random forests use tree predictors, where each tree depends on a random vector, and they vote for the most popular class as a prediction (33). For this model, we will follow the following steps; feature engineering, data splitting into test and train datasets, training the model, hyperparameter tuning, and lastly assessment of the model performance. The ensemble methods in random forests use a bagging method, where the training dataset it selected with a replacement. This means that individual data points can be used in more than one scenario (33). This study will use random forests to conduct feature engineering and identify the top features that highly influence trade volumes. The top features identified will then be used for predicting trade volumes using random forests algorithm. To avoid overfitting of the dataset, we will explore hyperparameter tuning, which makes the model not to memorize trends in the training data, thus providing more accurate results and reducing bias.

**Figure 6: Random Forests Technique (31)**

### 3.7.2 Extreme Gradient Boosting (XGBoost)

Extreme Gradient Boosting is a data training model that works well with tabular datasets. This makes it a better model to use for prediction of trade volumes using available variables. XGBoost has an in-built regularization tool, that prevents it from overfitting datasets and an integrated cross-validation model (12). This makes it great for running in systems like Hadoop and Spark (57).

Therefore, for this study, we will explore XGBoost in predicting trade values. These results will be compared with other prediction models to see the explainability of trade models using the given variables. The study will also use XGBoost model to conduct feature engineering to identify it’s model variables that affect trade volumes the most. We will also do some hyperparameter tuning such as GridSearchcv to find the optimal values for the model.

### 3.7.3 Long Short-Term Memory (LSTM) networks

Long Short-Term Memory (LSTM) networks are a special kind of recurrent neural network designed to learn long-term dependencies in sequence data (37). In this study, we will use LSTM to capture how the select macroeconomic variables and trade volume influence future trade flows. Each LSTM cell maintains a memory state Ct and uses gating mechanisms (forget gate ft, input gate it, etc.) to update this state: for example, the cell state updates as:

\[
C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t
\]

(4)

where Ct is a candidate cell state and ft, it ∈ [0, 1] are learned gates. This structure enables LSTM to “remember” significant economic patterns over long horizons while “forgetting” obsolete information. LSTMs have been shown to outperform traditional time-series models (like SARIMA) in GDP forecasting tasks, so they are well-suited for my research’s macroeconomic-variables-trade-volume forecasting context (37). For this study, LSTM will be tuned using GridSearch to prevent the model from memorizing patterns in the training set. This will help to provide a more accurate prediction of trade volumes.

### 3.7.4 K-Means Clustering

K-Means is an unsupervised learning algorithm that partitions data into K clusters. Each cluster is defined by its centroid, which is the mean of the points in that cluster. The algorithm assigns every data point (e.g. a country or time-period instance with features like inflation, GDP, trade) to the nearest centroid, then recomputes centroids and repeats until convergence (13). Formally, K-Means iteratively minimizes the sum of squared within-cluster distances:

\[
\min \sum_{j=1}^{K} \sum_{x_i \in C_j} \|x_i - \mu_j\|^2
\]

(5)

where j is the centroid of cluster Cj. This process optimizes cluster “cohesion” by reducing intra-cluster variance (13). In practice, we choose K by methods like the “elbow” criterion or silhouette analysis. K-Means is simple and fast but requires specifying K in advance; it works best when the clusters are roughly spherical and similar in size. This study will classify the country into different economic clusters based on the macroeconomic variables selected during feature engineering.

### 3.7.5 Density-Based Clustering

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is another unsupervised method that finds clusters of arbitrary shape by looking for dense regions in the data. DBSCAN defines a point as a core point if it has at least MinPts neighbors within a radius and then forms clusters by connecting core points and their neighbours (41). Points that are not reachable from any core point are treated as outliers. This means DBSCAN can automatically detect noise and do not require specifying the number of clusters. As one source notes, DBSCAN “groups data in the high-density region of the feature space” and requires two parameters: the radius and the density threshold minPts (41). Its ability to capture clusters of complex shapes and to filter noise makes DBSCAN useful for identifying unusual trade-volume patterns or outlier economies in the data. Since this is a better version for classification, the model will use this model to classify countries and economic blocs into their different economic clusters.

### 3.7.6 Monte Carlo Simulations

Monte Carlo Simulation is a computational algorithm that uses repeated random sampling to get data based on a likelihood of a range of results (17). It is used to give multiple predictions based on different uncertainties. Monte Carlo simulations use repeated random sampling of inputs of the random variable and the results are aggregated. In the finance and economics world, Monte Carlo simulations can be used to understand the adverse and uncertain economic events, and how they will affect the economic position of the countries. This means the feature engineered variables will be simulated and a study of how the countries behave under different shock scenarios. Changing the macroeconomic shocks will help study how trade volumes behave in extreme cases.

---

## 3.8 Model Valuation Metrics

We will evaluate regression/forecast models and clustering models using appropriate metrics. For regression (forecast) performance, common metrics include the coefficient of determination R2, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) (35). These are defined as:

\[
R^2 = 1 - \frac{\sum (y_i - \hat{y_i})^2}{\sum (y_i - \bar{y})^2}
\]

(6)

where yi are actual values, yˆi are predicted values, and y¯ is the mean of yi. A higher R2 (up to 1) indicates a better fit, while a lower MSE/RMSE indicates more accurate predictions.

For the performance of the clustering models, we will use internal validation indices like the Silhouette Score and the Davies–Bouldin (DB) Index. The silhouette score measures how similar an object is to its own cluster (cohesion) versus other clusters (separation); values range from -1 (poor clustering) to +1 (well-clustered) (35). In practice, a high average silhouette score (near 1) indicates tight, well-separated clusters. The DB Index is defined as the average ratio of intra-cluster scatter to inter-cluster separation; lower DB values indicate better clustering. These metrics help professionals quantify how well their clustering reflects underlying data structure (35).

---

## 3.9 Machine Learning Explainability

To interpret the machine learning models, the study will employ explainable AI techniques. Two popular frameworks are LIME (Local Interpretable Model-Agnostic Explanations) and SHAP (SHapley Additive exPlanations (20). SHAP is based on ideas from cooperative game theory and gives each feature a Shapley value, which basically measures how much that feature contributes to a model’s prediction. What’s unique about SHAP is that it follows fairness rules and can explain both the overall importance of features across the whole model and the reasoning behind individual predictions.

LIME works relatively differently, it builds a simple, easy-to-understand model (like a small linear regression) around one specific prediction. It tweaks the input features slightly to see how the output changes, creating a kind of “local snapshot” of how the complex model behaves. Together, SHAP and LIME help in understanding why models make the predictions in a manner that they do. In this case, they show which macroeconomic factors, like GDP or inflation, are driving trade volume forecasts or influencing how countries get grouped in clusters (20).

---

## 3.10 Model Deployment

The model will be deployed in a web-based dashboard. The dashboard will be set up with previous data and integrate live data directly from the World Bank website on the different macroeconomic variables. This way, in case of any shocks or changes in the variables, all stakeholders are able to see the trade outputs in live results.

This project will deploy the results in a streamlit dashboard since it integrates the different machine learning models used. Streamlit also allows for comparison of different models and live integration of country filters and comparisons.

The first step will be to prepare the local development environment by installing python and ensuring the environment has dependency isolation. The dependencies will be stored in a requirements.txt file to that its easy to reproduce when deploying the project. ‘

Next is to ensure the project has a well defined structure which will make the project easy to deploy. The dashboard will then be implemented by the python class using the Streamlit framework. The main components for the dashboard will include a sidebar that filters values according to country, economic bloc, time and time. Trade volumes and macroeconomic indicators will be the interactive visualizations, and the model predictions as outputs. The visualization will also include SHAP and LIME plots, and Monte Carlo simulation controls.

For this project, all trained models will be saved locally and then loaded during runtime. The tree-based models will be saved using joblib while the LSTM models using keras. The feature scalers and pre-processing pipelines will be saved and reused.

The next step will be pushing the project to a GitHub repository before deployment using the StreamLit community cloud to launch the application. Once the project has been deployed, the dashboard is tested and validated to ensure all the models run smoothly and there are correct outputs. This step also tests for how the dashboard responds when opened in different browsers. The dashboard will be accessed using a web url.

Lastly ensure the model has continuous updates by pulling macroeconomic data from the world bank and IMF APIs. This step ensures the dashboard is live and not just a static visualization that only uses past data.

The model will be monitored continuously to check its performance and ensure that the results given are accurate and explainable. Major obstacles such as explainability to the international economists and other stakeholders that use the data will also be documented. Recommendations on these struggles will be taken in and solutions pro26 vided. Other than predicting trade flows, the dashboard will offer explainability features to interpret and explain the main features that affect international trade and how this varies.

---

## 3.11 Research Quality and Validity

The reliability of this research will be based on the fact that we will used standardized preprocessing techniques, and the models can be reproduced at any time. This makes it reliable for using both past and future data to continuously evaluate the economic situations of the countries.

Validity of the research will be made on the fact that data used is obtained from authoritative and reliable sources such as the World Bank (49). Additionally, this study will narrow down the research to specific Sub-Saharan Africa countries, to make it easier to interpret the results and explore the different macroeconomic variables in the countries.