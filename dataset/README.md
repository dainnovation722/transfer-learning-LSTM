
<pre>
.
├── source
│   ├── air_quality
│   ├── appliances
│   ├── beijing_pm2.5
│   ├── electricity
│   ├── exchange_rate
│   ├── metro_interstate
│   ├── traffic
│   └── usv
└── target
    ├── debutanizer
    └── sru
</pre>


## dataset
|dataset|type|discription|citation|
|---|---|---|---|
|air_quality|source|The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area, at road level,within an Italian city. Data were recorded from March 2004 to February 2005 (one year)representing the longest freely available recordings of on field deployed air quality chemical sensor devices responses. Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) and were provided by a co-located reference certified analyzer. Evidences of cross-sensitivities as well as both concept and sensor drifts are present as described in De Vito et al., Sens. And Act. B, Vol. 129,2,2008 (citation required) eventually affecting sensors concentration estimation capabilities. Missing values are tagged with -200 value.This dataset can be used exclusively for research purposes. Commercial purposes are fully excluded.|[here](http://archive.ics.uci.edu/ml/datasets/Air+Quality)|
|appliances|source|The data set is at 10 min for about 4.5 months. The house temperature and humidity conditions were monitored with a ZigBee wireless sensor network. Each wireless node transmitted the temperature and humidity conditions around 3.3 min. Then, the wireless data was averaged for 10 minutes periods. The energy data was logged every 10 minutes with m-bus energy meters. Weather from the nearest airport weather station (Chievres Airport, Belgium) was downloaded from a public data set from Reliable Prognosis (rp5.ru), and merged together with the experimental data sets using the date and time column. Two random variables have been included in the data set for testing the regression models and to filter out non predictive attributes (parameters).|[here](http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)|
|beijing_pm2.5|source|The dataâ€™s time period is between Jan 1st, 2010 to Dec 31st, 2014. Missing data are denoted as â€œNAâ€.|[here](http://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)|
|electricity|source|This is the electricity consumption in kWh was recorded every 15 minutes from 2011 to 2014. Because the some dimensions are equal to 0. So we eliminate the records in 2011. Final we get data contains electircity consumption of 321 clients from 2012 to 2014. And we converted the data to reflect hourly consumption.|[here](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)|
|exchange_rate|source|the collection of the daily exchange rates of eight foreign countries including Australia, British, Canada, Switzerland, China, Japan, New Zealand and Singapore ranging from 1990 to 2016.|[here](https://github.com/laiguokun/multivariate-time-series-data)|
|metro_interstate|source|Hourly Interstate 94 Westbound traffic volume for MN DoT ATR station 301, roughly midway between Minneapolis and St Paul, MN. Hourly weather features and holidays included for impacts on traffic volume.|[here](http://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)|
|traffic|source|This data is a collection of 48 months (2015-2016) hourly data from the California Department of Transportation. The data describes the road occupancy rates (between 0 and 1) measured by different sensors on San Francisco Bay area freeways.|[here](http://pems.dot.ca.gov/)|
|sru|target||[here](https://www.springer.com/gp/book/9781846284793)|
|debutanizer|target||[here](https://www.springer.com/gp/book/9781846284793)|
