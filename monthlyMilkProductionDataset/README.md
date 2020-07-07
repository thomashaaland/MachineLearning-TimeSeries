# Monthly milk production

This project looks at monthly milkproduction. The dataset has three productions:
* Cheese
* Ice cream
* Milk

The data is stored in CADairyProduction.txt and is in csv format.

dairyProd.py looks at the production of milk and uses automatic tuning for ARIMA model prediction.

In the first part it can be seen that the dataset has a trend and seasonality. To remove the trend and seasonality we use differencing in the adjacent elements to remove the trend and differencing based on seasonality (12 month repetition) to remove seasonality.

It is then confirmed that the new dataset has neither trend nor seasonality.

This information is then used to tune the SARIMA model.