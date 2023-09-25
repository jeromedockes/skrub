"""
Interpolation join: infer missing rows when joining two tables
==============================================================

We illustrate the ``InterpolationJoin``, which is a type of join where values from the second table are inferred with machine-learning, rather than looked up in the table.
It is useful when exact matches are not available but we have rows that are close enough to make an educated guess -- in this sense it is a generalization of a ``fuzzy_join``.

The ``InterpolationJoin`` is therefore a transformer that adds the outputs of one or more machine-learning models as new columns to the table it operates on.

In this example we want our transformer to add weather data (temperature, rain, etc.) to the table it operates on.
We have a table containing information about commercial flights, and we want to add information about the weather at the time and place where each flight took off.
This could be useful to predict delays -- flights are often delayed by bad weather.

We have a table of weather data containing, at many weather stations, measurements such as temperature, rain and snow at many time points.
Unfortunately, our weather stations are not inside the airports, and the measurements are not timed according to the flight schedule.
Therefore, a simple equi-join would not yield any matching pair of rows from our two tables.
Instead, we use the ``InterpolationJoin`` to *infer* the temperature at the airport at take-off time.
We train supervised machine-learning models using the weather table, then query them with the times and locations in the flights table.

"""

######################################################################
# Load weather data
# -----------------
# We join the table containing the measurements to the table that contains the weather stations’ latitude and longitude.
# We subsample these large tables for the example to run faster.

from skrub.datasets import fetch_figshare
import pandas as pd

weather = fetch_figshare("41771457").X
weather = weather.sample(100_000, random_state=0, ignore_index=True)
stations = fetch_figshare("41710524").X
weather = stations.merge(weather, on="ID")[
    ["LATITUDE", "LONGITUDE", "YEAR/MONTH/DAY", "TMAX", "PRCP", "SNOW"]
]

######################################################################
# The TMAX is in tenths of degree Celsius -- a TMAX of 297 means the maximum temperature that day was 29.7℃.
# We convert it to degrees for readability

weather["TMAX"] /= 10

######################################################################
# InterpolationJoin with a ground truth: joining the weather table on itself
# --------------------------------------------------------------------------
# As a first simple example, we apply the ``InterpolationJoin`` in a situation where the ground truth is known.
# We split the weather table in half and join the second half on the first half.
# Thus, the values from the right side table of the join are inferred, whereas the corresponding columns from the left side contain the ground truth and we can compare them.

n_left = weather.shape[0] // 2
left_table = weather.iloc[:n_left]
left_table.head()

######################################################################
right_table = weather.iloc[n_left:]
right_table.head()


######################################################################
# Joining the tables
# ------------------
# Now we join our two tables and check how well the ``InterpolationJoin`` can reconstruct the matching rows that are missing from the right side table.
# To avoid clashes in the column names, we use the ``suffix`` parameter to append "predicted" to the right side table column names.

from skrub import InterpolationJoin

joiner = InterpolationJoin(
    right_table,
    on=["LATITUDE", "LONGITUDE", "YEAR/MONTH/DAY"],
    suffix="_predicted",
).fit()
join = joiner.transform(left_table)
join.head()

######################################################################
# Comparing the estimated values to the ground truth
# --------------------------------------------------

from matplotlib import pyplot as plt

join = join.sample(2000, random_state=0, ignore_index=True)
fig, axes = plt.subplots(
    3,
    1,
    figsize=(5, 9),
    gridspec_kw={"height_ratios": [1.0, 0.5, 0.5]},
    layout="compressed",
)
for ax, col in zip(axes.ravel(), ["TMAX", "PRCP", "SNOW"]):
    ax.scatter(
        join[col].values,
        join[f"{col}_predicted"].values,
        alpha=0.1,
    )
    ax.set_aspect(1)
    ax.set_xlabel(f"true {col}")
    ax.set_ylabel(f"predicted {col}")

######################################################################
# We see that in this case the interpolation join works well for the temperature, but not precipitation nor snow.
# So we will only add the temperature to our flights table.

right_table = right_table.drop(["PRCP", "SNOW"], axis=1)

######################################################################
# Loading the flights table
# -------------------------
# We load the flights table and join it to the airports table using the flights’ "Origin" which refers to the departure airport’s IATA code.
# We use only a subset to speed up the example.

flights = fetch_figshare("41771418").X[["Year_Month_DayofMonth", "Origin", "ArrDelay"]]
flights = flights.sample(20_000, random_state=0, ignore_index=True)
airports = fetch_figshare("41710257").X.loc[
    :, ["iata", "airport", "state", "lat", "long"]
]
flights = flights.merge(airports, left_on="Origin", right_on="iata")
# printing the first row is more readable than the head() when we have many columns
flights.iloc[0]

######################################################################
# Joining the flights and weather data
# ------------------------------------
# As before, we initialize our join transformer with the weather table.
# Then, we use it to transform the flights table -- it adds a "TMAX" column containing the predicted maximum daily temperature.
#

joiner = InterpolationJoin(
    right_table,
    left_on=["lat", "long", "Year_Month_DayofMonth"],
    right_on=["LATITUDE", "LONGITUDE", "YEAR/MONTH/DAY"],
)
join = joiner.fit_transform(flights)
join.head()

######################################################################
# Sanity checks
# -------------
# This time we do not have a ground truth for the temperatures.
# We can perform a few basic sanity checks.

state_temperatures = join.groupby("state")["TMAX"].mean().sort_values()

######################################################################
# states with the lowest average predicted temperatures: Alaska, Montana, North Dakota, Washington, Minnesota
state_temperatures.head()

######################################################################
# states with the highest predicted temperatures: Puerto Rico, Virgin Islands, Hawaii, Florida, Louisiana
state_temperatures.tail()

######################################################################
# Higher latitudes (farther up north) are colder -- the airports in this dataset are in the United States.
fig, ax = plt.subplots()
ax.scatter(join["lat"], join["TMAX"])
ax.set_xlabel("Latitude (higher is farther north)")
ax.set_ylabel("TMAX")

######################################################################
# Winter months are colder than spring -- in the north hemisphere January is colder than April
#

import seaborn as sns

join["month"] = join["Year_Month_DayofMonth"].dt.strftime("%m %B")
plt.figure()
sns.barplot(data=join.sort_values(by="month"), y="month", x="TMAX")

######################################################################
# Of course these checks do not guarantee that the inferred values in our ``join`` table’s ``TMAX`` column are accurate.
# But at least the ``InterpolationJoin`` seems to have learned a few reasonable trends from its training table.


######################################################################
# Conclusion
# ----------
# We have seen how to fit an ``InterpolationJoin`` transformer: we give it a table (the weather data) and a set of matching columns (here date, latitude, longitude) and it learns to predict the other columns’ values  (such as the max daily temperature).
# Then, it transforms tables by *predicting* values that a matching row would contain, rather than by searching for an actual match.
# It is a generalization of the ``fuzzy_join``, as ``fuzzy_join`` is the same thing as an ``InterpolationJoin`` where the estimators are 1-nearest-neighbor estimators.