## Visualizing Worldwide Energy Production Data from 2000-2024

**Project Description:** Worldwide energy production and consumption data year over year can be presented visually in multiple interesting and informative ways. The aim of this project is to tell a story with the data in visually pleasing charts and display skills in matplotlib and altair while doing so.

**Data Summary**: This [Kaggle](https://www.kaggle.com/datasets/elvisbui/renewable-energy-share-by-country-2000-2025) dataset contains energy production/consumption in TWh, population, and GDP for 309 countries and regions from 2000-2025. As much of the 2025 data is unpopulated or incomplete, the analysis and visualization was kept to 2000-2024. 

### Creating a Waffle Chart to Visualize Total Energy Production from 2000 to 2024

This visualization uses pywaffle, a library that partners with matplotlib to create waffle charts, which can help guide the human eye to more easily compare values from multiple categories and are generally more eye-catching than traditional stacked bar charts. While these graphs tend to shine with 3-4 categories, I thought to first look at what the chart looked like with all 9 categories of energy production from this data.

