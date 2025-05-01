# My Statistics Project

# Weather Effects on Bicycle Usage in Norwegian Cities

## Project Overview
This project analyzes the relationship between weather conditions (temperature and precipitation) and bicycle usage in three Norwegian cities: Bergen, Kristiansand, and Tromsø. Using Bayesian statistical methods, we investigate how temperature, precipitation, and day of week (weekday/weekend) affect cycling patterns across cities with different climatic profiles.

## Research Questions
1. How do temperature and precipitation affect bicycle usage in the three cities?
2. Are there differences in how weather affects cycling between the cities?
3. How does bicycle usage vary between weekdays and weekends?
4. To what extent can we predict bicycle usage based on weather forecasts?

## Data Sources
- Daily bicycle counts from counting stations in Bergen, Kristiansand, and Tromsø for 2024
- Weather data including temperature and precipitation from Meteorologisk institutt
- All data is organized by city in the respective folders

## Analysis Methods
The analysis employs several statistical techniques:

### Linear Regression and Visualization
- Time series plots of bicycle usage and weather variables
- Scatter plots with regression lines showing correlations between temperature and bicycle counts

### Gaussian Modeling and Time Aggregation
- Weekly and monthly averages of bicycle counts and weather data
- Seasonal patterns visualization and analysis

### Categorical Analysis
- Box plots showing distributions of bicycle usage across different categories:
  - Weekday vs weekend
  - Seasons
  - Temperature ranges
  - Precipitation levels
- Heatmaps highlighting interaction effects between:
  - Temperature and precipitation
  - Weekday and season

### Bayesian Statistical Analysis
- Bayesian regression modeling of factors influencing cycling
- Posterior probability distributions for model parameters
- Hypothesis tests evaluating specific claims about weather effects
- Quantification of uncertainty in parameter estimates

## Key Findings
- Temperature has a strong positive effect on bicycle usage in all three cities
- Precipitation has a negative effect, with varying strength across cities
- Bergen has the highest overall cycling numbers despite high rainfall
- Kristiansand shows the strongest weekday/weekend pattern, indicating commuter-focused cycling
- Tromsø demonstrates extreme seasonal variation with minimal winter cycling
- Weather effects are moderated by local cycling culture and infrastructure

## File Structure
- `Bergen.py`, `Kristiansand.py`, `Tromsø.py`: Analysis scripts for each city
- City folders (`/bergen`, `/kristiansand`, `/tromsø`): Raw data files
- Root directory: Generated visualization files and analysis outputs

## Visualizations
The project includes multiple visualization types:
- Time series plots (temperature, precipitation, bicycle counts)
- Scatter plots with regression lines
- Weekly and monthly average plots
- Box plots for categorical analysis
- Heatmaps showing interaction effects
- Posterior distribution plots
- Bayesian hypothesis test visualizations

## Requirements
- Python 3.10+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - pymc
  - arviz

## How to Run
1. Ensure all required libraries are installed
2. Run the individual city analysis scripts:
   ```
   python Bergen.py
   python Kristiansand.py
   python "Tromsø.py"
   ```
3. Review generated visualizations in the project root directory

## References
- Meteorologisk institutt for weather data
- "Statistikk - en bayesiansk tilnærming" for statistical methodology
