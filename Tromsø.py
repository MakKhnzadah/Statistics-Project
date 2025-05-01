import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import multiprocessing
from datetime import datetime

def run_analysis():
    # Load data (adjust path if needed)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    weather_file = os.path.join(BASE_DIR, "tromsø", "tromsø2024_vær.csv")
    bicycle_file = os.path.join(BASE_DIR, "tromsø", "tromsø2024_sykkeldata_clean.csv")

    print(f"Reading weather data from: {weather_file}")
    print(f"Reading bicycle data from: {bicycle_file}")

    # Read weather data with more careful handling
    try:
        weather_data = pd.read_csv(
            weather_file,
            sep=";",
            decimal=",",
            encoding='utf-8',
            on_bad_lines='warn'
        )
        
        # Print the shape and initial columns
        print("\nInitial weather data shape:", weather_data.shape)
        print("Initial weather data columns:", weather_data.columns.tolist())
        
        # Convert date column if it exists
        date_columns = [col for col in weather_data.columns if "Tid" in col or "tid" in col or "Date" in col or "dato" in col]
        if date_columns:
            date_col = date_columns[0]
            print(f"Found date column: {date_col}")
            weather_data['Date'] = pd.to_datetime(weather_data[date_col], format='%d.%m.%Y', errors='coerce')
        else:
            print("No date column found in weather data!")
        
        # Check for temperature column
        temp_columns = [col for col in weather_data.columns 
                       if "temp" in col.lower() or "temperatur" in col.lower()]
        
        if temp_columns:
            temp_col = temp_columns[0]
            print(f"Found temperature column: {temp_col}")
            weather_data = weather_data.rename(columns={temp_col: "Temp_C"})
            
            # Convert to numeric if needed
            weather_data["Temp_C"] = pd.to_numeric(
                weather_data["Temp_C"].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
        else:
            print("No temperature column found in weather data!")
        
        # Check for precipitation column
        precip_columns = [col for col in weather_data.columns 
                         if "nedb" in col.lower() or "precip" in col.lower()]
        if precip_columns:
            precip_col = precip_columns[0]
            print(f"Found precipitation column: {precip_col}")
            weather_data = weather_data.rename(columns={precip_col: "Precip_mm"})
            
            # Convert to numeric if needed
            weather_data["Precip_mm"] = pd.to_numeric(
                weather_data["Precip_mm"].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
        else:
            print("No precipitation column found in weather data!")
            weather_data["Precip_mm"] = 0.0  # Default value
        
        print("\nWeather data after processing:")
        print(weather_data[['Date', 'Temp_C', 'Precip_mm']].head())

    except Exception as e:
        print(f"Error reading weather data: {e}")
        # Create synthetic data for demonstration
        print("Creating synthetic weather data for demonstration")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31')
        weather_data = pd.DataFrame({
            'Date': dates,
            'Temp_C': np.sin((dates.dayofyear - 30) / 365 * 2 * np.pi) * 8 + 2,  # Synthetic seasonal pattern for Tromsø
            'Precip_mm': np.random.exponential(scale=2.0, size=len(dates))  # Random precipitation
        })

    # Read bicycle data
    try:
        print("\nReading bicycle data...")
        bicycle_data = pd.read_csv(
            bicycle_file,
            sep=";",
            encoding='utf-8',
            on_bad_lines='warn'
        )
        
        print("Bicycle data shape:", bicycle_data.shape)
        print("Bicycle data columns:", bicycle_data.columns.tolist())
        
        # Create sequential dates starting from January 1, 2024
        print("Creating date column for bicycle data (sequential days in 2024)")
        bicycle_data['Date'] = pd.date_range(start='2024-01-01', periods=len(bicycle_data))
        
        # Find bicycle count column
        count_columns = [col for col in bicycle_data.columns 
                        if "trafikk" in col.lower() or "count" in col.lower() 
                        or "mengde" in col.lower()]
        if count_columns:
            count_col = count_columns[0]
            bicycle_data = bicycle_data.rename(columns={count_col: "Bicycle_Count"})
            bicycle_data["Bicycle_Count"] = pd.to_numeric(bicycle_data["Bicycle_Count"], errors='coerce')
        else:
            # If no bicycle count column found, try to use the second column
            if len(bicycle_data.columns) >= 2:
                print(f"No bicycle count column found, using second column: {bicycle_data.columns[1]}")
                bicycle_data = bicycle_data.rename(columns={bicycle_data.columns[1]: "Bicycle_Count"})
                bicycle_data["Bicycle_Count"] = pd.to_numeric(bicycle_data["Bicycle_Count"], errors='coerce')
        
        print("\nBicycle data after processing:")
        print(bicycle_data[['Date', 'Bicycle_Count']].head())

    except Exception as e:
        print(f"Error reading bicycle data: {e}")
        # Create synthetic data for demonstration
        print("Creating synthetic bicycle data for demonstration")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31')
        bicycle_data = pd.DataFrame({
            'Date': dates,
            'Bicycle_Count': np.random.normal(loc=300, scale=150, size=len(dates)) * 
                            (1 + 0.7 * np.sin((dates.dayofyear - 30) / 365 * 2 * np.pi))  # Seasonal pattern
        })
        bicycle_data["Bicycle_Count"] = bicycle_data["Bicycle_Count"].clip(lower=0).astype(int)

    # Ensure there are no NaN/NaT values in the Date columns before merging
    weather_data = weather_data.dropna(subset=['Date'])
    bicycle_data = bicycle_data.dropna(subset=['Date'])

    # Create simplified datasets with only the necessary columns for merging
    weather_simple = pd.DataFrame({
        'Date': pd.to_datetime(weather_data['Date']),
        'Temp_C': weather_data['Temp_C'],
        'Precip_mm': weather_data['Precip_mm']
    })

    bicycle_simple = pd.DataFrame({
        'Date': pd.to_datetime(bicycle_data['Date']),
        'Bicycle_Count': bicycle_data['Bicycle_Count']
    })

    # Merge datasets on Date
    print("\nMerging datasets...")
    df = pd.merge(weather_simple, bicycle_simple, on="Date", how="inner")

    print("Final merged data shape:", df.shape)
    print("Final merged data columns:", df.columns.tolist())
    print("Sample of merged data:")
    print(df.head())

    # Clean up data before calculations (handle NaN, infinity, etc.)
    df = df.dropna(subset=['Temp_C', 'Bicycle_Count'])

    # Calculate correlation with error handling
    try:
        correlation = df["Temp_C"].corr(df["Bicycle_Count"])
        print(f"\nCorrelation between temperature and bicycle usage: {correlation:.4f}")
    except Exception as e:
        print(f"Could not calculate correlation: {e}")
        correlation = float('nan')

    # --- Plot 1: Time Series with Dual Axes ---
    print("\nCreating time series plot...")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot temperature
    temp_line = ax1.plot(df["Date"], df["Temp_C"], color="tab:blue", label="Temperature")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature (°C)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # Twin axis for bicycle count
    ax2 = ax1.twinx()
    bike_line = ax2.plot(df["Date"], df["Bicycle_Count"], color="tab:orange", label="Bicycle Count")
    ax2.set_ylabel("Bicycle Count", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    # Add combined legend
    lines = temp_line + bike_line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Daily Temperature and Bicycle Usage in Tromsø, 2024")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tromsø_time_series.png"))
    print("Time series plot saved as 'tromsø_time_series.png'")

    # --- Plot 2: Scatter Plot ---
    print("\nCreating scatter plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Temp_C"], df["Bicycle_Count"], alpha=0.6, color="tab:purple")

    # Add trend line with error handling
    try:
        # Filter out non-finite values
        mask = np.isfinite(df['Temp_C']) & np.isfinite(df['Bicycle_Count'])
        x = df.loc[mask, 'Temp_C'].values
        y = df.loc[mask, 'Bicycle_Count'].values
        
        if len(x) > 1:  # Need at least 2 points for a line
            # Use numpy's polyfit with explicit rcond value
            slope, intercept = np.polyfit(x, y, 1, rcond=1e-6)
            
            # Generate points for the line
            x_line = np.linspace(min(x), max(x), 100)
            y_line = slope * x_line + intercept
            
            plt.plot(x_line, y_line, 'r--', 
                    label=f'Trend: y = {slope:.1f}x + {intercept:.1f}')
            
            # Add R² value
            if not np.isnan(correlation):
                r_squared = correlation**2
                plt.text(0.05, 0.95, f'R² = {r_squared:.2f}', 
                        transform=plt.gca().transAxes)
        else:
            print("Not enough valid data points for trend line")
            plt.axhline(y=df['Bicycle_Count'].mean(), color='r', linestyle='--',
                       label=f"Mean: {df['Bicycle_Count'].mean():.1f}")
    except Exception as e:
        print(f"Error creating trend line: {e}")
        plt.axhline(y=df['Bicycle_Count'].mean(), color='r', linestyle='--',
                   label=f"Mean: {df['Bicycle_Count'].mean():.1f}")

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Bicycle Count")
    if not np.isnan(correlation):
        plt.title(f"Bicycle Usage vs Temperature in Tromsø, 2024 (r = {correlation:.2f})")
    else:
        plt.title("Bicycle Usage vs Temperature in Tromsø, 2024")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tromsø_scatter.png"))
    print("Scatter plot saved as 'tromsø_scatter.png'")

    # Show plots if running interactively
    plt.show()

    print("\nAnalysis for Tromsø completed successfully!")
    
    # ==== NEW CODE: WEEKLY AND MONTHLY ANALYSIS ====
    print("\n=== Analyzing Weekly and Monthly Averages ===")

    # Add week and month columns
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['Weekday'] >= 5

    # Create weekly averages
    weekly_avg = df.groupby('Week').agg({
        'Bicycle_Count': 'mean',
        'Temp_C': 'mean',
        'Precip_mm': 'mean'
    }).reset_index()

    print("Weekly averages computed. Sample data:")
    print(weekly_avg.head())

    # Create monthly averages
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mai', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Des'}
    df['Month_Name'] = df['Month'].map(month_names)

    monthly_avg = df.groupby('Month').agg({
        'Bicycle_Count': 'mean',
        'Temp_C': 'mean',
        'Precip_mm': 'mean'
    }).reset_index()
    monthly_avg['Month_Name'] = monthly_avg['Month'].map(month_names)

    print("Monthly averages computed. Sample data:")
    print(monthly_avg.head())

    # Visualize weekly averages
    print("\nCreating weekly average plot...")
    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(111)

    # Plot bicycle counts
    line1 = ax1.plot(weekly_avg['Week'], weekly_avg['Bicycle_Count'], 'b-', label='Sykkelantall (ukentlig gj.snitt)')
    ax1.set_ylabel('Gjennomsnittlig sykkelantall', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Uke')

    # Create second y-axis for temperature
    ax2 = ax1.twinx()
    line2 = ax2.plot(weekly_avg['Week'], weekly_avg['Temp_C'], 'r-', label='Temperatur (ukentlig gj.snitt)')
    ax2.set_ylabel('Gjennomsnittlig temperatur (°C)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add third y-axis for precipitation
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    line3 = ax3.bar(weekly_avg['Week'], weekly_avg['Precip_mm'], alpha=0.3, color='g', label='Nedbør (ukentlig gj.snitt)')
    ax3.set_ylabel('Gjennomsnittlig nedbør (mm)', color='g')
    ax3.tick_params(axis='y', labelcolor='g')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('Ukentlig gjennomsnitt for Tromsø, 2024')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tromsø_weekly_averages.png"))
    print("Weekly averages plot saved as 'tromsø_weekly_averages.png'")

    # Visualize monthly averages
    print("\nCreating monthly average plot...")
    plt.figure(figsize=(12, 6))

    # Create bar chart for monthly bicycle counts
    ax = plt.subplot(111)
    ax.bar(monthly_avg['Month_Name'], monthly_avg['Bicycle_Count'], color='blue', alpha=0.7)
    ax.set_ylabel('Gjennomsnittlig sykkelantall')

    # Add line for monthly temperature
    ax2 = ax.twinx()
    ax2.plot(monthly_avg['Month_Name'], monthly_avg['Temp_C'], 'ro-', linewidth=2)
    ax2.set_ylabel('Gjennomsnittlig temperatur (°C)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Månedlig gjennomsnitt for Tromsø, 2024')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tromsø_monthly_averages.png"))
    print("Monthly averages plot saved as 'tromsø_monthly_averages.png'")

    # ==== NEW CODE: BOX PLOTS AND HEATMAPS ====
    print("\n=== Creating Box Plots and Heatmaps ===")

    # Add season column
    df['Season'] = pd.cut(df['Month'], 
                        bins=[0, 3, 6, 9, 12],
                        labels=['Vinter', 'Vår', 'Sommer', 'Høst'])

    # Add temperature categories
    df['Temp_Category'] = pd.cut(df['Temp_C'], 
                                bins=[-20, 0, 10, 20, 30],
                                labels=['Under 0°', '0-10°', '10-20°', 'Over 20°'])

    # Add precipitation categories
    df['Precip_Category'] = pd.cut(df['Precip_mm'], 
                                bins=[-0.1, 0.1, 5, 15, 100],
                                labels=['Ingen', 'Lett', 'Moderat', 'Kraftig'])

    # Create boxplots
    plt.figure(figsize=(15, 10))

    # Boxplot by weekday/weekend
    plt.subplot(2, 2, 1)
    sns.boxplot(x=df['IsWeekend'].astype(int), y=df['Bicycle_Count'])
    plt.title('Sykkelbruk: Hverdag vs Helg')
    plt.xlabel('Helg')
    plt.ylabel('Sykkelantall')
    plt.xticks([0, 1], ['Hverdag', 'Helg'])

    # Boxplot by season
    plt.subplot(2, 2, 2)
    sns.boxplot(x=df['Season'], y=df['Bicycle_Count'])
    plt.title('Sykkelbruk per årstid')
    plt.xlabel('Årstid')
    plt.ylabel('Sykkelantall')

    # Boxplot by temperature category
    plt.subplot(2, 2, 3)
    sns.boxplot(x=df['Temp_Category'], y=df['Bicycle_Count'])
    plt.title('Sykkelbruk per temperaturkategori')
    plt.xlabel('Temperaturkategori')
    plt.ylabel('Sykkelantall')
    plt.xticks(rotation=45)

    # Boxplot by precipitation category
    plt.subplot(2, 2, 4)
    sns.boxplot(x=df['Precip_Category'], y=df['Bicycle_Count'])
    plt.title('Sykkelbruk per nedbørskategori')
    plt.xlabel('Nedbørskategori')
    plt.ylabel('Sykkelantall')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tromsø_boxplots.png"))
    print("Box plots saved as 'tromsø_boxplots.png'")

    # Create heatmaps
    print("\nCreating heatmaps...")
    plt.figure(figsize=(15, 10))

    # Create a pivot table for temperature vs precipitation
    pivot_temp_precip = df.pivot_table(values='Bicycle_Count', 
                                    index='Temp_Category', 
                                    columns='Precip_Category',
                                    aggfunc='mean')

    # Create heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(pivot_temp_precip, annot=True, cmap='YlGnBu', fmt='.0f')
    plt.title('Sykkelbruk: Temperatur vs Nedbør')

    # Create a pivot table for weekday vs season
    df['Weekday_Name'] = df['Date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_names_no = ['Mandag', 'Tirsdag', 'Onsdag', 'Torsdag', 'Fredag', 'Lørdag', 'Søndag']
    weekday_map = dict(zip(weekday_order, weekday_names_no))
    df['Weekday_Name_NO'] = df['Weekday_Name'].map(weekday_map)

    pivot_day_season = df.pivot_table(values='Bicycle_Count',
                                    index='Weekday_Name_NO',
                                    columns='Season',
                                    aggfunc='mean')
    pivot_day_season = pivot_day_season.reindex(weekday_names_no)

    plt.subplot(1, 2, 2)
    sns.heatmap(pivot_day_season, annot=True, cmap='YlGnBu', fmt='.0f')
    plt.title('Gjennomsnittlig sykkelbruk: Ukedag vs Årstid')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tromsø_heatmaps.png"))
    print("Heatmaps saved as 'tromsø_heatmaps.png'")

    # ==== NEW CODE: BAYESIAN HYPOTHESIS TESTING ====
    try:
        print("\n=== Performing Hypothesis Tests ===")
        
        # First, let's implement a simple Bayesian model if you haven't already
        import pymc as pm
        import arviz as az
        
        # Standardize variables for Bayesian analysis
        df_model = df.copy()
        df_model['Temp_std'] = (df_model['Temp_C'] - df_model['Temp_C'].mean()) / df_model['Temp_C'].std()
        df_model['Precip_std'] = (df_model['Precip_mm'] - df_model['Precip_mm'].mean()) / df_model['Precip_mm'].std()
        df_model['Weekend'] = df_model['IsWeekend'].astype(int)
        
        # Define and run the model
        with pm.Model() as model:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=100)
            beta_temp = pm.Normal('beta_temp', mu=0, sigma=10)
            beta_precip = pm.Normal('beta_precip', mu=0, sigma=10)
            beta_weekend = pm.Normal('beta_weekend', mu=0, sigma=10)
            sigma = pm.HalfCauchy('sigma', beta=10)
            
            # Expected value
            mu = (alpha + 
                  beta_temp * df_model['Temp_std'].values + 
                  beta_precip * df_model['Precip_std'].values + 
                  beta_weekend * df_model['Weekend'].values)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=df_model['Bicycle_Count'].values)
            
            # Sample with modified settings - use only 1 chain when not in the main guard
            print("Sampling from posterior distribution (this may take a few minutes)...")
            trace = pm.sample(1000, tune=1000, return_inferencedata=True, 
                          chains=1, cores=1, random_seed=42)
        
        # Extract posterior samples
        posterior_samples = az.extract(trace)
        
        # Hypothesis tests
        temp_effect = posterior_samples['beta_temp']
        precip_effect = posterior_samples['beta_precip']
        weekend_effect = posterior_samples['beta_weekend']
        
        # 1. Hypothesis test: Is temperature effect positive?
        prob_temp_positive = (temp_effect > 0).mean()
        
        # 2. Hypothesis test: Is precipitation effect negative?
        prob_precip_negative = (precip_effect < 0).mean()
        
        # 3. Hypothesis test: Is weekend effect negative?
        prob_weekend_negative = (weekend_effect < 0).mean()
        
        # 4. Hypothesis test: Is temperature effect greater than absolute value of precipitation effect?
        prob_temp_greater_than_precip = (temp_effect > abs(precip_effect)).mean()
        
        # Print hypothesis test results
        print("\nHypothesis Test Results for Tromsø:")
        print(f"H1: Temperature effect is positive - Probability: {prob_temp_positive:.2%}")
        print(f"H2: Precipitation effect is negative - Probability: {prob_precip_negative:.2%}")
        print(f"H3: Weekend effect is negative - Probability: {prob_weekend_negative:.2%}")
        print(f"H4: Temperature effect > |Precipitation effect| - Probability: {prob_temp_greater_than_precip:.2%}")
        
        # Create hypothesis test plots
        plt.figure(figsize=(15, 10))
        
        # Plot posterior distribution for temperature effect
        plt.subplot(2, 2, 1)
        sns.kdeplot(temp_effect, fill=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title(f'H1: Temperatureffekt er positiv\nSannsynlighet: {prob_temp_positive:.2%}')
        plt.xlabel('Temperatureffekt')
        
        # Plot posterior distribution for precipitation effect
        plt.subplot(2, 2, 2)
        sns.kdeplot(precip_effect, fill=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title(f'H2: Nedbørseffekt er negativ\nSannsynlighet: {prob_precip_negative:.2%}')
        plt.xlabel('Nedbørseffekt')
        
        # Plot posterior distribution for weekend effect
        plt.subplot(2, 2, 3)
        sns.kdeplot(weekend_effect, fill=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title(f'H3: Helgeeffekt er negativ\nSannsynlighet: {prob_weekend_negative:.2%}')
        plt.xlabel('Helgeeffekt')
        
        # Plot scatter of temperature effect vs precipitation effect
        plt.subplot(2, 2, 4)
        plt.scatter(temp_effect, precip_effect, alpha=0.1)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        # Add a diagonal line where |precip| = temp
        max_val = max(abs(temp_effect).max(), abs(precip_effect).max())
        plt.plot([-max_val, max_val], [max_val, -max_val], 'r--')
        
        plt.title(f'H4: Temperatureffekt > |Nedbørseffekt|\nSannsynlighet: {prob_temp_greater_than_precip:.2%}')
        plt.xlabel('Temperatureffekt')
        plt.ylabel('Nedbørseffekt')
        
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "tromsø_hypothesis_tests.png"))
        print("Hypothesis test plots saved as 'tromsø_hypothesis_tests.png'")
        
        # Also visualize the posterior distributions from the Bayesian analysis
        plt.figure(figsize=(12, 8))
        az.plot_posterior(trace, var_names=['alpha', 'beta_temp', 'beta_precip', 'beta_weekend', 'sigma'])
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "tromsø_posterior_distributions.png"))
        print("Posterior distributions saved as 'tromsø_posterior_distributions.png'")
        
    except Exception as e:
        print(f"Could not perform Bayesian analysis or hypothesis tests: {e}")
        print("If you have PyMC and ArviZ installed, check that the data is properly formatted.")

# Add the main guard to fix multiprocessing issue
if __name__ == "__main__":
    # Add the freeze_support for Windows
    multiprocessing.freeze_support()
    run_analysis()