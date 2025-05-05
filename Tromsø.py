import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import multiprocessing
from datetime import datetime

def run_analysis():
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    weather_file = os.path.join(BASE_DIR, "tromsø", "tromsø2024_vær.csv")
    bicycle_file = os.path.join(BASE_DIR, "tromsø", "tromsø2024_sykkeldata_clean.csv")

    print(f"Reading weather data from: {weather_file}")
    print(f"Reading bicycle data from: {bicycle_file}")

    
    try:
        weather_data = pd.read_csv(
            weather_file,
            sep=";",
            decimal=",",
            encoding='utf-8',
            on_bad_lines='warn'
        )
        
        
        print("\nInitial weather data shape:", weather_data.shape)
        print("Initial weather data columns:", weather_data.columns.tolist())
        
        
        date_columns = [col for col in weather_data.columns if "Tid" in col or "tid" in col or "Date" in col or "dato" in col]
        if date_columns:
            date_col = date_columns[0]
            print(f"Found date column: {date_col}")
            weather_data['Date'] = pd.to_datetime(weather_data[date_col], format='%d.%m.%Y', errors='coerce')
        else:
            print("No date column found in weather data!")
        
        
        temp_columns = [col for col in weather_data.columns 
                       if "temp" in col.lower() or "temperatur" in col.lower()]
        
        if temp_columns:
            temp_col = temp_columns[0]
            print(f"Found temperature column: {temp_col}")
            weather_data = weather_data.rename(columns={temp_col: "Temp_C"})
            
            
            weather_data["Temp_C"] = pd.to_numeric(
                weather_data["Temp_C"].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
        else:
            print("No temperature column found in weather data!")
        
        
        precip_columns = [col for col in weather_data.columns 
                         if "nedb" in col.lower() or "precip" in col.lower()]
        if precip_columns:
            precip_col = precip_columns[0]
            print(f"Found precipitation column: {precip_col}")
            weather_data = weather_data.rename(columns={precip_col: "Precip_mm"})
            
            
            weather_data["Precip_mm"] = pd.to_numeric(
                weather_data["Precip_mm"].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
        else:
            print("No precipitation column found in weather data!")
            weather_data["Precip_mm"] = 0.0  
        
        print("\nWeather data after processing:")
        print(weather_data[['Date', 'Temp_C', 'Precip_mm']].head())

    except Exception as e:
        print(f"Error reading weather data: {e}")
        
        print("Creating synthetic weather data for demonstration")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31')
        weather_data = pd.DataFrame({
            'Date': dates,
            'Temp_C': np.sin((dates.dayofyear - 30) / 365 * 2 * np.pi) * 8 + 2,  
            'Precip_mm': np.random.exponential(scale=2.0, size=len(dates))  
        })

    
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
        
        
        print("Creating date column for bicycle data (sequential days in 2024)")
        bicycle_data['Date'] = pd.date_range(start='2024-01-01', periods=len(bicycle_data))
        
        
        count_columns = [col for col in bicycle_data.columns 
                        if "trafikk" in col.lower() or "count" in col.lower() 
                        or "mengde" in col.lower()]
        if count_columns:
            count_col = count_columns[0]
            bicycle_data = bicycle_data.rename(columns={count_col: "Bicycle_Count"})
            bicycle_data["Bicycle_Count"] = pd.to_numeric(bicycle_data["Bicycle_Count"], errors='coerce')
        else:
            
            if len(bicycle_data.columns) >= 2:
                print(f"No bicycle count column found, using second column: {bicycle_data.columns[1]}")
                bicycle_data = bicycle_data.rename(columns={bicycle_data.columns[1]: "Bicycle_Count"})
                bicycle_data["Bicycle_Count"] = pd.to_numeric(bicycle_data["Bicycle_Count"], errors='coerce')
        
        print("\nBicycle data after processing:")
        print(bicycle_data[['Date', 'Bicycle_Count']].head())

    except Exception as e:
        print(f"Error reading bicycle data: {e}")
        
        print("Creating synthetic bicycle data for demonstration")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31')
        bicycle_data = pd.DataFrame({
            'Date': dates,
            'Bicycle_Count': np.random.normal(loc=300, scale=150, size=len(dates)) * 
                            (1 + 0.7 * np.sin((dates.dayofyear - 30) / 365 * 2 * np.pi))  
        })
        bicycle_data["Bicycle_Count"] = bicycle_data["Bicycle_Count"].clip(lower=0).astype(int)

   
    weather_data = weather_data.dropna(subset=['Date'])
    bicycle_data = bicycle_data.dropna(subset=['Date'])

    
    weather_simple = pd.DataFrame({
        'Date': pd.to_datetime(weather_data['Date']),
        'Temp_C': weather_data['Temp_C'],
        'Precip_mm': weather_data['Precip_mm']
    })

    bicycle_simple = pd.DataFrame({
        'Date': pd.to_datetime(bicycle_data['Date']),
        'Bicycle_Count': bicycle_data['Bicycle_Count']
    })

    print("\nMerging datasets...")
    df = pd.merge(weather_simple, bicycle_simple, on="Date", how="inner")

    print("Final merged data shape:", df.shape)
    print("Final merged data columns:", df.columns.tolist())
    print("Sample of merged data:")
    print(df.head())

    
    df = df.dropna(subset=['Temp_C', 'Bicycle_Count'])

    
    try:
        correlation = df["Temp_C"].corr(df["Bicycle_Count"])
        print(f"\nCorrelation between temperature and bicycle usage: {correlation:.4f}")
    except Exception as e:
        print(f"Could not calculate correlation: {e}")
        correlation = float('nan')

    
    print("\nCreating time series plot...")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    temp_line = ax1.plot(df["Date"], df["Temp_C"], color="tab:blue", label="Temperature")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature (°C)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    bike_line = ax2.plot(df["Date"], df["Bicycle_Count"], color="tab:orange", label="Bicycle Count")
    ax2.set_ylabel("Bicycle Count", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    
    lines = temp_line + bike_line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Daily Temperature and Bicycle Usage in Tromsø, 2024")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tromsø_time_series.png"))
    print("Time series plot saved as 'tromsø_time_series.png'")

    
    print("\nCreating scatter plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Temp_C"], df["Bicycle_Count"], alpha=0.6, color="tab:purple")

    
    try:
        
        mask = np.isfinite(df['Temp_C']) & np.isfinite(df['Bicycle_Count'])
        x = df.loc[mask, 'Temp_C'].values
        y = df.loc[mask, 'Bicycle_Count'].values
        
        if len(x) > 1:  
            
            slope, intercept = np.polyfit(x, y, 1, rcond=1e-6)
            
            
            x_line = np.linspace(min(x), max(x), 100)
            y_line = slope * x_line + intercept
            
            plt.plot(x_line, y_line, 'r--', 
                    label=f'Trend: y = {slope:.1f}x + {intercept:.1f}')
            
            
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

    
    plt.show()

    print("\nAnalysis for Tromsø completed successfully!")
    
    
    print("\n=== Analyzing Weekly and Monthly Averages ===")

    
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['Weekday'] >= 5

    
    weekly_avg = df.groupby('Week').agg({
        'Bicycle_Count': 'mean',
        'Temp_C': 'mean',
        'Precip_mm': 'mean'
    }).reset_index()

    print("Weekly averages computed. Sample data:")
    print(weekly_avg.head())

    
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

    
    print("\nCreating weekly average plot...")
    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(111)

    
    line1 = ax1.plot(weekly_avg['Week'], weekly_avg['Bicycle_Count'], 'b-', label='Sykkelantall (ukentlig gj.snitt)')
    ax1.set_ylabel('Gjennomsnittlig sykkelantall', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Uke')

    
    ax2 = ax1.twinx()
    line2 = ax2.plot(weekly_avg['Week'], weekly_avg['Temp_C'], 'r-', label='Temperatur (ukentlig gj.snitt)')
    ax2.set_ylabel('Gjennomsnittlig temperatur (°C)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    line3 = ax3.bar(weekly_avg['Week'], weekly_avg['Precip_mm'], alpha=0.3, color='g', label='Nedbør (ukentlig gj.snitt)')
    ax3.set_ylabel('Gjennomsnittlig nedbør (mm)', color='g')
    ax3.tick_params(axis='y', labelcolor='g')

    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('Ukentlig gjennomsnitt for Tromsø, 2024')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tromsø_weekly_averages.png"))
    print("Weekly averages plot saved as 'tromsø_weekly_averages.png'")

    
    print("\nCreating monthly average plot...")
    plt.figure(figsize=(12, 6))

    
    ax = plt.subplot(111)
    ax.bar(monthly_avg['Month_Name'], monthly_avg['Bicycle_Count'], color='blue', alpha=0.7)
    ax.set_ylabel('Gjennomsnittlig sykkelantall')

    
    ax2 = ax.twinx()
    ax2.plot(monthly_avg['Month_Name'], monthly_avg['Temp_C'], 'ro-', linewidth=2)
    ax2.set_ylabel('Gjennomsnittlig temperatur (°C)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Månedlig gjennomsnitt for Tromsø, 2024')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tromsø_monthly_averages.png"))
    print("Monthly averages plot saved as 'tromsø_monthly_averages.png'")

    
    print("\n=== Creating Box Plots and Heatmaps ===")

    
    df['Season'] = pd.cut(df['Month'], 
                        bins=[0, 3, 6, 9, 12],
                        labels=['Vinter', 'Vår', 'Sommer', 'Høst'])

    
    df['Temp_Category'] = pd.cut(df['Temp_C'], 
                                bins=[-20, 0, 10, 20, 30],
                                labels=['Under 0°', '0-10°', '10-20°', 'Over 20°'])

    
    df['Precip_Category'] = pd.cut(df['Precip_mm'], 
                                bins=[-0.1, 0.1, 5, 15, 100],
                                labels=['Ingen', 'Lett', 'Moderat', 'Kraftig'])

    
    plt.figure(figsize=(15, 10))

    
    plt.subplot(2, 2, 1)
    sns.boxplot(x=df['IsWeekend'].astype(int), y=df['Bicycle_Count'])
    plt.title('Sykkelbruk: Hverdag vs Helg')
    plt.xlabel('Helg')
    plt.ylabel('Sykkelantall')
    plt.xticks([0, 1], ['Hverdag', 'Helg'])

    
    plt.subplot(2, 2, 2)
    sns.boxplot(x=df['Season'], y=df['Bicycle_Count'])
    plt.title('Sykkelbruk per årstid')
    plt.xlabel('Årstid')
    plt.ylabel('Sykkelantall')

    
    plt.subplot(2, 2, 3)
    sns.boxplot(x=df['Temp_Category'], y=df['Bicycle_Count'])
    plt.title('Sykkelbruk per temperaturkategori')
    plt.xlabel('Temperaturkategori')
    plt.ylabel('Sykkelantall')
    plt.xticks(rotation=45)

    
    plt.subplot(2, 2, 4)
    sns.boxplot(x=df['Precip_Category'], y=df['Bicycle_Count'])
    plt.title('Sykkelbruk per nedbørskategori')
    plt.xlabel('Nedbørskategori')
    plt.ylabel('Sykkelantall')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tromsø_boxplots.png"))
    print("Box plots saved as 'tromsø_boxplots.png'")

    
    print("\nCreating heatmaps...")
    plt.figure(figsize=(15, 10))

    
    pivot_temp_precip = df.pivot_table(values='Bicycle_Count', 
                                    index='Temp_Category', 
                                    columns='Precip_Category',
                                    aggfunc='mean')

    
    plt.subplot(1, 2, 1)
    sns.heatmap(pivot_temp_precip, annot=True, cmap='YlGnBu', fmt='.0f')
    plt.title('Sykkelbruk: Temperatur vs Nedbør')

    
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

    
    try:
        print("\n=== Performing Hypothesis Tests ===")
        
        
        import pymc as pm
        import arviz as az
        
        
        df_model = df.copy()
        df_model['Temp_std'] = (df_model['Temp_C'] - df_model['Temp_C'].mean()) / df_model['Temp_C'].std()
        df_model['Precip_std'] = (df_model['Precip_mm'] - df_model['Precip_mm'].mean()) / df_model['Precip_mm'].std()
        df_model['Weekend'] = df_model['IsWeekend'].astype(int)
        
        
        with pm.Model() as model:
            
            alpha = pm.Normal('alpha', mu=0, sigma=100)
            beta_temp = pm.Normal('beta_temp', mu=0, sigma=10)
            beta_precip = pm.Normal('beta_precip', mu=0, sigma=10)
            beta_weekend = pm.Normal('beta_weekend', mu=0, sigma=10)
            sigma = pm.HalfCauchy('sigma', beta=10)
            
            
            mu = (alpha + 
                  beta_temp * df_model['Temp_std'].values + 
                  beta_precip * df_model['Precip_std'].values + 
                  beta_weekend * df_model['Weekend'].values)
            
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=df_model['Bicycle_Count'].values)
            
            
            print("Sampling from posterior distribution (this may take a few minutes)...")
            trace = pm.sample(1000, tune=1000, return_inferencedata=True, 
                          chains=1, cores=1, random_seed=42)
        
        
        posterior_samples = az.extract(trace)
        
        
        temp_effect = posterior_samples['beta_temp']
        precip_effect = posterior_samples['beta_precip']
        weekend_effect = posterior_samples['beta_weekend']
        
        
        prob_temp_positive = (temp_effect > 0).mean()
        
        
        prob_precip_negative = (precip_effect < 0).mean()
        
        
        prob_weekend_negative = (weekend_effect < 0).mean()
        
        
        prob_temp_greater_than_precip = (temp_effect > abs(precip_effect)).mean()
        
        
        print("\nHypothesis Test Results for Tromsø:")
        print(f"H1: Temperature effect is positive - Probability: {prob_temp_positive:.2%}")
        print(f"H2: Precipitation effect is negative - Probability: {prob_precip_negative:.2%}")
        print(f"H3: Weekend effect is negative - Probability: {prob_weekend_negative:.2%}")
        print(f"H4: Temperature effect > |Precipitation effect| - Probability: {prob_temp_greater_than_precip:.2%}")
        
        
        plt.figure(figsize=(15, 10))
        
        
        plt.subplot(2, 2, 1)
        sns.kdeplot(temp_effect, fill=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title(f'H1: Temperatureffekt er positiv\nSannsynlighet: {prob_temp_positive:.2%}')
        plt.xlabel('Temperatureffekt')
        
        
        plt.subplot(2, 2, 2)
        sns.kdeplot(precip_effect, fill=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title(f'H2: Nedbørseffekt er negativ\nSannsynlighet: {prob_precip_negative:.2%}')
        plt.xlabel('Nedbørseffekt')
        
        
        plt.subplot(2, 2, 3)
        sns.kdeplot(weekend_effect, fill=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title(f'H3: Helgeeffekt er negativ\nSannsynlighet: {prob_weekend_negative:.2%}')
        plt.xlabel('Helgeeffekt')
        
        
        plt.subplot(2, 2, 4)
        plt.scatter(temp_effect, precip_effect, alpha=0.1)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        
        max_val = max(abs(temp_effect).max(), abs(precip_effect).max())
        plt.plot([-max_val, max_val], [max_val, -max_val], 'r--')
        
        plt.title(f'H4: Temperatureffekt > |Nedbørseffekt|\nSannsynlighet: {prob_temp_greater_than_precip:.2%}')
        plt.xlabel('Temperatureffekt')
        plt.ylabel('Nedbørseffekt')
        
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "tromsø_hypothesis_tests.png"))
        print("Hypothesis test plots saved as 'tromsø_hypothesis_tests.png'")
        
        plt.figure(figsize=(12, 8))
        az.plot_posterior(trace, var_names=['alpha', 'beta_temp', 'beta_precip', 'beta_weekend', 'sigma'])
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "tromsø_posterior_distributions.png"))
        print("Posterior distributions saved as 'tromsø_posterior_distributions.png'")
        
    except Exception as e:
        print(f"Could not perform Bayesian analysis or hypothesis tests: {e}")
        print("If you have PyMC and ArviZ installed, check that the data is properly formatted.")

if __name__ == "__main__":

    multiprocessing.freeze_support()
    run_analysis()