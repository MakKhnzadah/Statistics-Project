import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pymc as pm
import arviz as az
import multiprocessing
import datetime
import warnings

warnings.filterwarnings('ignore')

def analyze_kristiansand_data():
    print("Starting analysis for Kristiansand...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weather_file = os.path.join(base_dir, "kristiansand", "krs2024_værdata (1).csv")
    bicycle_file = os.path.join(base_dir, "kristiansand", "krs2024_sykkeldata_clean.csv")
    
    print(f"Reading weather data from: {weather_file}")
    print(f"Reading bicycle data from: {bicycle_file}")
    
    
    weather_data = pd.read_csv(
        weather_file,
        sep=';',
        encoding='utf-8'
    )
    
    weather_data['Date'] = pd.to_datetime(weather_data['Tid(norsk normaltid)'], format='%d.%m.%Y')
    
    weather_data['Middeltemperatur (døgn)'] = weather_data['Middeltemperatur (døgn)'].astype(str)
    weather_data.loc[weather_data['Middeltemperatur (døgn)'] == '-', 'Middeltemperatur (døgn)'] = np.nan
    
    weather_data['Temp_C'] = weather_data['Middeltemperatur (døgn)'].str.replace(',', '.').astype(float)
    weather_data['Precip_mm'] = weather_data['Nedbør (døgn)'].astype(str).str.replace(',', '.').astype(float)
    
    weather_df = weather_data[['Date', 'Temp_C', 'Precip_mm']].copy()
    weather_df['Precip_mm'] = weather_df['Precip_mm'].fillna(0) 
    weather_df['Temp_C'] = weather_df['Temp_C'].interpolate(method='linear')  
    
    print(f"Weather data shape: {weather_df.shape}")
    print(f"Missing temperature values after processing: {weather_df['Temp_C'].isna().sum()}")
    print(f"Missing precipitation values: {weather_df['Precip_mm'].isna().sum()}")
    print("Weather data sample:")
    print(weather_df.head())
    
    bicycle_data = pd.read_csv(
        bicycle_file,
        sep=';',
        encoding='utf-8'
    )
    
    bicycle_data['Date'] = pd.to_datetime(bicycle_data['Dato'], format='%d.%m.%Y')
    bicycle_data['Bicycle_Count'] = pd.to_numeric(bicycle_data['Trafikkmengde_num'])
    
    bicycle_df = bicycle_data[['Date', 'Bicycle_Count']].copy()
    
    print(f"Bicycle data shape: {bicycle_df.shape}")
    print(f"Missing bicycle count values: {bicycle_df['Bicycle_Count'].isna().sum()}")
    print("Bicycle data sample:")
    print(bicycle_df.head())
    
    merged_df = pd.merge(weather_df, bicycle_df, on='Date', how='inner')
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Final dataset date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    
    merged_df['Month'] = merged_df['Date'].dt.month
    merged_df['Month_Name'] = merged_df['Date'].dt.month_name()
    merged_df['Week'] = merged_df['Date'].dt.isocalendar().week
    merged_df['Weekday'] = merged_df['Date'].dt.dayofweek
    merged_df['IsWeekend'] = merged_df['Weekday'] >= 5
    merged_df['Season'] = pd.cut(
        merged_df['Month'], 
        bins=[0, 3, 6, 9, 12],
        labels=['Vinter', 'Vår', 'Sommer', 'Høst']
    )
    
    
    print("Creating time series plot...")
    plt.figure(figsize=(14, 7))
    ax1 = plt.subplot(111)
    
    
    temp_line = ax1.plot(merged_df['Date'], merged_df['Temp_C'], color='tab:blue', linewidth=2, label='Temperatur (°C)')
    ax1.set_xlabel('Dato')
    ax1.set_ylabel('Temperatur (°C)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    
    ax2 = ax1.twinx()
    bike_line = ax2.plot(merged_df['Date'], merged_df['Bicycle_Count'], color='tab:orange', linewidth=1.5, label='Sykkelantall')
    ax2.set_ylabel('Sykkelantall', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
  
    lines = temp_line + bike_line
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Daglig temperatur og sykkelbruk i Kristiansand, 2024')
    plt.tight_layout()
    plt.savefig('kristiansand_time_series.png', dpi=300)
    plt.close()
    print("Time series plot saved")
    
    
    print("Creating scatter plot...")
    plt.figure(figsize=(10, 8))
    plt.scatter(merged_df['Temp_C'], merged_df['Bicycle_Count'], alpha=0.6, color='tab:orange')
    
    z = np.polyfit(merged_df['Temp_C'], merged_df['Bicycle_Count'], 1)
    p = np.poly1d(z)
    plt.plot(sorted(merged_df['Temp_C']), p(sorted(merged_df['Temp_C'])), 'r--', linewidth=2, 
             label=f'Trend linje: y = {z[0]:.1f}x + {z[1]:.1f}')
    
    plt.xlabel('Temperatur (°C)')
    plt.ylabel('Sykkelantall')
    plt.title('Sykkelbruk vs. Temperatur i Kristiansand, 2024')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kristiansand_scatter.png', dpi=300)
    plt.close()
    print("Scatter plot saved")
    
    
    print("Creating weekly averages plot...")
    weekly_avg = merged_df.groupby('Week').agg({
        'Bicycle_Count': 'mean',
        'Temp_C': 'mean',
        'Precip_mm': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(15, 7))
    ax1 = plt.subplot(111)
    
    
    line1 = ax1.plot(weekly_avg['Week'], weekly_avg['Bicycle_Count'], 'b-', 
                    linewidth=2, label='Sykkelantall (ukentlig gj.snitt)')
    ax1.set_ylabel('Gjennomsnittlig sykkelantall', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Uke')
    
   
    ax2 = ax1.twinx()
    line2 = ax2.plot(weekly_avg['Week'], weekly_avg['Temp_C'], 'r-', 
                    linewidth=2, label='Temperatur (ukentlig gj.snitt)')
    ax2.set_ylabel('Gjennomsnittlig temperatur (°C)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.bar(weekly_avg['Week'], weekly_avg['Precip_mm'], alpha=0.2, color='g', 
           label='Nedbør (ukentlig gj.snitt)')
    ax3.set_ylabel('Gjennomsnittlig nedbør (mm)', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Ukentlig gjennomsnitt for Kristiansand, 2024')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('kristiansand_weekly_averages.png', dpi=300)
    plt.close()
    print("Weekly averages plot saved")
    
    
    print("Creating monthly averages plot...")
    
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Des'
    }
    
    monthly_avg = merged_df.groupby('Month').agg({
        'Bicycle_Count': 'mean',
        'Temp_C': 'mean',
        'Precip_mm': 'mean'
    }).reset_index()
    
    monthly_avg['Month_Name'] = monthly_avg['Month'].map(month_names)
    
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    
   
    ax.bar(monthly_avg['Month_Name'], monthly_avg['Bicycle_Count'], color='blue', alpha=0.7)
    ax.set_ylabel('Gjennomsnittlig sykkelantall')
    
   
    ax2 = ax.twinx()
    ax2.plot(monthly_avg['Month_Name'], monthly_avg['Temp_C'], 'ro-', linewidth=2)
    ax2.set_ylabel('Gjennomsnittlig temperatur (°C)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Månedlig gjennomsnitt for Kristiansand, 2024')
    plt.tight_layout()
    plt.savefig('kristiansand_monthly_averages.png', dpi=300)
    plt.close()
    print("Monthly averages plot saved")
    
    
    print("Creating box plots...")
    
    merged_df['Temp_Category'] = pd.cut(
        merged_df['Temp_C'],
        bins=[-20, 0, 10, 20, 30],
        labels=['Under 0°', '0-10°', '10-20°', 'Over 20°']
    )
    
    merged_df['Precip_Category'] = pd.cut(
        merged_df['Precip_mm'],
        bins=[-0.1, 0.1, 5, 15, 100],
        labels=['Ingen', 'Lett', 'Moderat', 'Kraftig']
    )
    
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(x=merged_df['IsWeekend'].astype(int), y=merged_df['Bicycle_Count'])
    plt.title('Sykkelbruk: Hverdag vs Helg')
    plt.xlabel('Helg')
    plt.ylabel('Sykkelantall')
    plt.xticks([0, 1], ['Hverdag', 'Helg'])
    
    
    plt.subplot(2, 2, 2)
    sns.boxplot(x=merged_df['Season'], y=merged_df['Bicycle_Count'])
    plt.title('Sykkelbruk per årstid')
    plt.xlabel('Årstid')
    plt.ylabel('Sykkelantall')
    
    
    plt.subplot(2, 2, 3)
    sns.boxplot(x=merged_df['Temp_Category'], y=merged_df['Bicycle_Count'])
    plt.title('Sykkelbruk per temperaturkategori')
    plt.xlabel('Temperaturkategori')
    plt.ylabel('Sykkelantall')
    
    
    plt.subplot(2, 2, 4)
    sns.boxplot(x=merged_df['Precip_Category'], y=merged_df['Bicycle_Count'])
    plt.title('Sykkelbruk per nedbørskategori')
    plt.xlabel('Nedbørskategori')
    plt.ylabel('Sykkelantall')
    
    plt.tight_layout()
    plt.savefig('kristiansand_boxplots.png', dpi=300)
    plt.close()
    print("Box plots saved")
    
   
    print("Creating heatmaps...")
    plt.figure(figsize=(16, 8))
    
    
    plt.subplot(1, 2, 1)
    temp_precip_pivot = merged_df.pivot_table(
        values='Bicycle_Count',
        index='Temp_Category',
        columns='Precip_Category',
        aggfunc='mean'
    )
    
    sns.heatmap(temp_precip_pivot, annot=True, cmap='YlGnBu', fmt='.0f')
    plt.title('Sykkelbruk: Temperatur vs Nedbør')
    
    
    plt.subplot(1, 2, 2)
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_names_no = ['Mandag', 'Tirsdag', 'Onsdag', 'Torsdag', 'Fredag', 'Lørdag', 'Søndag']
    weekday_map = {i: name for i, name in enumerate(weekday_names_no)}
    merged_df['Weekday_Name'] = merged_df['Weekday'].map(weekday_map)
    
    day_season_pivot = merged_df.pivot_table(
        values='Bicycle_Count',
        index='Weekday_Name',
        columns='Season',
        aggfunc='mean'
    )
    
   
    day_season_pivot = day_season_pivot.reindex(weekday_names_no)
    
    sns.heatmap(day_season_pivot, annot=True, cmap='YlGnBu', fmt='.0f')
    plt.title('Gjennomsnittlig sykkelbruk: Ukedag vs Årstid')
    
    plt.tight_layout()
    plt.savefig('kristiansand_heatmaps.png', dpi=300)
    plt.close()
    print("Heatmaps saved")
    
    
    print("Running Bayesian analysis...")
    try:
       
        merged_df_model = merged_df.copy()
        merged_df_model['Temp_std'] = (merged_df_model['Temp_C'] - merged_df_model['Temp_C'].mean()) / merged_df_model['Temp_C'].std()
        merged_df_model['Precip_std'] = (merged_df_model['Precip_mm'] - merged_df_model['Precip_mm'].mean()) / merged_df_model['Precip_mm'].std()
        merged_df_model['Weekend'] = merged_df_model['IsWeekend'].astype(int)
        
        
        with pm.Model() as model:
            
            alpha = pm.Normal('alpha', mu=0, sigma=100)
            beta_temp = pm.Normal('beta_temp', mu=0, sigma=10)
            beta_precip = pm.Normal('beta_precip', mu=0, sigma=10)
            beta_weekend = pm.Normal('beta_weekend', mu=0, sigma=10)
            sigma = pm.HalfCauchy('sigma', beta=10)
            
            
            mu = (alpha + 
                  beta_temp * merged_df_model['Temp_std'].values + 
                  beta_precip * merged_df_model['Precip_std'].values + 
                  beta_weekend * merged_df_model['Weekend'].values)
            
            
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=merged_df_model['Bicycle_Count'].values)
            
            
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
        
        
        print("\nHypothesis Test Results for Kristiansand:")
        print(f"H1: Temperatureffekt er positiv - Sannsynlighet: {prob_temp_positive:.2%}")
        print(f"H2: Nedbørseffekt er negativ - Sannsynlighet: {prob_precip_negative:.2%}")
        print(f"H3: Helgeeffekt er negativ - Sannsynlighet: {prob_weekend_negative:.2%}")
        print(f"H4: Temperatureffekt > |Nedbørseffekt| - Sannsynlighet: {prob_temp_greater_than_precip:.2%}")
        
        
        plt.figure(figsize=(16, 12))
        
        
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
        plt.savefig('kristiansand_hypothesis_tests.png', dpi=300)
        plt.close()
        print("Hypothesis tests plot saved")
        
        plt.figure(figsize=(12, 8))
        az.plot_posterior(trace, var_names=['alpha', 'beta_temp', 'beta_precip', 'beta_weekend', 'sigma'])
        plt.tight_layout()
        plt.savefig('kristiansand_posterior_distributions.png', dpi=300)
        plt.close()
        print("Posterior distributions plot saved")
        
    except Exception as e:
        print(f"Error in Bayesian analysis: {e}")
        print("Skipping Bayesian analysis")
    
    print("All analyses completed for Kristiansand!")
    return True

if __name__ == "__main__":
    
    multiprocessing.freeze_support()
    analyze_kristiansand_data()
