import pandas as pd
import numpy as np

def calculate_trend_stats(group):
    # Calculate Standard Deviation on whole history
    full_std = group['Closing Rank'].std()
    
    # Identify 2024 data vs older data
    data_2024 = group[group['Year'] == 2024]
    data_past = group[group['Year'] < 2024]
    
    if not data_2024.empty and not data_past.empty:
        # Weighted Mean: 60% weight to most recent year, 40% to history
        w_mean = (data_2024['Closing Rank'].mean() * 0.6) + (data_past['Closing Rank'].mean() * 0.4)
    else:
        # Fallback if only one set exists
        w_mean = group['Closing Rank'].mean()
        
    return pd.Series({
        'mean': w_mean,
        'std': full_std,
        'count': len(group),
        'NIRF_Rank': group['NIRF Ranking 2025'].iloc[0]
    })

def generate_weights():
    print("🚀 Starting Trend-Aware Engine (2026 Edition)...")
    try:
        # Loading your updated database
        df = pd.read_csv("master_database_updated.csv")
        
        # Clean headers to prevent mapping errors
        df.columns = [c.strip() for c in df.columns]

        # Ensure numeric types
        df['Closing Rank'] = pd.to_numeric(df['Closing Rank'], errors='coerce')
        df['NIRF Ranking 2025'] = pd.to_numeric(df['NIRF Ranking 2025'], errors='coerce').fillna(200)

        # Apply Weighted Trend Logic
        print("📊 Calculating weighted means (60% weight to 2024)...")
        stats = df.groupby(['Institute', 'Academic Program Name', 'Seat Type', 'Gender', 'Quota']).apply(calculate_trend_stats).reset_index()
        
        # Rename for application compatibility
        stats.rename(columns={'Academic Program Name': 'Branch', 'Seat Type': 'Category'}, inplace=True)
        
        # --- ZERO DIVISION & NAN PROTECTION ---
        # Fill NaN (single year data) and Replace 0 (static ranks) with 500
        stats['std'] = stats['std'].fillna(500).replace(0, 500)

        # Branch Priority Scoring
        def get_priority(branch):
            b = str(branch).lower()
            if 'computer' in b or 'cse' in b: return 1
            if 'electronics' in b or 'ece' in b: return 2
            if 'electrical' in b: return 3
            if 'mechanical' in b: return 4
            return 10

        stats['Priority'] = stats['Branch'].apply(get_priority)

        # Save weights
        stats.to_csv("trend_weights.csv", index=False)
        print(f"✅ Success! Generated weights for {len(stats)} unique branch combinations.")

    except Exception as e:
        print(f"❌ Error in engine: {e}")

if __name__ == "__main__":
    generate_weights()