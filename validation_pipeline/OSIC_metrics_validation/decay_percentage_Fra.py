import pandas as pd


# Read csv train file
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def main():
    file_path = r'X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\test.csv'
    df = read_csv(file_path)

    # For each patient calculate the decay percentage between baseline and 52 weeks 
    # For the value at week 52 take only in the range of [-8,+8] weeks from week 52
    # For the value at week 0 take the closest value to week 0
    # Use Percent column (FVC as % of predicted) instead of FVC
    decay_data = []
    patient_ids = df['Patient'].unique()
    
    for patient_id in patient_ids:
        patient_data = df[df['Patient'] == patient_id]
        
        # Get baseline value (closest to week 0) - using Percent
        baseline_data = patient_data.iloc[(patient_data['Weeks'] - 0).abs().argsort()[:1]]
        baseline_percent = baseline_data['Percent'].values[0]
        baseline_week = baseline_data['Weeks'].values[0]
        
        # Get week 52 value (within [-8, +8] weeks from week 52) - using Percent
        week_52_data = patient_data[(patient_data['Weeks'] >= 44) & (patient_data['Weeks'] <= 60)]
        if not week_52_data.empty:
            week_52_closest = week_52_data.iloc[(week_52_data['Weeks'] - 52).abs().argsort()[:1]]
            week_52_percent = week_52_closest['Percent'].values[0]
            week_52_week = week_52_closest['Weeks'].values[0]
        else:
            # Place the closest value to week 52 even if out of range
            week_52_closest = patient_data.iloc[(patient_data['Weeks'] - 52).abs().argsort()[:1]]
            week_52_percent = week_52_closest['Percent'].values[0]
            week_52_week = week_52_closest['Weeks'].values[0]
        
        # Get week 42 value (closest to week 42)
        week_42_data = patient_data.iloc[(patient_data['Weeks'] - 42).abs().argsort()[:1]]
        week_42_percent = week_42_data['Percent'].values[0]
        week_42_week = week_42_data['Weeks'].values[0]
        
        # Calculate decay percentage in Percent units
        # Percent is already normalized, so we calculate absolute difference
        percent_decline = baseline_percent - week_52_percent
        
        # Also calculate relative decline percentage
        if baseline_percent > 0:
            relative_decline_percentage = ((baseline_percent - week_52_percent) / baseline_percent) * 100
        else:
            relative_decline_percentage = None
        
        decay_data.append({
            'PatientID': patient_id,
            'BaselinePercent': baseline_percent,
            'BaselineWeek': baseline_week,
            'Week42Percent': week_42_percent,
            'Week42Week': week_42_week,
            'Week52Percent': week_52_percent,
            'Week52Week': week_52_week,
            'PercentDecline_absolute': percent_decline,  # Absolute decline in percentage points
            'PercentDecline_relative': relative_decline_percentage,  # Relative decline in %
            'PercentChange_percentage': ((week_52_percent - baseline_percent) / baseline_percent) * 100 if baseline_percent != 0 else None,
            'Drop_1_year': baseline_percent - week_52_percent  # Drop dopo 1 anno
        })

    decay_df = pd.DataFrame(decay_data)
    
    # From dataframe filter the patients outside the range of [-5,10] for baselineweek and [42,62] for week52week
    filtered_decay_df = decay_df[
        (decay_df['BaselineWeek'] >= -5) & (decay_df['BaselineWeek'] <= 10) &
        (decay_df['Week52Week'] >= 42) & (decay_df['Week52Week'] <= 62)
    ]
    
    # Count the patients that have both baselineweek and week52week in the specified ranges
    count_filtered_patients = filtered_decay_df.shape[0]
    print(f"Number of patients with both baseline week and week 52 week in the specified ranges: {count_filtered_patients}")
    
    # Print statistics
    print(f"\nStatistics for Percent decline:")
    print(f"Mean baseline Percent: {filtered_decay_df['BaselinePercent'].mean():.1f}%")
    print(f"Mean week 42 Percent: {filtered_decay_df['Week42Percent'].mean():.1f}%")
    print(f"Mean week 52 Percent: {filtered_decay_df['Week52Percent'].mean():.1f}%")
    print(f"Mean absolute decline: {filtered_decay_df['PercentDecline_absolute'].mean():.1f} percentage points")
    print(f"Mean relative decline: {filtered_decay_df['PercentDecline_relative'].mean():.1f}%")
    print(f"Mean 1-year drop: {filtered_decay_df['Drop_1_year'].mean():.1f} percentage points")
    
    # Create dataset based on filtered patients
    # Add progressed label based on absolute decline > 5 percentage points OR relative decline > 10%
    # Two different progression definitions:
    filtered_decay_df['has_progressed_absolute'] = filtered_decay_df['PercentDecline_absolute'] > 5  # >5% points decline
    filtered_decay_df['has_progressed_relative'] = filtered_decay_df['PercentDecline_relative'] > 10  # >10% relative decline
    filtered_decay_df['has_progressed_combined'] = (filtered_decay_df['PercentDecline_absolute'] > 5) & (filtered_decay_df['PercentDecline_relative'] > 10)
    
    # Count progression rates
    print(f"\nProgression rates:")
    print(f"Patients with >5% points decline: {filtered_decay_df['has_progressed_absolute'].sum()} ({filtered_decay_df['has_progressed_absolute'].sum()/len(filtered_decay_df)*100:.1f}%)")
    print(f"Patients with >10% relative decline: {filtered_decay_df['has_progressed_relative'].sum()} ({filtered_decay_df['has_progressed_relative'].sum()/len(filtered_decay_df)*100:.1f}%)")
    print(f"Patients with both: {filtered_decay_df['has_progressed_combined'].sum()} ({filtered_decay_df['has_progressed_combined'].sum()/len(filtered_decay_df)*100:.1f}%)")
    
    # Save to csv with progression labels - PRIMO OUTPUT
    filtered_decay_df.to_csv('percent_decay_progressed_label.csv', index=False)
    print(f"\nFile created: percent_decay_progressed_label.csv")
    
    # Add from train.csv: age, sex, smoking status
    train_df = read_csv(file_path)
    merged_df = pd.merge(
        filtered_decay_df, 
        train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates(), 
        left_on='PatientID', 
        right_on='Patient', 
        how='left'
    )
    merged_df = merged_df.drop(columns=['Patient'])
    
    # Save to csv with demographics
    merged_df.to_csv('percent_decay_progressed_label_with_demographics.csv', index=False)
    print(f"File created: percent_decay_progressed_label_with_demographics.csv")
    
    # SECONDO OUTPUT: CSV con paziente, FVCpercent(week0), FVCpercent(week42), FVCpercent(drop 1 year)
    fvc_percent_summary = filtered_decay_df[['PatientID', 'BaselinePercent', 'Week42Percent', 'Drop_1_year']].copy()
    fvc_percent_summary.columns = ['Patient', 'FVCpercent(week0)', 'FVCpercent(week42)', 'FVCpercent(drop 1 year)']
    fvc_percent_summary.to_csv('fvc_percent_summary.csv', index=False)
    
    print(f"File created: fvc_percent_summary.csv")
    print(f"\nFinal datasets saved:")
    print(f"1. percent_decay_progressed_label.csv - {len(filtered_decay_df)} patients with progression labels")
    print(f"2. percent_decay_progressed_label_with_demographics.csv - With demographics")
    print(f"3. fvc_percent_summary.csv - Patient, FVCpercent(week0), FVCpercent(week42), FVCpercent(drop 1 year)")

    return merged_df, fvc_percent_summary


if __name__ == "__main__":
    main()