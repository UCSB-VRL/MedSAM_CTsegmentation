import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Read the CSV file
df = pd.read_csv('/data/home/umang/Vader_umang/Seg_models/MedSAM/files_store/Plot_sheet.csv')

# Convert date strings to datetime objects
df['Date of Scan'] = pd.to_datetime(df['Date of Scan'], format='%m/%d/%Y')

# Function to get aligned data for each subject
def get_aligned_data(subject_data):
    # Sort by date
    subject_data = subject_data.sort_values('Date of Scan')
    
    # Find post entries with filled 'Month difference after surgery'
    post_entries = subject_data[subject_data['Month difference after surgery'].notna()]
    
    if post_entries.empty:
        return pd.DataFrame()  # No post entries, return empty DataFrame
    
    # Find the pre entry just before the first post entry
    first_post_date = post_entries['Date of Scan'].min()
    pre_entry = subject_data[
        (subject_data['Date of Scan'] < first_post_date) & 
        (subject_data['file,'].str.contains('Pre'))
    ].iloc[-1:] if not subject_data[subject_data['Date of Scan'] < first_post_date].empty else pd.DataFrame()
    
    if not pre_entry.empty:
        pre_entry['Months After Surgery'] = 0
    
    # Set 'Months After Surgery' for post entries
    post_entries['Months After Surgery'] = post_entries['Month difference after surgery']
    
    # Combine pre entry and post entries
    return pd.concat([pre_entry, post_entries])

# Extract unique subject IDs
subject_ids = df['Sub ID'].dropna().unique()
# Specify the subject IDs you are interested in
desired_subject_ids = ['8588','8644','5486']

# Filter the DataFrame to include only these subject IDs and extract unique subject IDs
#subject_ids = df[df['Sub ID'].isin(desired_subject_ids)]['Sub ID'].unique()

print("subject ids", subject_ids)
# Filter and align data for all subjects
aligned_data = pd.concat([get_aligned_data(df[df['Sub ID'] == subject_id]) for subject_id in subject_ids])

# Normalize volumes by 1000 to display in cc
aligned_data['7ventr_vol_unet_cc'] = aligned_data['7ventr_vol_unet'] / 1000
aligned_data['7ventr_vol_medsam_cc'] = aligned_data['7ventr_vol_medsam'] / 1000

# Function to plot data for each measurement type
def plot_data(column_name, title, filename, long_term_only=False):
    plt.figure(figsize=(15, 10))
    for subject_id in subject_ids:
        subject_data = aligned_data[aligned_data['Sub ID'] == subject_id]
        if not subject_data.empty:
            max_months = subject_data['Months After Surgery'].max()
            if (long_term_only and max_months > 20) or (not long_term_only and max_months <= 20):
                plt.plot(subject_data['Months After Surgery'], subject_data[column_name],
                         marker='o', linestyle='-', label=f'Subject {subject_id}', linewidth=3)
                
                for _, row in subject_data.iterrows():
                    label = 'Pre' if row['Months After Surgery'] == 0 else f"{row['Months After Surgery']}m"
                    plt.annotate(label, (row['Months After Surgery'], row[column_name]),
                                 textcoords="offset points", xytext=(0,10), ha='center', rotation=45)

    plt.title(title)
    plt.xlabel('Months After Surgery')
    plt.ylabel('Ventricle volume (cc)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axvline(x=0, color='r', linestyle='--', label='Surgery')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Plot for 7ventr_vol_unet (short-term follow-up, <= 20 months)
plot_data('7ventr_vol_unet_cc', '7ventr_vol_unet over time (Aligned, <= 20 months follow-up)', 'ventricle_volumes_unet_plot_aligned_short_term.png')

# Plot for 7ventr_vol_medsam (short-term follow-up, <= 20 months)
plot_data('7ventr_vol_medsam_cc', '7ventr_vol_medsam over time (Aligned, <= 20 months follow-up)', 'ventricle_volumes_medsam_plot_aligned_short_term.png')

# Plot for 7ventr_vol_unet (long-term follow-up, > 20 months)
plot_data('7ventr_vol_unet_cc', '7ventr_vol_unet over time (Aligned)', 'ventricle_volumes_unet_plot_aligned_long_term.png', long_term_only=True)

# Plot for 7ventr_vol_medsam (long-term follow-up, > 20 months)
plot_data('7ventr_vol_medsam_cc', '7ventr_vol_medsam over time (Aligned)', 'ventricle_volumes_medsam_plot_aligned_long_term.png', long_term_only=True)

print("Plots have been saved as 'ventricle_volumes_unet_plot_aligned_short_term.png', 'ventricle_volumes_medsam_plot_aligned_short_term.png', 'ventricle_volumes_unet_plot_aligned_long_term.png', and 'ventricle_volumes_medsam_plot_aligned_long_term.png'.")