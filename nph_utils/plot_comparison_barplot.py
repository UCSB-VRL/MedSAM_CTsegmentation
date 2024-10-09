import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('/data/home/umang/Vader_umang/Seg_models/MedSAM/files_store/Plot_sheet.csv')

# Convert date strings to datetime objects
df['Date of Scan'] = pd.to_datetime(df['Date of Scan'], format='%m/%d/%Y')

# Function to get pre and latest post scans for each subject
def get_pre_post_data(subject_data):
    # Sort by date
    #subject_data = subject_data.sort_values('Date of Scan')
    # Find the pre entry (latest one)
    pre_entries = subject_data[subject_data['file,'].str.contains('Pre', na=False)]
    if pre_entries.empty:
        return None, None
    # Get the latest pre-entry by sorting and selecting the last one
    pre_entry = pre_entries.sort_values('Date of Scan').iloc[-1]
    
    # Find the post entry (earliest one)
    post_entries = subject_data[subject_data['Month difference after surgery'].notna()]
    if post_entries.empty:
        return pre_entry, None
    # Get the earliest post-entry by sorting and selecting the first one
    post_entry = post_entries.sort_values('Date of Scan').iloc[0]
    return pre_entry, post_entry

# Extract unique subject IDs
subject_ids = df['Sub ID'].dropna().unique()

# Prepare data for plotting
comparison_data = []

for subject_id in subject_ids:
    subject_data = df[df['Sub ID'] == subject_id]
    pre_entry, post_entry = get_pre_post_data(subject_data)
    
    if pre_entry is not None and post_entry is not None:
        dict ={
            'Subject ID': subject_id,
            'Pre UNet': pre_entry['7ventr_vol_unet'] / 1000,  # Convert to cc
            'Pre MedSAM': pre_entry['7ventr_vol_medsam'] / 1000,  # Convert to cc
            'Post UNet': post_entry['7ventr_vol_unet'] / 1000,  # Convert to cc
            'Post MedSAM': post_entry['7ventr_vol_medsam'] / 1000,  # Convert to cc
            'Months After Surgery': post_entry['Month difference after surgery']
        }
        print(dict)
        comparison_data.append(dict)
    else:
        print(f"Warning: Incomplete data for Subject ID {subject_id}")

comparison_df = pd.DataFrame(comparison_data)

def create_pre_post_comparison_plot(data, title, filename):
    if data.empty:
        print(f"No data available for {title}. Skipping this plot.")
        return
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    x = np.arange(len(data))  # Position of the subjects on the x-axis
    width = 0.2  # Width of each bar

    # Plotting the bars side by side for Pre and Post of both models
    ax.bar(x - width, data['Pre UNet'], width, label='Pre UNet', color='lightblue', alpha=0.7)
    ax.bar(x, data['Post UNet'], width, label='Post UNet', color='lightcoral', alpha=0.7)
    ax.bar(x + width, data['Pre MedSAM'], width, label='Pre MedSAM', color='blue', alpha=0.7)
    ax.bar(x + 2*width, data['Post MedSAM'], width, label='Post MedSAM', color='red', alpha=0.7)
    
    # Set labels and title
    ax.set_ylabel('Ventricle Volume (cc)')
    ax.set_title(title)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(
        [f"Subject {row['Subject ID']}\n{row['Months After Surgery']:.1f} months" for _, row in data.iterrows()],
        rotation=45, ha='right'
    )
    ax.legend()

    # Adding volume reduction on top of the bars
    for i, row in data.iterrows():
        # Calculate the reduction in volumes
        unet_reduction = -1*(row['Pre UNet'] - row['Post UNet'])
        medsam_reduction = -1*(row['Pre MedSAM'] - row['Post MedSAM'])
        
        # Display the reduction for UNet
        ax.text(i - width, row['Pre UNet'] + 0.05, f"{unet_reduction:.2f}", ha='center', va='bottom', fontsize=10)
        
        # Display the reduction for MedSAM
        ax.text(i + 2*width, row['Pre MedSAM'] + 0.05, f"{medsam_reduction:.2f}", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Create the plot with reductions displayed
create_pre_post_comparison_plot(comparison_df, 'Pre vs Post Surgery Ventricle Volumes: UNet vs MedSAM', 'pre_post_volumes_comparison.png')

print("Comparison plot has been saved as 'pre_post_volumes_comparison.png'.")

# Calculate and print average differences
avg_diff_unet = (comparison_df['Post UNet'] - comparison_df['Pre UNet']).mean()
avg_diff_medsam = (comparison_df['Post MedSAM'] - comparison_df['Pre MedSAM']).mean()

print(f"\nAverage volume difference (Post - Pre):")
print(f"UNet: {avg_diff_unet:.2f} cc")
print(f"MedSAM: {avg_diff_medsam:.2f} cc")

# Paired t-test for UNet and MedSAM differences
from scipy import stats

unet_diff = comparison_df['Post UNet'] - comparison_df['Pre UNet']
medsam_diff = comparison_df['Post MedSAM'] - comparison_df['Pre MedSAM']

t_statistic, p_value = stats.ttest_rel(unet_diff, medsam_diff)

print(f"\nPaired t-test results:")
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")