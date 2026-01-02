#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Randomly select one true neighbor and one false neighbor,
and plot their last_seen vs timestamp over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
# SI_STEP2_FILE = 'si_step2_after_unique_filter_links_reverse_ge2p5h.csv'
# Use active_scanned_from_si_si_to_x_lt2p5h_unique.csv which contains both si->x and x->si data
SI_STEP2_FILE = 'dogecoin_8th_Dec/active_scanned_from_si_si_to_x_lt2p5h_unique.csv'
SI_CONNECTIONS_FILE = 'dogecoin_8th_Dec/si_connections_8_dec.csv'

# SI Node IP
SI_NODE_IP = '109.123.232.246'

print("=" * 60)
print("Plotting last_seen vs timestamp for all neighbors")
print("=" * 60)

# Step 1: Load true neighbors
print("\n" + "=" * 60)
print("Step 1: Loading true neighbors")
print("=" * 60)

si_df = pd.read_csv(SI_CONNECTIONS_FILE)
si_df['ip'] = si_df['addr'].str.split(':').str[0]
si_true_neighbors = set(si_df['ip'].unique())
print(f"Total SI true neighbors: {len(si_true_neighbors):,}")

# Step 2: Load step2 file and filter
print("\n" + "=" * 60)
print("Step 2: Loading and filtering step2 file")
print("=" * 60)

print(f"Reading {SI_STEP2_FILE}...")
chunk_size = 100000
step2_chunks = []

total_rows = 0
for chunk in pd.read_csv(SI_STEP2_FILE, chunksize=chunk_size):
    total_rows += len(chunk)
    
    # Convert timestamps to datetime
    chunk['timestamp_dt'] = pd.to_datetime(chunk['timestamp'], utc=True, errors='coerce')
    chunk['last_seen_dt'] = pd.to_datetime(chunk['last_seen'], utc=True, errors='coerce')
    
    # Remove records with invalid timestamps
    chunk = chunk[chunk['timestamp_dt'].notna() & chunk['last_seen_dt'].notna()].copy()
    
    # Calculate time difference (use delta column if exists, otherwise calculate)
    if 'delta' in chunk.columns:
        chunk['time_diff_sec'] = chunk['delta']
    elif 'time_diff_sec' in chunk.columns:
        pass  # Already exists
    else:
        chunk['time_diff_sec'] = (chunk['timestamp_dt'] - chunk['last_seen_dt']).dt.total_seconds()
    
    # Determine direction based on source_ip and destination_ip
    chunk['direction'] = 'unknown'
    chunk.loc[chunk['source_ip'] == SI_NODE_IP, 'direction'] = 'si->x'
    chunk.loc[chunk['destination_ip'] == SI_NODE_IP, 'direction'] = 'x->si'
    
    # Only keep records where SI node is involved
    chunk = chunk[chunk['direction'] != 'unknown'].copy()
    
    # For si->x direction: filter to only include records with time_diff <= 2.5 hours
    # For x->si direction: keep all records (no filter)
    threshold_2p5h = 2.5 * 3600  # 2.5 hours in seconds
    si_to_x_mask = chunk['direction'] == 'si->x'
    # Filter si->x: time_diff <= 2.5h (x->si: keep all, no filter)
    if si_to_x_mask.any():
        chunk.loc[si_to_x_mask & (chunk['time_diff_sec'] > threshold_2p5h), 'direction'] = 'unknown'
    chunk = chunk[chunk['direction'] != 'unknown'].copy()
    
    if len(chunk) > 0:
        step2_chunks.append(chunk)
    
    if total_rows % 1000000 == 0:
        print(f"  Processed {total_rows:,} rows...")

print(f"Total rows processed: {total_rows:,}")

# Combine chunks
if step2_chunks:
    step2_df = pd.concat(step2_chunks, ignore_index=True)
    print(f"Total records after filtering: {len(step2_df):,}")
else:
    step2_df = pd.DataFrame()
    print("No records found after filtering")

# Step 3: Extract neighbors for both directions
print("\n" + "=" * 60)
print("Step 3: Extracting neighbors for both directions")
print("=" * 60)

# Separate by direction
si_to_x_df = step2_df[step2_df['direction'] == 'si->x'].copy()
x_to_si_df = step2_df[step2_df['direction'] == 'x->si'].copy()

print(f"Records with direction si->x: {len(si_to_x_df):,}")
print(f"Records with direction x->si: {len(x_to_si_df):,}")

# Collect records for x->si direction (for scatter plots)
x_to_si_true_records = []
x_to_si_false_records = []

for idx, row in x_to_si_df.iterrows():
    neighbor_ip = row['source_ip']  # For x->si, neighbor is source_ip
    
    record = {
        'neighbor_ip': neighbor_ip,
        'timestamp_dt': row['timestamp_dt'],
        'last_seen_dt': row['last_seen_dt'],
        'time_diff_sec': row['time_diff_sec'] if 'time_diff_sec' in row.index else None
    }
    
    if neighbor_ip in si_true_neighbors:
        x_to_si_true_records.append(record)
    else:
        x_to_si_false_records.append(record)

# Collect records for si->x direction (for histogram)
si_to_x_true_records = []
si_to_x_false_records = []

for idx, row in si_to_x_df.iterrows():
    neighbor_ip = row['destination_ip']  # For si->x, neighbor is destination_ip
    
    record = {
        'neighbor_ip': neighbor_ip,
        'timestamp_dt': row['timestamp_dt'],
        'last_seen_dt': row['last_seen_dt'],
        'time_diff_sec': row['time_diff_sec'] if 'time_diff_sec' in row.index else None
    }
    
    if neighbor_ip in si_true_neighbors:
        si_to_x_true_records.append(record)
    else:
        si_to_x_false_records.append(record)

# Prepare data for si->x direction (for first scatter plot)
si_to_x_true_records_df = pd.DataFrame(si_to_x_true_records) if si_to_x_true_records else pd.DataFrame()
si_to_x_false_records_df = pd.DataFrame(si_to_x_false_records) if si_to_x_false_records else pd.DataFrame()

# Prepare data for x->si direction (for second scatter plot)
# Filter x->si records: only keep records with delta t > 2.5h
threshold_2p5h = 2.5 * 3600  # 2.5 hours in seconds

x_to_si_true_records_df = pd.DataFrame(x_to_si_true_records) if x_to_si_true_records else pd.DataFrame()
x_to_si_false_records_df = pd.DataFrame(x_to_si_false_records) if x_to_si_false_records else pd.DataFrame()

# Filter x->si records: delta t > 2.5h
if len(x_to_si_true_records_df) > 0:
    initial_count = len(x_to_si_true_records_df)
    x_to_si_true_records_df = x_to_si_true_records_df[
        (x_to_si_true_records_df['time_diff_sec'].notna()) & 
        (x_to_si_true_records_df['time_diff_sec'] > threshold_2p5h)
    ].copy()
    print(f"\nx->si true records: {initial_count:,} -> {len(x_to_si_true_records_df):,} (filtered to delta > 2.5h)")

if len(x_to_si_false_records_df) > 0:
    initial_count = len(x_to_si_false_records_df)
    x_to_si_false_records_df = x_to_si_false_records_df[
        (x_to_si_false_records_df['time_diff_sec'].notna()) & 
        (x_to_si_false_records_df['time_diff_sec'] > threshold_2p5h)
    ].copy()
    print(f"x->si false records: {initial_count:,} -> {len(x_to_si_false_records_df):,} (filtered to delta > 2.5h)")

# For first plot, use si->x data
true_df = si_to_x_true_records_df.copy()
false_df = si_to_x_false_records_df.copy()
direction_label = "si->x"

# Sort by timestamp
if len(true_df) > 0:
    true_df = true_df.sort_values('timestamp_dt').reset_index(drop=True)
if len(false_df) > 0:
    false_df = false_df.sort_values('timestamp_dt').reset_index(drop=True)

# Count unique neighbors
true_unique_neighbors = true_df['neighbor_ip'].nunique() if len(true_df) > 0 else 0
false_unique_neighbors = false_df['neighbor_ip'].nunique() if len(false_df) > 0 else 0

print(f"True neighbors in data: {true_unique_neighbors:,}")
print(f"  Total records: {len(true_df):,}")
if len(true_df) > 0:
    print(f"  Timestamp range: {true_df['timestamp_dt'].min()} to {true_df['timestamp_dt'].max()}")

print(f"\nFalse neighbors in data: {false_unique_neighbors:,}")
print(f"  Total records: {len(false_df):,}")
if len(false_df) > 0:
    print(f"  Timestamp range: {false_df['timestamp_dt'].min()} to {false_df['timestamp_dt'].max()}")

# Step 5: Create plots
print("\n" + "=" * 60)
print("Step 5: Creating plots")
print("=" * 60)

# Plot 1: Combined plot - both neighbors in one figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot all true neighbors
if len(true_df) > 0:
    ax.scatter(true_df['timestamp_dt'], true_df['last_seen_dt'], 
              alpha=0.5, s=30, color='green', edgecolors='none', linewidth=0.3,
              label=f'True Neighbors (n={true_unique_neighbors} neighbors, {len(true_df)} records)')

# Plot all false neighbors
if len(false_df) > 0:
    ax.scatter(false_df['timestamp_dt'], false_df['last_seen_dt'], 
              alpha=0.5, s=30, color='red', edgecolors='none', linewidth=0.3,
              label=f'False Neighbors (n={false_unique_neighbors} neighbors, {len(false_df)} records)')

# Add diagonal reference line (last_seen = timestamp, i.e., time_diff = 0)
all_timestamps_list = []
all_last_seens_list = []
if len(true_df) > 0:
    all_timestamps_list.append(true_df['timestamp_dt'])
    all_last_seens_list.append(true_df['last_seen_dt'])
if len(false_df) > 0:
    all_timestamps_list.append(false_df['timestamp_dt'])
    all_last_seens_list.append(false_df['last_seen_dt'])

if all_timestamps_list:
    all_timestamps = pd.concat(all_timestamps_list)
    all_last_seens = pd.concat(all_last_seens_list)
    min_time = min(all_timestamps.min(), all_last_seens.min())
    max_time = max(all_timestamps.max(), all_last_seens.max())
    ax.plot([min_time, max_time], [min_time, max_time], 'k--', linewidth=1.5, alpha=0.5, label='last_seen = timestamp')

ax.set_xlabel('Timestamp', fontsize=12)
ax.set_ylabel('Last Seen', fontsize=12)
ax.set_title(f'Last Seen vs Timestamp: {direction_label} Direction\n({true_unique_neighbors} true, {false_unique_neighbors} false)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
output_file = 'last_seen_vs_timestamp_sample_neighbors.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {output_file}")

# Plot 1b: x->si direction Last Seen vs Timestamp (if x->si data exists)
print(f"\nChecking for x->si data to create separate plot...")
print(f"  x->si true records: {len(x_to_si_true_records_df):,}")
print(f"  x->si false records: {len(x_to_si_false_records_df):,}")

if len(x_to_si_true_records_df) > 0 or len(x_to_si_false_records_df) > 0:
    x_to_si_true_df_plot = x_to_si_true_records_df.copy()
    x_to_si_false_df_plot = x_to_si_false_records_df.copy()
    
    # Sort by timestamp
    if len(x_to_si_true_df_plot) > 0:
        x_to_si_true_df_plot = x_to_si_true_df_plot.sort_values('timestamp_dt').reset_index(drop=True)
    if len(x_to_si_false_df_plot) > 0:
        x_to_si_false_df_plot = x_to_si_false_df_plot.sort_values('timestamp_dt').reset_index(drop=True)
    
    x_to_si_true_unique = x_to_si_true_df_plot['neighbor_ip'].nunique() if len(x_to_si_true_df_plot) > 0 else 0
    x_to_si_false_unique = x_to_si_false_df_plot['neighbor_ip'].nunique() if len(x_to_si_false_df_plot) > 0 else 0
    
    fig_x_to_si, ax_x_to_si = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot all true neighbors
    if len(x_to_si_true_df_plot) > 0:
        ax_x_to_si.scatter(x_to_si_true_df_plot['timestamp_dt'], x_to_si_true_df_plot['last_seen_dt'], 
                  alpha=0.5, s=30, color='green', edgecolors='none', linewidth=0.3,
                  label=f'True Neighbors (n={x_to_si_true_unique} neighbors, {len(x_to_si_true_df_plot)} records)')
    
    # Plot all false neighbors
    if len(x_to_si_false_df_plot) > 0:
        ax_x_to_si.scatter(x_to_si_false_df_plot['timestamp_dt'], x_to_si_false_df_plot['last_seen_dt'], 
                  alpha=0.5, s=30, color='red', edgecolors='none', linewidth=0.3,
                  label=f'False Neighbors (n={x_to_si_false_unique} neighbors, {len(x_to_si_false_df_plot)} records)')
    
    # Add diagonal reference line
    all_timestamps_x_to_si = []
    all_last_seens_x_to_si = []
    if len(x_to_si_true_df_plot) > 0:
        all_timestamps_x_to_si.append(x_to_si_true_df_plot['timestamp_dt'])
        all_last_seens_x_to_si.append(x_to_si_true_df_plot['last_seen_dt'])
    if len(x_to_si_false_df_plot) > 0:
        all_timestamps_x_to_si.append(x_to_si_false_df_plot['timestamp_dt'])
        all_last_seens_x_to_si.append(x_to_si_false_df_plot['last_seen_dt'])
    
    if all_timestamps_x_to_si:
        all_timestamps_combined = pd.concat(all_timestamps_x_to_si)
        all_last_seens_combined = pd.concat(all_last_seens_x_to_si)
        min_time = min(all_timestamps_combined.min(), all_last_seens_combined.min())
        max_time = max(all_timestamps_combined.max(), all_last_seens_combined.max())
        ax_x_to_si.plot([min_time, max_time], [min_time, max_time], 'k--', linewidth=1.5, alpha=0.5, label='last_seen = timestamp')
    
    ax_x_to_si.set_xlabel('Timestamp', fontsize=12)
    ax_x_to_si.set_ylabel('Last Seen', fontsize=12)
    ax_x_to_si.set_title(f'Last Seen vs Timestamp: x->si Direction (delta > 2.5h)\n({x_to_si_true_unique} true, {x_to_si_false_unique} false)', fontsize=14, fontweight='bold')
    ax_x_to_si.grid(True, alpha=0.3)
    ax_x_to_si.tick_params(axis='x', rotation=45)
    ax_x_to_si.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    output_file_x_to_si = 'last_seen_vs_timestamp_x_to_si_direction.png'
    plt.savefig(output_file_x_to_si, dpi=300, bbox_inches='tight')
    print(f"Saved x->si direction plot to: {output_file_x_to_si}")
    plt.close(fig_x_to_si)
    
    # Plot 1c: x->si direction - True neighbors only
    if len(x_to_si_true_df_plot) > 0:
        fig_x_to_si_true, ax_x_to_si_true = plt.subplots(1, 1, figsize=(14, 10))
        
        ax_x_to_si_true.scatter(x_to_si_true_df_plot['timestamp_dt'], x_to_si_true_df_plot['last_seen_dt'], 
                  alpha=0.6, s=40, color='green', edgecolors='darkgreen', linewidth=0.5,
                  label=f'True Neighbors (n={x_to_si_true_unique} neighbors, {len(x_to_si_true_df_plot)} records)')
        
        # Add diagonal reference line
        min_time = min(x_to_si_true_df_plot['timestamp_dt'].min(), x_to_si_true_df_plot['last_seen_dt'].min())
        max_time = max(x_to_si_true_df_plot['timestamp_dt'].max(), x_to_si_true_df_plot['last_seen_dt'].max())
        ax_x_to_si_true.plot([min_time, max_time], [min_time, max_time], 'k--', linewidth=1.5, alpha=0.5, label='last_seen = timestamp')
        
        ax_x_to_si_true.set_xlabel('Timestamp', fontsize=12)
        ax_x_to_si_true.set_ylabel('Last Seen', fontsize=12)
        ax_x_to_si_true.set_title(f'Last Seen vs Timestamp: x->si Direction - TRUE Neighbors Only (delta > 2.5h)\n({x_to_si_true_unique} neighbors, {len(x_to_si_true_df_plot)} records)', fontsize=14, fontweight='bold')
        ax_x_to_si_true.grid(True, alpha=0.3)
        ax_x_to_si_true.tick_params(axis='x', rotation=45)
        ax_x_to_si_true.legend(fontsize=10, loc='upper left')
        
        plt.tight_layout()
        output_file_x_to_si_true = 'last_seen_vs_timestamp_x_to_si_true_only.png'
        plt.savefig(output_file_x_to_si_true, dpi=300, bbox_inches='tight')
        print(f"Saved x->si TRUE neighbors only plot to: {output_file_x_to_si_true}")
        plt.close(fig_x_to_si_true)
    
    # Plot 1d: si->x direction - True neighbors only
    if len(true_df) > 0:
        fig_si_to_x_true, ax_si_to_x_true = plt.subplots(1, 1, figsize=(14, 10))
        
        ax_si_to_x_true.scatter(true_df['timestamp_dt'], true_df['last_seen_dt'], 
                  alpha=0.6, s=40, color='green', edgecolors='darkgreen', linewidth=0.5,
                  label=f'True Neighbors (n={true_unique_neighbors} neighbors, {len(true_df)} records)')
        
        # Add diagonal reference line
        min_time = min(true_df['timestamp_dt'].min(), true_df['last_seen_dt'].min())
        max_time = max(true_df['timestamp_dt'].max(), true_df['last_seen_dt'].max())
        ax_si_to_x_true.plot([min_time, max_time], [min_time, max_time], 'k--', linewidth=1.5, alpha=0.5, label='last_seen = timestamp')
        
        ax_si_to_x_true.set_xlabel('Timestamp', fontsize=12)
        ax_si_to_x_true.set_ylabel('Last Seen', fontsize=12)
        ax_si_to_x_true.set_title(f'Last Seen vs Timestamp: si->x Direction - TRUE Neighbors Only\n({true_unique_neighbors} neighbors, {len(true_df)} records)', fontsize=14, fontweight='bold')
        ax_si_to_x_true.grid(True, alpha=0.3)
        ax_si_to_x_true.tick_params(axis='x', rotation=45)
        ax_si_to_x_true.legend(fontsize=10, loc='upper left')
        
        plt.tight_layout()
        output_file_si_to_x_true = 'last_seen_vs_timestamp_si_to_x_true_only.png'
        plt.savefig(output_file_si_to_x_true, dpi=300, bbox_inches='tight')
        print(f"Saved si->x TRUE neighbors only plot to: {output_file_si_to_x_true}")
        plt.close(fig_si_to_x_true)
else:
    print(f"Skipping x->si direction plot: No x->si data available in input file")

# Also create combined plot with time difference visualization
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))

# Prepare time difference data
if len(true_df) > 0 and 'time_diff_sec' in true_df.columns:
    true_df['time_diff_minutes'] = true_df['time_diff_sec'] / 60
if len(false_df) > 0 and 'time_diff_sec' in false_df.columns:
    false_df['time_diff_minutes'] = false_df['time_diff_sec'] / 60

# Plot all true neighbors time difference
if len(true_df) > 0 and 'time_diff_minutes' in true_df.columns:
    ax2.scatter(true_df['timestamp_dt'], true_df['time_diff_minutes'], 
               alpha=0.5, s=30, color='green', edgecolors='none', linewidth=0.3,
               label=f'True Neighbors (n={true_unique_neighbors} neighbors, {len(true_df)} records)')

# Plot all false neighbors time difference
if len(false_df) > 0 and 'time_diff_minutes' in false_df.columns:
    ax2.scatter(false_df['timestamp_dt'], false_df['time_diff_minutes'], 
               alpha=0.5, s=30, color='red', edgecolors='none', linewidth=0.3,
               label=f'False Neighbors (n={false_unique_neighbors} neighbors, {len(false_df)} records)')

# Add 2.5h threshold line
ax2.axhline(y=150, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='2.5h threshold')

ax2.set_xlabel('Timestamp', fontsize=12)
ax2.set_ylabel('Time Difference (timestamp - last_seen, minutes)', fontsize=12)
ax2.set_title(f'Time Difference vs Timestamp: {direction_label} Direction\n({true_unique_neighbors} true, {false_unique_neighbors} false)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
ax2.legend(fontsize=10, loc='upper right')

plt.tight_layout()
output_file2 = 'time_diff_vs_timestamp_sample_neighbors.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"Saved time difference plot to: {output_file2}")

# Create histogram distributions for both directions
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

# Prepare data for si->x direction
si_to_x_true_df = pd.DataFrame(si_to_x_true_records)
si_to_x_false_df = pd.DataFrame(si_to_x_false_records)

if len(si_to_x_true_df) > 0 and 'time_diff_sec' in si_to_x_true_df.columns:
    si_to_x_true_df['time_diff_minutes'] = si_to_x_true_df['time_diff_sec'] / 60
    si_to_x_true_minutes = si_to_x_true_df['time_diff_minutes'].values
else:
    si_to_x_true_minutes = np.array([])

if len(si_to_x_false_df) > 0 and 'time_diff_sec' in si_to_x_false_df.columns:
    si_to_x_false_df['time_diff_minutes'] = si_to_x_false_df['time_diff_sec'] / 60
    si_to_x_false_minutes = si_to_x_false_df['time_diff_minutes'].values
else:
    si_to_x_false_minutes = np.array([])

# Prepare data for x->si direction (use x_to_si filtered data)
x_to_si_true_df_hist = x_to_si_true_records_df.copy() if len(x_to_si_true_records_df) > 0 else pd.DataFrame()
x_to_si_false_df_hist = x_to_si_false_records_df.copy() if len(x_to_si_false_records_df) > 0 else pd.DataFrame()

if len(x_to_si_true_df_hist) > 0 and 'time_diff_sec' in x_to_si_true_df_hist.columns:
    x_to_si_true_df_hist['time_diff_minutes'] = x_to_si_true_df_hist['time_diff_sec'] / 60
    x_to_si_true_minutes = x_to_si_true_df_hist['time_diff_minutes'].values
else:
    x_to_si_true_minutes = np.array([])

if len(x_to_si_false_df_hist) > 0 and 'time_diff_sec' in x_to_si_false_df_hist.columns:
    x_to_si_false_df_hist['time_diff_minutes'] = x_to_si_false_df_hist['time_diff_sec'] / 60
    x_to_si_false_minutes = x_to_si_false_df_hist['time_diff_minutes'].values
else:
    x_to_si_false_minutes = np.array([])

# Plot 1: si->x direction histogram
ax3_1 = axes3[0]
# Calculate total count for si->x direction (all records)
si_to_x_total_count = len(si_to_x_true_minutes) + len(si_to_x_false_minutes)

if len(si_to_x_true_minutes) > 0:
    # Use weights to normalize by total si->x count
    weights_true = np.ones(len(si_to_x_true_minutes)) / si_to_x_total_count if si_to_x_total_count > 0 else None
    ax3_1.hist(si_to_x_true_minutes, bins=30, alpha=0.7, weights=weights_true,
              label=f'True Neighbors (n={len(si_to_x_true_minutes)})', 
              color='green', edgecolor='black')
if len(si_to_x_false_minutes) > 0:
    # Use weights to normalize by total si->x count
    weights_false = np.ones(len(si_to_x_false_minutes)) / si_to_x_total_count if si_to_x_total_count > 0 else None
    ax3_1.hist(si_to_x_false_minutes, bins=30, alpha=0.7, weights=weights_false,
              label=f'False Neighbors (n={len(si_to_x_false_minutes)})', 
              color='red', edgecolor='black')

ax3_1.set_xlabel('Delta_t (timestamp - last_seen, minutes)', fontsize=12)
ax3_1.set_ylabel('Relative Frequency', fontsize=12)
ax3_1.set_title(f'Delta_t Distribution: si->x Direction\nTrue vs False Neighbors (normalized by total si->x)', fontsize=14, fontweight='bold')
ax3_1.axvline(x=150, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='2.5h threshold')
ax3_1.legend(fontsize=10)
ax3_1.grid(axis='y', alpha=0.3)

# Plot 2: x->si direction histogram
ax3_2 = axes3[1]
# Calculate total count for x->si direction (all records)
x_to_si_total_count = len(x_to_si_true_minutes) + len(x_to_si_false_minutes)

if len(x_to_si_true_minutes) > 0:
    # Use weights to normalize by total x->si count
    weights_true = np.ones(len(x_to_si_true_minutes)) / x_to_si_total_count if x_to_si_total_count > 0 else None
    ax3_2.hist(x_to_si_true_minutes, bins=30, alpha=0.7, weights=weights_true,
              label=f'True Neighbors (n={len(x_to_si_true_minutes)})', 
              color='green', edgecolor='black')
if len(x_to_si_false_minutes) > 0:
    # Use weights to normalize by total x->si count
    weights_false = np.ones(len(x_to_si_false_minutes)) / x_to_si_total_count if x_to_si_total_count > 0 else None
    ax3_2.hist(x_to_si_false_minutes, bins=30, alpha=0.7, weights=weights_false,
              label=f'False Neighbors (n={len(x_to_si_false_minutes)})', 
              color='red', edgecolor='black')

ax3_2.set_xlabel('Delta_t (timestamp - last_seen, minutes)', fontsize=12)
ax3_2.set_ylabel('Probability Density', fontsize=12)
ax3_2.set_title(f'Delta_t Distribution: x->si Direction (delta > 2.5h)\nTrue vs False Neighbors', fontsize=14, fontweight='bold')
ax3_2.axvline(x=150, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='2.5h threshold')
ax3_2.legend(fontsize=10)
ax3_2.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_file3 = 'delta_t_histogram_both_directions.png'
plt.savefig(output_file3, dpi=300, bbox_inches='tight')
print(f"Saved delta_t histogram to: {output_file3}")

# Create frequency distribution bar charts (similar to getaddr timestamps distribution)
print("\nCreating frequency distribution bar charts...")

# Plot 4: si->x direction frequency distribution (bar chart)
fig4, ax4 = plt.subplots(1, 1, figsize=(14, 8))

# Prepare si->x data - need to combine true and false records with all necessary fields
si_to_x_all_records = si_to_x_true_records + si_to_x_false_records
if len(si_to_x_all_records) > 0:
    si_to_x_all_df = pd.DataFrame(si_to_x_all_records)
    # Calculate time_diff_minutes if not already present
    if 'time_diff_minutes' not in si_to_x_all_df.columns:
        if 'time_diff_sec' in si_to_x_all_df.columns:
            si_to_x_all_df['time_diff_minutes'] = si_to_x_all_df['time_diff_sec'] / 60
        elif 'timestamp_dt' in si_to_x_all_df.columns and 'last_seen_dt' in si_to_x_all_df.columns:
            si_to_x_all_df['time_diff_sec'] = (si_to_x_all_df['timestamp_dt'] - si_to_x_all_df['last_seen_dt']).dt.total_seconds()
            si_to_x_all_df['time_diff_minutes'] = si_to_x_all_df['time_diff_sec'] / 60
    
    if 'time_diff_minutes' in si_to_x_all_df.columns:
        si_to_x_all_df['is_peer'] = si_to_x_all_df['neighbor_ip'].isin(si_true_neighbors)
        
        # Define bins (use more bins for finer granularity)
        max_minutes = int(si_to_x_all_df['time_diff_minutes'].max()) + 25
        # Use 50 bins for better resolution
        num_bins = 50
        bin_width = max(5, (max_minutes + 25) // num_bins)  # At least 5 minutes per bin
        bins = range(0, max_minutes + bin_width, bin_width)
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
        
        # Count for true and false neighbors
        true_counts, _ = np.histogram(si_to_x_all_df[si_to_x_all_df['is_peer'] == True]['time_diff_minutes'].values, bins=bins)
        false_counts, _ = np.histogram(si_to_x_all_df[si_to_x_all_df['is_peer'] == False]['time_diff_minutes'].values, bins=bins)
        
        # Create grouped bar chart with side-by-side bars
        x_pos = np.arange(len(bin_centers))
        width = 0.35  # Relative bar width for side-by-side bars
        gap = 0.1  # Gap between bars for better separation
        
        ax4.bar(x_pos - width/2 - gap/2, true_counts, width, label='Is peer? T', color='green', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax4.bar(x_pos + width/2 + gap/2, false_counts, width, label='Is peer? F', color='red', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax4.set_xlabel('timestamp - last_seen (in minutes)', fontsize=12)
        ax4.set_ylabel('count', fontsize=12)
        ax4.set_title('SI Node GetAddr Timestamps Distribution: si->x Direction', fontsize=14, fontweight='bold')
        
        # Show every nth x-tick label to avoid crowding
        n_ticks = min(20, len(bin_centers))  # Show at most 20 tick labels
        step = max(1, len(bin_centers) // n_ticks)
        ax4.set_xticks(x_pos[::step])
        ax4.set_xticklabels([f'{int(bc)}' for bc in bin_centers[::step]], rotation=45, ha='right')
        ax4.legend(fontsize=10, loc='upper right')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file4 = 'si_getaddr_timestamps_distribution_si_to_x.png'
        plt.savefig(output_file4, dpi=300, bbox_inches='tight')
        print(f"Saved si->x frequency distribution to: {output_file4}")
        plt.close(fig4)
    else:
        print("Skipping si->x frequency distribution: Cannot calculate time_diff_minutes")
else:
    print("Skipping si->x frequency distribution: No data available")

# Plot 5: x->si direction frequency distribution (bar chart)
if len(x_to_si_true_records) > 0 or len(x_to_si_false_records) > 0:
    fig5, ax5 = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare x->si data - use records before filtering
    x_to_si_all_records = x_to_si_true_records + x_to_si_false_records
    x_to_si_all_df = pd.DataFrame(x_to_si_all_records)
    if len(x_to_si_all_df) > 0:
        # Calculate time_diff_minutes if not already present
        if 'time_diff_sec' not in x_to_si_all_df.columns:
            if 'timestamp_dt' in x_to_si_all_df.columns and 'last_seen_dt' in x_to_si_all_df.columns:
                x_to_si_all_df['time_diff_sec'] = (x_to_si_all_df['timestamp_dt'] - x_to_si_all_df['last_seen_dt']).dt.total_seconds()
        if 'time_diff_sec' in x_to_si_all_df.columns:
            x_to_si_all_df['time_diff_minutes'] = x_to_si_all_df['time_diff_sec'] / 60
            x_to_si_all_df['is_peer'] = x_to_si_all_df['neighbor_ip'].isin(si_true_neighbors)
            
            # Filter to delta > 2.5h for x->si
            x_to_si_all_df_filtered = x_to_si_all_df[x_to_si_all_df['time_diff_minutes'] > 150].copy()
            
            if len(x_to_si_all_df_filtered) > 0:
                # Check if we need log scale based on max value
                max_minutes = x_to_si_all_df_filtered['time_diff_minutes'].max()
                use_log_scale = max_minutes > 500  # Use log scale if max > 500 minutes
                
                if use_log_scale:
                    # Use log scale bins with fewer bins for better clarity
                    min_minutes = x_to_si_all_df_filtered['time_diff_minutes'].min()
                    # Create log-spaced bins - use fewer bins (20 instead of 40) for better separation
                    num_bins = 20
                    log_bins = np.logspace(np.log10(min_minutes), np.log10(max_minutes), num_bins + 1)
                    bin_centers = np.sqrt(log_bins[:-1] * log_bins[1:])  # Geometric mean for log scale
                    
                    # Count for true and false neighbors
                    true_counts, _ = np.histogram(x_to_si_all_df_filtered[x_to_si_all_df_filtered['is_peer'] == True]['time_diff_minutes'].values, bins=log_bins)
                    false_counts, _ = np.histogram(x_to_si_all_df_filtered[x_to_si_all_df_filtered['is_peer'] == False]['time_diff_minutes'].values, bins=log_bins)
                    
                    # For log scale bar chart, calculate bar widths as fraction of center values
                    # Use larger bar width ratio for fewer bins
                    bar_width_ratio = 0.25  # Width as 25% of center value
                    true_bar_widths = bin_centers * bar_width_ratio
                    false_bar_widths = bin_centers * bar_width_ratio
                    
                    # Shift positions more for better separation on log scale
                    shift_factor = 0.92  # Shift true bars more to the left
                    true_positions = bin_centers * shift_factor
                    false_positions = bin_centers / shift_factor
                    
                    # Filter out zero counts
                    true_mask = true_counts > 0
                    false_mask = false_counts > 0
                    
                    ax5.bar(true_positions[true_mask], true_counts[true_mask], 
                           width=true_bar_widths[true_mask], label='Is peer? T', 
                           color='salmon', alpha=0.9, edgecolor='darkred', linewidth=1, align='center')
                    ax5.bar(false_positions[false_mask], false_counts[false_mask], 
                           width=false_bar_widths[false_mask], label='Is peer? F', 
                           color='teal', alpha=0.9, edgecolor='darkblue', linewidth=1, align='center')
                    
                    ax5.set_xscale('log')
                    ax5.set_yscale('log')
                    ax5.set_xlabel('timestamp - last_seen (in minutes, log scale)', fontsize=12)
                    ax5.set_ylabel('count (log scale)', fontsize=12)
                    ax5.set_title('SI Node GetAddr Timestamps Distribution: x->si Direction (delta > 2.5h, log-log)', fontsize=14, fontweight='bold')
                    ax5.legend(fontsize=10, loc='upper right')
                else:
                    # Use linear scale bins (25-minute intervals)
                    max_minutes_int = int(max_minutes) + 50
                    bins = range(0, max_minutes_int + 25, 25)
                    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
                    
                    # Count for true and false neighbors
                    true_counts, _ = np.histogram(x_to_si_all_df_filtered[x_to_si_all_df_filtered['is_peer'] == True]['time_diff_minutes'].values, bins=bins)
                    false_counts, _ = np.histogram(x_to_si_all_df_filtered[x_to_si_all_df_filtered['is_peer'] == False]['time_diff_minutes'].values, bins=bins)
                    
                    # Create grouped bar chart
                    x_pos = np.arange(len(bin_centers))
                    width = 10  # Bar width
                    
                    ax5.bar(x_pos - width/2, true_counts, width, label='Is peer? T', color='salmon', alpha=0.8, edgecolor='black', linewidth=0.5)
                    ax5.bar(x_pos + width/2, false_counts, width, label='Is peer? F', color='teal', alpha=0.8, edgecolor='black', linewidth=0.5)
                    
                    ax5.set_xlabel('timestamp - last_seen (in minutes)', fontsize=12)
                    ax5.set_ylabel('count', fontsize=12)
                    ax5.set_title('SI Node GetAddr Timestamps Distribution: x->si Direction (delta > 2.5h)', fontsize=14, fontweight='bold')
                    ax5.set_xticks(x_pos)
                    ax5.set_xticklabels([f'{int(bc)}' for bc in bin_centers], rotation=45, ha='right')
                
                ax5.legend(fontsize=10, loc='upper right')
                ax5.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                output_file5 = 'si_getaddr_timestamps_distribution_x_to_si.png'
                plt.savefig(output_file5, dpi=300, bbox_inches='tight')
                print(f"Saved x->si frequency distribution to: {output_file5}")
                plt.close(fig5)
            else:
                print("Skipping x->si frequency distribution: No data after filtering to delta > 2.5h")
        else:
            print("Skipping x->si frequency distribution: Cannot calculate time_diff_minutes")
    else:
        print("Skipping x->si frequency distribution: No data available")
else:
    print("Skipping x->si frequency distribution: No data available")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)

