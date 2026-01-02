#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract records related to SI node from large scan file.
Add delta column (timestamp - last_seen).
Output to active_scanned_from_si_si_to_x_lt2p5h_unique.csv

New logic:
- si->x: keep records with delta < 2.5h, and no other nodes share same (destination, last_seen)
- x->si: keep all records, no delta filter
"""

import pandas as pd
import numpy as np

# File paths
INPUT_FILE_ALIBABA = 'dogecoin_8th_Dec/active_scanned_from_alibaba.csv'
INPUT_FILE_US = 'dogecoin_8th_Dec/active_scanned_from_us.csv'
OUTPUT_FILE = 'dogecoin_8th_Dec/active_scanned_from_si_si_to_x_lt2p5h_unique.csv'

# SI Node IP
SI_NODE_IP = '109.123.232.246'

# Chunk size for reading large file
CHUNK_SIZE = 100000

INPUT_FILES = [INPUT_FILE_ALIBABA, INPUT_FILE_US]

print("=" * 60)
print("Extracting SI-related records from scan files")
print(f"Input files: {INPUT_FILE_ALIBABA}, {INPUT_FILE_US}")
print(f"Output file: {OUTPUT_FILE}")
print(f"SI Node IP: {SI_NODE_IP}")
print("\nFilter logic:")
print("  si->x: delta < 2.5h, remove ALL records for destinations with ANY shared (destination, last_seen)")
print("  x->si: delta >= 2.5h AND last_seen_dt is unique (no two different x use same last_seen), where x satisfies si->x conditions")
print("=" * 60)

# First pass: collect all records to check for cross-node sharing
all_chunks = []
all_other_to_x_chunks = []  # Records from other nodes (not SI) to any destination
total_rows = 0
si_rows = 0

print("\nProcessing files in chunks...")
for input_file in INPUT_FILES:
    print(f"\nProcessing {input_file}...")
    file_chunk_idx = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=CHUNK_SIZE)):
        total_rows += len(chunk)
        file_chunk_idx += 1
        
        # Convert timestamps to datetime
        chunk['timestamp_dt'] = pd.to_datetime(chunk['timestamp'], utc=True, errors='coerce')
        chunk['last_seen_dt'] = pd.to_datetime(chunk['last_seen'], utc=True, errors='coerce')
        
        # Collect all SI-related records
        si_related = chunk[
            (chunk['source_ip'] == SI_NODE_IP) | 
            (chunk['destination_ip'] == SI_NODE_IP)
        ].copy()
        
        if len(si_related) > 0:
            # Calculate delta (timestamp - last_seen) in seconds
            si_related['delta'] = (si_related['timestamp_dt'] - si_related['last_seen_dt']).dt.total_seconds()
            si_related['delta_hours'] = si_related['delta'] / 3600
            all_chunks.append(si_related)
            si_rows += len(si_related)
        
        # Collect records from other nodes (not SI) to destinations (for cross-node check)
        other_to_x = chunk[
            (chunk['source_ip'] != SI_NODE_IP) & 
            chunk['destination_ip'].notna() &
            chunk['last_seen_dt'].notna()
        ].copy()
        
        if len(other_to_x) > 0:
            all_other_to_x_chunks.append(other_to_x)
        
        # Progress update
        if file_chunk_idx % 100 == 0:
            print(f"  Processed {file_chunk_idx} chunks from {input_file}, total {total_rows:,} rows, found {si_rows:,} SI-related records...")
    
    print(f"  Finished processing {input_file}: {file_chunk_idx} chunks")

print(f"\nTotal rows processed: {total_rows:,}")
print(f"SI-related records found: {si_rows:,}")

# Combine all chunks
print("\nCombining all chunks...")
if all_chunks:
    all_si_records = pd.concat(all_chunks, ignore_index=True)
    print(f"Total SI-related records: {len(all_si_records):,}")
else:
    all_si_records = pd.DataFrame()
    print("No SI-related records found!")

if all_other_to_x_chunks:
    all_other_to_x = pd.concat(all_other_to_x_chunks, ignore_index=True)
    print(f"Total other->x records: {len(all_other_to_x):,}")
else:
    all_other_to_x = pd.DataFrame()
    print("No other->x records found")

# Step 2: Apply filtering logic
print("\n" + "=" * 60)
print("Step 2: Applying filtering logic")
print("=" * 60)

if len(all_si_records) > 0:
    # Separate si->x and x->si
    si_to_x_records = all_si_records[all_si_records['source_ip'] == SI_NODE_IP].copy()
    x_to_si_records = all_si_records[all_si_records['destination_ip'] == SI_NODE_IP].copy()
    
    print(f"si->x records (before filtering): {len(si_to_x_records):,}")
    print(f"x->si records (all kept): {len(x_to_si_records):,}")
    
    # Filter si->x: delta < 2.5h
    threshold_2p5h = 2.5 * 3600  # 2.5 hours in seconds
    si_to_x_filtered = si_to_x_records[
        si_to_x_records['delta'].notna() & 
        (si_to_x_records['delta'] < threshold_2p5h)
    ].copy()
    print(f"si->x records with delta < 2.5h: {len(si_to_x_filtered):,}")
    
    # Filter si->x: if ANY (destination, last_seen) pair is shared by other nodes, remove ALL records for that destination
    if len(si_to_x_filtered) > 0 and len(all_other_to_x) > 0:
        # Create set of (destination_ip, last_seen_dt) pairs from other->x
        print("\nBuilding (destination, last_seen) pairs from other nodes...")
        other_to_x_pairs = set()
        for idx, row in all_other_to_x.iterrows():
            if pd.notna(row['destination_ip']) and pd.notna(row['last_seen_dt']):
                other_to_x_pairs.add((row['destination_ip'], row['last_seen_dt']))
        
        print(f"Unique (destination, last_seen) pairs from other nodes: {len(other_to_x_pairs):,}")
        
        # Find destination IPs that have ANY shared (destination, last_seen) pair
        print("\nFiltering si->x: removing ALL records for destinations with ANY shared (destination, last_seen)...")
        initial_count = len(si_to_x_filtered)
        conflicting_destinations = set()
        for idx, row in si_to_x_filtered.iterrows():
            if pd.notna(row['destination_ip']) and pd.notna(row['last_seen_dt']):
                if (row['destination_ip'], row['last_seen_dt']) in other_to_x_pairs:
                    conflicting_destinations.add(row['destination_ip'])
        
        print(f"Destination IPs with at least one shared (destination, last_seen) pair: {len(conflicting_destinations):,}")
        
        # Remove ALL records for these destination IPs
        si_to_x_filtered = si_to_x_filtered[~si_to_x_filtered['destination_ip'].isin(conflicting_destinations)].copy()
        
        removed_count = initial_count - len(si_to_x_filtered)
        print(f"Removed {removed_count:,} si->x records (ALL records for {len(conflicting_destinations):,} conflicting destinations)")
        print(f"si->x records after cross-node filter: {len(si_to_x_filtered):,}")
    elif len(si_to_x_filtered) > 0:
        print("No other->x records found, skipping cross-node filter")
    
    # Filter x->si: only keep records where source_ip (x) satisfies si->x conditions
    # AND delta >= 2.5h AND last_seen_dt is unique (no two different x use same last_seen)
    if len(si_to_x_filtered) > 0:
        # Get set of destination IPs that satisfy si->x conditions
        valid_x_ips = set(si_to_x_filtered['destination_ip'].unique())
        print(f"\nValid x IPs (satisfy si->x conditions): {len(valid_x_ips):,}")
        
        # Filter x->si records to only include those where source_ip is in valid_x_ips
        if len(x_to_si_records) > 0:
            initial_x_to_si_count = len(x_to_si_records)
            x_to_si_valid_x = x_to_si_records[
                x_to_si_records['source_ip'].isin(valid_x_ips)
            ].copy()
            print(f"x->si records (after filtering to valid x): {len(x_to_si_valid_x):,}")
            
            # Filter x->si: delta >= 2.5h
            x_to_si_case1_candidates = x_to_si_valid_x[
                x_to_si_valid_x['delta'].notna() & 
                (x_to_si_valid_x['delta'] >= threshold_2p5h)
            ].copy()
            print(f"x->si records with delta >= 2.5h: {len(x_to_si_case1_candidates):,}")
            
            # Filter x->si: last_seen_dt must be unique (no two different x use same last_seen)
            if len(x_to_si_case1_candidates) > 0:
                # Count how many different source_ip use each last_seen_dt
                last_seen_counts = x_to_si_case1_candidates.groupby('last_seen_dt')['source_ip'].nunique()
                unique_last_seen = set(last_seen_counts[last_seen_counts == 1].index)
                
                print(f"  last_seen_dt values used by only one source_ip: {len(unique_last_seen):,}")
                
                # Filter to only keep records where last_seen_dt is unique
                x_to_si_filtered = x_to_si_case1_candidates[
                    x_to_si_case1_candidates['last_seen_dt'].isin(unique_last_seen)
                ].copy()
                
                print(f"x->si records (delta >= 2.5h AND unique last_seen_dt): {len(x_to_si_filtered):,}")
                removed_by_unique = len(x_to_si_case1_candidates) - len(x_to_si_filtered)
                print(f"  Removed {removed_by_unique:,} records where last_seen_dt is shared by multiple source_ip")
            else:
                x_to_si_filtered = pd.DataFrame()
        else:
            x_to_si_filtered = pd.DataFrame()
    else:
        # No valid si->x records, so no valid x->si records either
        x_to_si_filtered = pd.DataFrame()
        print("\nNo valid si->x records found, so no x->si records will be included")
    
    # Combine filtered si->x and filtered x->si records
    output_chunks_list = []
    if len(si_to_x_filtered) > 0:
        output_chunks_list.append(si_to_x_filtered)
    if len(x_to_si_filtered) > 0:
        output_chunks_list.append(x_to_si_filtered)
    
    if output_chunks_list:
        output_df = pd.concat(output_chunks_list, ignore_index=True)
    else:
        output_df = pd.DataFrame()
    
    # Display summary statistics
    if len(output_df) > 0:
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        
        si_to_x = output_df[output_df['source_ip'] == SI_NODE_IP]
        x_to_si = output_df[output_df['destination_ip'] == SI_NODE_IP]
        
        print(f"\nDirection breakdown:")
        print(f"  si->x (SI as source, delta < 2.5h, destinations with ANY shared last_seen removed): {len(si_to_x):,} records")
        print(f"  x->si (SI as destination, where x satisfies si->x conditions): {len(x_to_si):,} records")
        
        if len(si_to_x) > 0:
            print(f"\nDelta statistics for si->x (timestamp - last_seen):")
            print(f"  Mean: {si_to_x['delta'].mean() / 3600:.2f} hours ({si_to_x['delta'].mean():.2f} seconds)")
            print(f"  Median: {si_to_x['delta'].median() / 3600:.2f} hours ({si_to_x['delta'].median():.2f} seconds)")
            print(f"  Min: {si_to_x['delta'].min() / 3600:.2f} hours ({si_to_x['delta'].min():.2f} seconds)")
            print(f"  Max: {si_to_x['delta'].max() / 3600:.2f} hours ({si_to_x['delta'].max():.2f} seconds)")
        
        # Save to CSV
        print(f"\nSaving to {OUTPUT_FILE}...")
        output_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✓ Saved {len(output_df):,} records to {OUTPUT_FILE}")
        
        print("\n" + "=" * 60)
        print("Extraction complete!")
        print("=" * 60)
    else:
        print("\nNo records to save after filtering!")
else:
    print("\nNo SI-related records found!")
