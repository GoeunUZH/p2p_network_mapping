# import pandas as pd

# # Load data as DataFrame
# data = pd.read_csv('data/processed_merge_20241211_datawith3nodes.txt')  # Replace with your file name

# # Define IP addresses to count
# ips = {
#     "Singapore": "109.123.232.246",
#     "US": "94.72.112.249",
#     "EU": "185.252.234.250"
# }

# # Initialize results dictionary
# results = {location: 0 for location in ips}

# # When counting, first create a new DataFrame to store deduplicated links
# # Ensure (ip1, ip2) is always sorted in dictionary order for deduplication
# data['link'] = data.apply(lambda row: tuple(sorted([row['ip1'], row['ip2']])), axis=1)

# # Deduplicated data
# unique_links = data.drop_duplicates(subset=['link'])

# # Count how many times each IP appears in ip1 and ip2
# for location, ip in ips.items():
#     results[location] = len(unique_links[(unique_links['ip1'] == ip) | (unique_links['ip2'] == ip)])

# # Print results
# print(results)

import pandas as pd

# Load data as DataFrame
data = pd.read_csv('/local/scratch/YG_monero_data/contabo-monero-data/data/processed_merge_20241211_datawith3nodes.txt')  # Replace with your file name

# Define IP addresses to count
ips = {
    "Singapore": "109.123.232.246",
    "US": "94.72.112.249",
    "EU": "185.252.234.250"
}

# Initialize results dictionary
results = {location: 0 for location in ips}

# Read connection data for each country
connections_eu = pd.read_csv('/local/scratch/YG_monero_data/contabo-monero-data/data/raw_peerlists_20241211/output_eu/connections_eu_deduplicated.csv')  # Read EU connection data
connections_us = pd.read_csv('/local/scratch/YG_monero_data/contabo-monero-data/data/raw_peerlists_20241211/output_us/connections_us_deduplicated.csv')  # Read US connection data
connections_si = pd.read_csv('/local/scratch/YG_monero_data/contabo-monero-data/data/raw_peerlists_20241211/output_si/connections_si_deduplicated.csv')  # Read Singapore connection data


print(connections_eu.head())

# Create set of connection addresses
eu_addresses = set(connections_eu['address'])
us_addresses = set(connections_us['address'])
si_addresses = set(connections_si['address'])


# Create link column in DataFrame, sorted in dictionary order for deduplication
data['link'] = data.apply(lambda row: tuple(sorted([row['ip1'], row['ip2']])), axis=1)
unique_links = data.drop_duplicates(subset=['link'])

# Count how many times each IP appears in ip1 and ip2, and calculate the percentage of ip2 in connection data
for location, ip in ips.items():
    # Count how many times ip1 and ip2 appear in the data
    ip2_addresses = unique_links[(unique_links['ip1'] == ip) | (unique_links['ip2'] == ip)]['ip2'].tolist()
    ip2_count = len(ip2_addresses)
    
    # Select corresponding connection data based on country
    if location == "Singapore":
        address_set = si_addresses
    elif location == "US":
        address_set = us_addresses
    else:
        address_set = eu_addresses
    
    # Count how many ip2s appear in connection data
    matched_count = sum(1 for ip2 in ip2_addresses if ip2 in address_set)
    
    # Calculate percentage
    if ip2_count > 0:
        percentage = (matched_count / ip2_count) * 100
    else:
        percentage = 0  # If no ip2 appears, percentage is 0
    
    results[location] = (ip2_count, matched_count, percentage)

# Print results
for location, result in results.items():
    ip2_count, matched_count, percentage = result
    print(f"{location}: Total ip2 count = {ip2_count}, Matched count = {matched_count}, Percentage = {percentage:.2f}%")

