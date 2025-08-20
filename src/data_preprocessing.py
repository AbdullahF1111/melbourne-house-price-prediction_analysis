#preprocessing and cleaning
# Select relevant features
features = ['Rooms', 'Bathroom', 'Car', 'Landsize', 'Distance', 'Type', 'Regionname', 'Price']
df = df[features].copy()

# Handle outliers
df = df[df['Distance'] != 0]
df = df[(df['Rooms'] >= 1) & (df['Rooms'] <= 5)]
df = df[(df['Bathroom'] >= 1) & (df['Bathroom'] <= 3)]
df = df[(df['Car'] >= 0) & (df['Car'] <= 4)]
df = df[(df['Landsize'] >= 10) & (df['Landsize'] <= 2000)]

# Remove rare regions
regions_to_drop = ['Eastern Victoria', 'Northern Victoria', 'Western Victoria']
df = df[~df['Regionname'].isin(regions_to_drop)]

# Drop missing values
df = df.dropna()

print("\n\u2705 Data shape after cleaning:", df.shape)
