import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/WELFake_Dataset.csv")
df = df.dropna()
# Split the dataset into features (X) and labels (y)
X = df['text']
y = df['label']

# Step 1: Split into 70% training and 30% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Split the 30% testing set into 50% validation and 50% test (15% each)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check the sizes of the resulting splits
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print(f"Validation set size: {len(X_val)}")

# Combine features and labels for training and testing
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
valid_data = pd.concat([X_val, y_val], axis=1)

# Save the combined data to CSV files
train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)
valid_data.to_csv('data/valid_data.csv', index=False)

print("Training, testing and validating data saved successfully!")

# Count the number of occurrences of each label in the training and testing data
train_label_counts = train_data['label'].value_counts()
test_label_counts = test_data['label'].value_counts()
valid_label_counts = valid_data['label'].value_counts()

# Create a DataFrame to combine both counts
label_comparison = pd.DataFrame({
    'Train': train_label_counts,
    'Test': test_label_counts,
    'valid': valid_label_counts
}).fillna(0)  # Fill NaN values with 0 for labels that don't exist in one of the sets

# Plotting the bar plot
label_comparison.plot(kind='bar', figsize=(8, 6), width=0.8)

# Customize the plot
plt.title('Label Distribution in Train and Test Data')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Dataset')

# Show the plot
plt.tight_layout()
plt.show()
