import pandas as pd
import numpy as np

# Load the CSV files
context_matrix = pd.read_csv('activity_context_matrix.csv', index_col='Context')
transition_matrix = pd.read_csv('activity_transition_matrix.csv', index_col='Current Activity')
time_probability = pd.read_csv('activity_probability_given_time_of_day.csv', index_col='Activity Variable_Name')

# Get all unique activities and convert to a sorted list
all_activities = sorted(set(context_matrix.columns) | set(transition_matrix.index) | set(time_probability.index))

# Create a new DataFrame for the expanded transition matrix
new_matrix = pd.DataFrame(0.0, index=all_activities, columns=all_activities)

# Fill in known transitions from the original matrix
for activity in transition_matrix.index:
    if activity in new_matrix.index:
        for next_activity in transition_matrix.columns:
            if next_activity in new_matrix.columns:
                new_matrix.loc[activity, next_activity] = transition_matrix.loc[activity, next_activity]

# For new activities, distribute probabilities evenly
for activity in new_matrix.index:
    if new_matrix.loc[activity].sum() == 0:
        new_matrix.loc[activity] = 1 / len(all_activities)
    else:
        # Normalize existing probabilities
        new_matrix.loc[activity] = new_matrix.loc[activity] / new_matrix.loc[activity].sum()

# Save the new matrix to a CSV file
new_matrix.to_csv('expanded_activity_transition_matrix.csv')

print("Expanded activity transition matrix has been generated and saved to 'expanded_activity_transition_matrix.csv'")
print(f"Total number of activities: {len(all_activities)}")
print("First few activities:")
print(all_activities[:10])