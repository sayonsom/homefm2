import numpy as np
import random

# Expanded list of activities
activities = [
    'Watching TV', 'Sleeping', 'Working from Home', 'Exercising',
    'Meditating', 'Having a Video Call', 'Playing Video Games',
    'Listening to Music', 'Doing Yoga', 'Reading a Book',
    'Entertaining Guests', 'Taking a Nap', 'Studying',
    'Dancing', 'Doing Household Chores', 'Practicing a Musical Instrument',
    'Writing', 'Painting or Drawing', 'Online Shopping',
    'Watching a Movie', 'Having a Romantic Dinner'
]

# Expanded list of interruptions
interruptions = [
    'Phone Call', 'Doorbell', 'Fire Alarm', 'Power Outage',
    'Unexpected Guest', 'Pet Needs Attention', 'Loud Noise Outside',
    'Received an Important Email', 'Food Delivery Arrival',
    'Neighbor Playing Loud Music', 'Water Leak Detection',
    'Baby Crying', 'Appliance Malfunction', 'Weather Alert',
    'Social Media Notification', 'Breaking News Alert'
]

# Define appliances with their possible states and settings
appliances = {
    'AC': {'states': ['Off', 'On'], 'settings': {'Temperature': range(16, 31)}},
    'TV': {'states': ['Off', 'On'], 'settings': {'Volume': range(0, 101, 5)}},
    'Lights': {'states': ['Off', 'On'], 'settings': {'Brightness': range(0, 101, 5)}}
}

def generate_activity_transition_matrix(activities):
    n = len(activities)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.6  # High probability of continuing the same activity
            elif abs(i - j) <= 2:
                matrix[i][j] = 0.1  # Higher probability for similar activities
            else:
                matrix[i][j] = 0.2 / (n - 3)  # Lower probability for dissimilar activities
    
    # Normalize probabilities
    for i in range(n):
        matrix[i] = matrix[i] / np.sum(matrix[i])
    
    return matrix

# Generate the activity transition matrix
activity_transition_matrix = generate_activity_transition_matrix(activities)

# Define the interruption-activity impact matrix
interruption_activity_matrix = np.random.rand(len(interruptions), len(activities))
interruption_activity_matrix = interruption_activity_matrix / interruption_activity_matrix.sum(axis=1)[:, np.newaxis]

def predict_next_action(current_activity, interruption, current_appliance_state, current_appliance_settings):
    current_activity_index = activities.index(current_activity)
    interruption_index = interruptions.index(interruption)
    
    # Probability of activity change due to interruption
    change_probability = interruption_activity_matrix[interruption_index][current_activity_index]
    
    if random.random() < change_probability:
        # Activity changes due to interruption
        new_activity_probs = interruption_activity_matrix[interruption_index]
        new_activity_index = np.random.choice(len(activities), p=new_activity_probs)
        new_activity = activities[new_activity_index]
    else:
        # Activity doesn't change, use activity transition matrix for possible change
        new_activity_probs = activity_transition_matrix[current_activity_index]
        new_activity_index = np.random.choice(len(activities), p=new_activity_probs)
        new_activity = activities[new_activity_index]
    
    # Predict appliance changes based on the new activity
    new_appliance_state, new_appliance_settings = predict_appliance_changes(new_activity, current_appliance_state, current_appliance_settings)
    
    return new_activity, new_appliance_state, new_appliance_settings

def predict_appliance_changes(new_activity, current_appliance_state, current_appliance_settings):
    new_appliance_state = current_appliance_state.copy()
    new_appliance_settings = current_appliance_settings.copy()
    
    # Define activity-specific appliance changes
    activity_appliance_changes = {
        'Watching TV': {'TV': 'On', 'Lights': 'On', 'AC': 'On'},
        'Sleeping': {'TV': 'Off', 'Lights': 'Off', 'AC': 'On'},
        'Working from Home': {'Lights': 'On', 'AC': 'On'},
        'Exercising': {'TV': 'On', 'Lights': 'On', 'AC': 'On'},
        'Meditating': {'TV': 'Off', 'Lights': 'On', 'AC': 'On'},
        'Having a Video Call': {'Lights': 'On', 'AC': 'On'},
        'Playing Video Games': {'TV': 'On', 'Lights': 'On', 'AC': 'On'},
        'Listening to Music': {'TV': 'Off', 'Lights': 'On', 'AC': 'On'},
        'Doing Yoga': {'TV': 'Off', 'Lights': 'On', 'AC': 'On'},
        'Reading a Book': {'Lights': 'On', 'AC': 'On'},
        'Entertaining Guests': {'TV': 'On', 'Lights': 'On', 'AC': 'On'},
        'Taking a Nap': {'TV': 'Off', 'Lights': 'Off', 'AC': 'On'},
        'Studying': {'Lights': 'On', 'AC': 'On'},
        'Dancing': {'TV': 'On', 'Lights': 'On', 'AC': 'On'},
        'Doing Household Chores': {'Lights': 'On', 'AC': 'On'},
        'Practicing a Musical Instrument': {'Lights': 'On', 'AC': 'On'},
        'Writing': {'Lights': 'On', 'AC': 'On'},
        'Painting or Drawing': {'Lights': 'On', 'AC': 'On'},
        'Online Shopping': {'Lights': 'On', 'AC': 'On'},
        'Watching a Movie': {'TV': 'On', 'Lights': 'Off', 'AC': 'On'},
        'Having a Romantic Dinner': {'Lights': 'On', 'AC': 'On', 'TV': 'Off'}
    }
    
    # Apply activity-specific changes
    for appliance, state in activity_appliance_changes.get(new_activity, {}).items():
        new_appliance_state[appliance] = state
    
    # Adjust settings based on the activity
    if new_activity in ['Sleeping', 'Taking a Nap']:
        new_appliance_settings['AC']['Temperature'] = random.randint(20, 24)
        new_appliance_settings['Lights']['Brightness'] = 0
    elif new_activity in ['Exercising', 'Dancing']:
        new_appliance_settings['AC']['Temperature'] = random.randint(18, 22)
        new_appliance_settings['TV']['Volume'] = random.randint(60, 80)
    elif new_activity in ['Reading a Book', 'Studying', 'Writing']:
        new_appliance_settings['Lights']['Brightness'] = random.randint(70, 100)
        new_appliance_settings['AC']['Temperature'] = random.randint(22, 26)
    elif new_activity in ['Watching TV', 'Playing Video Games', 'Watching a Movie']:
        new_appliance_settings['TV']['Volume'] = random.randint(40, 70)
        new_appliance_settings['Lights']['Brightness'] = random.randint(20, 50)
    elif new_activity == 'Having a Romantic Dinner':
        new_appliance_settings['Lights']['Brightness'] = random.randint(30, 50)
        new_appliance_settings['AC']['Temperature'] = random.randint(22, 24)
    
    return new_appliance_state, new_appliance_settings

# Example usage
current_activity = "Watching TV"
interruption = "Doorbell"
current_appliance_state = {'AC': 'On', 'TV': 'On', 'Lights': 'On'}
current_appliance_settings = {'AC': {'Temperature': 24}, 'TV': {'Volume': 50}, 'Lights': {'Brightness': 70}}

print(f"Current Activity: {current_activity}")
print(f"Current Appliance State: {current_appliance_state}")
print(f"Current Appliance Settings: {current_appliance_settings}")
print(f"Interruption: {interruption}")

next_activity, new_appliance_state, new_appliance_settings = predict_next_action(
    current_activity, interruption, current_appliance_state, current_appliance_settings)

print(f"\nPredicted Next Activity: {next_activity}")
print(f"Predicted Appliance State: {new_appliance_state}")
print(f"Predicted Appliance Settings: {new_appliance_settings}")