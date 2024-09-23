import pandas as pd
import random
from datetime import datetime, timedelta

# Assume these are loaded from CSV files
time_of_day_data = pd.read_csv('activity_probability_given_time_of_day.csv', index_col='Activity Variable_Name')
activity_transition_matrix = pd.read_csv('expanded_activity_transition_matrix.csv', index_col='Current Activity')
time_of_day_data = pd.read_csv('activity_probability_given_time_of_day.csv', index_col='Activity Variable_Name')

def get_time_of_day(current_time):
    hour = current_time.hour
    if 5 <= hour < 8:
        return 'Early Morning (5AM-8AM)'
    elif 8 <= hour < 12:
        return 'Morning (8AM-12PM)'
    elif 12 <= hour < 16:
        return 'Afternoon (12PM-4PM)'
    elif 16 <= hour < 20:
        return 'Evening (4PM-8PM)'
    elif 20 <= hour < 24:
        return 'Night (8PM-12AM)'
    else:
        return 'Late Night (12AM-5AM)'

def choose_initial_activity(current_time):
    time_of_day = get_time_of_day(current_time)
    probabilities = time_of_day_data[time_of_day]
    return random.choices(probabilities.index, weights=probabilities.values)[0]

# List of possible interruptions
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

def get_time_of_day(current_time):
    hour = current_time.hour
    if 5 <= hour < 8:
        return 'Early Morning (5AM-8AM)'
    elif 8 <= hour < 12:
        return 'Morning (8AM-12PM)'
    elif 12 <= hour < 16:
        return 'Afternoon (12PM-4PM)'
    elif 16 <= hour < 20:
        return 'Evening (4PM-8PM)'
    elif 20 <= hour < 24:
        return 'Night (8PM-12AM)'
    else:
        return 'Late Night (12AM-5AM)'

def predict_next_activity(current_activity, current_time):
    time_of_day = get_time_of_day(current_time)
    
    # Get probabilities based on time of day
    time_probs = time_of_day_data.loc[:, time_of_day]
    
    # Get probabilities based on current activity
    transition_probs = activity_transition_matrix.loc[current_activity]
    
    # Combine probabilities (you can adjust the weights as needed)
    combined_probs = 0.6 * time_probs + 0.4 * transition_probs
    
    # Remove any NaN values and normalize
    combined_probs = combined_probs.dropna()
    combined_probs = combined_probs / combined_probs.sum()
    
    # Choose next activity
    return random.choices(combined_probs.index, weights=combined_probs.values)[0]


def predict_appliance_changes(activity, current_appliance_state, current_appliance_settings):
    new_appliance_state = current_appliance_state.copy()
    new_appliance_settings = current_appliance_settings.copy()
    
    # Define activity-specific appliance changes
    activity_appliance_changes = {
        'sleep': {
            'TV': 'Off',
            'Lights': 'On',
            'AC': 'On',
            'settings': {
                'Lights': {'Brightness': 10},
                'AC': {'Temperature': random.randint(20, 22)}
            }
        },
        'prepare_for_bed': {
            'TV': 'Off',
            'Lights': 'On',
            'AC': 'On',
            'settings': {
                'Lights': {'Brightness': 30},
                'AC': {'Temperature': random.randint(21, 23)}
            }
        },
        'wake_up': {
            'Lights': 'On',
            'AC': 'On',
            'settings': {
                'Lights': {'Brightness': 70},
                'AC': {'Temperature': random.randint(22, 24)}
            }
        },
        'power_outage': {
            'TV': 'Off',
            'Lights': 'Off',
            'AC': 'Off',
            'settings': {
                'Lights': {'Brightness': 0}
            }
        },
        'study': {
            'Lights': 'On',
            'AC': 'On',
            'settings': {
                'Lights': {'Brightness': 100},
                'AC': {'Temperature': random.randint(23, 25)}
            }
        },
        'watch_media': {
            'TV': 'On',
            'Lights': 'On',
            'AC': 'On',
            'settings': {
                'TV': {'Volume': random.randint(30, 50)},
                'Lights': {'Brightness': 40},
                'AC': {'Temperature': random.randint(22, 24)}
            }
        },
        'exercise': {
            'AC': 'On',
            'Lights': 'On',
            'settings': {
                'Lights': {'Brightness': 100},
                'AC': {'Temperature': random.randint(18, 20)}
            }
        },
        'cook': {
            'Lights': 'On',
            'AC': 'On',
            'settings': {
                'Lights': {'Brightness': 100},
                'AC': {'Temperature': random.randint(23, 25)}
            }
        },
        'eat': {
            'Lights': 'On',
            'AC': 'On',
            'settings': {
                'Lights': {'Brightness': 80},
                'AC': {'Temperature': random.randint(22, 24)}
            }
        },
        'shower': {
            'Lights': 'On',
            'AC': 'Off',
            'settings': {
                'Lights': {'Brightness': 80}
            }
        },
        'socialize': {
            'Lights': 'On',
            'AC': 'On',
            'TV': 'On',
            'settings': {
                'Lights': {'Brightness': 70},
                'AC': {'Temperature': random.randint(22, 24)},
                'TV': {'Volume': random.randint(20, 40)}
            }
        },
        'game': {
            'TV': 'On',
            'Lights': 'On',
            'AC': 'On',
            'settings': {
                'TV': {'Volume': random.randint(40, 60)},
                'Lights': {'Brightness': 60},
                'AC': {'Temperature': random.randint(21, 23)}
            }
        }
    }
    
    # Apply activity-specific changes
    activity_changes = activity_appliance_changes.get(activity, {})
    for appliance, state in activity_changes.items():
        if appliance != 'settings':
            new_appliance_state[appliance] = state
    
    # Apply settings changes
    for appliance, settings in activity_changes.get('settings', {}).items():
        if appliance not in new_appliance_settings:
            new_appliance_settings[appliance] = {}
        new_appliance_settings[appliance].update(settings)
    
    return new_appliance_state, new_appliance_settings


def handle_interruption(current_activity, interruption, current_appliance_state, current_appliance_settings):
    # Define how different interruptions affect the current activity
    interruption_effects = {
        'Phone Call': {'change_probability': 0.8, 'new_activity': 'communicate'},
        'Doorbell': {'change_probability': 0.7, 'new_activity': 'socialize'},
        'Fire Alarm': {'change_probability': 1.0, 'new_activity': 'emergency'},
        'Power Outage': {'change_probability': 1.0, 'new_activity': 'emergency'},
        'Water Leak Detection': {'change_probability': 1.0, 'new_activity': 'emergency'},
        'Unexpected Guest': {'change_probability': 0.8, 'new_activity': 'socialize'},
        'Pet Needs Attention': {'change_probability': 0.6, 'new_activity': 'other'},
        'Loud Noise Outside': {'change_probability': 0.4, 'new_activity': None},
        'Received an Important Email': {'change_probability': 0.5, 'new_activity': 'communicate'},
        'Food Delivery Arrival': {'change_probability': 0.7, 'new_activity': 'eat'},
        'Neighbor Playing Loud Music': {'change_probability': 0.3, 'new_activity': None},
        'Baby Crying': {'change_probability': 0.8, 'new_activity': 'other'},
        'Appliance Malfunction': {'change_probability': 0.7, 'new_activity': 'other'},
        'Weather Alert': {'change_probability': 0.5, 'new_activity': None},
        'Social Media Notification': {'change_probability': 0.3, 'new_activity': 'browse_social_media'},
        'Breaking News Alert': {'change_probability': 0.4, 'new_activity': 'watch_media'}
    }
    
    effect = interruption_effects.get(interruption, {'change_probability': 0.5, 'new_activity': None})
    
    new_activity = effect['new_activity'] if random.random() < effect['change_probability'] else current_activity
    if new_activity is None:
        new_activity = predict_next_activity(current_activity, datetime.now())
    
    # Special cases for appliance changes during interruptions
    new_appliance_state = current_appliance_state.copy()
    new_appliance_settings = current_appliance_settings.copy()
    
    # Handle emergency situations
    if interruption in ['Fire Alarm', 'Water Leak Detection']:
        new_appliance_state = {'AC': 'Off', 'TV': 'Off', 'Lights': 'On'}
        new_appliance_settings = {'Lights': {'Brightness': 100}}
    elif interruption == 'Power Outage':
        new_appliance_state = {'AC': 'Off', 'TV': 'Off', 'Lights': 'On'}
        new_appliance_settings = {'Lights': {'Brightness': 50}}
    elif new_activity == 'emergency':
        new_appliance_state = {'AC': 'Off', 'TV': 'Off', 'Lights': 'On'}
        new_appliance_settings = {'Lights': {'Brightness': 100}}
    elif interruption in ['Baby Crying', 'Pet Needs Attention', 'Appliance Malfunction']:
        new_appliance_state['TV'] = 'Off'
        new_appliance_state['Lights'] = 'On'
        new_appliance_settings['Lights'] = {'Brightness': 100}
        new_appliance_settings['AC'] = {'Temperature': 22}  # Normal temperature
    
    return new_activity, new_appliance_state, new_appliance_settings

def display_current_state(current_time, current_activity, appliance_state, appliance_settings):
    print(f"\nCurrent Time: {current_time.strftime('%I:%M %p')}")
    print(f"Current Activity: {current_activity}")
    print("Appliance States:")
    for appliance, state in appliance_state.items():
        print(f"  {appliance}: {state}")
    print("Appliance Settings:")
    for appliance, settings in appliance_settings.items():
        print(f"  {appliance}: {settings}")


def interactive_simulation():
    current_time = datetime.now()
    current_activity = choose_initial_activity(current_time)
    current_appliance_state = {'AC': 'On', 'TV': 'Off', 'Lights': 'Off'}
    current_appliance_settings = {'AC': {'Temperature': 22}, 'TV': {'Volume': 0}, 'Lights': {'Brightness': 0}}
    is_emergency = False

    print(f"Starting simulation at {current_time.strftime('%I:%M %p')}")
    print(f"Initial activity based on time of day: {current_activity}")

    # Initialize appliance states based on the initial activity
    current_appliance_state, current_appliance_settings = predict_appliance_changes(
        current_activity, current_appliance_state, current_appliance_settings)

    while True:
        display_current_state(current_time, current_activity, current_appliance_state, current_appliance_settings)

        # Ask for context (new activity)
        print("\nAvailable activities:")
        for i, activity in enumerate(time_of_day_data.index, 1):
            print(f"{i}. {activity}")
        activity_choice = input("Enter the number of the new activity (or press Enter to keep current): ")
        
        if activity_choice:
            new_activity = time_of_day_data.index[int(activity_choice) - 1]
            
            # Reset appliance states if coming out of an emergency
            if is_emergency:
                current_appliance_state = {'AC': 'On', 'TV': 'Off', 'Lights': 'On'}
                current_appliance_settings = {'AC': {'Temperature': 22}, 'TV': {'Volume': 0}, 'Lights': {'Brightness': 50}}
                is_emergency = False
                print("\nEmergency situation resolved. Appliance settings reset to normal.")
            
            current_activity = new_activity
            current_appliance_state, current_appliance_settings = predict_appliance_changes(
                current_activity, current_appliance_state, current_appliance_settings)
            print("\nUpdated state after activity change:")
            display_current_state(current_time, current_activity, current_appliance_state, current_appliance_settings)

        # Ask for interruption
        print("\nPossible interruptions:")
        for i, interruption in enumerate(interruptions, 1):
            print(f"{i}. {interruption}")
        interruption_choice = input("Enter the number of the interruption (or press Enter for none): ")

        if interruption_choice:
            interruption = interruptions[int(interruption_choice) - 1]
            new_activity, new_appliance_state, new_appliance_settings = handle_interruption(
                current_activity, interruption, current_appliance_state, current_appliance_settings)
            
            print(f"\nInterruption occurred: {interruption}")
            print("Immediate effects:")
            display_current_state(current_time, new_activity, new_appliance_state, new_appliance_settings)
            
            confirm = input("Accept these changes? (y/n): ").lower()
            if confirm == 'y':
                current_activity = new_activity
                current_appliance_state = new_appliance_state
                current_appliance_settings = new_appliance_settings
                is_emergency = interruption in ['Fire Alarm', 'Power Outage', 'Water Leak Detection'] or new_activity == 'emergency'

        # Update time
        current_time += timedelta(minutes=30)

        # Ask if user wants to continue
        if input("\nContinue simulation? (y/n): ").lower() != 'y':
            break


# Run the interactive simulation
# interactive_simulation()