import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np
from interrruptionModel2 import handle_interruption, choose_initial_activity, get_time_of_day

# Assume these are already loaded
time_of_day_data = pd.read_csv('activity_probability_given_time_of_day.csv', index_col='Activity Variable_Name')
activity_transition_matrix = pd.read_csv('activity_transition_matrix.csv', index_col='Current Activity')


interruptions = [
    'Phone Call', 'Doorbell', 'Fire Alarm', 'Power Outage',
    'Unexpected Guest', 'Pet Needs Attention', 'Loud Noise Outside',
    'Received an Important Email', 'Food Delivery Arrival',
    'Neighbor Playing Loud Music', 'Water Leak Detection',
    'Baby Crying', 'Appliance Malfunction', 'Weather Alert',
    'Social Media Notification', 'Breaking News Alert'
]

def predict_appliance_changes(current_activity, previous_activity, is_emergency, current_appliance_state, current_appliance_settings):
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
        'wake_up': {
            'Lights': 'On',
            'AC': 'On',
            'settings': {
                'Lights': {'Brightness': 70},
                'AC': {'Temperature': random.randint(22, 24)}
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
        'emergency': {
            'TV': 'Off',
            'Lights': 'On',
            'AC': 'Off',
            'settings': {
                'Lights': {'Brightness': 100}
            }
        }
    }
    
    # If transitioning out of emergency, use previous activity's settings
    if is_emergency and current_activity != 'emergency':
        activity_changes = activity_appliance_changes.get(previous_activity, {})
    else:
        activity_changes = activity_appliance_changes.get(current_activity, {})
    
    # Apply activity-specific changes
    for appliance, state in activity_changes.items():
        if appliance != 'settings':
            new_appliance_state[appliance] = state
    
    # Apply settings changes
    for appliance, settings in activity_changes.get('settings', {}).items():
        if appliance not in new_appliance_settings:
            new_appliance_settings[appliance] = {}
        new_appliance_settings[appliance].update(settings)
    
    # Special handling for emergency
    if current_activity == 'emergency':
        new_appliance_state['TV'] = 'Off'
        new_appliance_state['Lights'] = 'On'
        new_appliance_state['AC'] = 'Off'
        new_appliance_settings['Lights']['Brightness'] = 100
    
    return new_appliance_state, new_appliance_settings



def predict_next_activity(current_activity, previous_activity, current_time):
    # If current activity is emergency, return to the previous activity
    if current_activity == 'emergency':
        return previous_activity

    time_of_day = get_time_of_day(current_time)
    
    # Get probabilities based on time of day
    time_probs = time_of_day_data.loc[:, time_of_day]
    
    # Get probabilities based on current activity
    if current_activity in activity_transition_matrix.index:
        transition_probs = activity_transition_matrix.loc[current_activity]
    else:
        # If the activity is not in the transition matrix, use a uniform distribution
        transition_probs = pd.Series(1.0 / len(activity_transition_matrix.columns), 
                                     index=activity_transition_matrix.columns)
    
    # Ensure both Series have the same index
    common_activities = time_probs.index.intersection(transition_probs.index)
    time_probs = time_probs[common_activities]
    transition_probs = transition_probs[common_activities]
    
    # Normalize probabilities
    time_probs = time_probs / time_probs.sum()
    transition_probs = transition_probs / transition_probs.sum()
    
    # Combine probabilities (you can adjust the weights as needed)
    combined_probs = 0.6 * time_probs + 0.4 * transition_probs
    
    # Normalize the combined probabilities
    combined_probs = combined_probs / combined_probs.sum()
    
    # Choose next activity
    return np.random.choice(combined_probs.index, p=combined_probs.values)


def simulate_day(start_time):
    current_time = start_time
    current_activity = choose_initial_activity(current_time)
    previous_activity = current_activity  # Initialize previous_activity
    current_appliance_state = {'AC': 'On', 'TV': 'Off', 'Lights': 'Off'}
    current_appliance_settings = {'AC': {'Temperature': 22}, 'TV': {'Volume': 0}, 'Lights': {'Brightness': 0}}
    is_emergency = False
    emergency_duration = 0  # Track how long the emergency has been ongoing
    
    day_data = []
    interruption_count = 0
    
    while current_time.date() == start_time.date():
        # Determine if an interruption occurs
        if interruption_count < 4 and random.random() < 0.1:  # 10% chance of interruption, max 4 per day
            interruption = random.choice(interruptions)
            interruption_count += 1
            new_activity, new_appliance_state, new_appliance_settings = handle_interruption(
                current_activity, interruption, current_appliance_state, current_appliance_settings)
            
            # Check if the new activity is an emergency
            is_emergency = new_activity == 'emergency'
            if is_emergency:
                emergency_duration = 1  # Start counting emergency duration
            
            # Ensure all keys exist in new_appliance_settings
            for appliance in ['AC', 'TV', 'Lights']:
                if appliance not in new_appliance_settings:
                    new_appliance_settings[appliance] = current_appliance_settings[appliance]
            
            day_data.append({
                'Time': current_time,
                'Activity': current_activity,
                'Interruption': interruption,
                'Changed_Activity': new_activity if not is_emergency else 'emergency_response',
                'AC_status': current_appliance_state['AC'],
                'AC_Setting': current_appliance_settings['AC'].get('Temperature', 22),
                'TV_status': current_appliance_state['TV'],
                'Light_status': current_appliance_state['Lights'],
                'After_Interruption_AC_status': new_appliance_state.get('AC', 'Unknown'),
                'After_Interruption_AC_Setting': new_appliance_settings['AC'].get('Temperature', 22),
                'After_Interruption_TV_status': new_appliance_state.get('TV', 'Unknown'),
                'After_Interruption_TV_Setting': new_appliance_settings['TV'].get('Volume', 0),
                'After_Interruption_Light_status': new_appliance_state.get('Lights', 'Unknown'),
                'After_Interruption_Light_Setting': new_appliance_settings['Lights'].get('Brightness', 0)
            })
            
            previous_activity = current_activity  # Store the current activity before changing
            current_activity = new_activity if not is_emergency else previous_activity
            current_appliance_state = new_appliance_state
            current_appliance_settings = new_appliance_settings
        else:
            day_data.append({
                'Time': current_time,
                'Activity': current_activity,
                'Interruption': 'emergency_ongoing' if is_emergency else None,
                'Changed_Activity': None,
                'AC_status': current_appliance_state['AC'],
                'AC_Setting': current_appliance_settings['AC'].get('Temperature', 22),
                'TV_status': current_appliance_state['TV'],
                'Light_status': current_appliance_state['Lights'],
                'After_Interruption_AC_status': None,
                'After_Interruption_AC_Setting': None,
                'After_Interruption_TV_status': None,
                'After_Interruption_TV_Setting': None,
                'After_Interruption_Light_status': None,
                'After_Interruption_Light_Setting': None
            })
        
        # Move to next time slot and predict next activity
        current_time += timedelta(minutes=30)
        
        if is_emergency:
            emergency_duration += 1
            if emergency_duration >= 2:  # Emergency lasts for 1 hour (2 x 30-minute intervals)
                is_emergency = False
                emergency_duration = 0
                # Transition back to a normal activity
                current_activity = predict_next_activity(previous_activity, previous_activity, current_time)
        else:
            previous_activity = current_activity  # Store the current activity before changing
            current_activity = predict_next_activity(current_activity, previous_activity, current_time)
        
        current_appliance_state, current_appliance_settings = predict_appliance_changes(
            current_activity, previous_activity, is_emergency, current_appliance_state, current_appliance_settings)
    
    return pd.DataFrame(day_data)


def monte_carlo_simulation(num_days=100):
    all_data = []
    for _ in range(num_days):
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=random.randint(1, 365))
        try:
            day_data = simulate_day(start_time)
            all_data.append(day_data)
        except Exception as e:
            print(f"Error simulating day: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data generated in the simulation.")
    
    return pd.concat(all_data, ignore_index=True)

# Run the simulation and save results
results = monte_carlo_simulation(100)
results.to_csv('monte_carlo_simulation_results.csv', index=False)
print("Simulation complete. Results saved to 'monte_carlo_simulation_results.csv'")

# Display a sample of the results
print(results.head())