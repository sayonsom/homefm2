import pandas as pd
import random
from datetime import datetime, timedelta
from interrruptionModel2 import handle_interruption, predict_next_activity, predict_appliance_changes,choose_initial_activity

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



def simulate_day(start_time):
    current_time = start_time
    current_activity = choose_initial_activity(current_time)
    previous_activity = current_activity  # Initialize previous_activity
    current_appliance_state = {'AC': 'On', 'TV': 'Off', 'Lights': 'Off'}
    current_appliance_settings = {'AC': {'Temperature': 22}, 'TV': {'Volume': 0}, 'Lights': {'Brightness': 0}}
    
    day_data = []
    interruption_count = 0
    
    while current_time.date() == start_time.date():
        # Determine if an interruption occurs
        if interruption_count < 4 and random.random() < 0.1:  # 10% chance of interruption, max 4 per day
            interruption = random.choice(interruptions)
            interruption_count += 1
            new_activity, new_appliance_state, new_appliance_settings = handle_interruption(
                current_activity, interruption, current_appliance_state, current_appliance_settings)
            
            # Ensure all keys exist in new_appliance_settings
            for appliance in ['AC', 'TV', 'Lights']:
                if appliance not in new_appliance_settings:
                    new_appliance_settings[appliance] = current_appliance_settings[appliance]
            
            day_data.append({
                'Time': current_time,
                'Activity': current_activity,
                'Interruption': interruption,
                'Changed_Activity': new_activity,
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
            current_activity = new_activity
            current_appliance_state = new_appliance_state
            current_appliance_settings = new_appliance_settings
        else:
            day_data.append({
                'Time': current_time,
                'Activity': current_activity,
                'Interruption': None,
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
        previous_activity = current_activity  # Store the current activity before changing
        current_activity = predict_next_activity(current_activity, previous_activity, current_time)
        current_appliance_state, current_appliance_settings = predict_appliance_changes(
            current_activity, current_appliance_state, current_appliance_settings)
    
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