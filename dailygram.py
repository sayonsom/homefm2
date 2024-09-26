import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

# Define mental states
STATES = ['Sleep', 'Relax', 'Normal', 'Concentrate', 'Tensed', 'Panic']

# Define events with corrected probabilities
EVENTS = [
    {"name": "Fire alarm", "annual_prob": 0.2, "duration": timedelta(minutes=30), "state": "Panic"},
    {"name": "Child crying at night", "annual_prob": 52, "duration": timedelta(minutes=20), "state": "Tensed"},
    {"name": "Loud noise/breaking sound", "annual_prob": 12, "duration": timedelta(minutes=10), "state": "Panic"},
    {"name": "Water leak", "annual_prob": 0.2, "duration": timedelta(hours=1), "state": "Panic"},
    {"name": "Power outage", "annual_prob": 2, "duration": timedelta(hours=2), "state": "Tensed"},
    {"name": "Insect/pest sighting", "annual_prob": 24, "duration": timedelta(minutes=15), "state": "Tensed"},
    {"name": "Smoke detector low battery beep", "annual_prob": 1, "duration": timedelta(minutes=45), "state": "Tensed"},
    {"name": "Food delivery", "annual_prob": 52, "duration": timedelta(minutes=10), "state": "Relax"},
    {"name": "Phone call", "annual_prob": 5, "duration": timedelta(minutes=15), "state": "Normal"},
    {"name": "Doorbell", "annual_prob": 1, "duration": timedelta(minutes=5), "state": "Normal"},
    {"name": "Text message", "annual_prob": 10, "duration": timedelta(minutes=2), "state": "Normal"},
    {"name": "Dog barking", "annual_prob": 5, "duration": timedelta(minutes=5), "state": "Normal"},
    {"name": "Neighbor's loud music", "annual_prob": 24, "duration": timedelta(minutes=30), "state": "Tensed"},
    {"name": "Unexpected guest", "annual_prob": 12, "duration": timedelta(hours=1), "state": "Normal"},
    {"name": "Minor argument", "annual_prob": 52, "duration": timedelta(minutes=20), "state": "Tensed"},
    {"name": "Remembering unfinished task", "annual_prob": 104, "duration": timedelta(minutes=15), "state": "Concentrate"},
]

def usual_state(hour):
    if 0 <= hour < 7:
        return 'Sleep'
    elif 7 <= hour < 9:
        return 'Relax'
    elif 9 <= hour < 18:
        return 'Normal'
    elif 18 <= hour < 22:
        return 'Relax'
    else:
        return 'Sleep'

def event_occurs(event, time_step):
    daily_prob = event['annual_prob'] / 365
    return random.random() < (daily_prob / (24 * 60 / time_step))

def simulate_day(time_step=15):
    start_time = datetime(2024, 1, 1, 0, 0)
    end_time = start_time + timedelta(days=4)
    current_time = start_time
    time_points = []
    states = []
    events_occurred = []

    while current_time < end_time:
        hour = current_time.hour + current_time.minute / 60
        base_state = usual_state(hour)
        
        # Check for new events
        for event in EVENTS:
            if event_occurs(event, time_step):
                event_end_time = current_time + event['duration']
                events_occurred.append((current_time, event_end_time, event['state'], event['name']))

        # Determine the current state
        current_state = base_state
        for start, end, state, name in events_occurred:
            if start <= current_time < end:
                current_state = state
                break

        time_points.append(current_time)
        states.append(current_state)
        
        current_time += timedelta(minutes=time_step)

    return time_points, states, events_occurred

# Run the simulation
time_points, states, events_occurred = simulate_day()

# Plotting
plt.figure(figsize=(15, 8))

# Convert states to numbers for plotting
state_numbers = [STATES.index(state) for state in states]

# Plot the mental states
plt.step([mdates.date2num(t) for t in time_points], state_numbers, where='post', color='blue', alpha=0.7)

# Create a color map for events
cmap = plt.get_cmap('tab10')
event_colors = {event['name']: cmap(i % 10) for i, event in enumerate(EVENTS)}

# Plot events as dots
for start, _, state, name in events_occurred:
    plt.plot(mdates.date2num(start), STATES.index(state), 'o', color=event_colors[name], markersize=8)

# Customize the plot
plt.yticks(range(len(STATES)), STATES)
plt.xlabel('Time of Day')
plt.ylabel('Mental State')
plt.title('Mental State Transitions over 24 Hours with Events')
plt.grid(True, alpha=0.3)

# Format x-axis to show hours
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))

# Create legend for events
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=event['name'], 
                              markerfacecolor=event_colors[event['name']], markersize=8)
                   for event in EVENTS if any(e[3] == event['name'] for e in events_occurred)]

# Add legend to the right of the plot
plt.legend(handles=legend_elements, title='Events', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Print event summary
print("Events that occurred:")
for start, end, state, name in events_occurred:
    print(f"{name}: {start.strftime('%H:%M')} - {end.strftime('%H:%M')} ({state})")