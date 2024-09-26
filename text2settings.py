import pandas as pd
import datetime
from collections import defaultdict
from transformers import pipeline, set_seed, AutoTokenizer
import os
from dateutil import parser
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set the base path for the CSV files
base_path = os.path.join('personas', 'college_student_shared_apartment')

# Load data from CSVs using pandas
contexts = pd.read_csv(os.path.join(base_path, 'daily_contexts.csv'))
activities = pd.read_csv(os.path.join(base_path, 'activity_weights.csv'))
activity_probabilities = pd.read_csv(os.path.join(base_path, 'activity_probability_given_time_of_day.csv'))
activity_transitions = pd.read_csv(os.path.join(base_path, 'activity_transition_matrix.csv'))
appliance_settings = pd.read_csv(os.path.join(base_path, 'appliance_settings_comfort.csv'))


# Convert probability columns to float
prob_columns = activity_probabilities.columns.drop('Activity Variable_Name')
activity_probabilities[prob_columns] = activity_probabilities[prob_columns].astype(float)

transition_columns = activity_transitions.columns.drop('Current Activity')
activity_transitions[transition_columns] = activity_transitions[transition_columns].astype(float)

# Initialize the language model and tokenizer
model_name = 'distilgpt2'
generator = pipeline('text-generation', model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
set_seed(42)

def parse_datetime(datetime_str):
    return parser.parse(datetime_str)

def get_time_of_day(time):
    hour = time.hour
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

def infer_health_condition(heart_rate, skin_temp):
    conditions = []
    if skin_temp > 99:
        conditions.append("fever")
    if skin_temp < 97:
        conditions.append("feeling cold")
    if skin_temp > 99.5:
        conditions.append("feeling hot")
    if heart_rate > 100:
        conditions.append("elevated heart rate")
    if heart_rate < 60:
        conditions.append("low heart rate")
    
    if not conditions:
        return "normal"
    return ", ".join(conditions)

def infer_activities_from_events(current_time, past_events, future_events):
    past_events = sorted(past_events, key=lambda x: x[1])
    future_events = sorted(future_events, key=lambda x: x[1])
    
    past_activity = past_events[-1][0] if past_events else "Unknown"
    next_activity = future_events[0][0] if future_events else "Unknown"
    
    time_to_next_event = (future_events[0][1] - current_time).total_seconds() / 3600 if future_events else None
    
    return past_activity, next_activity, time_to_next_event

def generate_context_prompt(locations, heart_rate, sleep_score, skin_temp, current_time, past_events, future_events):
    time_of_day = get_time_of_day(current_time)
    past_activity, next_activity, time_to_next_event = infer_activities_from_events(current_time, past_events, future_events)
    health_condition = infer_health_condition(heart_rate, skin_temp)
    
    prompt = f"Student data: Date/Time {current_time.strftime('%Y-%m-%d %I:%M %p')} ({time_of_day}), "
    prompt += f"Locations: {', '.join(locations)}, "
    prompt += f"Heart rate: {heart_rate}, Sleep: {sleep_score}/100, Temp: {skin_temp}F, "
    prompt += f"Health: {health_condition}, "
    prompt += f"Last: {past_activity} at {past_events[-1][1].strftime('%I:%M %p')}, "
    if time_to_next_event is not None:
        prompt += f"Next: {next_activity} in {time_to_next_event:.1f}h at {future_events[0][1].strftime('%I:%M %p')}. "
    else:
        prompt += "No upcoming events. "
    prompt += "Context: "

    return prompt

def guess_context(locations, heart_rate, sleep_score, skin_temp, current_time, past_events, future_events):
    prompt = generate_context_prompt(locations, heart_rate, sleep_score, skin_temp, current_time, past_events, future_events)
    
    # Get the number of tokens in the prompt
    prompt_tokens = tokenizer.encode(prompt)
    max_new_tokens = 1024 - len(prompt_tokens) - 1  # Subtract 1 for safety

    # Generate context prediction using DistilGPT-2
    response = generator(prompt, max_length=len(prompt_tokens) + max_new_tokens, num_return_sequences=1, truncation=True)[0]['generated_text']
    
    # Extract the predicted context from the response
    predicted_context = response.split("Context:")[-1].strip()
    
    # Find the closest matching context from our predefined list
    context_list = contexts['Context Description'].tolist()
    closest_context = min(context_list, key=lambda x: len(set(x.lower().split()) & set(predicted_context.lower().split())))
    
    return closest_context

def predict_activity(context, current_time, past_events, future_events):
    time_of_day = get_time_of_day(current_time)
    context_activities = activities[activities['Activity Description'].str.lower().str.contains(context.lower(), na=False)]['Activity Variable_Name'].tolist()
    
    if not context_activities:
        context_activities = activities['Activity Variable_Name'].tolist()
    
    past_activity = past_events[-1][0] if past_events else None
    
    activity_scores = {}
    
    if past_activity:
        transition_probs = activity_transitions[activity_transitions['Current Activity'] == past_activity]
        if not transition_probs.empty:
            activity_scores = {activity: transition_probs.iloc[0].get(activity, 0) for activity in context_activities}
    
    if not activity_scores:
        for activity in context_activities:
            prob_row = activity_probabilities[activity_probabilities['Activity Variable_Name'] == activity]
            if not prob_row.empty and time_of_day in prob_row.columns:
                activity_scores[activity] = float(prob_row[time_of_day].iloc[0])
    
    # Adjust scores based on upcoming events
    if future_events:
        next_event = future_events[0]
        time_to_event = (next_event[1] - current_time).total_seconds() / 3600  # in hours
        if time_to_event < 1:  # If the event is less than an hour away
            event_related_activities = [act for act in context_activities if act.lower() in next_event[0].lower()]
            for activity in event_related_activities:
                activity_scores[activity] = activity_scores.get(activity, 0) + 1  # Boost the score

    if not activity_scores:
        return 'study'
    
    return max(activity_scores, key=activity_scores.get)

def get_appliance_settings(activity):
    settings = appliance_settings[appliance_settings['Activity'] == activity]
    if settings.empty:
        return {
            'TV': 'Off',
            'AC Temperature (°F)': '72',
            'Smart Lights': 'Bright cool (70%)',
            'Washing Machine': 'N/A',
            'Fridge': 'N/A'
        }
    return settings.iloc[0].to_dict()

def predict_context_and_settings(current_datetime, past_events, future_events, locations, heart_rate, sleep_score, skin_temp):
    current_datetime = parse_datetime(current_datetime)
    past_events = [(event, parse_datetime(time)) for event, time in past_events]
    future_events = [(event, parse_datetime(time)) for event, time in future_events]
    
    context = guess_context(locations, heart_rate, sleep_score, skin_temp, current_datetime, past_events, future_events)
    activity = predict_activity(context, current_datetime, past_events, future_events)
    settings = get_appliance_settings(activity)
    
    return {
        'Context': context,
        'Predicted Activity': activity,
        'Appliance Settings': settings,
        'Health Condition': infer_health_condition(heart_rate, skin_temp)
    }

# Example usage
if __name__ == "__main__":
    current_datetime = input("Enter current date and time (YYYY-MM-DD HH:MM AM/PM): ")
    past_events = [
        ("Group project meeting", "2023-09-17 2:00 PM"),
        ("Lunch with roommate", "2023-09-17 12:30 PM"),
        ("Physics lecture", "2023-09-17 10:00 AM")
    ]
    future_events = [
        ("Birthday party", "2023-09-17 8:00 PM"),
        ("Call with academic advisor", "2023-09-18 10:00 AM"),
        ("Movie night with friends", "2023-09-18 9:00 PM")
    ]
    locations = ["university", "coffee shop", "home"]
    heart_rate = 72
    sleep_score = 90
    skin_temp = 96.43
    
    try:
        result = predict_context_and_settings(current_datetime, past_events, future_events, locations, heart_rate, sleep_score, skin_temp)
        print(f"\nCurrent Date/Time: {current_datetime}")
        print(f"Predicted Context: {result['Context']}")
        print(f"Predicted Activity: {result['Predicted Activity']}")
        print(f"Health Condition: {result['Health Condition']}")
        print("Recommended Appliance Settings:")
        for appliance, setting in result['Appliance Settings'].items():
            if appliance != 'Activity':
                print(f"  {appliance}: {setting}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your input data and CSV files.")