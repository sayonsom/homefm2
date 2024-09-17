import csv
from collections import defaultdict
import datetime
import random
from transformers import pipeline, set_seed

# Load data from CSVs (assuming they're in the same directory as this script)
def load_csv(filename):
    with open(filename, 'r') as f:
        return list(csv.DictReader(f))

contexts = load_csv('student-daily-contexts-csv.csv')
activities = load_csv('student-activities-csv.csv')
activity_probabilities = load_csv('activity-time-probabilities-csv.csv')
activity_transitions = load_csv('activity-transition-matrix.csv')
appliance_settings = load_csv('activity-appliance-settings-csv.csv')

# Convert probabilities to floats
for row in activity_probabilities:
    for key, value in row.items():
        if key != 'Activity Variable_Name':
            row[key] = float(value)

for row in activity_transitions:
    for key, value in row.items():
        if key != 'Current Activity':
            row[key] = float(value)

# Initialize the language model
generator = pipeline('text-generation', model='distilgpt2')
set_seed(42)

def generate_text_from_data(locations, heart_rate, sleep_score, skin_temp):
    prompt = f"Given the following data:\n"
    prompt += f"Recent locations: {', '.join(locations)}\n"
    prompt += f"Current heart rate: {heart_rate} bpm\n"
    prompt += f"Last night's sleep score: {sleep_score}/100\n"
    prompt += f"Current skin temperature: {skin_temp}°F\n"
    prompt += "Describe the person's current state and likely activity:"

    # Generate text using the language model
    generated_text = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    # Extract the generated part (excluding the prompt)
    generated_part = generated_text[len(prompt):].strip()
    
    return generated_part

# Function to guess context based on generated text and calendar events
def guess_context(text, past_events, future_events, locations, heart_rate, sleep_score, skin_temp):
    context_scores = defaultdict(float)
    for context in contexts:
        keywords = context['Context Description'].lower().split()
        for keyword in keywords:
            if keyword in text.lower():
                context_scores[context['Context Description']] += 1
        
        # Check calendar events for context clues
        all_events = past_events + future_events
        for event in all_events:
            if any(keyword in event.lower() for keyword in keywords):
                context_scores[context['Context Description']] += 0.5

    # Get current time
    current_time = datetime.datetime.now().time()
    
    # Adjust scores based on time of day
    time_of_day = get_time_of_day(current_time)
    for activity in activity_probabilities:
        activity_name = activity['Activity Variable_Name']
        probability = activity[time_of_day]
        for context, score in context_scores.items():
            if activity_name in context.lower():
                context_scores[context] += probability

    # Adjust scores based on additional data
    if 'home' in locations:
        for context in ['relaxing_day', 'study', 'sleep']:
            context_scores[context] += 0.5
    if heart_rate > 100:
        context_scores['exercise'] += 0.5
    if sleep_score < 50:
        context_scores['tired'] += 0.5
    if skin_temp > 99:
        context_scores['health_issue'] += 0.5

    return max(context_scores, key=context_scores.get)

def get_time_of_day(time):
    if datetime.time(5) <= time < datetime.time(8):
        return 'Early Morning (5AM-8AM)'
    elif datetime.time(8) <= time < datetime.time(12):
        return 'Morning (8AM-12PM)'
    elif datetime.time(12) <= time < datetime.time(16):
        return 'Afternoon (12PM-4PM)'
    elif datetime.time(16) <= time < datetime.time(20):
        return 'Evening (4PM-8PM)'
    elif datetime.time(20) <= time < datetime.time(0):
        return 'Night (8PM-12AM)'
    else:
        return 'Late Night (12AM-5AM)'

# Function to predict activity based on context and previous activity
def predict_activity(context, prev_activity):
    context_activities = [activity['Activity Variable_Name'] for activity in activities 
                          if activity['Activity Description'].lower() in context.lower()]
    
    if prev_activity:
        transition_probs = next(row for row in activity_transitions if row['Current Activity'] == prev_activity)
        activity_scores = {activity: transition_probs.get(activity, 0) for activity in context_activities}
    else:
        time_of_day = get_time_of_day(datetime.datetime.now().time())
        activity_scores = {activity['Activity Variable_Name']: float(activity[time_of_day]) 
                           for activity in activity_probabilities if activity['Activity Variable_Name'] in context_activities}
    
    return max(activity_scores, key=activity_scores.get)

# Function to get appliance settings for an activity
def get_appliance_settings(activity):
    return next(settings for settings in appliance_settings if settings['Activity'] == activity)

# Main function to predict context, activity, and appliance settings
def predict_context_and_settings(past_events, future_events, locations, heart_rate, sleep_score, skin_temp, prev_activity=None):
    generated_text = generate_text_from_data(locations, heart_rate, sleep_score, skin_temp)
    context = guess_context(generated_text, past_events, future_events, locations, heart_rate, sleep_score, skin_temp)
    activity = predict_activity(context, prev_activity)
    settings = get_appliance_settings(activity)
    
    return {
        'Generated Text': generated_text,
        'Context': context,
        'Predicted Activity': activity,
        'Appliance Settings': settings
    }

# Example usage
if __name__ == "__main__":
    past_events = ["Group project meeting", "Lunch with roommate", "Physics lecture"]
    future_events = ["Study session for finals", "Call with academic advisor", "Movie night with friends"]
    locations = ["university", "coffee shop", "home"]
    heart_rate = 75
    sleep_score = 80
    skin_temp = 98.6
    
    result = predict_context_and_settings(past_events, future_events, locations, heart_rate, sleep_score, skin_temp)
    print(f"Generated Text: {result['Generated Text']}")
    print(f"Predicted Context: {result['Context']}")
    print(f"Predicted Activity: {result['Predicted Activity']}")
    print("Recommended Appliance Settings:")
    for appliance, setting in result['Appliance Settings'].items():
        print(f"  {appliance}: {setting}")