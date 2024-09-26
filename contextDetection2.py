import datetime
import pandas as pd
from typing import List, Dict
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def analyze(self, text):
        result = self.analyzer(text)[0]
        return result['score'] if result['label'] == 'POSITIVE' else -result['score']

class ContextClassifier:
    def __init__(self):
        model_name = "facebook/bart-large-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.classifier = pipeline("zero-shot-classification", model=self.model, tokenizer=self.tokenizer)

    def classify(self, events: List[Dict[str, str]], candidate_contexts: List[str]) -> str:
        events_text = " ".join([f"{event['time']} - {event['event']}" for event in events])
        result = self.classifier(events_text, candidate_contexts)
        return result['labels'][0]

class PersonStateAnalyzer:
    def __init__(self):
        self.wearables_data: Dict[str, float] = {}
        self.phone_data: Dict[str, str] = {}
        self.calendar_data: Dict[str, any] = {}
        self.time_of_day: str = ""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.context_classifier = ContextClassifier()
        
        # Load CSV data
        self.activity_transition_matrix = pd.read_csv(r'personas/college_student/activity_transition_matrix.csv', index_col='Current Activity')
        self.time_based_probabilities = pd.read_csv(r'personas/college_student/time_based_probabilities.csv', index_col='Activity Variable_Name')
        self.context_based_probabilities = pd.read_csv(r'personas/college_student/context_based_probabilities.csv', index_col='Context')
        self.appliance_settings = pd.read_csv(r'personas/college_student/appliance_settings.csv', index_col='Activity')

        # Define candidate contexts
        self.candidate_contexts = ["hard_college_day", "deadline_tomorrow", "exam_preparation", "project_week", "relaxed_day", "social_events", "holiday_season"]

        # Initialize person_health and sleep_status
        self.person_health = "healthy"
        self.sleep_status = self.get_sleep_status()

    def set_wearables_data(self, heart_rate: float, skin_temperature: float):
        self.wearables_data = {
            "heart_rate": heart_rate,
            "skin_temperature": skin_temperature
        }
        self.update_person_health()

    def set_phone_data(self, l_60_ago: str, l_15_ago: str, l_now: str):
        self.phone_data = {
            "L_60_ago": l_60_ago,
            "L_15_ago": l_15_ago,
            "L_now": l_now
        }

    def set_calendar_data(self, l_15_later: str, l_60_later: str, scheduled_activities: List[Dict[str, str]], recent_activities: List[Dict[str, str]]):
        self.calendar_data = {
            "L_15_later": l_15_later,
            "L_60_later": l_60_later,
            "scheduled_activities": scheduled_activities,
            "recent_activities": recent_activities
        }

    def set_time_of_day(self, time_of_day: str):
        self.time_of_day = time_of_day
        self.sleep_status = self.get_sleep_status()

    def update_person_health(self):
        if self.wearables_data.get("skin_temperature", 98.6) > 99.4:
            self.person_health = "sick"
        else:
            self.person_health = "healthy"

    def get_sleep_status(self):
        hour = 1 # int(self.time_of_day.split(':')[0])
        if 0 <= hour < 6:
            return "high_sleep_probability"
        elif 6 <= hour < 9 or 22 <= hour < 24:
            return "medium_sleep_probability"
        else:
            return "low_sleep_probability"

    def predict_context(self, events: List[Dict[str, str]]) -> str:
        predicted_context = self.context_classifier.classify(events, self.candidate_contexts)
        if predicted_context not in self.context_based_probabilities.index:
            return self.find_closest_context(predicted_context)
        return predicted_context

    def find_closest_context(self, predicted_context: str) -> str:
        available_contexts = self.context_based_probabilities.index.tolist()
        print("Predicted Context: ", predicted_context)
        return predicted_context
        # for context in available_contexts:
        #     if predicted_context.lower() in context.lower() or context.lower() in predicted_context.lower():
        #         return context
        # return "normal_day"

    def get_context_probabilities(self, context: str) -> pd.Series:
        if context in self.context_based_probabilities.index:
            return self.context_based_probabilities.loc[context]
        else:
            return self.context_based_probabilities.mean()

    def get_likely_activities(self, context: str, top_n: int = 5) -> List[str]:
        context_probs = self.get_context_probabilities(context)
        
        # Get probabilities based on time of day
        hour = int(self.time_of_day.split(':')[0])
        if 5 <= hour < 8:
            time_column = 'Early Morning (5AM-8AM)'
        elif 8 <= hour < 12:
            time_column = 'Morning (8AM-12PM)'
        elif 12 <= hour < 16:
            time_column = 'Afternoon (12PM-4PM)'
        elif 16 <= hour < 20:
            time_column = 'Evening (4PM-8PM)'
        elif 20 <= hour < 24:
            time_column = 'Night (8PM-12AM)'
        else:
            time_column = 'Late Night (12AM-5AM)'
        
        time_probs = self.time_based_probabilities[time_column]

        # Combine probabilities
        combined_probs = context_probs * 0.6 + time_probs * 0.4

        # Adjust probabilities based on person_health and sleep_status
        if self.person_health == "sick":
            combined_probs['nap'] *= 2
            combined_probs['study'] *= 0.5
        if self.sleep_status == "high_sleep_probability":
            combined_probs['sleep'] *= 2

        # Return top N activities
        return combined_probs.nlargest(top_n).index.tolist()

    def get_appliance_settings(self, activity: str) -> Dict[str, str]:
        if activity in self.appliance_settings.index:
            return self.appliance_settings.loc[activity].to_dict()
        else:
            return self.appliance_settings.loc['wake_up'].to_dict()  # Default to wake_up settings

    def analyze_state(self) -> str:
        # Predict upcoming context
        upcoming_context = self.predict_context(self.calendar_data['scheduled_activities'])

        # Get likely activities based on context and time
        likely_activities = self.get_likely_activities(upcoming_context)

        # Get appliance settings for the most likely activity
        most_likely_activity = likely_activities[0]
        appliance_settings = self.get_appliance_settings(most_likely_activity)

        # Prepare the output
        result = f"Predicted upcoming context: {upcoming_context}\n"
        result += f"Most likely activity: {most_likely_activity}\n"
        result += "Suggested appliance settings:\n"
        for appliance, setting in appliance_settings.items():
            result += f"- {appliance}: {setting}\n"

        return result

# Usage example
analyzer = PersonStateAnalyzer()
analyzer.set_wearables_data(75, 98.6)
analyzer.set_phone_data("Home", "Home", "Office")
analyzer.set_calendar_data("Office", "Meeting Room", 
    scheduled_activities=[
        {"event": "Study for exam", "time": "14:00", "location": "Library"},
        {"event": "Submit a report", "time": "16:00", "location": "Office"},
        {"event": "Group project meeting", "time": "10:00", "location": "Study room"}
    ],
    recent_activities=[
        {"event": "Attended lecture", "time": "Yesterday 14:00"},
        {"event": "Worked on assignment", "time": "Yesterday 20:00"},
        {"event": "Had a study group", "time": "Today 09:00"}
    ]
)
analyzer.set_time_of_day("10:30")

result = analyzer.analyze_state()
print(result)