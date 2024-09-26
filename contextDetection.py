import datetime
from typing import List, Dict
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def analyze(self, text):
        result = self.analyzer(text)[0]
        return result['score'] if result['label'] == 'POSITIVE' else -result['score']

class PersonStateAnalyzer:
    def __init__(self):
        self.wearables_data: Dict[str, float] = {}
        self.phone_data: Dict[str, str] = {}
        self.calendar_data: Dict[str, any] = {}
        self.time_of_day: str = ""
        self.sentiment_analyzer = SentimentAnalyzer()

    def set_wearables_data(self, heart_rate: float, skin_temperature: float):
        self.wearables_data = {
            "heart_rate": heart_rate,
            "skin_temperature": skin_temperature
        }

    def set_phone_data(self, l_60_ago: str, l_15_ago: str, l_now: str):
        self.phone_data = {
            "L_60_ago": l_60_ago,
            "L_15_ago": l_15_ago,
            "L_now": l_now
        }

    def set_calendar_data(self, l_15_later: str, l_60_later: str, scheduled_activities: List[Dict[str, str]]):
        self.calendar_data = {
            "L_15_later": l_15_later,
            "L_60_later": l_60_later,
            "scheduled_activities": scheduled_activities
        }

    def set_time_of_day(self, time_of_day: str):
        self.time_of_day = time_of_day

    def analyze_state(self) -> str:
        state = "The person is "
        action = "and should be "
        recommendations = []

        # Check heart rate
        if self.wearables_data.get("heart_rate", 0) > 100:
            state += "experiencing elevated heart rate "
            recommendations.append("Practice deep breathing or meditation to calm down")
        
        # Check skin temperature
        if self.wearables_data.get("skin_temperature", 0) > 38:
            state += "potentially having a fever "
            recommendations.append("Consider taking fever-reducing medication and rest")
        
        # Check location changes
        if self.phone_data["L_now"] != self.phone_data["L_15_ago"]:
            state += "on the move "
        
        # Check upcoming activities
        now = datetime.datetime.strptime(self.time_of_day, "%H:%M")
        next_activity = next((act for act in self.calendar_data["scheduled_activities"] 
                              if datetime.datetime.strptime(act["time"], "%H:%M") > now), None)
        
        if next_activity:
            time_diff = datetime.datetime.strptime(next_activity["time"], "%H:%M") - now
            hours_until_next = time_diff.total_seconds() / 3600

            if hours_until_next < 1:
                action += f"preparing for {next_activity['event']} at {next_activity['location']}"
            else:
                action += f"considering preparation for the upcoming {next_activity['event']}"
                
                # Sentiment analysis for certain activities
                if next_activity['event'] in ["Create a report", "Submit a report", "Meeting with boss"]:
                    sentiment = self.sentiment_analyzer.analyze(next_activity['event'])
                    if sentiment < -0.3:
                        recommendations.append(f"The upcoming {next_activity['event']} might be challenging. Consider early preparation and positive affirmations.")
                    elif sentiment > 0.3:
                        recommendations.append(f"The upcoming {next_activity['event']} seems positive. Gather necessary materials and approach it with enthusiasm.")
                    else:
                        recommendations.append(f"Prepare for the upcoming {next_activity['event']} with a balanced approach.")
        else:
            action += "focusing on current tasks or personal development"
            recommendations.append("Consider reviewing your goals or learning something new")

        result = f"{state}{action}."
        if recommendations:
            result += " Recommendations: " + " ".join(recommendations)
        return result

# Usage example
analyzer = PersonStateAnalyzer()
analyzer.set_wearables_data(72, 37.3)
analyzer.set_phone_data("Home", "Home", "Home")
analyzer.set_calendar_data("Office", "Meeting Room", [
    {"event": "Birthday Party for Lauren", "time": "19:00", "location": "Alan's place"},
    {"event": "Submit a report", "time": "16:00", "location": "Office"}
])
analyzer.set_time_of_day("22:30")

result = analyzer.analyze_state()
print(result)