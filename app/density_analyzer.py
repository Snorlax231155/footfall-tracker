from datetime import datetime, timedelta
import os

class DensityAnalyzer:
    def __init__(self, threshold=None):
        self.density_threshold = threshold if threshold is not None else int(os.environ.get('DENSITY_THRESHOLD', 5))
        
    def analyze(self, footfall_data):
        high_density_periods = []
        
        # This is a simplified version. In reality, you'd need to track density over time
        for zone_id, count in footfall_data['zone_counts'].items():
            if count > self.density_threshold:
                # For demo purposes, using current time
                current_time = datetime.now()
                period = {
                    "start_time": current_time.strftime("%H:%M"),
                    "end_time": (current_time + timedelta(minutes=5)).strftime("%H:%M"),
                    "zone_id": zone_id
                }
                high_density_periods.append(period)
                
        return {
            'high_density_periods': high_density_periods
        } 