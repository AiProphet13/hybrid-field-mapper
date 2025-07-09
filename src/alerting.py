from slack_sdk import WebClient
import os
import pandas as pd

class PerformanceAlerter:
    def __init__(self, threshold_fidelity=0.8, threshold_p_value=0.05):
        self.threshold_fidelity = threshold_fidelity
        self.threshold_p_value = threshold_p_value
        self.client = WebClient(token=os.environ.get('SLACK_TOKEN', 'your-slack-token'))

    def check_alerts(self, results_df, tukey_results=None):
        alerts = []
        if not results_df.empty:
            mean_fidelity = results_df['fidelity'].mean()
            if mean_fidelity < self.threshold_fidelity:
                alerts.append({
                    'severity': 'warning',
                    'message': f"Fidelity dropped below {self.threshold_fidelity}: {mean_fidelity:.3f}",
                    'metric': 'fidelity',
                    'threshold': self.threshold_fidelity,
                    'actual': mean_fidelity
                })
            if tukey_results is not None and not tukey_results.empty:
                significant = tukey_results[tukey_results['p-adj'] < self.threshold_p_value]
                if not significant.empty:
                    alerts.append({
                        'severity': 'info',
                        'message': f"Significant differences: {significant[['group1', 'group2', 'p-adj']].to_dict('records')}",
                        'metric': 'tukey_p_value',
                        'threshold': self.threshold_p_value,
                        'actual': significant['p-adj'].min()
                    })
        return alerts

    def send_alert(self, message):
        self.client.chat_postMessage(channel='#hybrid-field-mapper-alerts', text=message)
