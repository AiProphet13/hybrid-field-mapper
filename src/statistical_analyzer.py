import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import linregress
import plotly.graph_objects as go

class AdvancedStatisticalAnalyzer:
    """Statistical analysis for HybridFieldMapper performance"""
    
    def __init__(self, harness):
        self.harness = harness
        self.results = pd.DataFrame()

    def perform_tukey_hsd(self, metric='fidelity'):
        """Perform Tukey's HSD test"""
        if self.results.empty:
            return go.Figure(layout_title_text=f"No data for {metric} Tukey's HSD")
        tukey = pairwise_tukeyhsd(endog=self.results[metric], groups=self.results['method'], alpha=0.05)
        summary = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        summary['significant'] = summary['p-adj'] < 0.05
        fig = go.Figure()
        for _, row in summary.iterrows():
            fig.add_trace(go.Bar(
                x=[row['group2']], y=[row['meandiff']],
                name=row['group1'], text=f"p={row['p-adj']:.4f}",
                marker_color='green' if row['significant'] else 'red'
            ))
        fig.update_layout(title=f"Tukey's HSD for {metric}", xaxis_title="Group", yaxis_title="Mean Difference")
        return fig

    def calculate_noise_sensitivity(self, metric='fidelity'):
        """Calculate sensitivity to noise"""
        sensitivities = {}
        for method in self.results['method'].unique():
            method_data = self.results[self.results['method'] == method]
            if 'noise_level' not in method_data.columns or method_data['noise_level'].isnull().all():
                continue
            x = method_data['noise_level'].values
            y = method_data[metric].values
            if len(np.unique(x)) < 2:
                continue
            slope, intercept, r_value, _, _ = linregress(x, y)
            sensitivities[method] = {'slope': slope, 'r_squared': r_value**2}
        fig = go.Figure()
        for method, result in sensitivities.items():
            x = np.linspace(0, 0.3, 10)
            y = result['intercept'] + result['slope'] * x
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"{method} (slope={result['slope']:.2f})"))
        fig.update_layout(title=f"Noise Sensitivity of {metric}", xaxis_title="Noise Level", yaxis_title=metric)
        return fig

    def analyze_topology_impact(self, results_df):
        """Analyze topology impact on performance"""
        fig = go.Figure()
        for topology in results_df['topology_type'].unique():
            df = results_df[results_df['topology_type'] == topology]
            fig.add_trace(go.Scatter3d(
                x=df['field_size'], y=df['fidelity'], z=df['execution_time'],
                mode='markers', name=topology, marker=dict(size=5)
            ))
        fig.update_layout(title="Topology Impact", scene=dict(
            xaxis_title='Field Size', yaxis_title='Fidelity', zaxis_title='Execution Time'
        ))
        return fig, results_df

    def generate_analysis_report(self):
        """Generate statistical report"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'mean_fidelity': self.results['fidelity'].mean() if not self.results.empty else 0,
                'mean_execution_time': self.results['execution_time'].mean() if not self.results.empty else 0
            }
        }

    def visualize_network(self, topology_type):
        """Visualize network topology"""
        return go.Figure(layout_title_text=f"2D Network Visualization for {topology_type}")
