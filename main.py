import os
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, request, abort
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from dash import Dash, dcc, html, Input, Output, State, exceptions
from dash.dash_table import DataTable
from dash_extensions import WebSocket
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import and_
import json
import hashlib
import networkx as nx
import numpy as np
from prometheus_client import generate_latest
import diskcache
from dash.long_callback import DiskcacheLongCallbackManager
from src.hybrid_field_mapper import ProductionReadyHybridFieldMapper
from src.statistical_analyzer import AdvancedStatisticalAnalyzer
from src.alerting import PerformanceAlerter
from src.topology_tester import create_complex_topologies, run_complex_topology_tests

server = Flask(__name__)
server.config.update(
    SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'sqlite:///results.db'),
    SECRET_KEY=os.environ.get('SECRET_KEY', 'your-secret-key'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False
)
db = SQLAlchemy(server)
login_manager = LoginManager(server)
login_manager.login_view = '/login'
limiter = Limiter(server, key_func=get_remote_address)
CORS(server)
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), default='viewer')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    def has_permission(self, permission):
        role_permissions = {
            'admin': ['view', 'execute', 'export'],
            'operator': ['view', 'execute'],
            'viewer': ['view']
        }
        return permission in role_permissions.get(self.role, [])

class TestResult(db.Model):
    __table_args__ = (db.UniqueConstraint('test_id', 'method', name='unique_test_method'),)
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.String(64))
    method = db.Column(db.String(50))
    fidelity = db.Column(db.Float)
    resonance_shift = db.Column(db.Float)
    execution_time = db.Column(db.Float)
    noise_level = db.Column(db.Float)
    topology_type = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime(timezone=True), default=lambda: datetime.utcnow().replace(tzinfo=timezone.utc))

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    severity = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)
    metric = db.Column(db.String(50))
    threshold_value = db.Column(db.Float)
    actual_value = db.Column(db.Float)
    test_id = db.Column(db.String(64), db.ForeignKey('test_results.test_id'))
    resolved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.utcnow().replace(tzinfo=timezone.utc))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@server.before_request
def verify_origin():
    origin = request.headers.get('Origin')
    if origin and not origin.startswith('https://hybrid-field-mapper.herokuapp.com'):
        abort(403)

app = Dash(__name__, server=server, url_base_pathname='/dashboard/', long_callback_manager=long_callback_manager)
harness = ProductionReadyHybridFieldMapper(quantum_backend='aer_simulator')
analyzer = AdvancedStatisticalAnalyzer(harness)
alerter = PerformanceAlerter()

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    WebSocket(id='ws', url=f'ws://{os.environ.get("HOST", "localhost")}:5000/ws'),
    dcc.Interval(id='update-interval', interval=15000, n_intervals=0)
])

login_layout = html.Div([
    html.H2("HybridFieldMapper Login", style={'color': '#00ff00'}),
    dcc.Input(id='username', type='text', placeholder='Username', style={'margin': '10px'}),
    dcc.Input(id='password', type='password', placeholder='Password', style={'margin': '10px'}),
    html.Button('Login', id='login-button', n_clicks=0, style={'margin': '10px', 'background': '#00ff00', 'color': '#000'}),
    html.Div(id='login-output', style={'color': 'red', 'margin': '10px'})
])

dashboard_layout = html.Div([
    html.H1("🚀 HybridFieldMapper - Quantum AI Propagation System", style={'text-align': 'center', 'color': '#00ff00'}),
    html.P("By AiProphet13 - FREE FOR ALL", style={'text-align': 'center', 'font-style': 'italic', 'color': '#00ff00'}),
    html.Div([
        html.Span(id='user-info', style={'float': 'right', 'color': '#00ff00'}),
        html.Button('Logout', id='logout-button', n_clicks=0, style={'float': 'right', 'margin': '10px', 'background': '#00ff00', 'color': '#000'})
    ]),
    html.Div([
        html.H3("🎛️ Control Panel", style={'color': '#00ff00'}),
        html.Label("Topology Type:"),
        dcc.Dropdown(id='topology-type-viz', options=[
            {'label': '⭐ Star Network', 'value': 'star'},
            {'label': '🔲 Grid Network', 'value': 'grid'},
            {'label': '⭕ Ring Network', 'value': 'ring'},
            {'label': '🌐 Hybrid Network', 'value': 'hybrid'}
        ], value='star', style={'width': '200px'}),
        html.Label("Field Size:"),
        dcc.Slider(id='field-size-slider', min=4, max=16, step=4, value=8, marks={i: f'{i}x{i}' for i in [4, 8, 12, 16]}),
        html.Label("Quantum Backend:"),
        dcc.Dropdown(id='backend-selector', options=[
            {'label': 'QASM Simulator', 'value': 'qasm_simulator'},
            {'label': 'Aer Simulator', 'value': 'aer_simulator'},
            {'label': 'IBM Hanoi', 'value': 'ibmq_hanoi'}
        ], value='aer_simulator', style={'width': '200px'}),
        html.Button('🔬 Run Quantum Simulation', id='run-test-button', n_clicks=0, style={'margin': '10px', 'padding': '10px', 'background': '#00ff00', 'color': '#000'}),
        html.Div(id='test-status', style={'margin': '10px', 'color': '#00ff00'})
    ], style={'background': '#2a2a2a', 'padding': '20px'}),
    dcc.Tabs([
        dcc.Tab(label='Real-Time Metrics', children=[
            html.H3("📊 Simulation Results", style={'color': '#00ff00'}),
            dcc.Graph(id='field-visualization'),
            dcc.Graph(id='metrics-plots'),
            DataTable(id='results-table', columns=[
                {'name': 'Test ID', 'id': 'test_id'},
                {'name': 'Method', 'id': 'method'},
                {'name': 'Fidelity', 'id': 'fidelity'},
                {'name': 'Resonance Shift', 'id': 'resonance_shift'},
                {'name': 'Execution Time', 'id': 'execution_time'},
                {'name': 'Topology', 'id': 'topology_type'}
            ], page_size=10, style_table={'overflowX': 'auto'}, style_data={'color': '#00ff00'}, style_header={'background': '#1e1e1e'})
        ]),
        dcc.Tab(label='Advanced Statistics', children=[
            html.Div([
                html.H3("Advanced Statistical Analysis", style={'color': '#00ff00'}),
                dcc.Dropdown(id='tukey-metric', options=[
                    {'label': 'Fidelity', 'value': 'fidelity'},
                    {'label': 'Resonance Shift', 'value': 'resonance_shift'},
                    {'label': 'Execution Time', 'value': 'execution_time'}
                ], value='fidelity'),
                dcc.Graph(id='tukey-plot'),
                dcc.Graph(id='sensitivity-plot'),
                dcc.Graph(id='topology-impact-plot'),
                dcc.Graph(id='network-visualization')
            ])
        ]),
        dcc.Tab(label='Network Topology', children=[
            html.Div([
                html.H3("Network Topology Visualization", style={'color': '#00ff00'}),
                dcc.RadioItems(id='topology-viz-type', options=[
                    {'label': '2D Network', 'value': '2d'},
                    {'label': '3D Network', 'value': '3d'}
                ], value='3d', style={'color': '#00ff00'}),
                dcc.Graph(id='network-visualization-3d')
            ])
        ]),
        dcc.Tab(label='Alerts', children=[
            html.Div(id='alert-messages', style={'color': 'red', 'padding': '20px'})
        ])
    ], style={'background': '#1e1e1e'}),
    html.Button('Export Report', id='export-button', n_clicks=0, style={'margin': '20px', 'background': '#00ff00', 'color': '#000'}),
    dcc.Download(id='download-report')
], style={'background': '#0a0a0a', 'min-height': '100vh'})

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'), Input('logout-button', 'n_clicks')],
    [State('username', 'value'), State('password', 'value')]
)
def display_page(pathname, logout_n_clicks, username, password):
    if logout_n_clicks > 0:
        logout_user()
        return login_layout
    if pathname == '/login' or not current_user.is_authenticated:
        if username and password:
            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                login_user(user)
                return dashboard_layout
            return login_layout, html.Div("Invalid credentials", style={'color': 'red'})
        return login_layout
    return dashboard_layout

@app.long_callback(
    output=[Output('test-status', 'children'), Output('field-visualization', 'figure'), Output('metrics-plots', 'figure'), Output('results-table', 'data')],
    inputs=[Input('run-test-button', 'n_clicks')],
    state=[State('topology-type-viz', 'value'), State('field-size-slider', 'value'), State('backend-selector', 'value')],
    running=[(Output('run-test-button', 'disabled'), True, False)],
    prevent_initial_call=True
)
def run_test(n_clicks, topology_type, field_size, backend):
    if n_clicks == 0 or not current_user.has_permission('execute'):
        raise exceptions.PreventUpdate("Permission denied")
    
    # Generate test wires based on topology
    wires = create_complex_topologies(topology_type, size=int(np.sqrt(field_size)))
    
    # Generate and process field
    env_params = {'conductivity': 5.96e7}
    field = harness.generate_test_field(wires[0], wires[1], env_params, field_size)
    corrected_field = harness.apply_quantum_correction(field, mitigate_noise=True)
    final_field = harness.apply_noise_mitigation(corrected_field)
    
    # Create field visualization
    field_fig = go.Figure(data=go.Heatmap(
        z=final_field, colorscale='Viridis', text=np.round(final_field, 3),
        texttemplate='%{text}', textfont={"size": 10}
    ))
    field_fig.update_layout(title='Quantum-Corrected EM Field', xaxis_title='X Position', yaxis_title='Y Position', template='plotly_dark')
    
    # Run full test suite
    results_df = run_complex_topology_tests((topology_type, field_size, backend))
    
    with server.app_context():
        for _, row in results_df.iterrows():
            db.session.merge(TestResult(
                test_id=str(row['test_id']), method=row['method'], fidelity=row['fidelity'],
                resonance_shift=row['resonance_shift'], execution_time=row['execution_time'],
                noise_level=row.get('noise_level'), topology_type=row['topology_type']
            ))
        db.session.commit()
        alerts = alerter.check_alerts(results_df, analyzer.perform_tukey_hsd('fidelity'))
        for msg in alerts:
            alerter.send_alert(msg['message'])
            db.session.add(Alert(
                severity=msg['severity'],
                message=msg['message'],
                metric=msg['metric'],
                threshold_value=msg['threshold'],
                actual_value=msg['actual'],
                test_id=results_df['test_id'].iloc[0]
            ))
        db.session.commit()
    
    metrics_fig = go.Figure()
    for m in ['fidelity', 'resonance_shift', 'execution_time']:
        metrics_fig.add_trace(go.Scatter(x=results_df['test_id'], y=results_df[m], mode='lines+markers', name=m))
    metrics_fig.update_layout(title='Live Performance Metrics', template='plotly_dark')
    
    table_data = results_df.to_dict('records')
    
    return (
        html.Div(f"Test completed for {topology_type} (Field: {field_size}x{field_size})", style={'color': 'green'}),
        field_fig,
        metrics_fig,
        table_data
    )

@app.callback(
    [Output('network-visualization', 'figure'), Output('network-visualization-3d', 'figure')],
    [Input('topology-type-viz', 'value'), Input('topology-viz-type', 'value')]
)
def update_network_viz(topology_type, viz_type):
    fig_2d = analyzer.visualize_network(topology_type)
    if viz_type == '2d':
        return fig_2d, go.Figure()
    wires = create_complex_topologies(topology_type, size=4)
    positions = {w['id']: w['position'] for w in wires}
    G = nx.Graph()
    for w in wires:
        G.add_node(w['id'], pos=w['position'])
    dist_matrix = np.array([[np.linalg.norm(np.array(p1)-np.array(p2)) for p2 in positions.values()]
                           for p1 in positions.values()])
    threshold = np.percentile(dist_matrix[dist_matrix > 0], 25)
    node_ids = list(positions.keys())
    for i, id1 in enumerate(node_ids):
        for j, id2 in enumerate(node_ids):
            if i < j and dist_matrix[i, j] < threshold:
                G.add_edge(id1, id2)
    node_x, node_y, node_z = [], [], []
    for node in G.nodes:
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(0)
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([0, 0, None])
    fig_3d = go.Figure(data=[
        go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(width=2, color='#888')),
        go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers', marker=dict(size=10, color='#007BFF'))
    ])
    fig_3d.update_layout(title=f'3D Network for {topology_type}', showlegend=False,
                         scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), template='plotly_dark')
    return fig_2d, fig_3d

@app.callback(
    [Output('alert-messages', 'children'), Output('download-report', 'data')],
    [Input('update-interval', 'n_intervals'), Input('export-button', 'n_clicks')]
)
def alert_and_export(n_intervals, export_clicks):
    df = TestResult.query.order_by(TestResult.timestamp.desc()).limit(100).all()
    df_data = pd.DataFrame([{'test_id': r.test_id, 'method': r.method, 'fidelity': r.fidelity,
                             'resonance_shift': r.resonance_shift, 'execution_time': r.execution_time,
                             'noise_level': r.noise_level} for r in df])
    alerts = alerter.check_alerts(df_data, analyzer.perform_tukey_hsd('fidelity'))
    alert_divs = [html.P(a['message']) for a in alerts]
    report_data = None
    if export_clicks:
        report = analyzer.generate_analysis_report()
        report_json = json.dumps(report, sort_keys=True)
        report['integrity_hash'] = hashlib.sha256(report_json.encode()).hexdigest()
        report_data = dict(content=json.dumps(report), filename='report.json')
    return alert_divs, report_data

@server.route('/api/metrics')
@limiter.limit("60 per minute")
def api_metrics():
    token = request.headers.get('X-API-TOKEN')
    if token != os.environ.get('METRICS_TOKEN'):
        abort(401)
    return generate_latest(), 200, {'Content-Type': 'text/plain; version=0.0.4'}

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 HYBRIDFIELDMAPPER STARTING UP 🚀")
    print("=" * 50)
    print("🌟 FREE FOR ALL - Built with love by AiProphet13")
    print("📡 Quantum systems: ONLINE")
    print("🔬 MEEP simulations: READY")
    print("💻 Dashboard: http://localhost:5000")
    print("=" * 50)
    with server.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            hashed_password = generate_password_hash('admin', method='pbkdf2:sha256')
            new_user = User(username='admin', password_hash=hashed_password, role='admin')
            db.session.add(new_user)
            db.session.commit()
    server.run(debug=True, host='0.0.0.0', port=5000)
