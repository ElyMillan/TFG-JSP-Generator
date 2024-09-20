import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import logging
from typing import List

logging.basicConfig(level=logging.DEBUG)

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("JSP Instance Generator", className='text-center'), width=12, className='mb-4')
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Input(id='size', type='text', placeholder='Size'), width=12, className='mb-3')
                ]),
                dbc.Row([
                    dbc.Col(dbc.Input(id='jobs', type='text', placeholder='Jobs'), width=6, className='mb-3'),
                    dbc.Col(dbc.Input(id='job-step', type='text', placeholder='Job Step'), width=6, className='mb-3')
                ]),
                dbc.Row([
                    dbc.Col(dbc.Input(id='machines', type='text', placeholder='Machines'), width=6, className='mb-3'),
                    dbc.Col(dbc.Input(id='machine-step', type='text', placeholder='Machine Step'), width=6, className='mb-3')
                ]),
                dbc.Row([
                    dbc.Col(html.H4("Distributions"), width=12, className='mb-2')
                ]),
                dbc.Row([
                    dbc.Col(dcc.Checklist(
                        id='Distributions',
                        options=[
                            {'label': 'Normal', 'value': 'normal'},
                            {'label': 'Uniform', 'value': 'uniform'},
                            {'label': 'Exponential', 'value': 'exponential'}
                        ],
                        style={'margin': '10px'}
                    ), width=12, className='mb-3')
                ]),
                dbc.Row([
                    dbc.Col(dbc.Input(id='Speed-Scaling', type='text', placeholder='Speed-Scaling'), width=12, className='mb-3')
                ]),
                dbc.Row([
                    dbc.Col(dbc.Input(id='ReleaseDueDate', type='text', placeholder='ReleaseDueDate'), width=12, className='mb-3')
                ]),
                dbc.Row([
                    dbc.Col(dbc.Button('Generate', id='generator', n_clicks=0, color='primary'), width=12, className='d-grid gap-2 mb-3')
                ]),
                dbc.Row([
                    dbc.Col(dcc.Loading(id='loading', children=[html.Div(id='output')], type='default'), width=12)
                ])
            ], fluid=True, style={'margin': '20px'}),
            width=10
        )
    ], justify='center')
], fluid=True, style={'margin': '20px'})

@app.callback(
    Output('output', 'children'),
    [Input('generator', 'n_clicks')],
    [State('size', 'value'),
     State('jobs', 'value'),
     State('job-step', 'value'),
     State('machines', 'value'),
     State('machine-step', 'value'),
     State('Distributions', 'value'),
     State('Speed-Scaling', 'value'),
     State('ReleaseDueDate', 'value')]
)
def generate_and_fetch(n_clicks, size, jobs, job_step, machines, machine_step, distributions, speedScaling, releaseDueDate):
    if n_clicks > 0:
        if not (size and jobs and job_step and machines and machine_step and distributions and speedScaling and releaseDueDate):
            return 'Please fill all fields before submitting.'

        jobsArray = list(range(int(job_step), int(jobs) + int(job_step), int(job_step)))
        machinesArray = list(range(int(machine_step), int(machines) + int(machine_step), int(machine_step)))

        payload = {
            "size": int(size),
            "jobs": jobsArray,
            "machines": machinesArray,
            "distributions": distributions,
            "speed_scaling": int(speedScaling),
            "release_due_date": int(releaseDueDate)
        }

        try:
            response = requests.post('http://127.0.0.1:8000/custom_generation', json=payload)
            response.raise_for_status()
            data = response.json()

            logging.debug(f"Request sent to: http://127.0.0.1:8000/custom_generation with payload: {payload}")
            logging.debug(f"Response received: {data}")

            if not data or 'zip_url' not in data:
                return 'Error: Invalid response format.'

            unique_id = data['unique_id']
            base_url = "http://127.0.0.1:8000/download/"
            zip_url = base_url + unique_id
            return dbc.Button('Download ZIP', href=zip_url, color='dark', target='_self', download="Archivo", className='mt-3')
        except requests.RequestException as e:
            return f'Error: {e}'

    return 'Please enter values and click Generate.'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
