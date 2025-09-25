import dash
from dash import html, dcc
import pandas as pd
imo     
import dash_bootstrap_components as dbc  # IMPORT NECESSÁRIO
from dash_bootstrap_templates import ThemeSwitchAIO


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sample data
df = pd.read_csv('insurance.csv')

app.layout = dbc.Container([
    html.H1('opa')
])

if __name__ == '__main__':
    app.run(debug=True, port=8050)  
