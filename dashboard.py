import dash
from dash import html, dcc  
import plotly.express as px
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

# Sample data
df = pd.read_csv('insrance_data.csv')

