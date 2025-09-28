import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd   
from dash_bootstrap_templates import ThemeSwitchAIO
from app import *

# stilo
url_theme1 = dbc.themes.BOOTSTRAP
url_theme2 = dbc.themes.CYBORG
template_theme1 = "bootstrap"
template_theme2 = "cyborg"
# Sample data
df = pd.read_csv('insurance.csv')
df["categoria_bmi"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, float("inf")],
    labels=["Abaixo do peso", "Peso normal", "Sobrepeso", "Obesidade"]
)
BMI_options = [{'label': x, 'value': x} for x in df['categoria_bmi'].unique()]


# Layout
            
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            ThemeSwitchAIO(aio_id="theme", themes=[url_theme1, url_theme2]),
            html.H3("Custo por BMI", className='text-center'),
            dcc.Dropdown(
                id='dropdown-bmi',
                value=[categoria_bmi['label'] for categoria_bmi in BMI_options],  # Default value
                multi=True,
                options=BMI_options
                ),
            dcc.Graph(id='graph-bmi'),
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Custo por Idade e Fumante", className='text-center'),
            dcc.Graph(id='graph-age-smoker')
        ])
    ])
    
])

#callbacks
# Grafico de barras - Custo por BMI
# Grafico scatter - Custo por idade e se é fumante
@app.callback(
    Output('graph-bmi', 'figure'),
    Output('graph-age-smoker', 'figure'),
    Input('dropdown-bmi', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), 'value')
    
)

def bar(categoria_bmi, toggle ):
    template = template_theme1 if toggle else template_theme2
    df_data = df.copy(deep=True)
    mask = df_data['categoria_bmi'].isin(categoria_bmi)
    fig2 = px.scatter(df, x='age', y='charges', color='smoker',template=template)

    fig = px.bar(df_data[mask],x='categoria_bmi', y='charges',
         color='categoria_bmi', 
         labels={'categoria_bmi':'Categoria BMI', 'charges':'Custo'},
         title='Custo por Categoria BMI',
         template=template,)

    return fig, fig2
# Run the app
if __name__ == '__main__':
    app.run(debug=True, port='8051')  

