import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px
from dash_bootstrap_templates import ThemeSwitchAIO
import joblib


df = pd.read_csv("insurance.csv")


df["categoria_bmi"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, float("inf")],
    labels=["Abaixo do peso", "Peso normal", "Sobrepeso", "Obesidade"]
)
BMI_options = [{'label': x, 'value': x} for x in df['categoria_bmi'].unique()]


random_forest = joblib.load("random_forest.pkl")
regressao_linear = joblib.load("regressão_linear.pkl")
xgboost_model = joblib.load("XGboost.pkl")

encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

categorical_cols = ["sex", "smoker", "region"]
numeric_cols = ["age", "bmi", "children"]


url_theme1 = dbc.themes.BOOTSTRAP
url_theme2 = dbc.themes.CYBORG
template_theme1 = "bootstrap"
template_theme2 = "cyborg"


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(ThemeSwitchAIO(aio_id="theme", themes=[url_theme1, url_theme2]), width=2),
        dbc.Col(html.H1("Dashboard de Previsão de Seguro Saúde", className="text-center my-4"), width=10)
    ]),

    #
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-age"), md=6),
        dbc.Col(dcc.Graph(id="fig-smoker"), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-bmi"), md=6),
        dbc.Col(dcc.Graph(id="fig-region"), md=6),
    ], className="my-4"),

    
    dbc.Row([
    dbc.Col([
        html.H3("Custo por BMI", className='text-center'),
        dcc.Dropdown(
            id='dropdown-bmi',
            value=["Abaixo do peso", "Peso normal", "Sobrepeso", "Obesidade"],
            multi=True,
            options=[
                {"label": "Abaixo do peso", "value": "Abaixo do peso"},
                {"label": "Peso normal", "value": "Peso normal"},
                {"label": "Sobrepeso", "value": "Sobrepeso"},
                {"label": "Obesidade", "value": "Obesidade"},
            ]
        ),
        dcc.Graph(id='graph-bmi', style={'height': '400px'}),  
    ], md=6),
    dbc.Col([
        html.H3("Custo por Idade e Fumante", className='text-center'),
        dcc.Graph(id='graph-age-smoker', style={'height': '400px'})  
    ], md=6)
]),
    html.Hr(),

    
    html.H3("Previsão de Charges - Insira os dados do paciente"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Idade"),
            dbc.Input(id="input-age", type="number", min=18, step=1, value=30),
        ], md=2),
        dbc.Col([
            dbc.Label("Sexo"),
            dcc.Dropdown(
                id="input-sex",
                options=[{"label": s, "value": s} for s in df["sex"].unique()],
                value="male"
            )
        ], md=2),
        dbc.Col([
            dbc.Label("BMI"),
            dbc.Input(id="input-bmi", type="number", step=0.1, value=25.0),
        ], md=2),
        dbc.Col([
            dbc.Label("Filhos"),
            dbc.Input(id="input-children", type="number", min=0, step=1, value=0),
        ], md=2),
        dbc.Col([
            dbc.Label("Fumante"),
            dcc.Dropdown(
                id="input-smoker",
                options=[{"label": s, "value": s} for s in df["smoker"].unique()],
                value="no"
            )
        ], md=2),
        dbc.Col([
            dbc.Label("Região"),
            dcc.Dropdown(
                id="input-region",
                options=[{"label": r, "value": r} for r in df["region"].unique()],
                value="southeast"
            )
        ], md=2),
    ], className="my-2"),

    dbc.Button("Prever", id="btn-predict", color="primary", className="my-3"),
    html.Div(id="output-predictions", className="alert alert-info")
], fluid=True)


@app.callback(
    Output("fig-age", "figure"),
    Output("fig-smoker", "figure"),
    Output("fig-bmi", "figure"),
    Output("fig-region", "figure"),
    Input(ThemeSwitchAIO.ids.switch("theme"), 'value')
)
def update_main_graphs(toggle):
    template = template_theme1 if toggle else template_theme2

    fig1 = px.histogram(df, x="age", nbins=20, title="Distribuição de Idade", template=template)
    fig2 = px.box(df, x="smoker", y="charges", title="Charges por Tabagismo", template=template)
    fig3 = px.scatter(df, x="bmi", y="charges", color="sex", title="Charges vs BMI", template=template)
    fig4 = px.violin(df, x="region", y="charges", box=True, title="Charges por Região", template=template)

    return fig1, fig2, fig3, fig4


@app.callback(
    Output('graph-bmi', 'figure'),
    Output('graph-age-smoker', 'figure'),
    Input('dropdown-bmi', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), 'value')
)
def update_extra_graphs(categoria_bmi, toggle):
    template = template_theme1 if toggle else template_theme2
    df_data = df.copy(deep=True)
    mask = df_data['categoria_bmi'].isin(categoria_bmi)

    
    df_bmi_mean = df_data[mask].groupby("categoria_bmi", as_index=False)["charges"].mean()
    fig_bar = px.bar(
        df_bmi_mean,
        x='categoria_bmi', y='charges',
        color='categoria_bmi',
        labels={'categoria_bmi': 'Categoria BMI', 'charges': 'Custo Médio'},
        title='Custo Médio por Categoria BMI',
        template=template
    )
    fig_bar.update_yaxes(range=[0, 50000])  

    
    df_age_smoker = df.groupby(["age", "smoker"], as_index=False)["charges"].mean()
    fig_scatter = px.scatter(
        df_age_smoker,
        x='age', y='charges', color='smoker',
        title='Custo Médio por Idade e Tabagismo',
        template=template
    )
    fig_scatter.update_yaxes(range=[0, 50000])  

    return fig_bar, fig_scatter



@app.callback(
    Output("output-predictions", "children"),
    Input("btn-predict", "n_clicks"),
    State("input-age", "value"),
    State("input-sex", "value"),
    State("input-bmi", "value"),
    State("input-children", "value"),
    State("input-smoker", "value"),
    State("input-region", "value"),
    prevent_initial_call=True
)
def predict(n, age, sex, bmi, children, smoker, region):
    try:
        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])

       
        x_encoded = encoder.transform(input_data[categorical_cols])
        x_encoded = pd.DataFrame(
            x_encoded.toarray(),
            columns=encoder.get_feature_names_out(categorical_cols),
            index=input_data.index
        )

        x_final = pd.concat([input_data[numeric_cols], x_encoded], axis=1)
        x_scaled = scaler.transform(x_final)

        pred1 = random_forest.predict(x_scaled)[0]
        pred2 = regressao_linear.predict(x_scaled)[0]
        pred3 = xgboost_model.predict(x_scaled)[0]

        return [
            html.H5("Previsões:"),
            html.P(f"Random Forest: {pred1:,.2f}"),
            html.P(f"Regressão Linear: {pred2:,.2f}"),
            html.P(f"XGBoost: {pred3:,.2f}")
        ]

    except Exception as e:
        return html.P(f"Erro no callback: {str(e)}", style={"color": "red"})


if __name__ == "__main__":
    app.run(debug=True, port=8050)
