# Day 69: Linking the Pickle Model

#importing libraries
import re
from dash.dependencies import Input, Output, State
import dash_html_components
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_html_components import H1
import pandas as pd
import dash
import dash_html_components as html
import webbrowser
import pickle
# Declaring GLobal Variables
def check_review(textData):
    recreated_model = pickle.load(open('model_data.pkl', 'rb'))
    myvocab = pickle.load(open('features_data.pkl', 'rb'))
    from sklearn.feature_extraction.text import CountVectorizer
    vocab = CountVectorizer(vocabulary= myvocab, decode_error='replace')
    from sklearn.feature_extraction.text import TfidfTransformer
    cvtotf = TfidfTransformer()
    return recreated_model.predict(cvtotf.fit_transform(vocab.transform([textData])))

project_name = None
app = dash.Dash()
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    # Favicon == 16x16 icon : favicon.ico --> assets
#Defining Function
def load_model():
    global scrapped_reviews
    


def create_app_ui():
    main_layout = html.Div(style={'backgroundImage':'linear-gradient(to right, cyan, green)','height':'100vh'},children=[
        html.H1(id='main_title', children = 'Sentiment Analysis for Movie Reviews', style={'margin-botton':'12px','text-align':'center','width':'100%'}),
       html.Div(children=dcc.Textarea(
           id='textarea_review',
           placeholder='Enter the Review Here',
           style={'width':'920px','height':'300px','border':'3px solid green','border-radius':'10px','margin':'20px'}
       ),style={'width':'50%','align-content':'center','margin':'auto'}),
       html.H1(children=None, id='result',style={'text-align':'center'}),
        html.Footer(children='Made by : TEAM_AI4U', style={'text-align':'center','margin-bottom':'0px'})
    ])
    return main_layout


def open_browser():
    webbrowser.open('http://127.0.0.1:8050/')

@app.callback(
    Output( 'result' , 'children' ),
    [
    Input( 'textarea_review', 'value')]
)
def update_app_ui(textarea_value):
    # print("Data Type = ", str(type(textarea_value)))
    # print("Value = ", str(textarea_value))
    if(textarea_value != "" or textarea_value[-1]==" "):
        response = check_review(textarea_value)
        if (response[0]==1):
            return "Looks Like You Loved the Movie"
        elif(response[0]==0):
            return "Movie wasn't Great..."
        else:
            return "Unknown or is it Blank"

#Main FunctiThe movie was worst.on
def main():
    print("Start of Your Project")
    load_model()
    open_browser()
    global scrapped_reviews
    global project_name
    project_name = "Sentiment Analysis"
    print(f"My Project Name is {project_name}")
    # print(f"My Scrapped Data: {scrapped_reviews.sample(5)}")
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()
    print("End of the Project")
    project_name = None
    scrapped_reviews = None
# Calling Main Function
if __name__ == '__main__':
    main()