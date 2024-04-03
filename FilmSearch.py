import streamlit as st
from chat_app import FilmSearch
import pandas as pd
import json

st.set_page_config(
    page_title="Film Search",
    page_icon="🎥",
)

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/EdIzaguirre/FilmSearchOpen)"
    "If there is nothing being generated after putting in your API key, typing your query, and hitting submit, check to make sure your API key is correct."

with open('./config.json') as f:
    config = json.load(f)

st.markdown("<h1 style='text-align: center;'>🎥 Film Search</h1>",
            unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>The better way to search for films.</h2>",
            unsafe_allow_html=True)

beginning_year = config["years"][0]
ending_year = config["years"][-1]

f"""
This film search bot has been given a database of roughly the 100 most popular
films from the years {beginning_year}-{ending_year}. It will only recommend films from this
database.
"""


def generate_response(input_text, openai_api_key):
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    pinecone_index_name = st.secrets["PINECONE_INDEX_NAME"]

    chat = FilmSearch(openai_api_key, pinecone_api_key, pinecone_index_name)
    st.write_stream(chat.ask(input_text))


st.markdown(
    "*Feel free to try one of these sample queries, or type your own below.*")

col1, col2, col3 = st.columns(3)

with col1:
    example = "Find me drama movies in English that are less than 2 hours long and feature pets."
    if st.button(example, key='button1'):
        st.session_state.query = example

with col2:
    example = "Films with very little dialogue made after 1950."
    if st.button(example, key='button2'):
        st.session_state.query = example

with col3:
    example = "Recommend some funny zombie movies that are streaming on Hulu."
    if st.button(example, key='button3'):
        st.session_state.query = example

with st.form('my_form'):
    text = st.text_area(
        label='Query:',
        placeholder='Type your query here.',
        value=st.session_state.query if 'query' in st.session_state else '',
        label_visibility='hidden')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    if submitted:
        generate_response(text, openai_api_key=openai_api_key)

st.divider()

st.header("Data Source")
st.write(""" All data was pulled from the The Movie Database (TMDB) and 
         compiled into a CSV file. Watch providers were pulled from JustWatch.
        Below is a snippet of data from the year 2023. """)

df = pd.read_csv('./data/2023_movie_collection_data.csv')

# Prevent commas from appearing in release year
df['Release Year'] = df['Release Year'].astype(str)

# Reverse order
df = df.iloc[::-1]

st.dataframe(df, hide_index=True)

st.write(
    "For the full data, download the full_movie_collection_data.csv [here](https://github.com/EdIzaguirre/FilmSearchOpen/blob/main/data/full_movie_collection_data.csv).""")
