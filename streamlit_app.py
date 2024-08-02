import base64
import streamlit as st
from rosebud_chat_model import rosebud_chat_model
import json
import wandb
import datetime
import threading


st.set_page_config(
    page_title="Rosebud",
    page_icon="🌹",
)


def local_css(file_name):
    with open(file_name, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")

st.markdown("#")

with open('./config.json') as f:
    config = json.load(f)

st.html("<h1 style='text-align: center;'>Rosebud 🌹</h1>")
st.html("<h2 style='text-align: center;'>Let's discover films.</h2>")

# Initialize session state
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'query_constructor' not in st.session_state:
    st.session_state.query_constructor = False
if 'context' not in st.session_state:
    st.session_state.context = ""
if 'response' not in st.session_state:
    st.session_state.response = ""
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False


def generate_response(query):
    with st.spinner(text="Generating awesome recommendations..."):
        chat_model = rosebud_chat_model()
        with st.chat_message("assistant"):
            response = st.write_stream(chat_model.predict_stream(query))
        st.session_state.query = query
        st.session_state.query_constructor = chat_model.query_constructor
        st.session_state.context = chat_model.context
        st.session_state.response = response
        st.session_state.sentiment = None
        st.session_state.feedback_given = False


def start_log_feedback(feedback):
    print("Logging feedback.")
    st.session_state.feedback_given = True
    st.session_state.sentiment = feedback
    thread = threading.Thread(target=log_feedback, args=(st.session_state.sentiment,
                                                         st.session_state.query,
                                                         st.session_state.query_constructor,
                                                         st.session_state.context,
                                                         st.session_state.response))
    thread.start()


def log_feedback(sentiment, query, query_constructor, context, response):
    ct = datetime.datetime.now()
    wandb.init(project="film-search",
               name=f"query: {ct}")
    table = wandb.Table(columns=["sentiment", "query", "query_constructor", "context", "response"])
    table.add_data(sentiment,
                   query,
                   query_constructor,
                   context,
                   response
                   )
    wandb.log({"Query Log": table})
    wandb.finish()


col1, col2, col3 = st.columns(3)

with col1:
    example1 = "Find me drama movies in English that are less than 2 hours long and feature dogs."
    button1_clicked = st.button(example1, key='button1')

with col2:
    example2 = "Films with very little dialogue made after 1970."
    button2_clicked = st.button(example2, key='button2')

with col3:
    example3 = "I'm looking for some highly rated horror films streaming on either Netflix or Hulu."
    button3_clicked = st.button(example3, key='button3')

# Input and button
query = st.chat_input(placeholder='Type your query here.',)

if query:
    with st.chat_message("human"):
        st.write(query)
    generate_response(query)
elif button1_clicked:
    with st.chat_message("human"):
        st.write(example1)
    generate_response(example1)
elif button2_clicked:
    with st.chat_message("human"):
        st.write(example2)
    generate_response(example2)
elif button3_clicked:
    with st.chat_message("human"):
        st.write(example3)
    generate_response(example3)

# Display response
if st.session_state.response and not st.session_state.feedback_given:
    # Feedback buttons
    col1, col2 = st.columns([1, 15])
    with col1:
        st.button('👍', key='positive_feedback', disabled=False, on_click=start_log_feedback, args=["positive"])

    with col2:
        st.button('👎', key='negative_feedback', disabled=False, on_click=start_log_feedback, args=["negative"])


if st.session_state.response and st.session_state.feedback_given:
    with st.chat_message("human"):
        st.write(st.session_state.query)
    with st.chat_message("ai"):
        st.write(st.session_state.response)

    # Feedback buttons
    col1, col2 = st.columns([1, 15])
    with col1:
        st.button('👍', key='positive_feedback_disabled', disabled=True)
    with col2:
        st.button('👎', key='negative_feedback_disabled', disabled=True)

    if st.session_state.sentiment == "positive":
        st.toast(body="Thanks for the positive feedback!", icon="🔥")
    else:
        st.toast(body="Thanks for the feedback. We'll try to improve!", icon="😔")


def render_svg(svg, width=200, height=50):
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" width="%s" height="%s"/>' % (
        b64, width, height)
    st.markdown(html, unsafe_allow_html=True)


with st.sidebar:
    st.header("About")
    beginning_year = config["years"][0]
    ending_year = config["years"][-1]

    f"""
    This film recommendation bot has been given a database of roughly the 100 most popular
    films from the years {beginning_year}-{ending_year}. **It will only recommend films from this
    database.** This bot is able to create metadata filters for your recommendations via natural language.
    For more information, please [see my article](https://towardsdatascience.com/how-to-build-a-rag-system-with-a-self-querying-retriever-in-langchain-16b4fa23e9ad)
    in Towards Data Science.
    """

    "To see the code repository for this project, [click here](https://github.com/EdIzaguirre/Rosebud)."

    st.header("FAQ")

    st.markdown(
        """
        + **How is this different then ChatGPT?**
        Good question. If you ask ChatGPT for movie recommendtions it will use its memory + occasional web searches to answer your query.
        While this might work out sometimes, some of the sources it uses on the web are outdated. ChatGPT's memory may also be faulty.
        This recommendation bot has access to a dedicated database of film data that is updated automatically on a weekly basis. In addition,
        this bot has the power to filter movies via natural language. As an example, if you ask for "*comedy films made after 1971 that take place
        on an island*", this bot is smart enought to filter out all films in the database that are not of the 'comedy' genre and that are not made
        after the year 1971, **before** searching for *'island films'*.
        + **What model is this application using?**
        It is currently using `gpt-4o-mini`. There are plans to give the user a choice of models including Claude and Llama.
        + **Are my queries logged?**
        By default queries are not logged. If you leave feedback by clicking the 👍 or 👎 buttons, then that query and response will
        be logged to help improve the application's performance.
        + **Where do the ratings come from?**
        The ratings are from the users of The Movie Database. Check out the [website here](https://www.themoviedb.org/?language=en-US).
        + **What are the attributes that I can filter by?**
        The attributes are:

        Actors: e.g. *['Christine Taylor', 'Ben Stiller', ...]*

        Buy: e.g. *['Apple TV', 'Amazon Video', ...]*

        Directors: e.g. *'Ben Stiller'*

        Genre: e.g. *'Comedy'*

        Language': e.g. *'English'*

        Production Companies: e.g. *['Paramount Pictures', 'Village Roadshow Pictures', ...]*

        Rating: e.g. *6.2*

        Release Year: e.g. *2001*

        Rent: e.g. *['Apple TV', 'Amazon Video']*

        Runtime (minutes): e.g. *90*

        Stream: e.g. *['Paramount Plus', ...]*

        Title: e.g. *'Zoolander'*
        """
    )

    st.header("Data Source")

    # Open the SVG file and read it into a variable
    with open('images/tmdb_logo.svg', 'r') as f:
        svg = f.read()

    # Call the function to display the SVG
    render_svg(svg)

    st.write(""" This application uses TMDB and the TMDB APIs but is
            not endorsed, certified, or otherwise approved by TMDB.
            Watch providers were pulled from JustWatch.
            """)
