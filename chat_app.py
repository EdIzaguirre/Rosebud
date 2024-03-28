# Langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.pinecone import PineconeTranslator
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

# Pinecone
from pinecone import Pinecone

# General
import json
from dotenv import load_dotenv
import os


class FilmSearch:
    with open('./config.json') as f:
        config = json.load(f)

    RETRIEVER_MODEL_NAME = config["RETRIEVER_MODEL_NAME"]
    SUMMARY_MODEL_NAME = config["SUMMARY_MODEL_NAME_GPT4"]
    constructor_prompt = None
    vectorstore = None
    retriever = None
    rag_chain_with_source = None

    def __init__(self):
        load_dotenv()
        self.initialize_query_constructor()
        self.initialize_vector_store()
        self.initialize_retriever()
        self.initialize_chat_model()

    def initialize_query_constructor(self):
        document_content_description = "Brief overview of a movie, along with keywords"

        # Define allowed comparators list
        allowed_comparators = [
            "$eq",  # Equal to (number, string, boolean)
            "$ne",  # Not equal to (number, string, boolean)
            "$gt",  # Greater than (number)
            "$gte",  # Greater than or equal to (number)
            "$lt",  # Less than (number)
            "$lte",  # Less than or equal to (number)
            "$in",  # In array (string or number)
            "$nin",  # Not in array (string or number)
        ]

        examples = [
            (
                "I'm looking for a sci-fi comedy released after 2021.",
                {
                    "query": "sci-fi comedy",
                    "filter": "and(eq('Genre', 'Science Fiction'), eq('Genre', 'Comedy'), gt('Release Year', 2021))",
                },
            ),
            (
                "Show me critically acclaimed dramas without Tom Hanks.",
                {
                    "query": "critically acclaimed drama",
                    "filter": "and(eq('Genre', 'Drama'), nin('Actors', ['Tom Hanks']))",
                },
            ),
            (
                "Recommend some films by Yorgos Lanthimos.",
                {
                    "query": "Yorgos Lanthimos",
                    "filter": 'in("Directors", ["Yorgos Lanthimos]")',
                },
            ),
            (
                "Films similar to Yorgos Lanthmios movies.",
                {
                    "query": "Dark comedy, absurd, Greek Weird Wave",
                    "filter": 'NO_FILTER',
                },
            ),
            (
                "Find me thrillers with a strong female lead released between 2015 and 2020.",
                {
                    "query": "thriller strong female lead",
                    "filter": "and(eq('Genre', 'Thriller'), gt('Release Year', 2015), lt('Release Year', 2021))",
                },
            ),
            (
                "Find me highly rated drama movies in English that are less than 2 hours long",
                {
                    "query": "Highly rated drama English under 2 hours",
                    "filter": 'and(eq("Genre", "Drama"), eq("Language", "English"), lt("Runtime (minutes)", 120))',
                },
            ),
        ]

        metadata_field_info = [
            AttributeInfo(
                name="Title", description="The title of the movie", type="string"),
            AttributeInfo(name="Runtime (minutes)",
                          description="The runtime of the movie in minutes", type="integer"),
            AttributeInfo(name="Language",
                          description="The language of the movie", type="string"),
            AttributeInfo(name="Release Year",
                          description="The release year of the movie", type="integer"),
            AttributeInfo(name="Genre", description="The genre of the movie",
                          type="string or list[string]"),
            AttributeInfo(name="Actors", description="The actors in the movie",
                          type="string or list[string]"),
            AttributeInfo(name="Directors", description="The directors of the movie",
                          type="string or list[string]"),
            AttributeInfo(name="Stream", description="The streaming platforms for the movie",
                          type="string or list[string]"),
            AttributeInfo(name="Buy", description="The platforms where the movie can be bought",
                          type="string or list[string]"),
            AttributeInfo(name="Rent", description="The platforms where the movie can be rented",
                          type="string or list[string]"),
            AttributeInfo(name="Production Companies",
                          description="The production companies of the movie", type="string or list[string]"),
        ]

        self.constructor_prompt = get_query_constructor_prompt(
            document_content_description,
            metadata_field_info,
            allowed_comparators=allowed_comparators,
            examples=examples,
        )

    def initialize_vector_store(self):
        # Create empty index
        PINECONE_KEY, PINECONE_INDEX_NAME = os.getenv(
            'PINECONE_API_KEY'), os.getenv('PINECONE_INDEX_NAME')

        pc = Pinecone(api_key=PINECONE_KEY)

        # Target index and check status
        pc_index = pc.Index(PINECONE_INDEX_NAME)

        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

        self.vectorstore = PineconeVectorStore(
            pc_index, embeddings
        )

    def initialize_retriever(self):
        query_model = ChatOpenAI(
            model=self.RETRIEVER_MODEL_NAME,
            temperature=0,
            streaming=True,
        )

        output_parser = StructuredQueryOutputParser.from_components()
        query_constructor = self.constructor_prompt | query_model | output_parser

        self.retriever = SelfQueryRetriever(
            query_constructor=query_constructor,
            vectorstore=self.vectorstore,
            structured_query_translator=PineconeTranslator(),
            search_kwargs={'k': 10}
        )

    def initialize_chat_model(self):
        def format_docs(docs):
            return "\n\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs)

        chat_model = ChatOpenAI(
            model=self.SUMMARY_MODEL_NAME,
            temperature=0,
            streaming=True,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    """
                    Your goal is to recommend films to users based on their 
                    query and the retrieved context. If a retrieved film doesn't seem 
                    relevant, omit it from your response. If your context is empty
                    or none of the retrieved films are relevant, do not recommend films, but instead
                    tell the user you couldn't find any films that match their query.
                    Aim for three to five film recommendations, as long as the films are relevant. You cannot 
                    recommend more than five films. Your recommendation should 
                    be relevant, original, and at least two to three sentences 
                    long.
                    
                    YOU CANNOT RECOMMEND A FILM IF IT DOES NOT APPEAR IN YOUR 
                    CONTEXT.

                    # TEMPLATE FOR OUTPUT
                    - **Title of Film**:
                        - Runtime:
                        - Release Year:
                        - Streaming:
                        - (Your reasoning for recommending this film)
                    
                    Question: {question} 
                    Context: {context} 
                    """
                ),
            ]
        )

        # Create a chatbot Question & Answer chain from the retriever
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: format_docs(x["context"])))
            | prompt
            | chat_model
            | StrOutputParser()
        )

        self.rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

    def ask(self, query: str):
        try:
            for chunk in self.rag_chain_with_source.stream(query):
                for key in chunk:
                    if key == 'answer':
                        yield chunk[key]
        except Exception as e:
            print(f"An error occurred: {e}")
