# Langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.query_constructor.base import AttributeInfo, StructuredQuery
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSerializable
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
from typing import Optional
from typing import Dict

# Weave
import weave
from weave import Model


class rosebud_chat_model(Model):
    RETRIEVER_MODEL_NAME: str = None
    SUMMARY_MODEL_NAME: str = None
    EMBEDDING_MODEL_NAME: str = None
    constructor_prompt: Optional[ChatPromptTemplate] = None
    vectorstore: Optional[PineconeVectorStore] = None
    retriever: Optional[SelfQueryRetriever] = None
    rag_chain_with_source: Optional[RunnableParallel] = None
    query_constructor: RunnableSerializable[Dict, StructuredQuery] = None
    context: str = None
    top_k: int = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        load_dotenv()
        with open('./config.json') as f:
            config = json.load(f)
            self.RETRIEVER_MODEL_NAME = config["RETRIEVER_MODEL_NAME"]
            self.SUMMARY_MODEL_NAME = config["SUMMARY_MODEL_NAME"]
            self.EMBEDDING_MODEL_NAME = config["EMBEDDING_MODEL_NAME"]
            self.top_k = config["top_k"]
        self.initialize_query_constructor()
        self.initialize_vector_store()
        self.initialize_retriever()
        self.initialize_chat_model(config)

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

        # Define allowed operators list
        allowed_operators = [
            "AND",
            "OR"
        ]

        examples = [
            (
                "Recommend some films similar to star wars movies but not part of the star wars universe.",
                {
                    "query": "space opera, adventure, epic battles",
                    "filter": "and(nin('Title', ['Star Wars']), in('Genre', ['Science Fiction', 'Adventure']))"
                }
            ),
            (
                "Show me critically acclaimed dramas without Tom Hanks.",
                {
                    "query": "critically acclaimed drama",
                    "filter": "and(eq('Genre', 'Drama'), nin('Actors', ['Tom Hanks']), gt('Rating', 7))",
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
                    "query": "Highly rated drama English",
                    "filter": 'and(eq("Genre", "Drama"), eq("Language", "English"), lt("Runtime (minutes)", 120), gt("Rating", 7))',
                },
            ),
            (
                "Short films that discuss the meaning of life.",
                {
                    "query": "meaning of life",
                    "filter": 'lt("Runtime (minutes)", 40))',
                },
            ),
        ]

        metadata_field_info = [
            AttributeInfo(name="Title", description="The title of the movie",
                          type="string"),
            AttributeInfo(name="Runtime (minutes)", description="The runtime of the movie in minutes",
                          type="integer"),
            AttributeInfo(name="Language", description="The language of the movie",
                          type="string"),
            AttributeInfo(name="Release Year", description="The release year of the movie",
                          type="integer"),
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
            AttributeInfo(name="Rating",
                          description="Rating of a film, out of 10", type="float"),
        ]

        self.constructor_prompt = get_query_constructor_prompt(
            document_content_description,
            metadata_field_info,
            allowed_comparators=allowed_comparators,
            allowed_operators=allowed_operators,
            examples=examples,
        )

    def initialize_vector_store(self):
        # Create empty index
        PINECONE_KEY, PINECONE_INDEX_NAME = os.getenv(
            'PINECONE_API_KEY'), os.getenv('PINECONE_INDEX_NAME')

        pc = Pinecone(api_key=PINECONE_KEY)

        # Target index and check status
        pc_index = pc.Index(PINECONE_INDEX_NAME)

        embeddings = OpenAIEmbeddings(model=self.EMBEDDING_MODEL_NAME)

        namespace = "film_search_prod"
        self.vectorstore = PineconeVectorStore(
            index=pc_index,
            embedding=embeddings,
            namespace=namespace
        )

    def initialize_retriever(self):
        query_model = ChatOpenAI(
            model=self.RETRIEVER_MODEL_NAME,
            temperature=0,
            streaming=True,
        )

        output_parser = StructuredQueryOutputParser.from_components()
        self.query_constructor = self.constructor_prompt | query_model | output_parser

        self.retriever = SelfQueryRetriever(
            query_constructor=self.query_constructor,
            vectorstore=self.vectorstore,
            structured_query_translator=PineconeTranslator(),
            search_kwargs={'k': self.top_k}
        )

    def initialize_chat_model(self, config):
        def format_docs(docs):
            return "\n\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs)

        chat_model = ChatOpenAI(
            model=self.SUMMARY_MODEL_NAME,
            temperature=config['TEMPERATURE'],
            streaming=True,
            max_retries=10
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
                        - **Runtime:**
                        - **Release Year:**
                        - **Streaming:**
                        - Your reasoning for recommending this film

                    Question: {question}
                    Context: {context}
                    """
                ),
            ]
        )

        # Create a chatbot Question & Answer chain from the retriever
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: format_docs(x["context"]))) | prompt | chat_model | StrOutputParser()
        )

        self.rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough(), "query_constructor": self.query_constructor}
        ).assign(answer=rag_chain_from_docs)

    # @weave.op()
    def predict_stream(self, query: str):
        weave.init('film-search')

        try:
            for chunk in self.rag_chain_with_source.stream(query):
                if 'answer' in chunk:
                    yield chunk['answer']
                elif 'context' in chunk:
                    docs = chunk['context']
                    self.context = "\n\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs)
                elif 'query_constructor' in chunk:
                    self.query_constructor = chunk['query_constructor'].json()

        except Exception as e:
            return {'answer': f"An error occurred: {e}"}

    @weave.op()
    async def predict(self, query: str):
        weave.init('film-search')

        try:
            result = self.rag_chain_with_source.invoke(query)
            return {
                'answer': result['answer'],
                'context': "\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in result['context'])
            }
        except Exception as e:
            return {'answer': f"An error occurred: {e}", 'context': ""}
