# Pinecone
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import NotFoundException

# Langchain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Prefect
from prefect import task, flow
from prefect.deployments import DeploymentImage

# Weave
import weave
from weave import Dataset

# General
import os
from dotenv import load_dotenv
import csv
from utils import get_id_list, get_data, write_file
import json


@task
def start():
    """
    Start-up: check everything works or fail fast!
    """

    # Print out some debug info
    print("Starting flow!")

    # Loading environment variables
    try:
        load_dotenv(verbose=True, dotenv_path='.env')
    except ImportError:
        print("Env file not found!")

    # Ensure user has set the appropriate env variables
    assert os.environ['LANGCHAIN_API_KEY']
    assert os.environ['OPENAI_API_KEY']
    assert os.environ['TMBD_API_KEY']
    assert os.environ['PINECONE_API_KEY']
    assert os.environ['PINECONE_INDEX_NAME']
    assert os.environ['TMDB_BEARER_TOKEN']
    assert os.environ['LANGCHAIN_TRACING_V2']
    assert os.environ['WANDB_API_KEY']


@task(retries=3, retry_delay_seconds=[1, 10, 100])
def pull_data_to_csv(config):
    TMBD_API_KEY = os.getenv('TMBD_API_KEY')
    YEARS = range(config["years"][0], config["years"][-1] + 1)
    CSV_HEADER = ['Title', 'Runtime (minutes)', 'Language', 'Overview',
                  'Release Year', 'Genre', 'Keywords',
                  'Actors', 'Directors', 'Stream', 'Buy', 'Rent',
                  'Production Companies', 'Rating']

    for year in YEARS:
        # Grab list of ids for all films made in {YEAR}
        movie_list = list(set(get_id_list(TMBD_API_KEY, year)))

        FILE_NAME = f'./data/{year}_movie_collection_data.csv'

        # Creating file
        with open(FILE_NAME, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

        # Iterate through list of ids to get data
        for id in movie_list:
            dict = get_data(TMBD_API_KEY, id)
            write_file(FILE_NAME, dict)

    print("Successfully pulled data from TMDB and created csv files in data/")


@task
def convert_csv_to_docs():
    # Loading in data from all csv files
    loader = DirectoryLoader(
        path="./data",
        glob="*.csv",
        loader_cls=CSVLoader,
        show_progress=True)

    docs = loader.load()

    metadata_field_info = [
        AttributeInfo(name="Title",
                      description="The title of the movie", type="string"),
        AttributeInfo(name="Runtime (minutes)",
                      description="The runtime of the movie in minutes", type="integer"),
        AttributeInfo(name="Language",
                      description="The language of the movie", type="string"),
        AttributeInfo(name="Release Year",
                      description="The release year of the movie as an integer", type="integer"),
        AttributeInfo(name="Genre",
                      description="The genre of the movie", type="string or list[string]"),
        AttributeInfo(name="Actors",
                      description="The actors in the movie", type="string or list[string]"),
        AttributeInfo(name="Directors",
                      description="The directors of the movie", type="string or list[string]"),
        AttributeInfo(name="Stream",
                      description="The streaming platforms for the movie", type="string or list[string]"),
        AttributeInfo(name="Buy",
                      description="The platforms where the movie can be bought", type="string or list[string]"),
        AttributeInfo(name="Rent",
                      description="The platforms where the movie can be rented",
                      type="string or list[string]"),
        AttributeInfo(name="Production Companies",
                      description="The production companies of the movie", type="string or list[string]"),
        AttributeInfo(name="Rating",
                      description="Rating of a film, out of 10", type="float"),
    ]

    def convert_to_list(doc, field):
        if field in doc.metadata and doc.metadata[field] is not None:
            doc.metadata[field] = [item.strip()
                                   for item in doc.metadata[field].split(',')]

    def convert_to_int(doc, field):
        if field in doc.metadata and doc.metadata[field] is not None:
            doc.metadata[field] = int(
                doc.metadata[field])

    def convert_to_float(doc, field):
        if field in doc.metadata and doc.metadata[field] is not None:
            doc.metadata[field] = float(
                doc.metadata[field])

    fields_to_convert_list = ['Genre', 'Actors', 'Directors',
                              'Production Companies', 'Stream', 'Buy', 'Rent']
    fields_to_convert_int = ['Runtime (minutes)', 'Release Year']
    fields_to_convert_float = ['Rating']

    # Set 'overview' and 'keywords' as 'page_content' and other fields as 'metadata'
    for doc in docs:
        # Parse the page_content string into a dictionary
        page_content_dict = dict(line.split(": ", 1)
                                 for line in doc.page_content.split("\n") if ": " in line)

        doc.page_content = (
            'Title: ' + page_content_dict.get('Title') +
            '. Overview: ' + page_content_dict.get('Overview') +
            ' Keywords: ' + page_content_dict.get('Keywords')
        )

        doc.metadata = {field.name: page_content_dict.get(
            field.name) for field in metadata_field_info}

        # Convert fields from string to list of strings
        for field in fields_to_convert_list:
            convert_to_list(doc, field)

        # Convert fields from string to integers
        for field in fields_to_convert_int:
            convert_to_int(doc, field)

        # Convert fields from string to floats
        for field in fields_to_convert_float:
            convert_to_float(doc, field)

    print("Successfully took csv files and created docs")

    return docs


@task
def upload_docs_to_pinecone(docs, config):
    # Create empty index
    PINECONE_KEY, PINECONE_INDEX_NAME = os.getenv(
        'PINECONE_API_KEY'), os.getenv('PINECONE_INDEX_NAME')

    pc = Pinecone(api_key=PINECONE_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ))

    # Target index and check status
    pc_index = pc.Index(PINECONE_INDEX_NAME)
    print(pc_index.describe_index_stats())

    embeddings = OpenAIEmbeddings(model=config['EMBEDDING_MODEL_NAME'])
    namespace = "film_search_prod"

    try:
        pc_index.delete(namespace=namespace, delete_all=True)
    except NotFoundException:
        print(f"Namespace '{namespace}' not found. Not deleting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    else:
        print("Namespace deleted successfully.")

    PineconeVectorStore.from_documents(
        docs,
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )

    print("Successfully uploaded docs to Pinecone vector store")


@task
def publish_dataset_to_weave(docs):
    # Initialize Weave
    weave.init('film-search')

    rows = []
    for doc in docs:
        row = {
            'Title': doc.metadata.get('Title'),
            'Runtime (minutes)': doc.metadata.get('Runtime (minutes)'),
            'Language': doc.metadata.get('Language'),
            'Overview': doc.page_content.split('. Keywords: ')[0].split('Overview: ')[-1],
            'Release Year': str(doc.metadata.get('Release Year')),
            'Genre': doc.metadata.get('Genre'),
            'Keywords': doc.page_content.split('. Keywords: ')[-1],
            'Actors': doc.metadata.get('Actors'),
            'Directors': doc.metadata.get('Directors'),
            'Stream': doc.metadata.get('Stream'),
            'Buy': doc.metadata.get('Buy'),
            'Rent': doc.metadata.get('Rent'),
            'Production Companies': doc.metadata.get('Production Companies'),
            'Rating': doc.metadata.get('Rating')
        }
        rows.append(row)

    dataset = Dataset(name='Movie Collection', rows=rows)
    weave.publish(dataset)
    print("Successfully published dataset to Weave")


@flow(log_prints=True)
def pinecone_flow():
    with open('./config.json') as f:
        config = json.load(f)

    start()
    pull_data_to_csv(config)
    docs = convert_csv_to_docs()
    upload_docs_to_pinecone(docs, config)
    publish_dataset_to_weave(docs)


if __name__ == "__main__":
    pinecone_flow.deploy(
        name="pinecone-flow-deployment",
        work_pool_name="my-aci-pool",
        cron="0 0 * * 0",
        image=DeploymentImage(
            name="prefect-flows:latest",
            platform="linux/amd64",
        )
    )

    # For testing purposes
    # pinecone_flow()
