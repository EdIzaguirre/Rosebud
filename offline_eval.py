import weave
import asyncio
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, ContextRelevancy, Faithfulness
from datasets import Dataset
from rosebud_chat_model import rosebud_chat_model
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import json

# Set environment variable to limit parallel workers
os.environ['WEAVE_PARALLELISM'] = '3'

with open('./config.json') as f:
    config = json.load(f)


@weave.op()
def evaluate_with_ragas(query, model_output):
    # Put data into a Dataset object
    data = {
        "question": [query],
        "contexts": [[model_output['context']]],
        "answer": [model_output['answer']]
    }
    dataset = Dataset.from_dict(data)

    # Define metrics to judge
    metrics = [
        AnswerRelevancy(),
        ContextRelevancy(),
        Faithfulness(),
        # harmfulness
    ]

    judge_model = ChatOpenAI(model=config['JUDGE_MODEL_NAME'])
    embeddings_model = OpenAIEmbeddings(model=config['EMBEDDING_MODEL_NAME'])

    evaluation = evaluate(dataset=dataset, metrics=metrics, llm=judge_model, embeddings=embeddings_model)

    return {
        "answer_relevancy": float(evaluation['answer_relevancy']),
        "context_relevancy": float(evaluation['context_relevancy']),
        "faithfulness": float(evaluation['faithfulness']),
        # "harmfulness": float(evaluation['harmfulness'])
    }


def run_evaluation():
    # Initialize chat model
    model = rosebud_chat_model()

    # Define evaluation questions
    questions = [
        {"query": "Suggest a good movie based on a book."},  # Adaptations
        {"query": "Suggest a film for a cozy night in."},  # Mood-Based
        {"query": "What are some must-watch horror movies?"},  # Genre-Specific
        {"query": "Recommend a film about overcoming adversity."},  # Theme-Based
        {"query": "Can you suggest a movie set in ancient Rome?"},  # Setting-Based
        {"query": "What are some essential movies to watch in the Marvel Cinematic Universe?"},  # Franchise Films
        {"query": "What are some lesser-known indie films worth watching?"},  # Indie Films
        {"query": "What's a good starting point if I want to explore Japanese cinema?"},  # Cultural Films
        {"query": "What's a good movie for a group with varied tastes?"},  # Contextual
        {"query": "I loved Inception and The Matrix. What should I watch next?"},  # Preference-Based
        {"query": "What are some highly rated documentary films?"},  # Ratings based
        {"query": "I'm looking for a good horror movie from the 1970s."},  # Year-Based
        {"query": "Can you suggest a good movie set in the 1980s?"},  # Era-Specific
        {"query": "What are some great Christmas movies?"},  # Seasonal
        {"query": "Recommend a movie that is set in the future."},  # Future-Set Films
        {"query": "Suggest a movie that blends horror and science fiction."},  # Multi-Genre
        {"query": "Recommend a film with a strong female lead."},  # Character-Based
        {"query": "Can you suggest a movie with a surprising twist at the end?"},  # Plot-Specific
        {"query": "Can you suggest a film with Leonardo DiCaprio in the lead role?"},  # Actor-Specific
        {"query": "I want some fantasy movies featuring dragons that are under 90 minutes long."}  # Multi-criteria
    ]

    # Create Weave Evaluation object
    evaluation = weave.Evaluation(dataset=questions, scorers=[evaluate_with_ragas])

    # Run the evaluation
    asyncio.run(evaluation.evaluate(model))


if __name__ == "__main__":
    weave.init('film-search')
    run_evaluation()
