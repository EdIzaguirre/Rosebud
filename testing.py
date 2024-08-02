from rosebud_chat_model import rosebud_chat_model


chat_model = rosebud_chat_model()
query = "Recommend some films similar to star wars movies but not part of the star wars universe"

query_constructor = chat_model.query_constructor.invoke(query)


print(f"query_constructor: {query_constructor}")

print("response:")
for chunk in chat_model.rag_chain_with_source.stream(query):
    print(chunk)
