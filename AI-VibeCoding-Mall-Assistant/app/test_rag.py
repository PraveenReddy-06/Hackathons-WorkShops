from rag_pipeline import ask_mall_assistant

question = "Suggest a laptop under 50000 with best discount"

response = ask_mall_assistant(question)

print("\nAI Response:\n")
print(response)