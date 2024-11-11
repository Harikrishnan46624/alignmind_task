# prompt_template="""
# Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """




prompt_template = """
Using only the provided context, answer the user's question directly and concisely. 
Do not repeat the question, provide additional information, or add explanations. 
If the context does not contain the answer, respond with "I'm not sure."

Context:
{context}

Question:
{question}

Answer:
"""