from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load the existing index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./doc_index",
    embedding_function=embeddings
)

# Set up Claude
llm = ChatAnthropic(model="claude-sonnet-4-5")

# Custom prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following excerpts from the document to answer the question.
If the answer isn't in the excerpts, say so — don't make things up.

Document excerpts:
{context}

Question: {question}

Answer:"""
)

# Build the chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Q&A chain ready!")
questions = [
    "What is the main topic of this document?",
    "What are the key recommendations?",
    "Who is the intended audience?",
]

for q in questions:
    print(f"\nQ: {q}")
    answer = qa_chain.invoke(q)
    print(f"A: {answer}")