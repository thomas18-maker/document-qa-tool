import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.title("Document Q&A")
st.write("Ask any question about your document.")

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./doc_index",
        embedding_function=embeddings
    )
    llm = ChatAnthropic(model="claude-sonnet-4-5")
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following excerpts from the document to answer the question.
If the answer isn't in the excerpts, say so — don't make things up.

Document excerpts:
{context}

Question: {question}

Answer:"""
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

qa_chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question about your document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = qa_chain.invoke(question)
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})