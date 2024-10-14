import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from secret_key import openapi_key
os.environ["OPENAI_API_KEY"] = openapi_key
llm = OpenAI(temperature = 0.9, max_tokens = 500)

st.title("News Research Tool 📈")
st.sidebar.title("New Article URLs")
main_placeholder = st.empty()

urls = [] 
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
    
process_url_clicked = st.sidebar.button("Process URL")
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started....✅✅✅")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", ".", ","],
        chunk_size = 1000
    )
    main_placeholder.text("Text Splitter...Started....✅✅✅")
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vector_index = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vectors Started Building....✅✅✅")
    time.sleep(2)
    vector_index.save_local("vector_index")
    
    
query = main_placeholder.text_input("Ask anything about the blogs...")
if query:
    x = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = x.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = retriever)
    result = chain({"question": query}, return_only_outputs = True)
    st.header("Answer")
    st.subheader(result['answer'])
    
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources")
        sources_list = sources.split("\n")
        for source in sources_list:
            st.write(source)