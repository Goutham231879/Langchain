from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

os.environ["OPENAI_API_KEY"] = "AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe"

sample_text = """
The solar system consists of the Sun and eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.
Each planet has unique characteristics. For example, Jupiter is the largest planet, known for its Great Red Spot.
"""

texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(sample_text)
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(texts, embeddings)
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(), return_source_documents=True)

def ask_question(question):
    result = qa_chain({"query": question})
    return result["result"], result["source_documents"]

question = "What is the largest planet in the solar system?"
answer, sources = ask_question(question)
print(answer)
for doc in sources:
    print(doc.page_content)
