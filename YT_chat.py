from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

video_id = "VKGQ0TgFmBs"
transcript = ""
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id,languages=["en"])

    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript,"    kaslkjdlkas ")
    if not transcript.strip():
        print("Transcript is empty. Exiting.")
        exit()
except:
    print("No Captions available for this video")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(chunks)
#print(len(chunks))

embeddings = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2" )
print(embeddings)
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.index

# print(vector_store.index_to_docstore_id)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

#print(retriever.invoke("what colud gujarat have done differently to control mumbais batting"))

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

# question = "what colud gujarat have done differently to control mumbais batting"
# retrieved_docs = retriever.invoke(question)



#final_prompt = prompt.invoke({"context": context_text, "question": question})

llm = ChatGroq(model="qwen-qwq-32b")

# answer = llm.invoke(final_prompt)
# print(answer)

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(format_docs),
    'question' : RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

print(main_chain.invoke("what colud gujarat have done differently to control mumbai's batting"))
