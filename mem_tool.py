from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool
from langchain.agents import initialize_agent
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm
from uuid import uuid4
import pinecone
import tiktoken
import openai
import os


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

dataset = load_dataset("Nickgannon10/langchain-docs", split="train")

data = dataset.to_pandas()

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai.api_key
)

YOUR_API_KEY = os.getenv("PINECONE_API_KEY")
YOUR_ENV = os.getenv("ENVIRONMENT")

index_name = 'langchain-retrieval-agent'
pinecone.init(
    api_key=YOUR_API_KEY,
    environment=YOUR_ENV
)

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=1536  
    )

index = pinecone.Index(index_name)

documents = [{
    'id': doc['id'],
    'text': doc['text'],
    'metadata': {'url': doc['source']}
} for doc in dataset]

batch_size = 100

for i in tqdm(range(0, len(documents), batch_size)):
    # get end of batch
    i_end = min(len(documents), i+batch_size)
    batch = documents[i:i_end]
    # first get metadata fields for this record
    metadatas = [{**doc['metadata'], 'text': doc['text']} for doc in batch]
    # get the list of text / documents
    texts = [doc['text'] for doc in batch]
    # create document embeddings
    embeds = embed.embed_documents(texts)
    # get IDs
    ids = [doc['id'] for doc in batch]
    # add everything to pinecone
    index.upsert(vectors=zip(ids, embeds, metadatas))

from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

query = "Tell me about the LLMChain documentation"

vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=openai.api_key,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering general knowledge queries about langchain documentation '
            'more information about the topic'
        )
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

query = "Tell me about the LLMChain documentation"
agent(query)

