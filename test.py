from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def initialize_retriever():
    
    #document loading
    loader = PyPDFLoader("AN IV_Cap 1_Sangerari_txt.pdf")
    documents = loader.load()
    
    #text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    #embedding
    emb = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=emb)
    
    #emb=OpenAIEmbeddings()
    #vectorstore = Chroma.from_documents(documents=splits, embedding=emb)

    # Retrieve and generate using the relevant snippets of the document.
    retriever = vectorstore.as_retriever()
    return retriever

def get_chatbot():
    llm = ChatOpenAI(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        openai_api_key="EMPTY",
        openai_api_base="http://localhost:8000/v1",
        temperature=0,
    )
    
    
    conversational_memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True,input_key="input")
    
    
    system_message_template='''
    You are a helpful assistant. Your task is to help the user.
    Use the following pieces of retrieved context to answer questions:
    {context}
    
    {chat_history}
    '''
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
    human_template="{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    llmchain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt]), memory = conversational_memory)
    
    
    retriever = initialize_retriever()
    
    return create_retrieval_chain(retriever, llmchain)


agent=get_chatbot()

while True:
    query = input(">> ")
    #Ask me a question about GAN, tell me the answer right after
    rep=agent.invoke({"input": query})
    question_anwer=rep['answer']['text']
    print(rep['answer']['text'])
    print()
    #print(rep['context'])
    #print(rep['context'][0].metadata['page'])
    
    
    rep_2=agent.invoke({"input": 'Quote me the context that gives the answer to this questions:'+question_anwer})
    print(rep_2)
    
    
