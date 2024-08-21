from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    temperature=1,
)


print(llm.invoke('tell me a joke'))

