import os
import base64
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import  pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, dotenv_values
import streamlit as st
from PIL import Image
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_chat import message

# Load environment variables and auto-reload on change
load_dotenv()
env_vars = dotenv_values()
langchain_api_key=env_vars["LANGCHAIN_API_KEY"]

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "10.STB-6000_chat"

print("------- load embedding started------")
embedding = HuggingFaceBgeEmbeddings(model_name="C:/Users/arasu/Workspace/Projects/GenAI/embeddings/hkunlp_instructor-large/")
vector_db = Chroma(persist_directory="text_vector_db",embedding_function=embedding)
print("------- load embedding completed------")
print("------- load model started------")
# Create LLM model
model = "C:/Users/arasu/Workspace/Projects/GenAI/models/MBZUAILaMini-Flan-T5-248M/"
# model = "MBZUAI/LaMini-T5-738M"
tokenizer = T5Tokenizer.from_pretrained(model,truncation=True)
base_model = T5ForConditionalGeneration.from_pretrained(model)
pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 500,
        do_sample = True,
        temperature = 0.5,
        top_p= 0.95
    )
llm = HuggingFacePipeline(pipeline=pipe)
print("------- load model ended------")
print("------- load document started------")
text = []
loader = TextLoader("C:/Users/arasu/Workspace/Projects/GenAI/10.STB-6000_chat/manual/STB-6000_USER_MANUAL.txt",encoding="UTF-8")
text.extend(loader.load())
text_spiltter = RecursiveCharacterTextSplitter(chunk_size = 1400, chunk_overlap = 350)
text_documents = text_spiltter.split_documents(text)
print("------- load document ended------")

def check_greeting(input_dict):
    if not isinstance(input_dict, dict) or 'query' not in input_dict:
        return "Invalid input. Please provide a dictionary with a 'query' key."

    greetings_synonyms = ["hi","hello", "greetings", "hey", "howdy", "hiya", "salutations", "aloha", "good day",
                          "hola", "what's up", "yo", "wassup", "hi there", "how's it going", "how are you",
                          "hi-oh", "hi-de-ho", "hi-ya", "ciao", "how's tricks"]
    
    synonyms_for_thank_you = ["thank you","gratitude", "appreciation", "acknowledgment", "recognition", "thanks", "blessings", "cheers", "kudos", "praise", "acknowledgement", "applause", "gratefulness", "tribute", "ovation", "hail", "salute", "admiration", "thanksgiving", "credit", "props"]

    input_lower = input_dict['query'].lower()

    for synonym in greetings_synonyms:
        if input_lower in synonym:
            print("------- found greeting------")
            return "Hi, how can I assist you?"
        if input_lower in synonyms_for_thank_you:
            print("------- found greeting------")
            return "you are welcome!"
    print("------- no greeting found------")
    return "No common greeting found"

def get_qa_chain(retr):

    # Create a retriever for querying the vector database
    if retr=="normal":
        retriever = vector_db.as_retriever(search_kwargs={"k":2})
    else:
        retriever = vector_db.as_retriever(search_kwargs={"k":2})
        bm25_retriever = BM25Retriever.from_documents(text_documents)
        retriever = EnsembleRetriever(retrievers=[retriever, bm25_retriever], weights=[0.5, 0.5])
    
    template = """
    You are friendly customer care assistant trying to help user on the context provided.\
    if the answer is not found in the context then reply "I Dont Know".\
    context: {context}
    question: {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        input_key="query",
        chain_type_kwargs=chain_type_kwargs
    )
    return qa,retriever  

def process_answer(instruction):
    response = ''
    image = ""
    instruction = instruction
    print("-------check greeting started------")
    greetings = check_greeting(instruction)
    print("-------check greeting completed------")
    if greetings == "No common greeting found":
        print("------- load qa chain started------")
        qa,retriever = get_qa_chain('normal')
        print("------- load qa chain ended------")
        docs = retriever.get_relevant_documents(instruction['query'])
        if docs and 'image' in docs[0].metadata:
            image = docs[0].metadata['image']
        else:
            image = ""
        print("------- LLM call started------")    
        generated_text = qa(instruction)
        print("------- LLM call ended------")
        answer = generated_text['result']

    else:
         answer =  greetings 
    return answer,image

def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_images(image):
    if image!="":
        st.markdown(f"""<img src="data:png;base64,{image}" width='350' height='350' >""", True)


# Display conversation history using Streamlit messages
def display_conversation(history,image):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            message(history["generated"][i],key=str(i))
        with col2:
            if i == (len(history["generated"])-1):
                display_images(image) 
        

    # Function to convert image to base64
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()