import streamlit as st
from langchain.llms import OpenAI
import os
import sys
import pandas as pd
from io import StringIO
sys.path.append('./')
from urllib.parse import unquote
from constants import MODES, MODEL_TYPES, OPENAI_CHAT_MODELS, OPENAI_COMPLETION_MODELS, HUGGINGFACE_MODELS
from data_loader import render_prompts, load_data
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


params = st.experimental_get_query_params()
st.experimental_set_query_params()


st.title('🦜🔗 F-Chat')

# openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# def generate_response(input_text):
#     llm = OpenAI(temperature=0.2, openai_api_key=os.environ['OPENAI_API_KEY'])
#     st.info(llm(input_text))

# with st.form('my_form'):
#     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#     submitted = st.form_submit_button('Submit')
#     if not openai_api_key.startswith('sk-'):
#         st.warning('Please enter your OpenAI API key!', icon='⚠')
#     if submitted and openai_api_key.startswith('sk-'):
#         generate_response(text)




################# Functions
    ############# Modules
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
        

def text_splitter(text, chunk_size=850, chunk_overlap=150, splitter=RecursiveCharacterTextSplitter()):
    # TODO: Split text into chunks
    # Chunks should be passed from Streamlit to the model
    # Splitting method should be chosen by user
    if splitter == 'RecursiveCharacterTextSplitter':
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n"],
            length_function=len
        )
    
    rc_splitter = splitter.split_text(text)
    return rc_splitter
   
    
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def retrieve():
    return  None


def chat():
    return None
        









################# Main

def main():
    with st.sidebar:
        # Choose mode
        if "mode" not in st.session_state and "mode" in params:
            st.session_state.mode = unquote(params["mode"][0])
        mode = st.radio("Choose a mode", MODES, key="mode")
        
        # ##
        if mode != 'Model Comparison':
            # Model Type
            if "model_type" not in st.session_state and "models" in params:
                st.session_state.model_type = unquote(params["model_type"][0])
            
            ## Upload file
            uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=True)
            if st.button('Process'):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(uploaded_file)
                    # st.write(raw_text)
                    
                    text_chunks = text_splitter(raw_text)
                    st.write(len(text_chunks[6]))
            
                
            model_type = st.selectbox("Modelltyp", MODEL_TYPES, key="model_type")
            
            model, api_key = None, None
            
            # Model
            if "model" not in st.session_state and "model" in params:
                st.session_state.model = unquote(params["model"][0])
            
            # If HF => Ask for Repo ID and API Key
            if model_type == "HuggingFace Hub":
                if "model" not in st.session_state and "model" in params:
                    st.session_state.model = unquote(param["model"][0])
                model = st.selectbox("Model", HUGGINGFACE_MODELS, key="model")
                api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]
                # model = st.text_input("Repo ID", key="model")
                # api_key = st.text_input("HuggingFace Hub API Key", type="password")
            # If OpenAI => Choose Model and provide API key
            elif model_type == "OpenAI Chat":
                if "model" not in st.session_state and "model" in params:
                    st.session_state.model = unquote(params["model"][0])
                model = st.selectbox("Model", OPENAI_CHAT_MODELS, key="model")
                # api_key = st.text_input("Provide OpenAI API Key", type="password")
                api_key = os.environ['OPENAI_API_KEY']
            # elif model_type == "OpenAI Completion":
            #     model = st.selectbox("Model", OPENAI_COMPLETION_MODELS, key="model")
            #     api_key = st.text_input("OpenAI API Key", type="password")
                
            
            # Prompt Template
            variable_count = 0
            if mode == "Chat with your data":
                # Add Prompt Template
                instruction_count = st.number_input("Add Template", step=1, min_value=1, max_value=5)
                # User Input
                prompt_count = st.number_input("Add User Input", step=1, min_value=1, max_value=10)
                # Variables to Prompt
                variable_count = len(params["var_names"][0].split(",")) if "var_names" in params else 1
                variable_count = st.number_input("Add Variable", step=1, min_value=1, max_value=10, value=variable_count)
            elif model_type == "OpenAI Chat":
                instruction_count = st.number_input("Add System Message", step=1, min_value=1, max_value=5)
                prompt_count = st.number_input("Add User Message", step=1, min_value=1, max_value=10)
            else:
                instruction_count = st.number_input("Add Instruction", step=1, min_value=1, max_value=5)
                prompt_count = st.number_input("Add Prompt", step=1, min_value=1, max_value=10)
                
            # Variables to prompt
            var_names = []
            if "var_names" in params:
                var_names = unquote(params["var_names"][0]).split(",")
            for i in range(variable_count):
                if f"varname_{i}" not in st.session_state:
                    if len(var_names) > i:
                        st.session_state[f"varname_{i}"] = var_names[i]
                    else:
                        st.session_state[f"varname_{i}"] = f"Variable {i+1}"

            for i in range(variable_count):
                var_names.append(
                    st.text_input(
                        f"Variable {i+1} Name",
                        key=f"varname_{i}",
                    )
                )
            
            # Temperature
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperature")
            top_p = None
            max_tokens = None
            presence_penalty = None
            frequency_penalty = None
            if model_type == "OpenAI Chat" or model_type == "OpenAI Completion":
                # top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key="top_p")
                # max_tokens = st.number_input("Max Tokens", min_value=0, value=, step=1, key="max_tokens")
                presence_penalty = st.slider(
                    "Presence Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.01, key="presence_penalty"
                )
                frequency_penalty = st.slider(
                    "Frequency Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.01, key="frequency_penalty"
                )
                
                
        ## Else: Compare models panels
        ## How many models to compare?
        ## Prompts to models to compare
        ## Credentials for such models 
        
        # else:
        #     model_count = st.number_input("Add Model", step=1, min_value=1, max_value=5)
        #     prompt_count = st.number_input("Add Prompt", step=1, min_value=1, max_value=10)
        #     openai_api_key = st.text_input("OpenAI API Key", type="password")
        #     hf_api_key = st.text_input("HuggingFace Hub Key", type="password")
        
        
        
    
    
    # MAIN PANEL
        
    ## Create grid for inputs and outputs
    if mode == "Chat with your data":
        instruction = None
        
        # Set default prompt
        if model_type == "OpenAI Chat":
            instruction = st.text_area(
                "System Message",
                key="instruction",
                value=["instruction"][0] if "instruction" in params else "Du är en hjälpsam chattbot",
            )
            
        elif model_type == "HuggingFace Hub":
            instruction = st.text_area(
                "Instruction",
                key="instruction",
                value=params["instruction"][0] if "instruction" in params else "You are a helpful AI assistant.",
            )
            
        
        # Add placeholders for output
        placeholders = [[st.empty() for _ in range(instruction_count + variable_count)] for _ in range(prompt_count)]

        cols = st.columns(instruction_count + variable_count)
        
        # Create top row for instructions or system messages
        with cols[0]:
            a = None
        templates = []
        for j in range(variable_count, instruction_count + variable_count):
            with cols[j]:
                templates.append(
                    st.text_area(
                        "Prompt Template",
                        key=f"col_{j-variable_count}",
                        value=params["template"][0] if "template" in params else "",
                    )
                )
        
        ## Create rows for prompts, and output placeholders
        vars = []
        varlist = []
        if "vars" in params:
            varlist = params["vars"][0].split(",")
        for i in range(prompt_count):
            cols = st.columns(instruction_count + variable_count)
            vars.append(dict())
            for j in range(variable_count):
                with cols[j]:
                    vars[i][var_names[j]] = st.text_area(
                        var_names[j], key=f"var_{i}_{j}", value=varlist[j] if len(varlist) > 0 else ""
                    )
            for j in range(variable_count, instruction_count + variable_count):
                with cols[j]:
                    placeholders[i][j] = st.empty()  # placeholders for the future output
            st.divider()
            
        
        ## Buttons
        run_button, clear_button, share_button = st.columns([1, 1, 1], gap="small")
        with run_button:
            run = st.button("Kör")
        with clear_button:
            clear = st.button("Rensa")
        with share_button:
            share = st.button("Dela")
            
            
        #### Link
        
        
        
        
        
        # ## Run
        # if run:
        #     prompts = render_prompts(templates, vars)
        #     df = load_data(model_type, model, [instruction], prompts, temperature, api_key=api_key)
        #     st.session_state.prompts = prompts
        #     st.session_state.df = df
        #     for i in range(len(vars)):
        #         for j in range(len(templates)):
        #             placeholders[i][j + variable_count].markdown(df["response"][i + len(vars) * j])
        # elif "df" in st.session_state and "prompts" in st.session_state and not clear:
        #     df = st.session_state.df
        #     prompts = st.session_state.prompts
        #     for i in range(len(vars)):
        #         for j in range(len(templates)):
        #             placeholders[i][j + variable_count].markdown(df["response"][i + len(vars) * j])

            

                
            
        
        

if __name__ == "__main__":
    main()    