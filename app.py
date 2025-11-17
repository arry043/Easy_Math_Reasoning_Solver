# import os
# from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains.llm_math.base import LLMMathChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentType
from langchain_classic.agents import initialize_agent
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.tools import Tool


# Streamlit App
st.set_page_config(page_title="Text-to-Math Problem Solver", page_icon="🦜")

st.title("Text-to-Math Problem Solver Using LangChain")

st.sidebar.markdown("## Authentication")
groq_api_key=st.sidebar.text_input("Groq API Key",value="",type="password")
if groq_api_key:
    st.spinner("Authenticating...")
    # delay half second
    import time
    time.sleep(0.5)
    st.info("You are authenticated")
else:
    st.error("Please enter your Groq API Key to proceed.")
    st.stop()

# llm=ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
llm=ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# initializing tools
wikipedia_wrapper=WikipediaAPIWrapper()
wiki_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find the various information on the topics mentioned in the text."
)


# initialize the math tool
math_chain=LLMMathChain.from_llm(llm=llm, verbose=False)
# math_chain=LLMMathChain.from_llm(llm=llm, verbose=True, return_only_outputs=True)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description=(
    "Use this tool to compute exact results for arithmetic expressions. "
    "Supports +, -, *, /, **, and parentheses. Input must be a string "
    "containing a valid math expression, and the tool returns the evaluated "
    "numeric result. Use it whenever precise mathematical computation is needed."
    )
)


prompt="""
You are an agent tasked for solving users mathematical questions. 
Logically arrive at the solution and provide a detailed explanation and display it point by point for the question below. 
If the question is not math related, please respond with 'I am sorry, I cannot help you with that.'
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all Tool into chain
chain=LLMChain(
    llm=llm,
    prompt=prompt_template
)

reasoning_tool=Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# initialize the agent
assistent_agen=initialize_agent(
    tools=[wiki_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # max_iterations=10,
    # max_execution_time=10,
    verbose=True,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role": "assistant", "content": "Hey, I'm the Math Problem Solver. I can solve all your math problems. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Func to generate the response
# def generate_response(user_question):
#     respone=assistent_agen.invoke({"input": user_question})
#     return respone

# question
user_question=st.text_area("Enter your question", "What is the square root of 144?")

# Interaction
if st.button("Find Answer"):
    if user_question:
        with st.success(st.spinner("Generating Answer...")):
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.chat_message("user").write(user_question)
            
            st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            # response=assistent_agen.run(st.session_state.messages, callbacks=[st_cb])
            response = assistent_agen.run(user_question, callbacks=[st_cb])
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("Answer Generated")
            st.success(response)
    else:
        st.warning("Please enter a question.")