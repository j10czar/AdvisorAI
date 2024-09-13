import streamlit as st
from model import process_input

st.set_page_config(
    page_title="AdvisorAI",
    page_icon="🐊",
    layout="centered",
    initial_sidebar_state="collapsed",
)



st.title("🐊 AdvisorAI")
st.write("An AI-powered academic advisor for University of Florida students. Built by Jason Tenczar.")
st.write("")
st.write("")



col1, col2 = st.columns([5,1])


with col1:
    if st.button("New Conversation"):
        st.session_state.messages = []  # Clear chat history
        st.experimental_rerun()
with col2:
    st.link_button("👾 GitHub","https://github.com/j10czar/AdvisorAI")

st.divider()
with st.chat_message(name="assistant"):
    st.write("👋 Hello I'm Alberta, your personal student advisor! Welcome to AdvisorAI. How may I help you today?")


# pull chat history from local storage

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



#respond to user prompt
if prompt := st.chat_input("How may I help you?"):
    #prompt the model with user chat message
    with st.chat_message("user"):
        st.write(prompt)
    #add prompt to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})



    response = process_input(prompt, st.session_state.messages)
    #add response to chat history
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})





