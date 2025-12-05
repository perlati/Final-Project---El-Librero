import streamlit as st

st.set_page_config(page_title="EditorialCopilot", page_icon="📚")

st.title("EditorialCopilot – Internal Editorial Assistant")

st.write(
    "This is a placeholder UI. "
    "In the next steps we'll connect it to the RAG backend over the book catalogue."
)

user_input = st.text_input("Ask something about the catalogue:", "")

if user_input:
    st.write("Answer (stub): this will call the RAG pipeline later.")
