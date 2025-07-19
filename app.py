# app.py
import streamlit as st
from jain_gpt.retrieval import query_jain_gpt

st.set_page_config(page_title="Jain GPT", layout="centered")
st.title("☸️ Jain GPT")
st.write("Ask questions from the Jain text in Hindi.")

question = st.text_input("प्रश्न यहाँ लिखें:")
if st.button("Get Answer"):
    if not question.strip():
        st.warning("कृपया प्रश्न लिखें।")
    else:
        with st.spinner("सोच रहा हूँ..."):
            try:
                answer, context= query_jain_gpt(question)
                st.markdown(f"**जवाब:**\n\n{answer}")
                st.markdown(f"**reffernce:**\n\n{context}")
            except Exception as e:
                st.error(f"त्रुटि हुई: {e}")
