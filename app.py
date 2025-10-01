from dotenv import load_dotenv

import streamlit as st
from utils import parse_markdown_table_to_df
from rag_chain import build_rag_chain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()



# Page title
st.set_page_config(
    page_title="شات بوت كلية الذكاء الاصطناعي - جامعة كفر الشيخ",
    page_icon="🤖",
    layout="wide"
)

# CSS to support Arabic (RTL) and improve UI
st.markdown("""
    <style>
    .main {
        direction: rtl;
        text-align: right;
    }
    .stChatMessage {
        direction: rtl;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

# Main interface
st.title("🤖 شات بوت كلية الذكاء الاصطناعي - جامعة كفر الشيخ")
st.markdown("---")

# Build the Chain
rag_chain = build_rag_chain()

if rag_chain is None:
    st.stop()

# Chat session (session state)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "table" in message.get("type", ""):
            # If the reply contains a table, display it as a DataFrame
            df = parse_markdown_table_to_df(message["content"])
            if df is not None:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.markdown(message["content"])
        else:
            st.markdown(message["content"])

# User input
if prompt := st.chat_input("اكتب سؤالك هنا... (مثل: ما هي الجدول الدراسي للمجموعة 1 يوم الأحد؟)"):
    # Add the question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # The reply
    with st.chat_message("assistant"):
        with st.spinner("جاري الرد..."):
            response = rag_chain.invoke(prompt)
        
        # Parse the response: if there's a Markdown table, convert it to a DataFrame
        df = parse_markdown_table_to_df(response)
        if df is not None:
            st.dataframe(df, use_container_width=True, hide_index=True)
            # Add the reply with type "table" for storage
            st.session_state.messages.append({"role": "assistant", "content": response, "type": "table"})
        else:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
with st.sidebar:
   
    st.markdown("---")
    st.info("هذه المعلومات غير مؤكده تماما ولا يعتد عليها للتأكد من المعلومات يرجي الرجوع للكلية")