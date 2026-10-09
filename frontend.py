import streamlit as st
import requests
import os

st.set_page_config(page_title="MirAI Policy Advisor", layout="wide")
st.title("Autonomous MirAI Student Policy Advisor")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Sidebar for controls and status
with st.sidebar:
    st.header("Admin Controls")
    
    st.subheader("System Health")
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success("Backend: Online")
            if health_data.get("policy_handbook_indexed"):
                st.info("Status: Handbook Indexed")
            else:
                st.warning("Status: Handbook NOT Indexed")
            st.text(f"Model: {health_data.get('model')}")
        else:
            st.error("Backend: Error")
    except Exception:
        st.error("Backend: Unreachable")
        
    st.subheader("Ingest Handbook")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("Ingest File"):
        if uploaded_file is not None:
            with st.spinner("Ingesting document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Ingested! Chunks: {data['chunk_count']}")
                    else:
                        st.error(f"Error: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"Failed to ingest: {str(e)}")
        else:
            st.warning("Please upload a file first.")
            
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Excerpts"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}** (Page {src['page']})")
                    st.markdown(f"> {src['content']}")

if prompt := st.chat_input("Ask a policy question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching policy handbook..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/chat", json={"question": prompt}, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("View Retrieved Excerpts"):
                            for src in sources:
                                st.markdown(f"**{src['source']}** (Page {src['page']})")
                                st.markdown(f"> {src['content']}")
                                
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    try:
                        error_detail = resp.json().get("detail", resp.text)
                    except:
                        error_detail = resp.text
                    error_msg = f"Backend error: {error_detail}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"Connection failed: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
