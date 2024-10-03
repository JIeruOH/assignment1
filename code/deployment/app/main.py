import streamlit as st
import requests

size = st.number_input("Apple size(cm)")
weight = st.number_input("Apple weight(g)")

if st.button("Predict juiciness"):
    resp = requests.post("http://127.0.0.1:8000/", json={"size": size, 'weight': weight})
    st.write(resp.json()['juiciness'])
