import json, streamlit as st
from .graphviz_export import render

st.set_page_config(layout="wide")
data_file = st.sidebar.file_uploader("Load analyzer JSON")
if not data_file: st.stop()

data = json.load(data_file)
edges = data.get("edges", [])
dot_path = render(edges, "_tmp.dot")
st.graphviz_chart(open(dot_path).read())
