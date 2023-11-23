import streamlit as st

from modules.detector import detect
from modules.plotly import plot_heatmap
from modules.utils import plag_percentage

st.set_page_config(
    page_title='WSD-based Plagarism Detection',
    page_icon='🔎',
    layout="wide"
)

st.title('WSD-based Plagarism Detection')

# set ngrams
n = 4

with st.form("text_area"):
    col1, col2 = st.columns(2)
    train_text = col1.text_area("Original text")
    test_text = col2.text_area("Text to be checked")

    submitted = st.form_submit_button("Check plagarism")
    if submitted:
        scores, testing_data = detect(train_text, test_text, n)
        st.divider()
        st.subheader(str(plag_percentage(scores)) +
                     "% plagarism w.r.t original text")
        st.plotly_chart(plot_heatmap(scores, testing_data, n),
                        use_container_width=True)
