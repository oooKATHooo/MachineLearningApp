import streamlit as st
import sklearn.datasets as ds

from pathlib import Path
from ml_pipeline import MLPipeline
from streamlit_app import title_area


st.set_page_config(page_title="ML App",
                   page_icon='magic_wand',
                   layout="centered",
                   initial_sidebar_state='expanded'
                   )

state = st.session_state

@st.cache_data
def get_info(set:str):
    if set == "Breast_cancer":
        file = "info_cancer_set.md"
    elif set == "Iris":
        file = "info_iris_set.md"
    elif set == "Digits":
        file = "info_digits_set.md"
    with open(Path(__file__).parent.parent / "Dataset_Infos" / file, mode ="r") as f:
            content = f.read()
    return content


@st.cache_data
def get_df(set:str):
    if set == "Iris":
        data = ds.load_iris(as_frame=True)
    elif set == "Breast_cancer":
        data = ds.load_breast_cancer(as_frame=True)
    else:        # "Digits"
        data = ds.load_digits(as_frame=True)
    return data.frame


def show_info_page():
    with st.sidebar:
        st.selectbox("choose a dataset:",
                        MLPipeline.Datasets,
                        index=MLPipeline.Datasets.index(state.data_choice),
                        key="set")


    title_area("Machine Learning Application")
    st.markdown(f"### Selected Dataset: {state.set}.")

    # info area:
    try:
        info = get_info(state.set)
    except:
        info = "Info not available."
    st.container(height=200).markdown(info)

    # dataframe:
    st.dataframe(get_df(state.set))
    

show_info_page()
