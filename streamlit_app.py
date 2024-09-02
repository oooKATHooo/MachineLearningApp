from typing import Callable, Literal, Iterable
from ml_pipeline import MLPipeline
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import csv
import pandas as pd
import numpy as np
import itertools
import os
from pathlib import Path


st.set_page_config(page_title="ML App",
                   page_icon='magic_wand',
                   layout="centered",
                   initial_sidebar_state='expanded'
                   )


# set state:
state = st.session_state
state.color = "MediumAquaMarine"
colormap = sns.light_palette(state.color, as_cmap=True)

if "df_" not in state:
        state.df_ = pd.DataFrame(columns=["dataset","model",
                                        "scaling",
                                        "pca_method","pca_arg",
                                        "columns_after_pca",
                                        "evaluation"])


@st.experimental_fragment()
def select_parameters():
    choice_info = ("The datasets that can be selected here "
                   "are suitable for classification. "
                   "Linear Regression and Clustering are for comparison only. "
                   "They are not ideal for classification problems."
                  )
    # Dataset:
    data_choice = st.selectbox("choose a dataset:",
                    MLPipeline.Datasets,
                    index=0)
    
    # Model:
    model_choice = st.selectbox("Choose the desired model:",
                    ["Classification", "Linear Regression", "Clustering"],
                    index=0,
                    help = choice_info)
    if model_choice == "Classification":
        class_model_choice = st.selectbox("Choose a Classification model:",
                    ["Logistic Regression", "Decission Tree"],
                    index=0)
        model_name = class_model_choice
    else:
        model_name = model_choice
    
    # Preprocessing:
    choose_preprocessing = st.checkbox("Preprocess the data.", value=False)
    if choose_preprocessing:
        choose_scaling = st.selectbox("Choose a scaling method:",
                    ["MinMaxScaler", "StandardScaler"],
                    index=0)
        pca_choice = st.checkbox("Perform PCA", value=False)
        if pca_choice:
            pca_method = st.radio("Perform PCA with",
                    ["Percentage of Variance", "Number of components"],
                    index=0)
            if pca_method == "Percentage of Variance":
                pca_arg = st.slider("Percentage of variance:",
                    min_value=1, max_value=99, step=1, value=75)
                st.write(f"Your choice: {pca_arg}%")
                
            elif pca_method == "Number of components":
                max_values = {"Iris":4, "Digits":64, "Breast_cancer":30}
                pca_arg = st.number_input("Number of components",
                        min_value=1,
                        max_value=max_values[data_choice],
                        value=max_values[data_choice])
                st.write(f"Your choice: {pca_arg} components.")
        else:
            pca_arg = None
            pca_method = None
    else:
        choose_scaling = None
        pca_choice = False
        pca_method = None
        pca_arg = None
    
    # state:
    state["data_choice"] = data_choice
    state["model_name"] = model_name
    state["choose_scaling"] = choose_scaling
    state["pca_method"] = pca_method
    state["pca_arg"] = pca_arg


def run():
    """lead through ML Pipeline Process."""
    sc = state.choose_scaling
    pca = state.pca_method
    arg = state.pca_arg
    name = state.model_name

    mlp = MLPipeline(state["data_choice"])
    mlp.preprocess_data(sc,pca,arg)
    mlp.create_fit_model(name)
    ev = mlp.evaluate()
    
    state.mlp = mlp


def add_to_df_():
    """add results to dataframe if not yet existing."""
    docu = state.mlp.docu.copy()
    if isinstance(docu["evaluation"],Iterable):
        ev = (docu["evaluation"],)
    else:
        ev = ((docu["evaluation"],),)
    docu["evaluation"] = ev
                
    df = pd.DataFrame(docu)
    df = pd.concat([state.df_,df], ignore_index=True)
    state.df_ = df.drop_duplicates()
    return None


def title_area(text:str):
    colorbox = stylable_container(
        key="colorbox",
        css_styles=f"""
            {{   border-radius: 1px;
                 padding-bottom: 1.0rem;
            }}
            h1{{ background-color: {state.color};
                 text-align: center;
                 color: white;
                 text-shadow: 2.5px 2.5px rgb(0, 67, 45);
            }}
            """
        )
    colorbox.markdown('# ' + text)
    return colorbox


def evaluation_area():
    """show summary and confusion matrix."""
    c1,c2 = st.columns([7,4])
    box = c1.container(height=280)
    if state.btn_run:
        docu = state.mlp.docu
        fieldnames = ["dataset","model",
                      "scaling",
                      "pca_method","pca_arg",
                      "columns_after_pca",
                      "evaluation",
                      ]
        
        with box:
            # summary:
            for el in fieldnames[0:-1]:
                st.text(el + f" : {docu[el]}")
            st.text("evaluation" + f" : {docu["evaluation"]}",
                help=state.mlp.ev_info)
        
        with c2:
            # confusion matrix:
            st.write("**Confusion Matrix:**")
            labels = {"Iris" : [0,1,2],
                      "Digits" : range(10),
                      "Breast_cancer" : [0,1]}
            
            if  state.model_name in ["Logistic Regression", "Decission Tree"]:
                pred = state.mlp.y_pred
                test = state.mlp.y_test
            else:
                pred = np.array(np.round(state.mlp.y_pred,0).astype(int))
                test = np.array(state.mlp.y_test)
            conf_matrix = confusion_matrix(
                    test,
                    pred,
                    labels=labels[state.mlp.name])

            figure = plt.figure()
            sns.heatmap(conf_matrix, annot=True,
                        xticklabels=state.mlp.target_names,
                        yticklabels=state.mlp.target_names,
                        cmap=colormap)
            plt.xlabel("prediction")
            plt.ylabel("test")
            st.pyplot(figure)


@st.experimental_fragment()
def retain_area():
    box = st.container(height=40, border=False)
    if state.btn_run:
        col1, col2, col3 = box.columns([7,7,8])
        with col1:
            styled_button("**Retain Results**",
                        on_click=add_to_df_,
                        use_container_width=True)

        with col2:
            styled_button("**Download Model**",
                        on_click="download",
                        use_container_width=True)
    
    st.dataframe(state.df_,hide_index=True,use_container_width=True)


def styled_button(label:str,
                  on_click:Callable|Literal["download"]|None=None,
                  args:tuple|None=None,
                  kwargs:dict|None=None,
                  key:str|int|None=None,
                  use_container_width:bool=False):
    box = stylable_container(
        key="btn_",
        css_styles=f"""
            button {{background-color: {state.color};
                    color: white; }}
            """
        )
    if on_click == "download":
        btn = box.download_button(label,
                        data=pickle.dumps(state.mlp.model),
                        file_name="model.pkl",
                        use_container_width=use_container_width)
    else:
        btn = box.button(label,
                         on_click=on_click,
                         args=args,
                         kwargs=kwargs,
                         key=key,
                         use_container_width=use_container_width)
    return btn

def make_interface():
    with st.sidebar:
        select_parameters()
        state.btn_run = styled_button("**Run ML Process**",
                                on_click=run,
                                use_container_width=True,
                                key="run")
    
    title_area("Machine Learning Application")
    evaluation_area()
    retain_area()
    print("finished app_script\n"+ ("="*50),"\n\n")

    


make_interface()
