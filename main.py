import os
import openai
from tqdm import tqdm
from pprint import pprint
import requests
import json
import streamlit as st

openai_engine = "text-davinci-003"


def normalize_text(text) -> str:
    return " ".join(text.split()) if isinstance(text, str) else text


def get_new_texts(text, openai, params) -> list:
    prompt = "Paraphrase this"
    if params.get('language'):
        prompt += f" in {params['language']}"
    paraphrase_λ = lambda text: openai.Completion.create(
        engine=openai_engine,
        prompt=f"{prompt} : {text}",
        max_tokens=1_024,
        n=params['n'],
        stop=None,
        temperature=params['temp'],
    )
    
    try:
        completion = paraphrase_λ(text)
        new_texts = [normalize_text(choice.text) for choice in completion.choices]
    except:
        new_texts = ""
        
    
    return dict(zip(range(params['n']), new_texts))

def main():
    openai.api_key= st.secrets["api_key"]
    st.header('OpenAI 한국어 paraphrase 페이지')
    
    with st.expander('모델 파라미터', expanded=True):
    
        temperature = st.slider('온도', min_value=0.0, max_value = 2.0, value=0.5, format=None,
        help='''
            temperature number Optional Defaults to 1
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic
            ''')
        n  = st.number_input('생성할 문장 수', min_value=1, max_value=16, value=1, step=1, format=None,
        help='''
            How many completions to generate for each prompt.
            Note: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for max_tokens and stop.
            ''')
        language = st.selectbox('언어', ('Korean', 'English'), index=0)

    params=dict(temp=temperature, n=n, language=language)
    
    prompt = st.text_area('Prompt', height = 200,\
                             placeholder = '문장 입력')

    generate= st.button('생성')
    if generate:
        with st.spinner('생성중'):
            new_texts = get_new_texts(prompt, openai, params)

        st.write(new_texts)


if __name__ == "__main__":
    main()