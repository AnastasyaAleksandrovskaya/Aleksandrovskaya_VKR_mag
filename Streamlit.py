import numpy as np
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import warnings
import random
import math
import seaborn as sns; sns.set()
import matplotlib as mpl
import streamlit as st
from PIL import Image

from navec import Navec
from slovnet import NER

from natasha import NamesExtractor

from spacy import displacy
import spacy
import stanza
#from spacy_stanza import StanzaLanguage
import datetime
import os


navec = Navec.load('./navec_news_v1_1B_250K_300d_100q.tar')
ner = NER.load('./slovnet_ner_news_v1.tar')
ner.navec(navec)

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

todo_selectbox = st.sidebar.selectbox("",("Результаты", "Распознать именованные сущности в тексте"))

if todo_selectbox == "Результаты":
    st.markdown('<p class="big-font">Результаты</p>', unsafe_allow_html=True)
    data21 = pd.read_csv('results.csv')
    st.write(data21)
    st.markdown('В таблице присутствуют следующие данные:')
    st.markdown('1) Дата публикации новости')
    st.markdown('2) Время публикации новости')
    st.markdown('3) Заголовок новости')
    st.markdown('4) Распознанные именованные сущности из заголовка новости, такие как: ')
    st.markdown('1. дата')
    st.markdown('2. денежные единицы')
    st.markdown('3. персонаж')
    st.markdown('4. локация')
    st.markdown('5. организация')


    st.markdown('<p class="big-font">Парсинг</p>', unsafe_allow_html=True)
    st.markdown('Парсинг данных осуществлялся с новостного сайта Вести (https://www.vesti.ru/news). ')
    st.markdown('Были извлечены заголовки новостей, дата и время создания новости')
    image = Image.open('Screenshot_2.png')
    st.image(image, caption='Example Image', use_column_width=True)

elif todo_selectbox == "Распознать именованные сущности в тексте":
    visualize_selectbox = st.sidebar.selectbox(
        "", ("Natasha", "Spacy"))
    if visualize_selectbox == "Natasha":
        st.markdown('<p class="big-font">Распознать именованные сущности в тексте</p>', unsafe_allow_html=True)
        text_input = st.text_area(label='Введите текст', height=100)
        #st.write(text_input)
        if (text_input>''):
            markup = ner(text_input)
            #markup
            persons = [text_input[s.start:s.stop] for s in markup.spans if s.type == 'PER']
            locations = [text_input[s.start:s.stop] for s in markup.spans if s.type == 'LOC']
            organizations = [text_input[s.start:s.stop] for s in markup.spans if s.type == 'ORG']
            st.write('Найденные персоны:')
            st.write(persons)
            st.write('Найденные организации:')
            st.write(organizations)
            st.write('Найденные локации:')
            st.write(locations)
            #persons1 = pd.DataFrame(persons)
            #persons1.to_csv('persons1.csv', index=False)
            #persons1 = pd.read_csv('persons1.csv')
            #st.write(persons1)
    elif visualize_selectbox == "Spacy":
        st.markdown('<p class="big-font">Распознать именованные сущности в тексте</p>', unsafe_allow_html=True)
        text_input = st.text_area(label='Введите текст', height=100)
        # st.write(text_input)
        if (text_input > ''):
            # Загружаем модель для английского языка
            nlp = spacy.load("en_core_web_sm")

            # Создаем поле для ввода текста
            #text = st.text_area("Введите текст:", value="", height=100, max_chars=None)

            # Если текст был введен, обрабатываем его
            if text_input:
                # Используем модель для обработки текста
                doc = nlp(text_input)

                # Отображаем результаты распознавания именованных сущностей
                for ent in doc.ents:
                    st.write(ent.text, ent.label_)

