import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from PIL import Image


st.set_page_config(layout="wide")

def process_sentences(sentences):
       stop_words = set(stopwords.words('english'))
       word_tokens = nltk.word_tokenize(sentences)
       tokenized_sentence = [w for w in word_tokens if not w.lower() in stop_words]
       remove_punctuation = [word for word in tokenized_sentence if word.isalnum()]
       cleaned_text = ''
       for word in remove_punctuation:
              cleaned_text = cleaned_text +' ' + word 
       return cleaned_text

head_image = Image.open('images/AromasImage03.png')

@st.experimental_memo
def load_data():
       main_df = pd.read_csv('data/table_10k')
       df_des = (pd.read_csv('models/final_description_matrix_fp16').values).astype('float32')
       df_non_des = pd.read_csv('models/non_description_matrix').values
       return main_df, df_des, df_non_des

[main_df, df_des, df_non_des] = load_data()

rename_cols = {'country':'Country','variety':'Variety', 'winery':'Winery', 'points':'Points', 
                     'price':'Price($)', 'designation':'Designation','description':'Description'}
main_df.rename(columns=rename_cols, inplace=True)

st.title('Hello!')
st.image(head_image, width=900)
with st.sidebar:
       st.write('Select the country, points, price and province of the wine you are looking for:')
       country =st.selectbox("Select country :", ['any_country', 'Argentina',
              'Australia', 'Austria', 'Chile', 'France', 'Italy', 'Other_country', 'Portugal',
              'Spain', 'US'])
       points = st.selectbox("Select points :", ['any_points', '79-85', '85-90', '90-95', '95-100'])
       price = st.selectbox("Select price :", ['any_price', '0-10', '10-20',
              '20-30', '30-60', '>60'])
       province = st.selectbox("Select province :", ['any_province', 'Bordeaux', 'California', 'Mendoza Province',
              'Northeastern Italy', 'Northern Spain', 'Oregon', 'Other_province', 'Piedmont',
              'Sicily & Sardinia', 'Tuscany', 'Washington'])

input_values = ['any_country', 'any_points', 'any_price', 'any_province', country, points, price, province]
input_dict = {'any_country':0, 'any_points':1, 'any_price':2, 'any_province':3, 'Argentina':4,
       'Australia':5, 'Austria':6, 'Chile':7, 'France':8, 'Italy':9, 'Other_country':10, 'Portugal':11,
       'Spain':12, 'US':13, '79-85':14, '85-90':15, '90-95':16, '95-100':17, '0-10':18, '10-20':19,
       '20-30':20, '30-60':21, '>60':22, 'Bordeaux':23, 'California':25, 'Mendoza Province':25,
       'Northeastern Italy':26, 'Northern Spain':27, 'Oregon':28, 'Other_province':29, 'Piedmont':30,
       'Sicily & Sardinia':31, 'Tuscany':32, 'Washington':33}

input_array = np.zeros(len(input_dict))
for item in input_values:
       index = input_dict[item]
       input_array[index] = 1
st.write('Write a description of the flavours you want : (e.g. Smokey, oaky, citrus, earthy, black pepper...)')
user_input = st.text_area('', )
button_pressed = st.button('Give me wine recommendations')

if button_pressed:
       model = SentenceTransformer('distilbert-base-nli-mean-tokens')
       out = np.dot(df_non_des, input_array)
       max_ids = np.argwhere(out == np.max(out))
       
       processed_input = model.encode(process_sentences(user_input))
       #dotted_vec = np.dot(des_vec, processed_input)
       dotted_vec = util.cos_sim(df_des, processed_input)
       best_index_from_description = np.array(dotted_vec).T[0].argsort()[-60:][::-1]   #get top matches
       recommendation = pd.DataFrame(columns = ['Country','Variety', 'Winery', 'Points', 'Price($)', 'Designation','Description'])
       intersection = np.intersect1d(max_ids, best_index_from_description)
       useful_features_from_main_df = ['Country','Variety', 'Winery', 'Points', 'Price($)', 'Designation','Description']
       useful_features = ['country','variety', 'winery', 'points', 'price', 'designation','description']

       if len(intersection) == 0:
              id = best_index_from_description[0:3]
              st.write('Umm... We do not have any exact matches... But here are some which we think you may like')
              recommendation = main_df.loc[id, useful_features_from_main_df]
              st.write(recommendation)
       else:
              recommendation = main_df.loc[intersection[0:5], useful_features_from_main_df]
              st.write("Here are our recommendations:")
              st.write(recommendation)

