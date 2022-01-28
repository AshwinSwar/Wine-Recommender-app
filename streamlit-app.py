import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import nltk
#from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from PIL import Image

STOP_WORDS = {'m', 'hadn', 'yourself', 'this', 'what', 'from', 'will', 'herself', 
       "aren't", 'too', "that'll", "didn't", 'did', 'how', 'ain', "you'll", 
       'into', 'off', 'than', 'now', 're', 'shan', "shan't", 'where', "you'd", 
       "needn't", 'being', 'so', 'can', 'of', 'isn', 'or', "hadn't", 'then', 
       'he', 'with', 'won', "wasn't", 'wouldn', 'before', 'between', 'which', 
       'very', 'under', "won't", 'hers', "you're", 'it', 'over', 've', 'him', 
       'yourselves', 'was', 'himself', "isn't", 'ours', 'these', 'no',
       'down', 'they', 'about', 'through', 'other', 'don', 'ourselves', 'my', 
       "mustn't", "weren't", 'because', 'i', 'who', 'same', 'just', 'wasn', 
       'not', 'to', 'those', "doesn't", 'hasn', 'be', 'were', 'further', 'y', 
       'if', 'nor', 'am', "wouldn't", 's', 'theirs', 'most', "should've", 'her',
       'only', 'our', 'below', 'haven', 'a', 'when', 'why', 'o', 'more', 'had',
       'are', 'an', 'again', 'some', 'itself', 'mightn', 'been', 'after', 
       "don't", 'didn', 'ma', 'she', 'have', 'against', 'is', 'yours', 'both', 
       'its', 'your', 'doesn', 'his', 'but', 'until', 'do', 'on', 'that', 
       'each', "it's", 'themselves', 'such', 'any', 't', 'couldn', 'the', 
       "she's", 'does', 'their', 'doing', 'and', 'once', 'whom', 'we', 'all', 
       "you've", 'has', 'aren', 'as', 'you', 'few', 'should', 'll', 'shouldn', 
       'there', 'above', 'own', "hasn't", 'at', "haven't", 'mustn', 'them', 
       'for', 'in', 'needn', 'me', "couldn't", 'during', "mightn't", 'weren', 
       'myself', 'here', 'by', 'out', "shouldn't", 'having', 'd',
       'up', 'while'}

st.set_page_config(layout="wide")

@st.experimental_memo
def get_punkt():
       nltk.download('punkt')
get_punkt()

@st.experimental_memo
def load_model():
       return SentenceTransformer('distilbert-base-nli-mean-tokens')

def process_sentences(sentences):
       word_tokens = nltk.word_tokenize(sentences)
       tokenized_sentence = [w for w in word_tokens if not w.lower() in STOP_WORDS]
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

rename_cols = {'country':'Country','variety':'Variety', 'winery':'Winery', 
              'points':'Points', 'price':'Price($)', 'designation':'Designation',
              'description':'Description'}
main_df.rename(columns=rename_cols, inplace=True)

st.title('Hello!')
st.image(head_image, width=900)
with st.sidebar:
       st.write('Select the country, points, price and province of the wine \
              you are looking for:')
       country =st.selectbox("Select country :", ['any_country', 'Argentina',
              'Australia', 'Austria', 'Chile', 'France', 'Italy', \
              'Other_country', 'Portugal','Spain', 'US'])
       points = st.selectbox("Select points :", ['any_points', '79-85', \
              '85-90', '90-95', '95-100'])
       price = st.selectbox("Select price :", ['any_price', '0-10', '10-20',
              '20-30', '30-60', '>60'])
       province = st.selectbox("Select province :", ['any_province', \
              'Bordeaux', 'California', 'Mendoza Province',
              'Northeastern Italy', 'Northern Spain', 'Oregon', \
              'Other_province', 'Piedmont', 'Sicily & Sardinia', 'Tuscany', \
              'Washington'])

user_selections = [country, points, price, province]
input_values = ['any_country', 'any_points', 'any_price', 'any_province'] + user_selections
input_dict = {'any_country':0, 'any_points':1, 'any_price':2, 'any_province':3,
       'Argentina':4, 'Australia':5, 'Austria':6, 'Chile':7, 'France':8, 
       'Italy':9, 'Other_country':10, 'Portugal':11, 'Spain':12, 'US':13, 
       '79-85':14, '85-90':15, '90-95':16, '95-100':17, '0-10':18, '10-20':19,
       '20-30':20, '30-60':21, '>60':22, 'Bordeaux':23, 'California':25, 
       'Mendoza Province':25,'Northeastern Italy':26, 'Northern Spain':27, 
       'Oregon':28, 'Other_province':29, 'Piedmont':30,
       'Sicily & Sardinia':31, 'Tuscany':32, 'Washington':33}

user_selection_input_array = np.zeros(len(input_dict))
for item in input_values:
       index = input_dict[item]
       user_selection_input_array[index] = 1

user_text_input = st.text_area('Write a description of the flavours you want \
                     : (e.g. Smokey, oaky, citrus, earthy, black pepper...)', )
button_pressed = st.button('Give me wine recommendations')
def user_description_is_empty(user_text_input):
       if len(process_sentences(user_text_input)) > 0:
              return False
       return True

def user_selections_are_default(user_selections):
       if (user_selections[0]=='any_country' and \
                            user_selections[1]=='any_points' and \
                            user_selections[2]=='any_price' and \
                            user_selections[3]=='any_province'):
              return True
       return False

useful_features_from_main_df = ['Country','Variety', 'Winery', 'Points', \
                                   'Price($)', 'Designation','Description']
useful_features = ['country','variety', 'winery', 'points', 'price', \
                                   'designation','description']

if button_pressed:
       out = np.dot(df_non_des, user_selection_input_array)
       best_ids_from_nondes_matrix = np.argwhere(out == np.max(out)).T[0]
       if user_description_is_empty(user_text_input):
              if not user_selections_are_default(user_selections):
                     id = best_ids_from_nondes_matrix[0:6]
                     recommendation = main_df.loc[id, useful_features_from_main_df]
                     st.write("Here are our recommendations:")
                     st.write(recommendation)
              else:
                     id = np.random.randint(low=0, high=len(best_ids_from_nondes_matrix), size=6)
                     recommendation = main_df.loc[id, useful_features_from_main_df]
                     st.write("Here are our recommendations:")
                     st.write(recommendation)
       else:
              model = load_model()
              processed_input = model.encode(process_sentences(user_text_input))
              #dotted_vec = np.dot(des_vec, processed_input)
              dotted_vec = util.cos_sim(df_des, processed_input)
              best_ids_from_des_matrix = np.array(dotted_vec).T[0].argsort()[-60:][::-1]   #get top matches
              recommendation = pd.DataFrame(columns = useful_features_from_main_df)
              intersection = np.intersect1d(best_ids_from_nondes_matrix, best_ids_from_des_matrix)

              if len(intersection) == 0:
                     id = best_ids_from_des_matrix[0:3]
                     recommendation = main_df.loc[id, useful_features_from_main_df]
                     recommendation.reset_index(drop=True, inplace=True)
                     st.write('Umm... We do not have any exact matches... But \
                             here are some which we think you may like')
                     st.write(recommendation)
              else:
                     id = intersection[0:5]
                     recommendation = main_df.loc[id, useful_features_from_main_df]
                     recommendation.reset_index(drop=True, inplace=True)
                     st.write("Here are our recommendations:")
                     st.write(recommendation)

