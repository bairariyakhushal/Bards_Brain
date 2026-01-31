import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

## Load model & tokenizer 
model=load_model('Bards-Brain.h5')

with open('tokenizer.pickle','rb') as file:
    tokenizer=pickle.load(file)
    
    
# Function to predict the next word
def predict_next_word(mdl,tkn,text,max_seq_len):
    # Convert to lowercase to match training data
    text = text.lower()
    
    token_list=tkn.texts_to_sequences([text])[0]
   
    if len(token_list) >= max_seq_len :
        token_list=token_list[-(max_seq_len-1):]# Ensure the sequence length matches max_seq_len-1
    
    token_list=pad_sequences([token_list],padding='pre',maxlen=max_seq_len-1)
    
    predicted=mdl.predict(token_list,verbose=0)
    
    predicted_word_index=np.argmax(predicted,axis=1)
    
    for word,index in tkn.word_index.items() :
        if index==predicted_word_index[0] :
            return word
    
    return None


st.title('Next Word Prediction with LSTM')
st.write('This model is trained on Shakespeare\'s Hamlet. Use words from the play for best results.')

input_text=st.text_input('Enter sentence', placeholder='e.g., to be or not to')

if st.button('Predict Next Word') :
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.success(f'Next Word: **{next_word}**')
        
 