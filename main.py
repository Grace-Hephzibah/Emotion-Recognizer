import streamlit as st
from model_code import Emotion

# Page Setups
st.set_page_config(layout="wide")
st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)


# Setting some default stuffs
e = Emotion()
ans =  {0.84251994: 'joy', 0.073605336: 'love', 0.06991012: 'surprise',
        0.007311212: 'fear', 0.006328419: 'sadness', 0.00032493653: 'anger'} # default values
default_sentence = "I am happy that you are here!"

st.title("Emotion Recognizer ")
c1, c2, c3 = st.columns([1, 3, 1])

emoji_code = {'sadness': '😟', 'anger': '😡', 'love': '💖', 'surprise': '😮', 'fear': '😱', 'joy': '🥳'}

# Initialization
if 'key' not in st.session_state:
    st.session_state['key'] = default_sentence

# Code logic starts here
with c1:
    st.write("Emoji Codes : ", emoji_code)

with c2:
    sentence = st.text_area("Enter the sentence here ⬇️",
                            value=default_sentence,
                            height=10,
                            key = "key"
                            )
    state = st.button("Emotion Check")
    if state:
        sentence = st.session_state["key"]
        ans = e.predict_emotion([sentence])
        #print("Ans : ", ans)

with c3:
    st.write("Your Sentence Emotion: ")
    for key in ans:
        val = ans[key]
        prob = round(key, 2)
        show = str(emoji_code[val]) + "\t : \t" + str(prob)

        st.markdown('<p class="big-font">' + show + '</p>', unsafe_allow_html=True)