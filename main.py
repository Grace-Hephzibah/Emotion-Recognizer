import streamlit as st
from model_code import Emotion

# Setting some default stuffs
st.set_page_config(layout="wide")
st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)

# Code logic starts here
emoji_code = {'sadness': '😟', 'anger': '😡', 'love': '💖', 'surprise': '😮', 'fear': '😱', 'joy': '🥳'}

st.title("Emotion Recognizer ")

e = Emotion()
c1, c2, c3 = st.columns([1, 3, 1])

with c1:
    st.write("Emoji Codes : ", emoji_code)

with c2:
    sentence = st.text_area("Enter the sentence here ⬇️",
                            value="I am happy that you are here !",
                            height=10
                            )

with c3:
    ans = e.predict_emotion(sentence)
    st.write("Your Sentence Emotion: ")
    for key in ans:
        val = ans[key]
        prob = round(key, 2)
        show = str(emoji_code[val]) + "\t : \t" + str(prob)

        st.markdown('<p class="big-font">' + show + '</p>', unsafe_allow_html=True)
