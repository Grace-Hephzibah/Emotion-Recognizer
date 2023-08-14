import streamlit as st
from helper_classes import Emotion

# Page Setups
st.set_page_config(layout="wide", page_icon = '❓')
st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)
st.title("Emotion Recognizer ")
st.write('''###### Explore My Code Here: https://github.com/Grace-Hephzibah/Emotion-Recognizer/''')
st.write("-----------------")

# Setting some default stuffs
e = Emotion()
ans =  {"joy" : 1, 'sadness': 0, 'anger': 0, 'love': 0, 'surprise': 0, 'fear': 0} # default values


c1, c2, c3 = st.columns([1, 3, 1])

emoji_code = {'sadness': '😟', 'anger': '😡', 'love': '💖', 'surprise': '😮', 'fear': '😱', 'joy': '🥳'}

# # Initialization
# if 'key' not in st.session_state:
#     st.session_state['key'] = default_sentence

# Code logic starts here
with c1:
    st.write("Emoji Codes : ", emoji_code)

with c2:
    sentence = st.text_area("Enter the sentence here ⬇️",
                            # value=default_sentence,
                            height=10,
                            # key = "key"
                            )
    state = st.button("Emotion Check")
    if state:
        # sentence = st.session_state["key"]
        ans = e.predict_emotion(sentence)
        #print("Ans : ", ans)

with c3:
    st.write("Your Sentence Emotion: ")
    for key in ans:
        val = ans[key]
        prob = round(val, 2)
        show = str(emoji_code[key]) + "\t : \t" + str(prob)

        st.markdown('<p class="big-font">' + show + '</p>', unsafe_allow_html=True)

st.write("------------------")
st.subheader("✨ By Grace Hephzibah ✨")
st.write("------------------")