import streamlit as st 
from streamlit_chat import message
from helper import img_to_base64, displayPDF,process_answer,display_conversation

    
def main():
    st.set_page_config(
        page_title="STB-600 Smart Assistant",
        page_icon="images/avatar_streamly.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": """
                ## STB-600 Smart Assistant
                                
                The AI Assistant named, Streamly, aims to provide the latest updates from Streamlit,
                generate code snippets for Streamlit widgets,
                and answer questions about Streamlit's latest features, issues, and more.
                Streamly has been trained on the latest Streamlit updates and documentation.
            """
        }
    )

    # Inject custom CSS for glowing border effect
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load and display sidebar image with glowing effect
    img_path = "images/stb-6000-2.png"
    img_base64 = img_to_base64(img_path)
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    expander = st.expander("STB-6000 User Manual")
    with expander:
        displayPDF("manual/STB-6000.pdf")

    st.sidebar.markdown("""
    ### Basic Interactions
    - **Ask About STB-6000**: Type your questions about STB-6000 setup box's latest updates, features, or issues.
    - This is an AI powered chatbot which try to answer your questions.
    """)

    # Streamlit Updates and Expanders
    st.title("STB-600 Smart Assistant")

    # user_input = st.text_input("",key="input")
    #initialize session state for generted response and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am an AI assitance how can I help?"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]    
    user_input = st.chat_input("Ask me about STB-6000 setup box:")
    image = ""

    # Search the database for a response based on user input and update session state    
    if user_input:
        answer,image = process_answer({'query': user_input})
        st.session_state["past"].append(user_input)
        response = answer
        st.session_state["generated"].append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state,image)

        

if __name__ == "__main__":
    main()