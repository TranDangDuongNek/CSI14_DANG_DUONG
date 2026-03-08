import os
import pandas as pd
from dotenv import load_dotenv
# import google.generativeai as genai
from google import genai
from google.genai import types
import json
import streamlit as st

# ---------------------------------------
# setup api
# ---------------------------------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=google_api_key)


# ---------------------------------------
# load menu
# ---------------------------------------
menu_df = pd.read_csv("menu.csv", index_col=[0])
# ---------------------------------------
# tao model
# ---------------------------------------
MODEL_VERSION = "gemini-2.5-flash"
SYSTEM_INTRO = f"""
                Bạn tên là PhoBot, một trợ lý AI có nhiệm vụ hỗ trợ giải đáp thông tin cho khách hàng đến nhà hàng Viet Cuisine.
                Các chức năng mà bạn hỗ trợ gồm:
                1. Giới thiệu nhà hàng Viet Cuisine: là một nhà hàng thành lập bởi người Việt, ở địa chỉ 329 Scottmouth, Georgia, USA
                2. Giới thiệu menu của nhà hàng, gồm các món: {', '.join(menu_df["name"].to_list())}.
                3. Lịch mở cửa của nhà hàng: từ T2 -> T6 sẽ hoạt động từ 9:30 sáng tới 8:30 tối, T7 + CN hoạt động từ 8:30 sáng tới 10:00 tối. 
                Ngoài các chức năng trên, bạn không hỗ trợ chức năng nào khác. Đối với các câu hỏi ngoài chức năng mà bạn hỗ trợ, trả lời bằng 'Tôi đang không hỗ trợ chức năng này. Xin liên hệ nhân viên nhà hàng qua hotline 318-237-3870 để được trợ giúp.'
                Hãy có thái độ thân thiện và lịch sự khi nói chuyện với khác hàng, vì khách hàng là thượng đế.
                """
# model = client.models.generate_content(model = MODEL_VERSION,
#                         contents="",
#                         config=types.GenerateContentConfig(
#                                 system_instruction=SYSTEM_INTRO
#                                 )
#                         )
# ---------------------------------------
# loda câu nói khi mở đầu LLM
# ---------------------------------------
with open('config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)
    functions = config.get('functions','giới thiệu nhà hàng')
    initial_bot_message = config.get('initial_bot_message','Chào bạn, tôi là BOT ANH BA CHÀ PHÚ. Tôi có thể giúp bạn điều gì?')
    
# ---------------------------------------
# hàm trò chuyện với ANH BA CHÀ PHÚ
# ---------------------------------------
def retaurant_chatbot():
    st.title("PhoBot - Trợ lý ANH BA CHÀ PHÚ")
    st.write("Chào bạn, tôi là PhoBot siêu trí tuệ nhân tạo. Tôi có thể giúp bạn điều gì?")
    st.write("Bạn có thể hỏi tôi về nhà hàng Viet Cuisine hoặc menu của chúng tôi.")
    st.write("Hãy nhập câu hỏi của bạn vào ô dưới đây:")

    # nếu chưa cp1 lịch sử trò chuyện
    if 'conversation_log' not in st.session_state:
        st.session_state.conversation_log = [
            {"role": "assistant", "content": initial_bot_message},
        ]
    
    # nếu đã có lịch sử trò chuyện, hiển thị lịch sử đó

    for message in st.session_state.conversation_log:
        if message['role'] == 'system':
            with st.chat_message(message['role']):
                st.write(message['content'])

    # user input (promt)
    if promt := st.chat_input("Nhập câu hỏi của bạn ở đây..."):
        # hiển thị promt của user
        with st.chat_message("user"):
            st.write(promt)
        #  thêm vào log
        st.session_state.conversation_log.append({"role": "user", "content": promt})
        

        
        # check prompt có đề cập đến meu (kĩ thuật cắt chữ)
        response = ""
        bot_reply = ""
        if "menu" in promt.lower() or "món" in promt.lower():
            bot_reply = '\n\n'.join([f"**{row['name']}** - {row['description']}" for index, row in menu_df.iterrows()])
        else:
            # tạo llm
            response = client.models.generate_content(
                            model = MODEL_VERSION,
                            contents=promt,
                            config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_INTRO
                                    )
                            )
            bot_reply = response.text
        
        # hiển thị câu trả lời của bot
        with st.chat_message("assistant"):
            st.write(bot_reply)
        # thêm câu trả lời của bot vào log
        st.session_state.conversation_log.append({"role": "assistant", "content": bot_reply})

# hiển thị ra màn hình
if __name__ == "__main__":
    retaurant_chatbot()