from flask import Flask, render_template, request
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import speech_recognition as sr

recognizer = sr.Recognizer()

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
chat_history_ids = None  # Define chat history globally

@app.route("/")
def index():
    global chat_history_ids
    chat_history_ids = None  # Reset chat history
    welcome_msg = "Hi there! I'm your community assistant. How can I help you today?"
    return render_template('chat.html', welcome_msg=welcome_msg)

@app.route("/get", methods=["POST"])
def chat():
    global chat_history_ids  # Access the global chat history
    user_input = request.form["msg"]

    # Generate response
    response = get_chat_response(user_input)

    return response

def get_chat_response(user_input):
    global chat_history_ids
    # Encode the user input
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Concatenate the user input with chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    
    # Generate response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode and return response
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Conversation flow
    if "hello" in user_input.lower() or "hi" in user_input.lower():
        bot_response = "Hi! I'm your community assistant. How can I help you?"

    elif "apply" in user_input.lower():
        bot_response = "Thank you for interested to Join our community! Drop your resume to careers@iqinternz.com"
    elif "join" in user_input.lower():
        bot_response = "Thank you for interested to Join our community! Drop your resume to careers@iqinternz.com"
    elif "hai" in user_input.lower() or "hi" in user_input.lower():
        bot_response = "Hai! I'm your community assistant. How can I help you?"
    elif "courses" in user_input.lower() or "course" in user_input.lower():
        bot_response = "We offer courses in Machine Learning, Artificial Intelligence, Web Development, Data Analytics, and more."
    elif "company" in user_input.lower() or "about" in user_input.lower():
        bot_response = """At IQInternz, we're committed to bridging the gap between education and professional success. 
        Our mission is to empower students with a comprehensive platform that encompasses training and internship opportunities. 
        Recognizing the challenges students face transitioning from classrooms to the workforce, we equip them with the necessary tools to thrive in today's competitive job market.

        Through our training programs, students acquire the skills and knowledge essential for their chosen fields. 
        Our internship opportunities provide hands-on experience, allowing students to apply their learning in real-world scenarios. 
        At IQInternz, we are dedicated to nurturing tomorrow's talent, assisting students in transforming aspirations into tangible achievements."""
        
    elif "help" in user_input.lower() or "contact" in user_input.lower():
        bot_response = "Contact us at support@iqinternz.com."
    elif "community" in user_input.lower() or "purpose" in user_input.lower() or  "what is the purpose of this community" in user_input.lower():
        bot_response = """The purpose of a community for your IQInternz startup AI company is to foster collaboration and knowledge-sharing among members, driving innovation and growth. 
        It serves as a platform for networking, support, and learning, empowering individuals to leverage collective intelligence to solve complex problems and drive progress in the AI industry."""
    elif "founder" in user_input.lower():
        bot_response = "Ruthvik Sree Ranga S is the Founder and CEO of IQinternz."
    elif "co-ceo" in user_input.lower() or "co founder" in user_input.lower() or "cofounder" in user_input.lower():
        bot_response = "V Chandu Chowdary is the CO-Founder of IQinternz."
    elif "i'm" in user_input.lower() or "iam" in user_input.lower() or "my name is" in user_input.lower() or "myself" in user_input.lower():
    # Extracting the name from the user input
        if "i'm" in user_input.lower():
            name_start_index = user_input.lower().find("i'm") + 4
        elif "iam" in user_input.lower():
            name_start_index = user_input.lower().find("iam") + 3
        elif "my name is" in user_input.lower():
            name_start_index = user_input.lower().find("my name is") + 11
        elif "myself" in user_input.lower():
            name_start_index = user_input.lower().find("myself") + 7
    
        name_end_index = user_input.find(" ", name_start_index)
        if name_end_index == -1:
            name = user_input[name_start_index:].strip()
        else:
            name = user_input[name_start_index:name_end_index].strip()
    
        bot_response = f"Hello {name}, welcome to Internz community assistance."

    elif "head of operations" in user_input.lower():
        bot_response = "Tharun Bisai is the Head of Operations at IQinternz."
    elif "mentor" in user_input.lower():
        bot_response = "G Dinnesh is the Mentor - B.A and HR at IQinternz."
    elif "python syntax" in user_input.lower() or "python" in user_input.lower():
        bot_response = "Python syntax: print('Hello, World!')"
    elif "java syntax" in user_input.lower() or "java" in user_input.lower():
        bot_response = "Java syntax: public static void main(String[] args) { System.out.println('Hello, World!'); }"
    elif "date" in user_input.lower():
        bot_response = f"Today's date is {datetime.date.today()}"
    elif "prime minister" in user_input.lower():
        bot_response = "The current Prime Minister of India Narendra Modi."
    # elif "navigation" in user_input.lower():
    # # Return navigation buttons
    #     bot_response = "Select a topic:"
    #     bot_response += "\n<button onclick=\"get_chat_response('founder')\">Founder</button>"
    #     bot_response += "\n<button onclick=\"get_chat_response('company')\">About Company</button>"
    #     bot_response += "\n<button onclick=\"get_chat_response('co-ceo')\">Co-CEO</button>"
    #     bot_response += "\n<button onclick=\"get_chat_response('community')\">Community</button>"
    #     bot_response += "\n<button onclick=\"get_chat_response('contact')\">Contact Us</button>"
    #     bot_response += "\n<button onclick=\"get_chat_response('mentor')\">Mentor</button>"
    #     bot_response += "\n<button onclick=\"get_chat_response('apply')\">Apply Now</button>"
    #     bot_response += "\n<button onclick=\"get_chat_response('courses')\">Courses</button>"
    #     bot_response += "\n<button onclick=\"get_chat_response('help')\">Help</button>"

    else:
        bot_response = "I'm not sure about that. Please visit our community website https://iqinternz.com/our_team.html"

    return bot_response

if __name__ == '__main__':
    app.run(debug=True)
