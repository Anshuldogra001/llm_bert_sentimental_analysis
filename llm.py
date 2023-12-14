import PyPDF2
import pandas as pd
import plotly.express as px
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

import speech_recognition as sr
from transformers import BartForConditionalGeneration, BartTokenizer
import streamlit as st
import base64
st.title("It's Sentiment Analysis!")
# import gdown

# url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vNm9zMTJWOGxobkU'
# output = '20150428_collected_images.tgz'
# gdown.download(url, output, quiet=False)



# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to generate summary and save it to a PDF file
def generate_and_save_summary(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
# Example usage
model_name = "facebook/bart-large-cnn"
model1 = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer1 = BartTokenizer.from_pretrained(model_name)




# Token Initialization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Model Initialization
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model =model.to(device)


# Load the tokenizer and model from the saved directory
# model_name ="D:\Work\llm_bert_streamlt\saved_model"
model_name ="saved_model"
Bert_Tokenizer = BertTokenizer.from_pretrained(model_name)
Bert_Model = BertForSequenceClassification.from_pretrained(model_name).to(device)


def predict_user_input(input_text, model=Bert_Model, tokenizer=Bert_Tokenizer, device=device):
    user_input = [input_text]

    user_encodings = tokenizer(user_input, truncation=True, padding=True, return_tensors="pt")

    user_dataset = TensorDataset(user_encodings['input_ids'], user_encodings['attention_mask'])

    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch in user_loader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy().tolist()[0]



def read_csv(file):
    df = pd.read_csv(file)
    return df

def text_input():
    return st.text_input("Enter Text:", "")


def csv_input(file):
    df = pd.read_csv(file)

    if df.empty:
        st.warning("CSV file is empty.")
        return []

    st.write("Column names in the CSV file:")
    st.write(df.columns)

    text_column_name = st.text_input("Enter the column name containing text:", "")

    if not text_column_name:
        st.warning("Please enter a valid column name.")
        return []

    if text_column_name not in df.columns:
        st.error(f"Column '{text_column_name}' not found in the CSV file.")
        return []

    text_column = df[text_column_name].astype(str).str.cat(sep=' ')
    return text_column



def record_and_transcribe():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=10)

    try:
        transcript = recognizer.recognize_google(audio)
        return transcript
    except sr.UnknownValueError:
        return "Unable to recognize speech"
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"





input_type = st.radio("Select Input Type:", ["Text", "Speech", "PDF", "CSV"])
text=""
if input_type == "Text":
    text = text_input()

elif input_type == "Speech":
    st.write("Recording...")
    text = record_and_transcribe()
    st.write(text)

elif input_type == "PDF":
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
        text = generate_and_save_summary(text, model1, tokenizer1)
        st.write(text)

elif input_type == "CSV":
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if csv_file is not None:
        text = csv_input(csv_file)




if text:
    # Make predictions
    predictions = predict_user_input(text)

    val=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    fig = px.bar(x=val,
                y=predictions,
                color=val,
                color_discrete_sequence=px.colors.qualitative.Dark24_r,
                title='<b>Predictions of Target Labels')

    fig.update_layout(title='Predictions of Target Labels',
                    xaxis_title='Toxicity Labels',
                    yaxis_title='Prediction',
                    template='plotly_dark')

    # Show the bar chart
    st.plotly_chart(fig)

else:
    st.warning("Please enter some text.")






@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_bg = get_img_as_base64("back_img.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img_bg}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}


[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# st.sidebar.header("Configuration")
