import plotly.express as px
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import streamlit as st
import speech_recognition as sr
# Token Initialization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Model Initialization
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model =model.to(device)


# Load the tokenizer and model from the saved directory
model_name ="saved_model"
Bert_Tokenizer = BertTokenizer.from_pretrained(model_name)
Bert_Model = BertForSequenceClassification.from_pretrained(model_name).to(device)


def predict_user_input(input_text, model=Bert_Model, tokenizer=Bert_Tokenizer,device=device):
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
            predictions = torch.sigmoid(logits)

    predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    return predicted_labels[0].tolist()







def text_input():
    return st.text_input("Enter Text:", "")

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





input_type = st.radio("Select Input Type:", ["Text", "Speech"])

if input_type == "Text":
    text = text_input()
elif input_type == "Speech":
    st.write("Recording...")
    text = record_and_transcribe()
    st.write(text)




# Load the pre-trained model
# text = st.text_input('Enter text')

if text:
    # Make predictions
    predictions = predict_user_input(text)
    predictions.append(0)
    st.write("My predictions says the text is:")
    if(predictions[0]==1):
        st.write("Toxic")
    if(predictions[1]==1):
        st.write("Severe_toxic")
    if(predictions[2]==1):
        st.write("obscene")
    if(predictions[3]==1):
        st.write("threat")
    if(predictions[4]==1):
        st.write("insult")
    if(predictions[5]==1):
        st.write("identity_hate")
    if(predictions.count(predictions[0]) == len(predictions) and len(predictions)>0):
        st.write("It is a Good Comment")
        predictions[-1]=1
    else:
        predictions[-1]=0


    val=["toxic","severe_toxic","obscene","threat","insult","identity_hate","Good Comments"]
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
