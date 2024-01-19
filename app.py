from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

def load_model(model_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

model_path = "" # add the path
tokenizer, model = load_model(model_path)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    text = request.form["text"]
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    translated_code = tokenizer.decode(outputs[0])
    return render_template("index.html", translated_code=translated_code)

if __name__ == "__main__":
    app.run(debug=True)
