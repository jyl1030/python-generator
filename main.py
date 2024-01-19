from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def translate_code(code_snippet, model):
    # Tokenize the input code snippet
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    input_ids = tokenizer.encode(code_snippet, return_tensors='pt')

    # Generate the translation
    output = model.generate(input_ids)

    # Decode the output and return the translated code
    translated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    return translated_code

if __name__ == '__main__':
    # Load the pretrained model
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Test the model
    code_snippet = "for i in range(10):\n    print(i)"
    translated_code = translate_code(code_snippet, model)
    print("Original code:\n", code_snippet)
    print("Translated code:\n", translated_code)


