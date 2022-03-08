# Note. For language Detection i have used fasttext library.
# fastText is a library for learning of word embeddings and text classification created by Facebook's AI Research lab.

import fasttext
from pycountry import languages

def readDataFromFile():

    with open('data.txt') as f:
        lines = f.readlines()
    return lines
def detect_language():

    sentences = readDataFromFile()
    model_path = 'lid.176.bin'
    model = fasttext.load_model(model_path)
    predictions = model.predict(sentences)
    print(predictions)
    val = predictions[0]
    val1 = val[0]
    val2 = val1[0]
    x = val2.split("__")


    lang_name = languages.get(alpha_2=x[2]).name
    print("---------Output--------")
    print("---Language of document is:---- " + lang_name)


if __name__ == "__main__":
    detect_language()
