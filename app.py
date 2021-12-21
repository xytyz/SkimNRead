import tensorflow as tf
import spacy
import PyPDF2

from flask import Flask,render_template,request,flash,redirect
nlp = spacy.load("en_core_web_sm")

def split_chars(text):
    return " ".join(list(text))
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def main():
  return render_template('index.html')

@app.route('/submit', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        print(request.files)
        #check if the post request has the file part
        if 'myFile' not in request.files:
            print('no file')
            return 'no file given'
    
        abstract = request.files['myFile']
        file_path = './' + abstract.filename
        abstract.save(file_path)
        pdfFileObject = open(file_path, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
        count = pdfReader.numPages
        output = " "
        for i in range(count):
            page = pdfReader.getPage(i)
            output+= page.extractText()

    doc = nlp(output)
    abstract_lines = [str(sent) for sent in list(doc.sents)]
    print(abstract_lines)

    # Get total number of lines
    total_lines_in_sample = len(abstract_lines)

    # Go through each line in abstract and create a list of dictionaries containing features for each line
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)
    #print(sample_lines)
    # Get all line_number values from sample abstract
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15) 

    # Get all total_lines values from sample abstract
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)
    #print(test_abstract_total_lines_one_hot)

    #Split abstract lines into characters
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]
    #print(abstract_chars)

    loaded_model_5 = tf.keras.models.load_model('skimlit_model_5')

    test_abstract_pred_probs = loaded_model_5.predict(x=(test_abstract_line_numbers_one_hot,
                                                    test_abstract_total_lines_one_hot,
                                                    tf.constant(abstract_lines),
                                                    tf.constant(abstract_chars)))
    #print(test_abstract_pred_probs)

    # Turn prediction probabilities into prediction classes
    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
    #print(test_abstract_preds)

    labels = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    # Turn prediction class integers into string class names
    test_abstract_pred_classes = [labels[i] for i in test_abstract_preds]
    #print(test_abstract_pred_classes)

    # Visualize abstract lines and predicted sequence labels
    results = [f"{test_abstract_pred_classes[i]}: {line}" for i, line in enumerate(abstract_lines)]
    OBJECTIVE = []
    METHODS = []
    BACKGROUND = []
    RESULTS = []
    CONCLUSIONS = []
    for line in results:
        if line.startswith('OBJECTIVE'):
            line = line.lstrip('OBJECTIVE')
            line = line.lstrip('\\n')
            OBJECTIVE.append(line)
        elif line.startswith('METHODS'):
            line = line.lstrip('METHODS')
            line = line.lstrip('\\n')
            METHODS.append(line)
        elif line.startswith('BACKGROUND'):
            line = line.lstrip('BACKGROUND')
            line = line.lstrip('\\n')
            BACKGROUND.append(line)
        elif line.startswith('RESULTS'):
            line = line.lstrip('RESULTS')
            line = line.lstrip('\\n')
            RESULTS.append(line) 
        elif line.startswith('CONCLUSIONS'):
            line = line.lstrip('CONCLUSIONS')
            line = line.lstrip('\\n')
            CONCLUSIONS.append(line)
    results = str([(f'Objective: {OBJECTIVE}'),
    (f'Methods: {METHODS}'),
    (f'Background: {BACKGROUND}'),
    (f'Results: {RESULTS}'),
    (f'Conclusions: {CONCLUSIONS}')])
     
    return render_template('index.html', result = results)
    

if __name__ =='__main__':
  app.run(debug=False)
