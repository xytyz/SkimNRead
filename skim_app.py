import tensorflow as tf
import spacy
import PyPDF2
import os

import pdfkit


from flask import Flask,render_template,request, make_response, jsonify


OBJECTIVE=[]
METHODS = []
BACKGROUND = []
RESULTS = []
CONCLUSIONS = []
output=''
file_path=''

model = tf.keras.models.load_model("skimlit_model_5")

nlp = spacy.load("en_core_web_sm")

def split_chars(text):
    return " ".join(list(text))
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def main():
  return render_template('index.html')

@app.route('/aboutus')
def about_us():
    return render_template('aboutus.html')

@app.route('/submit2', methods=['GET','POST'])
def load_page():
    global file_path
    global output
    output=''
    filepath=''
    if request.method == 'POST':
        if 'myfile'  in request.files:
            abstract = request.files['myfile']
            file_path = 'database/' + abstract.filename
            abstract.save(file_path)
        else:
            output = request.form.get("paragraph_text")
    return render_template('redirect.html')

@app.route('/submit',methods=['GET','POST'])
def index():
    global output
    if output:
        output=output
    else:
        pdfFileObject = open(file_path, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
        count = pdfReader.numPages
        output = " "
        for i in range(count):
            page = pdfReader.getPage(i)
            output+= page.extractText()
        output = output.lstrip('\n')
        
    doc = nlp(output)
    abstract_lines = [str(sent) for sent in list(doc.sents)]
    

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

    
    test_abstract_pred_probs = model.predict(x=(test_abstract_line_numbers_one_hot,
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
    global OBJECTIVE
    global METHODS 
    global BACKGROUND
    global RESULTS 
    global CONCLUSIONS 
    OBJECTIVE=[]
    METHODS = []
    BACKGROUND = []
    RESULTS = []
    CONCLUSIONS = []
    for line in results:
        if line.startswith('OBJECTIVE'):
            line = line.lstrip('n\OBJECTIVE')
            OBJECTIVE.append(line)
        elif line.startswith('METHODS'):
            line = line.lstrip('n\METHODS')
            METHODS.append(line)
        elif line.startswith('BACKGROUND'):
            line = line.lstrip('n\BACKGROUND')
            BACKGROUND.append(line)
        elif line.startswith('RESULTS'):
            line = line.lstrip('n\RESULTS')
            RESULTS.append(line)
        elif line.startswith('CONCLUSIONS'):
            line = line.lstrip('n\CONCLUSIONS')
            CONCLUSIONS.append(line)
    # OBJECTIVE = str(OBJECTIVE)
    # BACKGROUND = str(BACKGROUND)
    # RESULTS = str(RESULTS)
    # CONCLUSIONS = str(CONCLUSIONS)
    
    OBJECTIVE = "".join(OBJECTIVE)
    OBJECTIVE = OBJECTIVE.replace(':','')
    
    METHODS = "".join(METHODS)
    METHODS = METHODS.replace(':','')

    CONCLUSIONS = "".join(CONCLUSIONS)
    CONCLUSIONS = CONCLUSIONS.replace(':','')

    RESULTS = "".join(RESULTS)
    RESULTS = RESULTS.replace(':','')

    BACKGROUND = "".join(BACKGROUND)
    BACKGROUND = BACKGROUND.replace(':','')

     
    return jsonify('so slow')

@app.route('/done')
def done_page():    
    global OBJECTIVE
    global METHODS 
    global BACKGROUND
    global RESULTS 
    global CONCLUSIONS  
    return render_template('skimmit.html',
                            objective = OBJECTIVE,
                            methods = METHODS,
                            background=BACKGROUND,
                            results=RESULTS,
                            conclusions=CONCLUSIONS)                     

@app.route('/submit/get_pdf')
def get_pdf():
    config = pdfkit.configuration(wkhtmltopdf='C:\Program Files\wkhtmltopdf/bin\wkhtmltopdf.exe')

    rendered = render_template('pdf_file.html',objective = OBJECTIVE,
                            methods = METHODS,
                            background=BACKGROUND,
                            results=RESULTS,
                            conclusions=CONCLUSIONS)
    pdf = pdfkit.from_string(rendered, False,configuration=config, css='static/cs/style_result.css')
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=output.pdf'
    
    return response
    # return pdfkit.from_url('http://127.0.0.1:5000/submit', 'out.pdf',configuration=config,)


if __name__ =='__main__':
  app.run(debug=True)
