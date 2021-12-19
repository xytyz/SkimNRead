
import PyPDF2
pdfFileObject = open('Error_prob_pcm.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
count = pdfReader.numPages
output = " "
for i in range(count):
    page = pdfReader.getPage(i)
    output+= page.extractText()

from flask import Flask

example_abstract = '''This RCT examined the efficacy of a manualized social intervention for children with HFASDs. Participants were randomly assigned to treatment or wait-list conditions. Treatment included instruction and therapeutic activities targeting social skills, face-emotion recognition, interest expansion, and interpretation of non-literal language. A response-cost program was applied to reduce problem behaviors and foster skills acquisition. Significant treatment effects were found for five of seven primary outcome measures (parent ratings and direct child measures). Secondary measures based on staff ratings (treatment group only) corroborated gains reported by parents. High levels of parent, child and staff satisfaction were reported, along with high levels of treatment fidelity. Standardized effect size estimates were primarily in the medium and large ranges and favored the treatment group.", "source": "https://pubmed.ncbi.nlm.nih.gov/20232240/", "details": "RCT of a manualized social treatment for high-functioning autism spectrum disorders'''

def split_chars(text):
    return " ".join(list(text))
app = Flask(__name__)

@app.route('/')
def index():
    import tensorflow as tf
    import spacy
    nlp = spacy.load("en_core_web_sm")
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
    print(test_abstract_line_numbers_one_hot)

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
    results = str([f"{test_abstract_pred_classes[i]}: {line}" for i, line in enumerate(abstract_lines)])
    return results
    

app.run(host='0.0.0.0', port=81)
