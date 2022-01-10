import numpy as np
from flask import Flask, request, jsonify, render_template
from kg_2_network import wiki_scrape, entity_pairs, refine_ent, draw_kg
import spacy

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/kg',methods=['POST'])
def kg():
    '''
    For rendering results on HTML GUI
    '''
    query = request.form.values()
    wiki_data = wiki_scrape(query)

    output_1 = wiki_data.shape[0]

    entities_total = []
    for i in range(50):
        entities_per_iteration = entity_pairs(wiki_data.loc[i,'text'])
        entities_total.append(entities_per_iteration)
    entities_total_df = pd.concat(entities_total)

    output_2 = entities_total_df.sample(15)

    return render_template('index.html',
    dataframe_sample='Number of pages scraped $ {}, some extracted entities are $ {} '.format(output_1,output_2),
    knowledge_graph = draw_kg(entities_total_df))


if __name__ == "__main__":
    app.run(debug=True)
