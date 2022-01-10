import wikipediaapi
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import re
import spacy
import networkx as nx
import matplotlib.pyplot as plt
#TODO incorporate neuralcoref

nlp = spacy.load('en_core_web_sm')

def wiki_scrape(topic_name, verbose=True):
    def wiki_link(link):
        try:
            page = wiki_api.page(link)
            if page.exists():
                d = {'page': link, 'text': page.text, 'link': page.fullurl,
                     'categories': list(page.categories.keys())}
                return d
        except:
            return None

    wiki_api = wikipediaapi.Wikipedia(language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI)
    page_name = wiki_api.page(topic_name)
    if not page_name.exists():
        print('page {} does not exist'.format(topic_name))
        return
    page_links = list(page_name.links.keys())
    progress = tqdm(desc='Links Scraped', unit='', total=len(page_links)) if verbose else None
    sources = [{'page': topic_name, 'text': page_name.text, 'link': page_name.fullurl,
                'categories': list(page_name.categories.keys())}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_link = {executor.submit(wiki_link, link): link for link in page_links}
        for future in concurrent.futures.as_completed(future_link):
            data = future.result()
            progress.update(1) if verbose else None
            if data:
                sources.append(data)
    progress.close() if verbose else None
    blacklist = ('Template', 'Help:', 'Category:', 'Portal:', 'Wikipedia:', 'Talk:')
    sources = pd.DataFrame(sources)
    sources = sources[(len(sources['text']) > 20)
                      & ~(sources['page'].str.startswith(blacklist))]
    sources['categories'] = sources.categories.apply(lambda x: [y[9:] for y in x])
    sources['topic'] = topic_name
    print ('Wikipedia pages scraped:', len(sources))
    return sources


#wiki_data.to_csv('wiki_gard.csv',index = False)
#print(wiki_data.head(10))
######


def entity_pairs(text, coref=True):
    text = re.sub(r'\n+', '.', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    text = nlp(text)
    sentences = [sent.string.strip() for sent in text.sents]
    ent_pairs = list()
    for sent in sentences:
        sent = nlp(sent)
        spans = list(sent.ents) + list(sent.noun_chunks)
        spans = spacy.util.filter_spans(spans)
        with sent.retokenize() as retokenizer:
            [retokenizer.merge(span) for span in spans]
        dep = [token.dep_ for token in sent]
        if (dep.count('obj')+dep.count('dobj'))==1 \
                and (dep.count('subj')+dep.count('nsubj'))==1:
            for token in sent:
                if token.dep_ in ('obj', 'dobj'):
                    subject = [w for w in token.head.lefts if w.dep_
                               in ('subj', 'nsubj')]
                    if subject:
                        subject = subject[0]
                        relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
                        if relation:
                            relation = relation[0]
                            if relation.nbor(1).pos_ in ('ADP', 'PART'):
                                relation = ' '.join((str(relation),
                                        str(relation.nbor(1))))
                        else:
                            relation = 'unknown'
                        subject, subject_type = refine_ent(subject, sent)
                        token, object_type = refine_ent(token, sent)
                        ent_pairs.append([str(subject), str(relation), str(token),
                                str(subject_type), str(object_type)])
    filtered_ent_pairs = [sublist for sublist in ent_pairs
                          if not any(str(x) == '' for x in sublist)]
    pairs = pd.DataFrame(filtered_ent_pairs, columns=['subject',
                         'relation', 'object', 'subject_type',
                         'object_type'])
    print('Entity pairs extracted:', str(len(filtered_ent_pairs)))
    return pairs


def refine_ent(ent, sent):
    unwanted_tokens = (
        'PRON',  # pronouns
        'PART',  # particle
        'DET',  # determiner
        'SCONJ',  # subordinating conjunction
        'PUNCT',  # punctuation
        'SYM',  # symbol
        'X',  # other
        )
    ent_type = ent.ent_type_  # get entity type
    if ent_type == '':
        ent_type = 'NOUN_CHUNK'
        ent = ' '.join(str(t.text) for t in
                nlp(str(ent)) if t.pos_
                not in unwanted_tokens and t.is_stop == False)
    elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
        t = ''
        for i in range(len(sent) - ent.i):
            if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                t += ' ' + str(ent.nbor(i))
            else:
                ent = t.strip()
                break
    return ent, ent_type
# only for orignal link
#pairs = entity_pairs(wiki_data.loc[0,'text'])
#print(pairs)

def draw_kg(entities_total_df):
    """ outputs a netowork of nodes connected by edges.  """
    k_graph = nx.from_pandas_edgelist(entities_total_df, 'subject', 'object',
            'relation',create_using=nx.MultiDiGraph())
    node_deg = nx.degree(k_graph)
    layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
    plt.figure(num=None, figsize=(120, 90), dpi=80)
    nx.draw_networkx(
        k_graph,
        node_size=[int(deg[1]) * 500 for deg in node_deg],
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color='red',
        edgecolors='black',
        node_color='yellow',
        )
    labels = dict(zip(list(zip(entities_total_df.subject, entities_total_df.object)),
                  entities_total_df['relation'].tolist()))
    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,
                                 font_color='red')
    plt.axis('off')
    #plt.savefig("Outpatient clinic_1.png")
    plt.show()


def draw_kg_filter(entities_total_df):
    """ filtered network graph. Input a subset of orignal data you want to make the network for """
    k_graph = nx.from_pandas_edgelist(entities_total_df, 'subject', 'object',
            'relation',create_using=nx.MultiDiGraph())
    node_deg = nx.degree(k_graph)
    layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
    plt.figure(num=None, figsize=(20, 12), dpi=80)
    nx.draw_networkx(
        k_graph,
        node_size=[int(deg[1]) * 500 for deg in node_deg],
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color='blue',
        edgecolors='black',
        node_color='green',
        )
    labels = dict(zip(list(zip(entities_total_df.subject, entities_total_df.object)),
                  entities_total_df['relation'].tolist()))
    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,
                                 font_color='red')
    plt.axis('off')
    #plt.savefig("Diabetes_filter.png")
    plt.show()

if __name__ == "__main__":
    query = input("Please enter a query you want to be scraped off wikipedia")
    print("**************")
    print('\n')
    print("Note that all wikipedia page with the query name will be scraped")
    print("**************")
    wiki_data = wiki_scrape(query)
    entities_total = []
    for i in range(50):
        entities_per_iteration = entity_pairs(wiki_data.loc[i,'text'])
        entities_total.append(entities_per_iteration)
    entities_total_df = pd.concat(entities_total)
    # to ensure that the scraped data is of correct dimensions:

    #entities_total_df.shape
    #entities_total_df.head(10)
    #entities_total_df.to_csv("triple_gard.csv", index = False)
    draw_kg(entities_total_df)

    # example use case for the draw_kg_filter method; uncomment to explore further
    # draw_kg_filter(entities_total_df[entities_total_df["subject"]=='symptoms'])
