import html
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(
    subset='test',
    categories=[
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'comp.windows.x'
        ],
    download_if_missing=True
    )
vectorizer = TfidfVectorizer(
    min_df=0.005,
    max_df=0.2,
    binary=False,
    use_idf=True,
    norm='l2',
    stop_words='english'
    )
vectorizer.fit(newsgroups.data)
inv_voc = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = []
for i in range(len(inv_voc)):
    vocabulary.append(inv_voc[i])
out = {
    'vocabulary': vocabulary,
    'data':[]
    }
for d in newsgroups.data:
    vec = vectorizer.transform([d])
    #vec = vec / np.sum(vec)
    out['data'].append({
        'text': html.escape(d),
        'features': [{'idx': int(i), 'value': '{:.3f}'.format(vec[0, i])} for i in vec.indices]
    })
outstr = json.dumps(out)
print('Output size {}'.format(len(outstr)))
with open('data.json', 'w') as f:
    json.dump(out, f)
print('Done')
