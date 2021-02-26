from indicators.core.nlp_utils import fit_topic_model
from indicators.two import arxiv_topics, nih_topics

# for module in (arxiv_topics, nih_topics):  # Can append new modules as they arise
for module in (nih_topics,):  # Can append new modules as they arise
    fit_topic_model(module)
