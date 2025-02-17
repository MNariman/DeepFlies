import gzip
import pandas as pd
from GraphProcessor import *
from NeuralModels import *

with gzip.open("traced-roi-connections.csv.gz", "rt") as f:
    df = pd.read_csv(f)


G=nx.from_pandas_edgelist(df, source='bodyId_pre', target='bodyId_post',create_using=nx.DiGraph())
nx.is_directed_acyclic_graph(G) # expecting No.


# compute input neurons
import re
neurons=pd.read_csv('traced-neurons.csv')
# 1) Define keywords
potential_input_keywords = [
    # Olfactory
    r"\bPN\b", "vPN", "mPN", "lPN", r"\bORN\b", r"\bAL\b", r"\bLN\b", "LHN",
    # Visual
    r"\bME\b", r"\bLO\b", "LOP", "AME", r"\bTm\b", r"\bT4\b", r"\bT5\b", r"\bMi\b", "TmY", r"\bMt\b",
    # Mechanosensory
    "AMMC",
    # Early visual integration
    "AOTU", "WED", "AVLP", "PVLP", "PLP"
]
pattern = "|".join(potential_input_keywords)

# Assuming 'neurons' is a pandas DataFrame with columns 'type' and 'instance'

# 2) Subset to those that mention it in 'type' or 'instance'
input_like = neurons[
    neurons['type'].str.contains(pattern, flags=re.IGNORECASE, regex=True) |
    neurons['instance'].str.contains(pattern, flags=re.IGNORECASE, regex=True)
]

# 3) Exclude known "deep" areas if desired
exclude_kw = ["FB", "EB", "MB", "SMP", "SIP", "SLP", "NO", "PB"]
exclude_pat = "|".join(exclude_kw)

robust_input_guesses = input_like[
    ~input_like['type'].str.contains(exclude_pat, flags=re.IGNORECASE, regex=True) &
    ~input_like['instance'].str.contains(exclude_pat, flags=re.IGNORECASE, regex=True)
]

# 'robust_input_guesses' is your final filtered DataFrame


# Get the input nodes that are actually in the graph G
input_nodes = set(robust_input_guesses['bodyId']) & set(G.nodes())

# instantiate GraphProcessor
GP=GraphProcessor(G)
GP.process_graph(input_nodes)
fly_brain=EfficientDAGNN(GP.layers,GP.layer_connections,GP.connection_masks)
fly_brain(torch.randn(1,len(GP.layers[0])))
# test the model with benchmark datasets below.
