
#Divide-and-Conquer NLG from Structured Data

Code for decomposing a large domain for natural language generation into a subset of smaller sub-domains based on the similarity of knowledge graphs.

Compares 2 models: a sequence-to-sequence set-up (bi-LSTM with attention) and a Mem2Seq setup (see https://github.com/HLTCHKUST/Mem2Seq).

To generate clusters, you need to get the knowledge graph first and the target outputs, see here for download:

https://universityofhull.box.com/s/l1yo755g3cq1ohczeb0e4aj8a5e8g950

With this, "embeddings" has methods for embeddings the knowledge graph.

generate_clusters.py generates a set of n clusters.

generate_hierarchical_clusters.py generates clusters across 2 levels of a hierarchy.


For further documentation, please see publications [HERE].




