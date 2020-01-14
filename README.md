
# Code for "A Divide-and-Conquer Approach to Neural Natural Language Generation from Structured Data"

(in submission)


The idea with this paper is that a large domain of text for natural language generation can be decomposed automatically into 
a subset of smaller sub-domains based on the similarity of knowledge graphs. The intuition here is that parts of a knowledge 
graph are similar and therefore can be used to identify semantic slots that often occur together with a specific semantic entity. 

<img src="/img/kgraph.png" alt="drawing" width="600"/>

The sub-domains, or "generation spaces" are based on a notion of similarity of embedded partial knowledge graphs that 
represent a domain and feed into a hierarchy of sequence-to-sequence or memory-to-sequence learners for concept-to-text 
generation. Experiments show that our hierarchical approach overcomes issues of data sparsity and learns robust 
lexico-syntactic patterns, consistently outperforming flat baselines and previous work by up to 30%

A particular focus with this paper is the comparison of the hierarchical setup for sequence-to-sequence and 
memory-to-sequence models, see:

<img src="/img/models.png" alt="drawing" width="700"/>

# Code

Code is given for a sequence-to-sequence bidirectional LSTm with attention, see <code>Seq2Seq</code> folder.
The code for the memory-to-sequence model, see <code>Mem2Seq-master<code> is taken from https://github.com/HLTCHKUST/Mem2Seq. 

To generate clusters (and estimate generation spaces), you need to get the knowledge graph first and the target outputs, see here for download:

https://universityofhull.box.com/s/l1yo755g3cq1ohczeb0e4aj8a5e8g950

With this, <code>embeddings</code> contains methods for embeddings the knowledge graph.

After that, <code>generate_clusters.py</code> generates a set of <em>n</em> clusters on a single level. For a 
hierarchical setup, <code>generate_hierarchical_clusters.py</code> generates clusters across 2 levels of a hierarchy.

This repository is work in progress, more details to follow once published.




