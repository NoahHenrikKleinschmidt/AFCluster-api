from afcluster import AFCluster
from afcluster.utils import read_a3m
from pathlib import Path

parent = Path(__file__).parent
input_file = Path(__file__).parents[2] / "tests/test.a3m"

input_data = read_a3m(input_file)
clusterer = AFCluster()
eps = clusterer.gridsearch_eps(
    input_data,
    min_samples=10,
)
print(f"Optimal eps: {eps}")
df = clusterer.cluster(
    input_data,
    eps=eps,
    resample=False,
    consensus_sequence=True,
    levenshtein=True,
    min_samples=10,
)
outdir = parent / "afcluster_search_eps"
clusterer.write_a3m(outdir)
clusterer.write_cluster_table(outdir / "cluster_table.csv")
pca = clusterer.pca()
pca.figure.savefig(outdir / "pca.png")
pca.figure.clf()
