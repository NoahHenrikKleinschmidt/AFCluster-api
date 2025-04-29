from pathlib import Path

test_data = Path(__file__).parent / "test.a3m"
seqs = []
with open(test_data, "r") as f:
    for line in f:
        if line.startswith(">"):
            continue
        seqs.append(line.strip())


def test_can_cluster_simple():
    from afcluster import AFCluster

    clusterer = AFCluster()
    df = clusterer.cluster(seqs, eps=5, resample=True, resample_frac=0.2)
    assert df is not None
    assert "cluster_id" in df.columns
    assert "consensus_sequence" in df.columns
    assert "levenshtein_query" in df.columns
    assert "levenshtein_consensus" in df.columns


def test_can_cluster_call_impl():
    from afcluster import AFCluster

    clusterer = AFCluster()
    df = clusterer(
        seqs,
        eps=5,
        resample=True,
        resample_frac=0.2,
        consensus_sequence=False,
        levenshtein=False,
    )
    assert df is not None
    assert "cluster_id" in df.columns
    assert not "consensus_sequence" in df.columns
    assert not "levenshtein_query" in df.columns
    assert not "levenshtein_consensus" in df.columns


def test_can_gridsearch_eps_simple():
    from afcluster import AFCluster

    clusterer = AFCluster()
    eps = clusterer.gridsearch_eps(
        seqs,
        min_eps=3,
        max_eps=5,
        step=1,
    )
    assert eps is not None
    assert clusterer._eps == eps


def test_can_gridsearch_eps_multiprocess():
    from afcluster import AFCluster

    clusterer = AFCluster()
    eps = clusterer.gridsearch_eps(
        seqs,
        min_eps=3,
        max_eps=5,
        step=1,
        n_processes=3,
    )
    assert eps is not None
    assert clusterer._eps == eps


def test_can_use_searched_eps_for_cluster():
    from afcluster import AFCluster

    clusterer = AFCluster()
    eps = clusterer.gridsearch_eps(
        seqs,
        min_eps=3,
        max_eps=20,
        step=0.5,
        n_processes=5,
    )
    assert eps is not None
    assert clusterer._eps == eps
    df = clusterer.cluster(
        seqs,
        resample=True,
        resample_frac=0.2,
        consensus_sequence=True,
        levenshtein=True,
    )
    assert df is not None
    assert "cluster_id" in df.columns
    assert "consensus_sequence" in df.columns
    assert "levenshtein_query" in df.columns
    assert "levenshtein_consensus" in df.columns


def test_plot_pca():
    from afcluster import AFCluster
    from afcluster.visuals import pca

    clusterer = AFCluster()
    df = clusterer.cluster(
        seqs,
        eps=6,
        resample=False,
        consensus_sequence=True,
        levenshtein=True,
        min_samples=10,
    )
    assert df is not None

    ax = pca(clusterer, inplace=False)
    assert ax is not None
    import matplotlib.pyplot as plt

    plt.gcf().savefig("test_pca.png")
    plt.close()


def test_plot_tsne():
    from afcluster import AFCluster
    from afcluster.visuals import tsne

    clusterer = AFCluster()
    df = clusterer.cluster(
        seqs,
        eps=7,
        resample=False,
        consensus_sequence=True,
        levenshtein=True,
        min_samples=10,
    )
    assert df is not None

    ax = tsne(clusterer, inplace=False)
    assert ax is not None
    import matplotlib.pyplot as plt

    plt.gcf().savefig("test_tsne.png")
    plt.close()


def test_write_cluster_table():
    from afcluster import AFCluster
    from afcluster.utils import write_cluster_table

    clusterer = AFCluster()
    df = clusterer.cluster(
        seqs,
        eps=8,
        resample=False,
        consensus_sequence=True,
        levenshtein=True,
        min_samples=10,
    )
    assert df is not None

    write_cluster_table(clusterer, "test_cluster_table.csv")
    assert Path("test_cluster_table.csv").exists()


def test_write_clusters_to_a3m():
    from afcluster import AFCluster
    from afcluster.utils import write_clusters_to_a3m

    clusterer = AFCluster()
    df = clusterer.cluster(
        seqs,
        eps=8,
        resample=False,
        consensus_sequence=True,
        levenshtein=True,
        min_samples=10,
    )
    assert df is not None

    write_clusters_to_a3m(clusterer, "test_clusters")
    assert Path("test_clusters").exists()


def test_output_list():
    from afcluster import AFCluster

    clusterer = AFCluster()
    df = clusterer.cluster(
        seqs,
        eps=8,
        resample=False,
        consensus_sequence=True,
        levenshtein=True,
        min_samples=10,
        return_type="list",
    )
    assert df is not None
    assert isinstance(df, list)


def test_clusterer_method_api():
    from afcluster import AFCluster

    clusterer = AFCluster()
    df = clusterer.cluster(
        seqs,
        eps=8,
        resample=False,
        consensus_sequence=True,
        levenshtein=True,
        min_samples=10,
    )

    ax = clusterer.pca()
    assert ax is not None

    clusterer.write_a3m("test_clusters")
    assert Path("test_clusters").exists()


def test_from_a3m_df():
    from afcluster import AFCluster
    from afcluster.utils import read_a3m

    clusterer = AFCluster()
    df = read_a3m(test_data)
    assert df is not None
    assert "sequence" in df.columns
    assert "header" in df.columns

    df = clusterer.cluster(
        df,
        eps=8,
        resample=False,
        consensus_sequence=True,
        levenshtein=True,
        min_samples=10,
    )
    assert df is not None
    assert "cluster_id" in df.columns
    assert "consensus_sequence" in df.columns
    assert "levenshtein_query" in df.columns
    assert "levenshtein_consensus" in df.columns
    assert "header" in df.columns
    assert "sequence" in df.columns
