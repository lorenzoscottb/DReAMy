

def reduce_space(embeddings ,method="pca", n_components=2):
	
	if method == "pca":

		from sklearn.decomposition import PCA

		pca = PCA(n_components=n_components)
		pca.fit(embeddings)
		X, Y = zip(*pca.transform(embeddings))

	elif method == "t-sne":

		from sklearn.manifold import TSNE

		X_embedded = TSNE(n_components=n_components,
		                  init='random', perplexity=3
		).fit_transform(embeddings)
		X, Y = zip(*X_embedded)

	else:
		
		print("Sorry, {} method is not yet available")
		return None

	return X, Y


def get_KMeans(embeddings, n_clusters, random_state=31):

	from sklearn.cluster import KMeans

	kmeans = KMeans(n_clusters, random_state=random_state).fit(embeddings)
	kmen_labels = kmeans.labels_

	return kmen_labels

    