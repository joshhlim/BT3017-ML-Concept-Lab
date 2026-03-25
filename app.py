from flask import Flask, render_template, jsonify, request
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics.pairwise import rbf_kernel
import random

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# ─── Kernel ───────────────────────────────────────────────────────────────────

@app.route("/api/kernel/classify", methods=["POST"])
def kernel_classify():
    data = request.json
    points = np.array(data["points"])
    kernel = data.get("kernel", "rbf")
    C = float(data.get("C", 1.0))
    X = points[:, :2]
    y = points[:, 2].astype(int)
    if len(np.unique(y)) < 2:
        return jsonify({"error": "Need points from both classes"}), 400

    clf = SVC(kernel=kernel, C=C, gamma='scale')
    clf.fit(X, y)

    # Send a flat grid of predicted labels (2.7KB vs 37KB for float grid)
    n = 40
    margin = 0.5
    x_min, x_max = float(X[:,0].min()) - margin, float(X[:,0].max()) + margin
    y_min, y_max = float(X[:,1].min()) - margin, float(X[:,1].max()) + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    labels = clf.predict(np.c_[xx.ravel(), yy.ravel()]).tolist()

    lift3d = None
    if kernel == "rbf":
        gv = 1.0 / (X.shape[1] * max(X.var(), 1e-6))
        r2 = np.sum(X**2, axis=1)
        lift3d = {"points": np.column_stack([X, np.exp(-gv*r2)]).tolist(), "labels": y.tolist()}

    return jsonify({
        "labels": labels, "n": n,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "support_vectors": clf.support_vectors_.tolist(),
        "lift3d": lift3d,
        "accuracy": float(clf.score(X, y))
    })

@app.route("/api/kernel/dataset", methods=["POST"])
def kernel_dataset():
    data = request.json
    dataset = data.get("dataset","circles")
    n = 60
    if dataset=="moons": X,y = make_moons(n_samples=n,noise=0.12,random_state=42)
    elif dataset=="circles": X,y = make_circles(n_samples=n,noise=0.08,factor=0.45,random_state=42)
    elif dataset=="xor":
        rng=np.random.RandomState(42); X=rng.randn(n,2)
        y=((X[:,0]>0)^(X[:,1]>0)).astype(int)
    else:
        X,y=make_blobs(n_samples=n,centers=2,cluster_std=0.6,random_state=42)
        y=(y>0).astype(int)
    return jsonify({"points":np.column_stack([X,y]).tolist()})

# ─── PCA ──────────────────────────────────────────────────────────────────────

@app.route("/api/pca/compute", methods=["POST"])
def pca_compute():
    data = request.json
    points = np.array(data["points"])
    n_components = int(data.get("n_components", 2))
    n_features = points.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(points)
    pca_full = PCA(); pca_full.fit(X_scaled)
    explained_full = pca_full.explained_variance_ratio_.tolist()
    eigenvalues_full = pca_full.explained_variance_.tolist()
    k = min(n_components, n_features)
    pca_k = PCA(n_components=k)
    X_transformed = pca_k.fit_transform(X_scaled)
    X_reconstructed = scaler.inverse_transform(pca_k.inverse_transform(X_transformed))
    pca_2d = PCA(n_components=min(2,n_features))
    X_2d = pca_2d.fit_transform(X_scaled)
    return jsonify({
        "transformed": X_transformed.tolist(),
        "transformed_2d": X_2d.tolist(),
        "reconstructed": X_reconstructed.tolist(),
        "components": pca_k.components_.tolist(),
        "mean": scaler.mean_.tolist(),
        "explained_variance": explained_full,
        "eigenvalues": eigenvalues_full,
        "cumulative_variance": np.cumsum(explained_full).tolist(),
        "n_features": n_features,
        "n_components_used": k
    })

@app.route("/api/pca/dataset", methods=["POST"])
def pca_dataset():
    data = request.json
    dataset = data.get("dataset","correlated")
    n = 120
    rng = np.random.RandomState(42)
    if dataset=="correlated":
        t=rng.randn(n); noise=0.25
        X=np.column_stack([t+rng.randn(n)*noise, t*1.4+rng.randn(n)*noise,
                           t*0.8+rng.randn(n)*noise*1.5, t*1.1+rng.randn(n)*noise,
                           t*0.6+rng.randn(n)*noise*2])
    elif dataset=="faces":
        expr=rng.randn(n); light=rng.randn(n); noise=0.3
        X=np.column_stack([expr+rng.randn(n)*noise, expr*0.9-light*0.3+rng.randn(n)*noise,
                           light+rng.randn(n)*noise, expr*0.5+light*0.7+rng.randn(n)*noise,
                           -expr*0.4+light*0.9+rng.randn(n)*noise, expr*0.3-light*0.5+rng.randn(n)*noise])
    elif dataset=="elongated":
        t=rng.randn(n)*3
        X=np.column_stack([t+rng.randn(n)*0.2, t*0.7+rng.randn(n)*0.3,
                           t*0.4+rng.randn(n)*0.5, rng.randn(n)*0.4])
    else:
        X=rng.randn(n,4)
    return jsonify({"points":X.tolist(),"n_features":X.shape[1]})

# ─── Spectral ─────────────────────────────────────────────────────────────────

@app.route("/api/spectral/cluster", methods=["POST"])
def spectral_cluster():
    data = request.json
    points = np.array(data["points"])
    n_clusters = int(data.get("n_clusters", 2))
    gamma = float(data.get("gamma", 1.0))
    n = len(points)

    # Subsample only for large datasets (> 100 pts)
    MAX_EIGEN = 100
    if n > MAX_EIGEN:
        sub_idx = np.random.RandomState(42).choice(n, MAX_EIGEN, replace=False)
        pts_sub = points[sub_idx]
    else:
        sub_idx = np.arange(n)
        pts_sub = points

    W = rbf_kernel(pts_sub, gamma=gamma)
    np.fill_diagonal(W, 0)
    d = W.sum(axis=1)
    d_safe = np.maximum(d, 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_safe))
    L_sym = np.eye(len(pts_sub)) - D_inv_sqrt @ W @ D_inv_sqrt

    # Only compute the eigenvalues we need
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        k_eig = min(n_clusters + 4, len(pts_sub) - 1)
        vals, vecs = eigsh(csr_matrix(L_sym), k=k_eig, which='SM')
        order = np.argsort(vals)
        eigenvalues = vals[order]
        eigenvectors = vecs[:, order]
    except Exception:
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

    U = eigenvectors[:, :n_clusters]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U_norm = U / np.maximum(norms, 1e-10)

    km_spec = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    spectral_labels_sub = km_spec.fit_predict(U_norm)
    km_raw = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans_labels_sub = km_raw.fit_predict(pts_sub)

    if n > MAX_EIGEN:
        from sklearn.neighbors import KNeighborsClassifier
        spectral_labels = KNeighborsClassifier(n_neighbors=3).fit(pts_sub, spectral_labels_sub).predict(points).tolist()
        kmeans_labels = KNeighborsClassifier(n_neighbors=3).fit(pts_sub, kmeans_labels_sub).predict(points).tolist()
    else:
        spectral_labels = spectral_labels_sub.tolist()
        kmeans_labels = kmeans_labels_sub.tolist()

    n_disp = min(40, len(pts_sub))
    disp_idx = np.linspace(0, len(pts_sub)-1, n_disp, dtype=int)

    return jsonify({
        "spectral_labels": spectral_labels,
        "kmeans_labels": kmeans_labels,
        "eigenvalues": eigenvalues[:min(10, len(eigenvalues))].tolist(),
        "embedding": U[:, :2].tolist(),
        "embedding_labels": spectral_labels_sub.tolist(),
        "affinity_matrix": W[np.ix_(disp_idx, disp_idx)].tolist(),
    })

@app.route("/api/spectral/dataset", methods=["POST"])
def spectral_dataset():
    data = request.json
    dataset = data.get("dataset","circles"); n=100
    if dataset=="circles": X,_=make_circles(n_samples=n,noise=0.05,factor=0.4,random_state=42)
    elif dataset=="moons": X,_=make_moons(n_samples=n,noise=0.07,random_state=42)
    elif dataset=="blobs": X,_=make_blobs(n_samples=n,centers=3,cluster_std=0.5,random_state=42)
    elif dataset=="spiral":
        # Archimedean spiral — 1.25 turns, arms 180° apart, clearly spiral-shaped
        rng=np.random.RandomState(42)
        t=np.linspace(np.pi/3, 2.5*np.pi, n//2)
        r=(t - np.pi/3)/(2*np.pi)*0.7 + 0.08
        noise=0.018
        X1=np.column_stack([r*np.cos(t)+rng.randn(n//2)*noise, r*np.sin(t)+rng.randn(n//2)*noise])
        X2=np.column_stack([-r*np.cos(t)+rng.randn(n//2)*noise, -r*np.sin(t)+rng.randn(n//2)*noise])
        X=np.vstack([X1,X2])
    else: X,_=make_circles(n_samples=n,noise=0.05,factor=0.4,random_state=42)
    return jsonify({"points":X.tolist()})

# ─── GNN ──────────────────────────────────────────────────────────────────────

@app.route("/api/gnn/forward", methods=["POST"])
def gnn_forward():
    data = request.json
    nodes=data["nodes"]; edges=data["edges"]; n_layers=int(data.get("n_layers",2))
    n=len(nodes)
    if n==0: return jsonify({"error":"No nodes"}),400
    node_ids=[nd["id"] for nd in nodes]; id_to_idx={nid:i for i,nid in enumerate(node_ids)}
    A=np.zeros((n,n))
    for e in edges:
        i=id_to_idx.get(e["source"]); j=id_to_idx.get(e["target"])
        if i is not None and j is not None: A[i,j]=1; A[j,i]=1
    A_hat=A+np.eye(n); d=A_hat.sum(axis=1)
    A_norm=np.diag(1.0/np.maximum(d,1e-10))@A_hat
    feat_dim=len(nodes[0].get("features",[0.5,0.5]))
    H=np.array([nd.get("features",[0.5]*feat_dim) for nd in nodes],dtype=float)
    rng=np.random.RandomState(7); hidden_dim=4
    layer_states=[{"embeddings":H.tolist(),"layer":0}]
    H_curr=H.copy()
    for layer in range(n_layers):
        W=rng.randn(H_curr.shape[1],hidden_dim)*0.5
        H_curr=np.tanh(A_norm@H_curr@W)
    layer_states.append({"embeddings":H_curr.tolist(),"layer":n_layers})
    message_trace=[{"node":nd["id"],
                    "neighbours":[node_ids[j] for j in range(n) if A[id_to_idx[nd["id"]],j]>0],
                    "n_neighbours":int(A[id_to_idx[nd["id"]]].sum())} for nd in nodes]
    return jsonify({"layer_states":layer_states,"message_trace":message_trace,
                    "final_embeddings":H_curr.tolist(),"adjacency":A.tolist()})

@app.route("/api/gnn/preset", methods=["POST"])
def gnn_preset():
    data=request.json; preset=data.get("preset","network")
    if preset=="network":
        nodes=[{"id":"A","features":[0.9,0.2],"label":0,"x":180,"y":160},
               {"id":"B","features":[0.3,0.8],"label":0,"x":320,"y":80},
               {"id":"C","features":[0.7,0.6],"label":1,"x":460,"y":160},
               {"id":"D","features":[0.2,0.4],"label":1,"x":400,"y":290},
               {"id":"E","features":[0.8,0.1],"label":0,"x":220,"y":300},
               {"id":"F","features":[0.5,0.9],"label":1,"x":540,"y":290}]
        edges=[{"source":"A","target":"B"},{"source":"A","target":"E"},
               {"source":"B","target":"C"},{"source":"B","target":"D"},
               {"source":"C","target":"D"},{"source":"C","target":"F"},
               {"source":"D","target":"E"},{"source":"D","target":"F"}]
    elif preset=="star":
        nodes=[{"id":"Hub","features":[1.0,1.0],"label":0,"x":340,"y":210},
               {"id":"S1","features":[0.8,0.2],"label":1,"x":180,"y":120},
               {"id":"S2","features":[0.2,0.9],"label":1,"x":500,"y":120},
               {"id":"S3","features":[0.6,0.3],"label":1,"x":180,"y":300},
               {"id":"S4","features":[0.3,0.7],"label":1,"x":500,"y":300}]
        edges=[{"source":"Hub","target":"S1"},{"source":"Hub","target":"S2"},
               {"source":"Hub","target":"S3"},{"source":"Hub","target":"S4"}]
    elif preset=="chain":
        nodes=[{"id":"N1","features":[1.0,0.0],"label":0,"x":100,"y":210},
               {"id":"N2","features":[0.7,0.3],"label":0,"x":230,"y":210},
               {"id":"N3","features":[0.5,0.5],"label":1,"x":360,"y":210},
               {"id":"N4","features":[0.3,0.7],"label":1,"x":490,"y":210},
               {"id":"N5","features":[0.0,1.0],"label":1,"x":620,"y":210}]
        edges=[{"source":"N1","target":"N2"},{"source":"N2","target":"N3"},
               {"source":"N3","target":"N4"},{"source":"N4","target":"N5"}]
    else:
        nodes=[{"id":"A","features":[1,0],"label":0,"x":200,"y":200},
               {"id":"B","features":[0,1],"label":1,"x":400,"y":120},
               {"id":"C","features":[1,1],"label":0,"x":480,"y":280}]
        edges=[{"source":"A","target":"B"},{"source":"B","target":"C"}]
    return jsonify({"nodes":nodes,"edges":edges})

# ─── Quiz ─────────────────────────────────────────────────────────────────────

@app.route("/api/quiz/questions", methods=["GET"])
def quiz_questions():
    topic=request.args.get("topic","kernel")
    bank=QUIZ_BANK.get(topic,[])
    selected=random.sample(bank, min(5,len(bank)))
    return jsonify({"questions":selected})


QUIZ_BANK = {
    "kernel": [
        {"id":"k1","question":"You have data arranged in concentric circles. Which kernel is most appropriate for an SVM?","options":["Linear","RBF (Gaussian)","Polynomial degree 1","No kernel needed"],"answer":1,"explanation":"RBF maps points into infinite-dimensional space, effectively separating concentric rings that are inseparable in 2D."},
        {"id":"k2","question":"The kernel trick avoids computing the feature map φ(x) explicitly. What does it compute instead?","options":["The inverse of the Gram matrix","The inner product K(x,x') = φ(x)·φ(x') directly","The distance in input space","The gradient of the loss"],"answer":1,"explanation":"K(x,x') = φ(x)·φ(x') can be computed cheaply in input space, even when φ maps to an infinite-dimensional space."},
        {"id":"k3","question":"Increasing the C parameter in an SVM does what?","options":["Makes the margin wider, allowing more errors","Makes the margin narrower, penalising misclassifications more","Switches the kernel function","Always reduces overfitting"],"answer":1,"explanation":"C controls the trade-off: high C → hard margin (fits training data tightly, risk of overfitting). Low C → soft margin (tolerates more errors, better generalisation)."},
        {"id":"k4","question":"Which of the following is a valid Mercer kernel?","options":["K(x,y) = x − y","K(x,y) = exp(−γ‖x−y‖²)","K(x,y) = ‖x−y‖","K(x,y) = x·y − 1"],"answer":1,"explanation":"The RBF kernel exp(−γ‖x−y‖²) is always positive semi-definite. The others can produce indefinite Gram matrices, violating Mercer's condition."},
        {"id":"k5","question":"What are 'support vectors' in an SVM?","options":["All training points","Training points on or within the margin","Centroids of each class","Eigenvectors of the kernel matrix"],"answer":1,"explanation":"Support vectors are the training points closest to the boundary. Only they influence the decision surface — removing all other points would not change it."},
        {"id":"k6","question":"The RBF kernel implicitly maps data into a feature space of what dimensionality?","options":["Same as input d","2d","d²","Infinite dimensional"],"answer":3,"explanation":"The RBF kernel corresponds to an infinite-dimensional feature space via its Taylor expansion. This is what makes it capable of representing arbitrarily complex boundaries."},
        {"id":"k7","question":"A linear SVM achieves 95% train accuracy but 60% test accuracy. What is the most likely cause?","options":["C is too low","The kernel is too simple for the data's non-linear structure","The margin is too wide","Too few support vectors"],"answer":1,"explanation":"A large train-test gap with a linear kernel typically means the data is not linearly separable. The kernel cannot capture the true decision boundary."},
        {"id":"k8","question":"For 2D input, polynomial kernel K(x,y) = (x·y + 1)² maps to how many features?","options":["2","4","6","Infinite"],"answer":2,"explanation":"Expanding (x·y + 1)² for 2D input gives 6 terms: {1, x₁, x₂, x₁², x₂², x₁x₂}. A degree-2 kernel on 2D input implicitly maps to 6D feature space."},
        {"id":"k9","question":"In the dual SVM, the decision function is f(x) = Σᵢ αᵢ yᵢ K(xᵢ,x) + b. What role do the αᵢ play?","options":["Kernel hyperparameters","Learned weights indicating each support vector's influence on the boundary","Class probabilities","Feature map coefficients"],"answer":1,"explanation":"αᵢ are Lagrange multipliers. Only support vectors have αᵢ > 0 — all other points have αᵢ = 0 and contribute nothing to predictions."},
        {"id":"k10","question":"Why can't we just compute polynomial feature expansions directly instead of using the kernel trick?","options":["Polynomial features are not useful","Explicitly computing high-degree expansions is computationally intractable for large d","Polynomial kernels violate Mercer's condition","Feature expansion needs more data"],"answer":1,"explanation":"For degree-p polynomial in d dimensions, explicit features have O(dᵖ) entries. For d=1000, p=3: 10⁹ features — impossible. The kernel trick achieves the same in O(d) time."}
    ],
    "pca": [
        {"id":"p1","question":"PCA finds principal components by maximising which quantity?","options":["The mean of the data","The variance of projected data","The reconstruction error","The correlation between features"],"answer":1,"explanation":"Each principal component maximises the variance of data projected onto it, subject to being orthogonal to previous components."},
        {"id":"p2","question":"After applying PCA, the principal components are always:","options":["Correlated with each other","Orthogonal (uncorrelated) to each other","Equal in explained variance","Normalised to unit length only"],"answer":1,"explanation":"PCA components are orthogonal eigenvectors of the covariance matrix — they are completely uncorrelated by construction."},
        {"id":"p3","question":"Why should you standardise features before PCA?","options":["PCA requires integer inputs","Features with large scales dominate variance and distort components","Standardisation makes data linearly separable","It only speeds up computation"],"answer":1,"explanation":"Without standardisation, high-variance features dominate the principal components regardless of their actual importance. Z-scoring puts all features on equal footing."},
        {"id":"p4","question":"The explained variance ratio of PC1 is 0.85. What does this mean?","options":["85% of data points are captured","85% of total variance is explained by PC1","PC1 has 85 features","Model accuracy is 85%"],"answer":1,"explanation":"Explained variance ratio tells you the fraction of total dataset variance captured by that component. 0.85 means PC1 alone accounts for 85% of all variation."},
        {"id":"p5","question":"You have 100 samples and 50 features. What is the maximum number of non-zero PCs?","options":["50","100","99","min(99,50) = 50"],"answer":3,"explanation":"PCA can produce at most min(n−1, d) non-zero components. Here min(99, 50) = 50."},
        {"id":"p6","question":"PC1 explains 45%, PC2 explains 40%. You use only PC1. What is the reconstruction error?","options":["55%","40%","15%","60%"],"answer":0,"explanation":"If PC1 explains 45%, then 100%−45% = 55% of variance is NOT captured. The reconstruction error equals the unexplained variance."},
        {"id":"p7","question":"Which task is PCA NOT well-suited for?","options":["Reducing storage of high-dimensional data","Visualising data in 2D","Classifying data into categories","Removing correlated features"],"answer":2,"explanation":"PCA is unsupervised — it doesn't use class labels. It cannot perform classification. For that, you need a supervised method like LDA or an SVM."},
        {"id":"p8","question":"What happens if you add a perfectly duplicated feature before running PCA?","options":["PCA fails with an error","The duplicate adds a zero-variance component","The eigenvalue doubles","The covariance matrix becomes rank-deficient, adding a zero eigenvalue"],"answer":3,"explanation":"A duplicate feature creates perfect linear dependence in the covariance matrix, making it rank-deficient. One eigenvalue becomes exactly 0 — that component captures no variance."},
        {"id":"p9","question":"A scree plot shows eigenvalues [3.8, 3.6, 0.2, 0.15, 0.1]. How many components should you retain?","options":["1","2","4","5"],"answer":1,"explanation":"The eigengap occurs after the 2nd value — there's a sharp drop from 3.6 to 0.2. The heuristic says retain 2 components, which together capture the vast majority of variance."},
        {"id":"p10","question":"Eigenfaces in face recognition are the principal components of pixel data. What do they represent?","options":["Individual faces stored in the model","Directions of maximum pixel variance across the face dataset","Randomly generated templates","Average face per person"],"answer":1,"explanation":"Each eigenface is a PC of the pixel covariance matrix — a direction of high variance in face space, capturing things like lighting variation or facial expression."}
    ],
    "spectral": [
        {"id":"s1","question":"Why does k-means fail on concentric circles but spectral clustering succeeds?","options":["K-means is slower","K-means assumes convex clusters; spectral uses graph connectivity","Spectral uses more memory","K-means can only find 2 clusters"],"answer":1,"explanation":"K-means partitions by distance to centroids, assuming spherical clusters. Spectral clustering uses the graph Laplacian to capture non-convex connectivity structure."},
        {"id":"s2","question":"What does the graph Laplacian L = D − W encode?","options":["Pairwise distances","Connectivity — how similar each point is to its neighbours","PCA components","Cluster centroids"],"answer":1,"explanation":"L = D − W, where W is the affinity matrix and D is the degree matrix. The Laplacian encodes graph connectivity and is the key to finding natural clusters."},
        {"id":"s3","question":"The number of near-zero eigenvalues of the Laplacian tells you what?","options":["Number of dimensions needed","Number of connected components","Optimal gamma value","Number of outliers"],"answer":1,"explanation":"Each connected component contributes one zero eigenvalue. The eigengap heuristic (jump in eigenvalues) reveals the natural number of clusters."},
        {"id":"s4","question":"After computing the spectral embedding, what step comes next?","options":["Apply PCA again","Run k-means on the embedding","Return eigenvectors as labels","Apply SVM"],"answer":1,"explanation":"The spectral embedding maps points to a space where connected components are well-separated. K-means on these embeddings then cleanly identifies the clusters."},
        {"id":"s5","question":"Increasing γ in the RBF affinity W_ij = exp(−γ‖xᵢ−xⱼ‖²) does what?","options":["Makes distant points more similar","Makes similarity drop off faster, creating sparser connections","Has no effect","Increases cluster count"],"answer":1,"explanation":"Large γ means the exponential decays rapidly with distance — only very close points get high similarity weights, creating a sparser, more locally-connected graph."},
        {"id":"s6","question":"Why does spectral clustering row-normalise the eigenvector matrix U before k-means?","options":["To speed up k-means","To project each point onto the unit sphere, improving separation","K-means requires unit-norm inputs","To remove the first eigenvector's effect"],"answer":1,"explanation":"Row-normalisation maps points onto the unit sphere in embedding space. This makes angular separation between clusters more pronounced, helping k-means find cleaner partitions."},
        {"id":"s7","question":"Spectral clustering is related to finding a minimum cut in a graph. What does 'cut' mean here?","options":["Minimising cluster sizes","Minimising total edge weight crossing between clusters","Finding shortest paths","Removing low-weight edges"],"answer":1,"explanation":"A graph cut's value is the total weight of edges crossing the partition boundary. Spectral clustering finds balanced minimum cuts — separating weakly-connected regions."},
        {"id":"s8","question":"What is the main computational bottleneck of spectral clustering on large datasets?","options":["Running k-means on embeddings","Computing and storing the n×n affinity matrix W","Normalising eigenvectors","Choosing k"],"answer":1,"explanation":"The affinity matrix W is n×n. For n=10,000 this is 10⁸ entries. Full eigendecomposition is O(n³). In practice, approximate methods like Nyström or sparse graphs are used."},
        {"id":"s9","question":"You run spectral clustering with k=3 and notice near-equal cluster sizes despite one group being naturally larger. Most likely cause?","options":["Gamma too small","Normalised Laplacian enforces balanced cuts, suppressing imbalanced solutions","K-means always balances clusters","Eigenvalues computed incorrectly"],"answer":1,"explanation":"The normalised Laplacian (Lₛ = D⁻½ L D⁻½) implicitly penalises imbalanced partitions. This is a feature for most use cases but a drawback when true clusters are highly unequal in size."},
        {"id":"s10","question":"Spectral clustering is applied to a friendship graph (nodes=people, edges=friendships). What does W encode?","options":["Geographic distances","Friendship connections (1 if friends, 0 if not)","Number of mutual friends","Community membership probabilities"],"answer":1,"explanation":"In a friendship graph, the adjacency matrix directly serves as W. Spectral clustering then finds communities as densely-connected subgraphs with few cross-group edges."}
    ],
    "gnn": [
        {"id":"g1","question":"What does one round of message passing in a GNN accomplish?","options":["Trains weights via backpropagation","Each node aggregates information from its immediate neighbours","Adds new edges to the graph","Computes the graph Laplacian"],"answer":1,"explanation":"In each layer, every node collects feature vectors from its neighbours and combines them to update its own representation."},
        {"id":"g2","question":"After 2 layers of message passing, a node's embedding captures information from:","options":["Only itself","Direct neighbours only","Nodes up to 2 hops away","All nodes in the graph"],"answer":2,"explanation":"Each GNN layer expands the receptive field by 1 hop. After k layers, a node's embedding incorporates information from all nodes within k hops."},
        {"id":"g3","question":"Why are GNNs preferred over flattening the adjacency and using an MLP?","options":["GNNs are always faster","GNNs are permutation-invariant and scale to any graph size","MLPs can't handle numerical features","GNNs need no training data"],"answer":1,"explanation":"Flattening loses permutation invariance and doesn't generalise to different-sized graphs. GNNs handle both naturally by operating on local neighbourhoods."},
        {"id":"g4","question":"In the GCN update H' = σ(Â H W), what is Â?","options":["The learned weight matrix","The normalised adjacency with self-loops","The node feature matrix","The activation function"],"answer":1,"explanation":"Â = D̂⁻¹(A + I) is the degree-normalised adjacency with self-loops. It controls how each node aggregates its own features plus its neighbours'."},
        {"id":"g5","question":"Why are self-loops (adding I to adjacency) important in GCNs?","options":["They prevent disconnected graphs","Each node includes its own features when aggregating","They make the matrix invertible","Only needed for undirected graphs"],"answer":1,"explanation":"Without self-loops, a node would only aggregate neighbours' information, ignoring its own features entirely. Adding I ensures every node considers itself as one of its own 'neighbours'."},
        {"id":"g6","question":"A 3-layer GNN is applied to a graph with diameter 10. What does a node's embedding capture?","options":["The entire graph","Nodes within 3 hops","Nodes within 10 hops","Only the node's own features"],"answer":1,"explanation":"Each GNN layer expands receptive field by 1 hop. With 3 layers, a node only 'sees' its 3-hop neighbourhood. Capturing the full diameter of 10 would require at least 10 layers."},
        {"id":"g7","question":"What is the 'over-smoothing' problem in deep GNNs?","options":["Model trains too slowly","After many layers, all node embeddings converge to the same value","Graph edges become too weighted","The activation saturates"],"answer":1,"explanation":"With many layers, each node aggregates from an increasingly large neighbourhood. Eventually all nodes have seen the entire graph and their embeddings converge — losing local structural differences."},
        {"id":"g8","question":"In graph classification, how are node embeddings combined into a prediction?","options":["Only the first node's embedding is used","Node embeddings are pooled (sum/mean/max) into a single graph-level vector","The adjacency matrix is used directly","Each node is classified independently"],"answer":1,"explanation":"Graph-level tasks require a readout function aggregating all node embeddings into a fixed-size vector. Common choices: global sum, mean, or max pooling, or learned hierarchical pooling."},
        {"id":"g9","question":"You want to recommend new friends in a social network. Which GNN task fits?","options":["Node classification","Link prediction","Graph classification","Edge regression"],"answer":1,"explanation":"Friend recommendation is link prediction — predicting whether an edge should exist between two nodes. GNNs learn node embeddings; pairs with similar embeddings are likely to be friends."},
        {"id":"g10","question":"A GNN trained on a 10-node graph is applied to a 1000-node graph at inference. Will this work?","options":["No — GNNs require fixed-size input","Yes — GNN weights are shared across all nodes and are graph-size independent","Only if topology matches","Only if node features match"],"answer":1,"explanation":"GNN weights W operate on node feature vectors, not the graph structure directly. The same weights apply to any node in any graph — GNNs are naturally inductive and generalise across graph sizes."}
    ]
}

# ─── AI Tutor ─────────────────────────────────────────────────────────────────

@app.route("/api/tutor", methods=["POST"])
def tutor():
    import urllib.request, urllib.error, json as _json, os

    data     = request.json
    messages = data.get("messages", [])
    system   = data.get("system", "You are a helpful ML tutor.")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set. Run: export GEMINI_API_KEY=YOUR_KEY then restart the server."}), 500

    # Convert OpenAI-style messages to Gemini format
    # Prepend system prompt as the first user turn
    gemini_contents = [{"role": "user", "parts": [{"text": system}]},
                       {"role": "model", "parts": [{"text": "Understood. I'm ready to help."}]}]
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        gemini_contents.append({"role": role, "parts": [{"text": m["content"]}]})

    payload = _json.dumps({
        "contents": gemini_contents,
        "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.7}
    }).encode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read())
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return jsonify({"text": text})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return jsonify({"error": f"Gemini API error {e.code}: {body}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(debug=True, port=5050)
