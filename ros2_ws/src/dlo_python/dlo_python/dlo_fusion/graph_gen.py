#!/home/lar/visionenv/bin/python
import cv2
import numpy as np
import os
import torch
from torch import nn
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt


torch.manual_seed(0)
np.random.seed(0)


def edge_projection(point_test, point1, point2):

    # v1 -> vector from the closer node to the test point
    v1 = point_test - point1
    # v2 -> vector from the closer node to the other closer node (edge)
    v2 = point2 - point1
    dot_product = np.multiply(v1, v2).sum()
    segment_length = np.linalg.norm(v2)
    projection = dot_product / segment_length**2
    return point1 + projection * v2


def points_association(points2d, nodes, graph, dist_th=50):
    new_points = np.zeros_like(points2d)
    for it, point_test in enumerate(points2d):
        closer_nodes = np.argsort(np.linalg.norm(nodes - point_test, axis=1))
        node0 = closer_nodes[0]
        node1 = closer_nodes[1]
        neighs = list(nx.neighbors(graph, node0))
        for cn in closer_nodes[1:]:
            if cn in neighs:
                node1 = cn
                break

        # length_1 = np.linalg.norm(nodes[node0] - point_test)
        # length_2 = np.linalg.norm(nodes[node1] - point_test)
        # if np.mean([length_1, length_2]) > dist_th:
        #    new_points[it] = point_test
        # else:
        new_points[it, :] = edge_projection(point_test, nodes[node0], nodes[node1])

    return new_points


def correction(points2d_model, new_points, mask):
    mask_w, mask_h = mask.shape

    # validate new points

    valid_ids = [0, 1, len(new_points) - 2, len(new_points) - 1]
    for it in range(2, len(new_points) - 2):
        point_test = new_points[it].astype(int)
        if point_test[0] < 0 or point_test[0] >= mask_h or point_test[1] < 0 or point_test[1] >= mask_w:
            continue

        if mask[int(new_points[it][1]), int(new_points[it][0])] > 127:
            valid_ids.append(it)

    invalid_ids = [i for i in range(len(new_points)) if i not in valid_ids]

    validity_vector = np.zeros(len(new_points))
    validity_vector[valid_ids] = 1

    corrections = np.linalg.norm(new_points - points2d_model, axis=1)

    dirs = np.diff(points2d_model, axis=0)
    dirs = dirs / np.linalg.norm(dirs, axis=1).reshape(-1, 1)
    dirs_90 = np.array([-dirs[:, 1], dirs[:, 0]]).T

    # Compute the dot products to determine the correct direction
    dot_products = np.einsum("ij,ij->i", dirs_90, new_points[:-1] - points2d_model[:-1])
    # Create a mask where dot products are negative (indicating incorrect direction)
    mask_incorrect_rotations = dot_products < 0
    # Invert the perpendicular directions where the mask is True
    dirs_90[mask_incorrect_rotations] = -dirs_90[mask_incorrect_rotations]

    # Identify contiguous gaps where invalid_ids are contiguous
    contiguous_gaps = []
    gap = []

    for it in invalid_ids:
        if not gap:  # If the gap list is empty
            gap.append(it)
        elif it - gap[-1] == 1:  # If the current ID is contiguous with the last ID in the gap
            gap.append(it)
        else:  # If the current ID is not contiguous, save the current gap and start a new one
            contiguous_gaps.append(gap)
            gap = [it]

    # Append the last gap if it exists
    if gap:
        contiguous_gaps.append(gap)

    for gap in contiguous_gaps:

        # update directions
        id_below = min(gap) - 1
        prev_dir = dirs_90[id_below]
        for it in gap:
            if np.dot(dirs_90[it], prev_dir) < 0:
                dirs_90[it] = -dirs_90[it]
            prev_dir = dirs_90[it]

    # update not valid points
    for gap in contiguous_gaps:
        id_below = min(gap) - 1
        id_above = max(gap) + 1

        for it in gap:
            dist_below = it - id_below
            dist_above = id_above - it

            correction_below = corrections[id_below]
            correction_above = corrections[id_above]

            interp_corr = (correction_below * dist_above + correction_above * dist_below) / (dist_below + dist_above)
            new_points[it] = points2d_model[it] + interp_corr * dirs_90[it]

    return new_points, validity_vector


class AngleNet(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.conv_w1 = nn.Conv2d(1, hidden_dim // 4, 3)
        self.bn_w1 = nn.BatchNorm2d(hidden_dim // 4)
        self.pool_w1 = nn.MaxPool2d(3)
        self.conv_w2 = nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3)
        self.bn_w2 = nn.BatchNorm2d(hidden_dim // 4)
        self.flatten = nn.Flatten()

        self.lin_reg = nn.Linear(hidden_dim, 180)

    def encode_w(self, w):
        y = self.bn_w1(self.conv_w1(w.unsqueeze(dim=1))).relu()
        y = self.pool_w1(y)
        y = self.bn_w2(self.conv_w2(y)).relu()
        return self.flatten(y)

    def forward(self, windows):
        y = self.encode_w(windows)
        return self.lin_reg(y)


class GraphGeneration:

    def __init__(self, n_knn, th_edges_similarity, th_mask=127, wsize=15, sampling_ratio=0.1, network=None):
        self.n_knn = n_knn
        self.th_edges_similarity = th_edges_similarity
        self.th_mask = th_mask
        self.wsize = wsize
        self.sampling_ratio = sampling_ratio

        if network is None:
            self.load_network_model()
        else:
            self.network = network

    def load_network_model(self):
        script_path = os.path.dirname(os.path.abspath(__file__))
        self.network = AngleNet(hidden_dim=128)
        self.network.load_state_dict(torch.load(os.path.join(script_path, "CP_angle.pth"), weights_only=True))
        self.network.eval()

    def exec(self, mask, edges_filtering=True):
        mask[mask < self.th_mask] = 0
        mask[mask != 0] = 255

        # DISTANCE IMAGE
        dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist_img = cv2.GaussianBlur(dist_img, (3, 3), 0)

        nodes, edges = self.run_graph_generation(mask, dist_img)

        if nodes is None or edges is None:
            return None, None

        if edges.shape[0] == 0:
            return None, None

        if edges_filtering:
            edges = self.filter_edges_from_mask(nodes, edges, mask)

        ######################
        # path from start/end points
        # graph = nx.Graph()
        # nodes = nodes[:, [1, 0]]
        # graph.add_nodes_from([(it, {"pos": np.array(x)}) for it, x in enumerate(nodes)])
        # graph.add_edges_from(edges)

        return nodes, edges

    def compute_closer_node(self, graph, point):
        nodes = nx.get_node_attributes(graph, "pos")
        distances = np.linalg.norm(np.array(list(nodes.values())) - point, axis=1)
        return np.argmin(distances)

    def edges_similarity_fast(self, matrix_scores, X, edges, edges_norm, adj_fake, th=0.25):
        num_nodes = matrix_scores.shape[0]
        num_edges = matrix_scores.shape[1]

        edges_tuples = edges.T
        E = np.repeat(edges_norm.reshape(1, -1), num_nodes, axis=0)
        X2 = np.repeat(X[edges[0], edges[1]].reshape(1, -1), num_nodes, axis=0)

        M = matrix_scores * X2 * adj_fake
        pos_matrix_scores = M > th
        E_pos_zero = E.copy()
        E_pos_zero[pos_matrix_scores == False] = 0
        if E_pos_zero.shape[1] > 0:
            E_pos_max = np.repeat(np.max(E_pos_zero, axis=1).reshape(-1, 1), num_edges, axis=1)

            E_pos_inf = E.copy()
            E_pos_inf[pos_matrix_scores == False] = np.inf
            E_pos_min = np.repeat(np.min(E_pos_inf, axis=1).reshape(-1, 1), num_edges, axis=1)
            W_pos = 1 - (E - E_pos_min) / E_pos_max
            W_pos[pos_matrix_scores == False] = 0
            S_pos = M * W_pos
            S_pos[pos_matrix_scores == False] = 0

            pos_indeces = np.argmax(S_pos, axis=1)
            pos_edges = edges_tuples[pos_indeces]
        else:
            pos_edges = []

        neg_matrix_scores = M < -th
        E_neg_zero = E.copy()
        E_neg_zero[neg_matrix_scores == False] = 0
        if E_neg_zero.shape[1] > 0:
            E_neg_max = np.repeat(np.max(E_neg_zero, axis=1).reshape(-1, 1), num_edges, axis=1)

            E_neg_inf = E.copy()
            E_neg_inf[neg_matrix_scores == False] = np.inf
            E_neg_min = np.repeat(np.min(E_neg_inf, axis=1).reshape(-1, 1), num_edges, axis=1)
            W_neg = 1 - (E - E_neg_min) / E_neg_max
            W_neg[neg_matrix_scores == False] = 0
            S_neg = M * W_neg
            S_neg[neg_matrix_scores == False] = 0

            neg_indeces = np.argmin(S_neg, axis=1)
            neg_edges = edges_tuples[neg_indeces]
        else:
            neg_edges = []

        new_edges_fast = np.concatenate([pos_edges, neg_edges])
        return new_edges_fast

    def cosine_sim(self, a, b, eps=1e-8):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def similarity_score_nodes_edges(self, nodes, edges, nodes_angles, debug=False):
        edges_dirs, edges_norm, adj_fake = self.edges_prep(edges, nodes)
        edges_dirs = torch.Tensor(edges_dirs)

        angles = torch.Tensor(np.array(nodes_angles))
        nodes_dir = torch.stack([torch.sin(angles), torch.cos(angles)]).T

        X = torch.abs(self.cosine_sim(nodes_dir, nodes_dir))
        similarity_score_edges = self.cosine_sim(nodes_dir, edges_dirs) * torch.abs(torch.Tensor(adj_fake))

        return similarity_score_edges, X, edges_norm, adj_fake

    def edges_prep(self, edges, nodes, debug=False):
        nodes_0 = nodes[edges[0, :]]
        nodes_1 = nodes[edges[1, :]]
        edge_dirs = (nodes_1 - nodes_0).astype(np.float32)
        edge_norms = np.linalg.norm(edge_dirs, axis=1)
        edge_dirs[:, 0] = edge_dirs[:, 0] / edge_norms
        edge_dirs[:, 1] = edge_dirs[:, 1] / edge_norms
        edge_dirs = edge_dirs.reshape(-1, 2)

        edges_list = list(zip(edges[0], edges[1]))
        adj_fake = np.zeros((len(nodes), len(edges_list)))
        for it, (e0, e1) in enumerate(edges_list):
            adj_fake[e0, it] = 1
            adj_fake[e1, it] = -1

        return edge_dirs, edge_norms, adj_fake

    def orientations_from_network(self, windows):

        windows = torch.Tensor(np.array(windows))
        preds = self.network(windows).sigmoid().detach().cpu().numpy()

        angles = np.deg2rad(np.argmax(preds, axis=1))
        dirs = np.stack([np.sin(angles), np.cos(angles)]).T
        return dirs, angles

    def run_graph_generation(self, mask, dist_img):

        # local maximus
        max_image = cv2.dilate(dist_img, np.ones((3, 3)))
        maxmask = (dist_img == max_image) & mask

        x, y = np.nonzero(maxmask)
        points_maxmask = np.stack([x, y]).T

        # fps
        points = torch.Tensor(points_maxmask)
        indices_fps = torch_geometric.nn.fps(points, ratio=self.sampling_ratio)
        reduced = points[indices_fps].detach().cpu().numpy().astype(int)

        # windows
        l = (self.wsize - 1) // 2
        windows, points = [], []
        for p in reduced:
            w = dist_img[p[0] - l : p[0] + l + 1, p[1] - l : p[1] + l + 1]
            if w.shape == (self.wsize, self.wsize):
                windows.append(w)
                points.append(p)
        points = np.array(points)

        if points.shape[0] == 0:
            return None

        # orientations
        W = np.array(windows) / np.max(windows)
        points_dirs, points_angles = self.orientations_from_network(W)

        # edges with knn
        edges_knn = torch_geometric.nn.knn_graph(torch.Tensor(points), self.n_knn).detach().cpu().numpy()

        # matrix score
        matrix_scores, X, edges_norm, adj_fake = self.similarity_score_nodes_edges(points, edges_knn, points_angles)

        # edges computation
        new_edges = self.edges_similarity_fast(
            matrix_scores, X, edges_knn, edges_norm, adj_fake, th=self.th_edges_similarity
        )

        return points, new_edges

    def filter_edges_from_mask(self, nodes, edges, mask):

        mask_large = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

        nodes_0 = nodes[edges[:, 0]]
        nodes_1 = nodes[edges[:, 1]]

        distances = np.linalg.norm(nodes_0 - nodes_1, axis=1)
        indeces = np.where(distances > np.mean(distances))[0]

        mid_points_to_test = ((nodes_0[indeces] + nodes_1[indeces]) / 2).astype(int)
        indeces_zero = np.where(mask_large[mid_points_to_test[:, 0], mid_points_to_test[:, 1]] == 0)[0]

        mask = np.ones(edges.shape[0], bool)
        mask[indeces[indeces_zero]] = 0
        new_edges = edges[mask]
        return new_edges


def plot_graph(nodes, edges, ax=None):

    graph = nx.Graph()
    graph.add_nodes_from([(it, {"pos": x}) for it, x in enumerate(nodes)])
    for edge in edges:
        graph.add_edge(edge[0], edge[1])

    pos = nx.get_node_attributes(graph, "pos")
    r = nx.get_node_attributes(graph, "r")

    if ax is not None:
        nx.draw(graph, pos, node_size=10, ax=ax)
        nx.draw_networkx_edges(graph, pos, ax=ax)
    else:
        nx.draw(graph, pos, node_size=10, node_color=[r[key] for key in pos.keys()])
        nx.draw_networkx_edges(graph, pos)
        plt.show()


if __name__ == "__main__":

    g = GraphGeneration(n_knn=8, th_edges_similarity=0.25, th_mask=127, wsize=15, sampling_ratio=0.1)

    mask = cv2.imread("synthetic/new_renderings/mask1.png", cv2.IMREAD_GRAYSCALE)

    print(mask.shape)

    nodes_dict, edges = g.exec(mask)

    plot_graph(nodes_dict, edges)
