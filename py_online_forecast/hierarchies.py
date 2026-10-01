"""Transformations and tools for hierarchical forecast reconciliation.

This module provides transformations and tools for working with hierarchical forecast
reconciliation. The main class is ``RidgeReconciliation``, which uses the ``RRR`` (or
optionally ``SRRR``) prediction to model the bottom level forecast errors and then
constructs reconciled forecasts at all levels of the hierarchy. Behind the scenes, these
classes use the ``HierarchicalForecastReconciliation`` transformation, which provides
a generic way to setup forecast reconciliation using any prediction model.

For working with temporal hierarchies, the ``TemporalRidgeReconciliation`` class is a
thin wrapper around ``RidgeReconciliation`` that uses a ``BackShift`` transformation to
construct a bottom level variable for a temporal hierarchy.

The module also includes some tools for constructing hierarchies using a ``Node`` class.
"""

import numpy as np

from . import core as c
from . import features as f
from . import prediction as p


class _ShrinkagePrior(c.Transformation):
    """Transformation for computing shrinkage priors for forecast reconciliation."""

    def __init__(self, S_top, X, Y, memory, l_shrink = "auto"):
        self.S_top = S_top
        n, m = S_top.shape
        npm = n + m

        self.n, self.m = n, m
        self.sigma_bot_hat_d = np.zeros(m)
        self.sigma_top_hat_d = np.zeros(n)
        self.sigma_hat = np.zeros((npm, npm))
        self.var_sigma_hat = np.zeros((npm, npm))
        self.diag_mask = np.eye(npm, dtype=bool)


        prior = c.Transformation(self._compute, X, Y, memory, l_shrink, ITERATE = ("x", "y"))
        super().__init__(p.stack, prior)

    def _compute(self, x, y, memory, l_shrink):
        # Compute the shrinkage prior for the ridge regression
        
        # If data is not ready, return None,
        if np.isnan(x).any() or np.isnan(y).any():
            return {"Q": np.zeros((self.n, self.n)), "theta0": np.zeros((self.n, self.m))}

        # Compute optimal Q and theta0 based on current error estimates
        err_top = self.S_top @ y - x

        if (
            l_shrink == "auto"
        ):  # Estimate shrinkage parameter using full covariance matrix

            if memory == 1:
                raise ValueError(
                    "Automatic shrinkage parameter estimation is not supported for no forgetting (mem=1)."
                )

            err_full = np.append(err_top, y)

            self.sigma_hat = memory * self.sigma_hat + (1 - memory) * np.outer(
                err_full, err_full
            )

            err_sq = err_full**2
            sigma_sq = self.sigma_hat**2

            # Update higer order variance estimates
            self.var_sigma_hat = (
                memory * (1 - memory) ** 2 * (np.outer(err_sq, err_sq) - sigma_sq)
                + memory**2 * self.var_sigma_hat
            )

            # Sum non-diagonal elements to estimate shrinkage parameter
            l_shrink = (
                np.where(self.diag_mask, 0, self.var_sigma_hat).sum()
                / np.where(self.diag_mask, 0, sigma_sq).sum()
            )

            cov_top_hat_d = np.diag(np.diag(self.sigma_hat[: self.n, : self.n]))
            cov_bot_hat_d = np.diag(np.diag(self.sigma_hat[self.n :, self.n :]))

        else:  # Estimate covariance diagonal

            # y_i contains bottom level errors
            self.sigma_bot_hat_d = self.sigma_bot_hat_d * memory + (1 - memory) * y**2

            # compute top level covariance from top level errors
            self.sigma_top_hat_d = self.sigma_top_hat_d * memory + (1 - memory) * err_top**2

            cov_top_hat_d = np.diag(self.sigma_top_hat_d)
            cov_bot_hat_d = np.diag(self.sigma_bot_hat_d)

        b0 = cov_top_hat_d + self.S_top @ cov_bot_hat_d @ self.S_top.T
        b1 = self.S_top @ cov_bot_hat_d

        Q = (
            1 / (1 - memory) * (l_shrink / (1 - l_shrink)) * b0
        )  # Consider using a better reperesentative of current memory than the limit 1/(1 - mem).

        theta0 = np.linalg.solve(b0, b1)

        return {"Q": Q, "theta0": theta0}


class RidgeReconciliation(c.Transformation):

    def __init__(self, S_top, Y_bot, Y_hat, horizon = 1, memory = 0.99, full_hierarchy_cov=False, 
                 optimal_shrinkage=False, full_cov = True, l_shrink = None, **kwargs):

        # Store inputs
        self.S_top = S_top
        self.S = np.vstack((S_top, np.eye(S_top.shape[1])))
        self.Y_bot = Y_bot
        self.Y_hat = Y_hat
        self.full_hierarchy_cov = full_hierarchy_cov

        # Construct the regression problem
        nsm, m = S_top.shape

        Y_hat_arr = f.ToArray(Y_hat)  # Ensure output of Y_hat is a numpy array.
        Y_hat_top = Y_hat_arr[:, :nsm]  # First n-m columns
        Y_hat_bot = Y_hat_arr[:, -m:]  # Last m columns

        # Incoherence at top level
        X = Y_hat_top - Y_hat_bot @ S_top.T

        # Bottom level forecast errors
        Y = Y_bot - f.Lag(Y_hat_bot, horizon)

        if optimal_shrinkage:
            # Check that kwargs does not contain any to be modified
            for k in ["Q", "theta0", "iterate_Q", "iterate_theta0"]:
                if k in kwargs:
                    raise ValueError(
                        f"Cannot specify {k} when optimal_shrinkage is True"
                    )

            # Compute shrinkage prior for ridge regression
            shrink_kwargs = {} if l_shrink is None else {"l_shrink": l_shrink}
            shrinkage_prior = _ShrinkagePrior(S_top, X, Y, memory, **shrink_kwargs)
            kwargs["Q"] = shrinkage_prior["Q"]
            kwargs["theta0"] = shrinkage_prior["theta0"]
            kwargs["iterate_Q"] = True
            kwargs["iterate_theta0"] = True

            self.Q = shrinkage_prior["Q"]

        elif l_shrink is not None:
            raise ValueError(
                "Cannot specify l_shrink when optimal_shrinkage is False"
            )

        # Construct prediction of bottom level errors
        self.prediction = p.RRR(X, Y, horizon, full_cov=full_cov, **kwargs)

        # Get the predicted bottom level errors
        err_bot_hat = self.prediction["mean"]

        # Construct the bottom level reconciled forecasts
        Y_hat_rec_bot = Y_hat_bot + err_bot_hat

        # Aggregate the reconciled forecasts to all levels of the hierarchy.
        Y_hat_rec = Y_hat_rec_bot @ self.S.T

        super().__init__(self._finalize, self.prediction, Y_hat_rec)

    def _finalize(self, prediction, Y_hat_rec):
        # Make a few additional computations and prepare the output dict

        result = {"mean": Y_hat_rec}

        cov_bot = prediction["cov"]

        # If only variances are provided, expand to full covariance matrix
        if cov_bot.ndim == 2:

            t, n = cov_bot.shape

            cov_bot_full = np.full((t, n, n), np.nan)

            # Get mask for valid rows
            valid_rows = ~np.any(np.isnan(cov_bot), axis=1)

            # Initialise valid rows to zero
            cov_bot_full[valid_rows] = 0

            # Assign diagonal elements
            ran = np.arange(n)
            cov_bot_full[:, ran, ran] = cov_bot[:, ran]

            cov_bot = cov_bot_full

        # Compute reconciled variance S V[x] S.T
        rec_var_est = self.S @ cov_bot @ self.S.T

        # Fetch only diagonal? (TODO: consider doing self.S @ cov_bot @ self.S.T more efficiently if only diagonal is needed)
        if not self.full_hierarchy_cov:
            rec_var_est = np.diagonal(rec_var_est, axis1=1, axis2=2)

        result["cov"] = rec_var_est

        return result

class _TemporalUpdateSchema(c.Transformation):

    def __init__(self, horizon, reference):
        self.horizon = horizon
        super().__init__(self._increment, reference, STATE = "offset")

    def _increment(self, reference, offset = None):
        # Use vectorized version instead
        if offset is None:
            offset = 0

        n = len(reference)

        bools = ((offset + np.arange(1, n + 1)) % self.horizon == 0).tolist()

        offset = (offset + n) % self.horizon

        return bools, offset

class TemporalRidgeReconciliation(RidgeReconciliation):

    def __init__(self, S_top, B, Y, Y_hat, horizon = None, skip_duplicates = False, **kwargs):
        if horizon is None:
            if isinstance(B, list):
                horizon = max(B) + 1
            elif isinstance(B, dict):
                horizon = max(B.values()) + 1

        # Construct backshift operator
        Y_bot = p.BackShift(B, data=Y)

        if skip_duplicates:
            # Check that kwargs does not contain update_theta or iterate_update_theta
            if "update_theta" in kwargs or "iterate_update_theta" in kwargs:
                raise ValueError(
                    "Cannot specify update_theta or iterate_update_theta when skip_duplicates is True"
                )

            # Use _TemporalUpdateSchema to skip some updates
            kwargs["update_theta"] = _TemporalUpdateSchema(horizon, reference=Y)
            kwargs["iterate_update_theta"] = True

        super().__init__(S_top, Y_bot, Y_hat, horizon=horizon, **kwargs)

class Node:
    """Class for constructing hierarchies of variables.

    Parameters
    ----------
    sources : list of Node
        List of source nodes.
    name : str, optional
        Name of the node. If None (default), a unique name is generated automatically.
    """

    _counter = 0

    def __init__(self, *sources: "Node", name=None):
        level = 0
        for node in sources:
            level = max(level, node.level + 1)
        self.level = level
        self.sources = list(sources)
        self.lags = {}
        if name is None:
            name = str(Node._counter)
            Node._counter += 1
        self.name = name

    def find_node(self, name):
        """Find node with given name in the hierarchy."""
        return next(n for n in self.get_nodes() if n.name == name)

    def __add__(self, other):
        """Combine two nodes into a new node with both as sources."""
        if not isinstance(other, Node):
            raise ValueError("Can only add Node to Node")
        return Node(self, other)

    def __radd__(self, other):
        """Combine two nodes into a new node with both as sources."""
        return self.__add__(other)

    def shift(self, h: int):
        """Return a lagged version of the node with lag h."""
        if len(self.sources) > 0:
            raise ValueError("Can only shift leaf nodes")
        if h < 0:
            raise ValueError("Shift must be non-negative")
        if h not in self.lags:
            self.lags[h] = LaggedNode(h, self)
        return self.lags[h]

    def is_parent(self, other):
        """Return True if self is a parent of other, False otherwise."""
        if other in self.sources:
            return True
        for src in self.sources:
            if src.is_parent(other):
                return True
        return False

    def get_nodes(self, nodes=None):
        """Return a list of all nodes in the hierarchy with self as root.

        Parameters
        ----------
        nodes : list of Node or None, optional
            List of nodes to consider when determining nodes. If None (default), all
            nodes in the hierarchy are considered. If provided, also specifies the
            ordering of the nodes in the output.
        """
        all_nodes = [self]
        current_nodes = all_nodes
        while len(current_nodes) > 0:
            new_nodes = []
            for node in current_nodes:
                new_nodes.extend(node.sources)
            all_nodes.extend(new_nodes)
            current_nodes = new_nodes

        if nodes is not None:
            # Find nodes if strings
            nodes = [self.find_node(n) if isinstance(n, str) else n for n in nodes]
            all_nodes = [n for n in nodes if n in all_nodes]

        return all_nodes

    def get_bot_nodes(self, nodes=None):
        """Return a list of leaf nodes in the hierarchy with self as root.

        Parameters
        ----------
        nodes : list of Node or None, optional
            List of nodes to consider when determining leaf nodes. If None (default),
            all nodes in the hierarchy are considered. If provided, also specifies the
            ordering of the nodes in the output.
        """
        # Get all leaf nodes (nodes without sources)
        if not self.sources:
            return [self]
        else:
            leafs = []
            for src in self.sources:
                # Add leaves not already in list
                for leaf in src.get_bot_nodes():
                    if leaf not in leafs:
                        leafs.append(leaf)

            if nodes is not None:
                # Find nodes if strings
                nodes = [self.find_node(n) if isinstance(n, str) else n for n in nodes]
                leafs = [n for n in nodes if n in leafs]

            return leafs

    def get_top_nodes(self, nodes=None):
        """Return a list of top nodes in the hierarchy with self as root.

        Parameters
        ----------
        nodes : list of Node or None, optional
            List of nodes to consider when determining top nodes. If None (default), all
            nodes in the hierarchy are considered. If provided, also specifies the
            ordering of the nodes in the output.

        """
        bot_nodes = self.get_bot_nodes()
        top_nodes = []
        for n in self.get_nodes():
            if n not in bot_nodes:
                top_nodes.append(n)

        if nodes is not None:
            # Find nodes if strings
            nodes = [self.find_node(n) if isinstance(n, str) else n for n in nodes]
            top_nodes = [n for n in nodes if n in top_nodes]

        return top_nodes

    def get_remaining_nodes(self, obs_nodes):
        """Return a list of remaining nodes in order.

        The function returns all nodes in the hierarchy that are not in the given list
        of observed nodes, in the order they appear in the hierarchy, as if returned by
        get_nodes().

        Parameters
        ----------
        obs_nodes : list of Node
            List of observed nodes to determine latent nodes.
        """
        all_nodes = self.get_nodes()
        remaining_nodes = [n for n in all_nodes if n not in obs_nodes]
        return remaining_nodes

    def build_summation_matrix(self, full=False, nodes=None):
        """Return summation matrix for hierarchy.

        Parameters
        ----------
        full : bool, optional
            Whether to return the full summation matrix including the identity for
            observed (bottom) nodes. If False (default), only the latent (top) nodes
            are included in the output.
        nodes : list of Node or None, optional
            List of nodes to include in the summation matrix. If None (default), all
            nodes in the hierarchy are included. If provided, also specifies the
            ordering of the nodes in the summation matrix.

        Returns
        -------
        S : np.ndarray of shape (n_top, n_bot) or (n_top + n_bot, n_bot)
            Summation matrix for top levels (if full=False) or all nodes (if full=True).
        """
        bot_nodes = self.get_bot_nodes()
        top_nodes = self.get_top_nodes()

        if nodes is not None:
            # Find nodes if strings
            nodes = [self.find_node(n) if isinstance(n, str) else n for n in nodes]

            bot_nodes = [n for n in nodes if n in bot_nodes]
            top_nodes = [n for n in nodes if n in top_nodes]

        return build_summation_matrix(bot_nodes, top_nodes, full=full)

    def build_aggregation_matrix(self, obs_nodes, lat_nodes=None, full=False):
        """Return aggregation matrix ordered by obs_nodes and lat_nodes.

        Parameters
        ----------
        obs_nodes : list of Node
            List of observable nodes to include.
        lat_nodes : list of Node or None, optional
            List of latent nodes to include. If None (default), all remaining
            descendants of self that are not in obs_nodes are used.
        full : bool, optional
            Whether to return the full aggregation matrix including the identity for
            observed (bottom) nodes. If False (default), only the latent (top) nodes
            are included in the output.

        Returns
        -------
        A : np.ndarray of shape (n_lat, n_obs) or (n_lat + n_obs, n_obs)
            Aggregation matrix for latent nodes (if full=False) or all nodes (if
            full=True).
        """
        # Find obs nodes if any are strings
        obs_nodes = [self.find_node(o) if isinstance(o, str) else o for o in obs_nodes]

        if lat_nodes is None:
            # Get latent nodes if not provided
            lat_nodes = [n for n in self.get_nodes() if not n in obs_nodes]
        else:
            # Find lat nodes if any are strings
            lat_nodes = [
                self.find_node(l) if isinstance(l, str) else l for l in lat_nodes
            ]

            # Check that lat nodes don't include obs nodes
            for lat in lat_nodes:
                if lat in obs_nodes:
                    raise ValueError("Latent nodes cannot include observed nodes")

        # Get a list of all nodes to be used (not necessarily all descendants of self)
        all_nodes = obs_nodes + lat_nodes

        # Get bottom and top level nodes
        bot_nodes = [node for node in self.get_bot_nodes() if node in all_nodes]
        top_nodes = [node for node in self.get_top_nodes() if node in all_nodes]

        # Build summation matrix from selected nodes
        S = build_summation_matrix(bot_nodes, top_nodes, full=True)

        # Build lists of old and new nodes for permutation matrix
        old_nodes = top_nodes + bot_nodes
        new_nodes = lat_nodes + obs_nodes

        # Build permutation matrix to reorder from old_nodes to new_nodes
        P = build_permutation_matrix(old_nodes, new_nodes)

        # Construct and return aggegation matrix
        return construct_aggregation_matrix(S, P, full=full)

    def build_backshift(self, output_nodes, input_nodes=None):
        """Return backshift structure for given output nodes.

        Parameters
        ----------
        output_nodes : list of Node
            List of output nodes to determine backshift structure.
        input_nodes : list of Node or None, optional
            List of input nodes to determine backshift structure. If None (default), all
            required input nodes are inferred using get_input_nodes(output_nodes).
        """
        output_nodes = [
            self.find_node(o) if isinstance(o, str) else o for o in output_nodes
        ]

        if input_nodes is None:
            input_nodes = get_input_nodes(output_nodes)
        else:
            input_nodes = [
                self.find_node(i) if isinstance(i, str) else i for i in input_nodes
            ]

        return build_backshift(input_nodes, output_nodes)

    def print_hierarchy(self, n=0):
        """Print the hierarchy structure starting from self as root."""
        indent = "  " * n
        print(f"{indent}- {self}")
        for node in self.sources:
            node.print_hierarchy(n + 1)

    def __repr__(self):
        """Return a string representation of the node."""
        return f"Node({self.name})"


def build_summation_matrix(bot_nodes, top_nodes, full=False):
    """Return summation matrix for hierarchy.

    Parameters
    ----------
    bot_nodes : list of Node or str
        List of observable (bottom) nodes to include in the summation matrix.
    top_nodes : list of Node or str
        List of latent (top) nodes to include in the summation matrix.

    full : bool, optional
        Whether to return the full summation matrix including the identity for
        observed (bottom) nodes. If False (default), only the latent (top) nodes
        are included.

    Returns
    -------
    S : np.ndarray of shape (m, n_bot)
        Summation matrix for latent nodes (if full=False) with m = n_lat, or for all
        nodes (if full=True) with m = n_lat + n_bot.
    """
    S = np.zeros((len(top_nodes), len(bot_nodes)))
    for i, lat in enumerate(top_nodes):
        for j, obs in enumerate(bot_nodes):
            if lat.is_parent(obs):
                S[i, j] = 1
    if full:
        S = np.vstack((S, np.eye(len(bot_nodes))))

    return S


def build_permutation_matrix(old_nodes, new_nodes):
    """Return permutation matrix for reordering nodes in hierarchy.

    Parameters
    ----------
    old_nodes : list of Node
        List of nodes in the original order.
    new_nodes : list of Node
        List of nodes in the new order.

    Returns
    -------
    C : np.ndarray of shape (n, n)
        Permutation matrix such that x_new = C @ x_old, where x_old and x_new are state
        vectors of the nodes in the old and new order, respectively.
    """
    # Check that old and new nodes contain the same nodes
    if set(old_nodes) != set(new_nodes):
        raise ValueError("Old and new nodes must contain the same nodes")
    n = len(old_nodes)
    C = np.zeros((n, n))
    for i, new in enumerate(new_nodes):
        for j, old in enumerate(old_nodes):
            if new is old:
                C[i, j] = 1
    return C


def construct_aggregation_matrix(S, P, full=False):
    """Return aggregation matrix for latent nodes in hierarchy.

    Parameters
    ----------
    S : np.ndarray of shape (n_tot, n_bot)
        Summation matrix for latent nodes in terms of observable nodes.
    P : np.ndarray of shape (n_tot, n_tot)
        Permutation matrix for reordering top and bottom level nodes to latent and
        observable nodes.

    Returns
    -------
    A: np.ndarray of shape (m, n_bot)
        Aggregation matrix for latent nodes in terms of observable nodes. If full=False
        (default), only the latent nodes are included, with m = n_top. If full=True, all
        nodes are included, with m = n_tot.
    """
    n_tot, n_bot = S.shape
    n_top = n_tot - n_bot
    P_lat = P[:n_top]
    P_obs = P[n_top:]

    #    A = P_lat @ S @ np.linalg.inv(P_obs @ S)
    tmp1 = P_lat @ S
    tmp2 = P_obs @ S
    A = np.linalg.solve(tmp2.T, tmp1.T).T

    if full:
        A = np.vstack((A, np.eye(n_bot)))

    return A.round().astype(int)


def get_input_nodes(output_nodes):
    """Return a list of required input nodes for given output nodes.

    Parameters
    ----------
    output_nodes : list of Node or None, optional
        List of output nodes to determine required input nodes.
    """
    input_nodes = []
    for out in output_nodes:
        if isinstance(out, LaggedNode):
            out = out.var
        if out not in input_nodes:
            input_nodes.append(out)

    return input_nodes


def build_backshift(input_nodes, output_nodes):
    """Return backshift matrix structure for temporal hierarchy.

    Constructs a backshift operator B such that y_bot = B z, where z are the states
    of the input nodes. The output is a list of pairs (i_k, j_k), k = 1, ..., m
    specifying that the i_k-th input state is lagged by j_k steps to construct the
    k-th output variable.

    Parameters
    ----------
    input_nodes : list of Node
        List of input nodes in the hierarchy.
    output_nodes : list of Node
        List of output nodes in the hierarchy.
    """
    B = []
    for out_node in output_nodes:

        if isinstance(out_node, LaggedNode):
            in_node = input_nodes.index(out_node.var)
            lag = out_node.lag
        else:
            in_node = input_nodes.index(out_node)
            lag = 0

        B.append((in_node, lag))

    return B

class LaggedNode(Node):
    """Class for lagged nodes in a temporal hierarchy.

    Parameters
    ----------
    lag : int
        Lag of the node, must be non-negative.
    source : Node
        Source node for the lagged variable.
    """

    def __init__(self, lag: int, source: Node):
        if lag < 0:
            raise ValueError("Lag must be non-negative")
        name = f"{source.name}(-{lag})"
        super().__init__(name=name)
        self.lag = lag
        self.var = source

    def shift(self):
        """Return a further lagged version of the node with lag increased by 1."""
        return self.sources[0].shift(self.lag + 1)

def make_hierarchy(edges):
    """Construct a hierarchy of nodes from a dictionary of edges.

    Parameters
    ----------
    edges : dict
        Dictionary of edges defining the hierarchy, where keys are parent node names and
        values are lists of child node names or tuples of (child node name, lag) for
        temporal hierarchies. For example, for a simple hierarchy with top node "A" and
        bottom nodes "B" and "C", the edges could be defined as ``{"A": ["B", "C"]}``.
        For a temporal hierarchy with top node "A" and bottom node "B" lagged by 1,
        the edges could be defined as ``{"A": [("B", 1)]}``. Nodes that are not parents
        (i.e. do not appear as keys) are assumed to be leaf nodes.

    Returns
    -------
    nodes : dict
        Dictionary mapping node names to Node instances.
    bot_nodes : list of Node
        List of bottom-level nodes in the hierarchy.
    top_nodes : list of Node
        List of top-level nodes in the hierarchy.
    """
    top_nodes = list(edges.keys())
    node_names = top_nodes.copy()
    for nodes in edges.values():
        for node in nodes:
            if isinstance(node, tuple):
                if node[0] not in node_names:
                    node_names.append(node[0])
            if node not in node_names:
                node_names.append(node)

    # Then create top nodes
    to_process = node_names.copy()
    nodes = {}

    while to_process:
        for node in to_process:
            if isinstance(node, tuple):
                if node[0] in nodes:
                    nodes[node] = nodes[node[0]].shift(node[1])
                    to_process.remove(node)
            elif node in edges:
                children = edges[node]
                if all(child in nodes for child in children):
                    nodes[node] = Node(*[nodes[child] for child in children], name=node)
                    to_process.remove(node)
            else:
                nodes[node] = Node(name=node)
                to_process.remove(node)

    return nodes
