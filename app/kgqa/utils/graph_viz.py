import base64
import enum
import logging
import os
import pickle
import time
from abc import ABC
from itertools import groupby
from typing import List, Optional
from urllib.parse import urlparse

import graphviz
import networkx as nx
import requests
from pywikidata import Entity

DEFAULT_CACHE_PATH = '/tmp/cache'
SPARQL_ENDPOINT = 'https://query.wikidata.org/sparql'


class CacheBase(ABC):
    """CacheBase - Abstract base class for storing something in cache file"""

    def __init__(
        self,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
        cache_filename: str = "cache.pkl",
    ) -> None:
        self.cache_dir_path = cache_dir_path
        self.cache_filename = cache_filename
        self.cache = None

        self.cache_file_path = os.path.join(self.cache_dir_path, self.cache_filename)

        self.load_from_cache()

    def load_from_cache(self):
        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, "rb") as file:
                self.cache = pickle.load(file)

    def save_cache(self):
        if not os.path.exists(self.cache_dir_path):
            os.makedirs(self.cache_dir_path)

        with open(self.cache_file_path, "wb") as file:
            pickle.dump(self.cache, file)


class WikidataBase(CacheBase):
    """
    WikidataBase - Abstract base class for working with Wikidata SPARQL endpoints
    and storing results in cache file
    """

    def __init__(
        self,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
        cache_filename: str = "cache.pkl",
        sparql_endpoint: str = None,
    ) -> None:
        super().__init__(cache_dir_path, cache_filename)

        self.sparql_endpoint = sparql_endpoint
        if self.sparql_endpoint is None:
            self.sparql_endpoint = SPARQL_ENDPOINT

        parsed_sparql_endpoint = urlparse(self.sparql_endpoint)
        path_from_sparql_endpoint = (
            parsed_sparql_endpoint.netloc + "|" + parsed_sparql_endpoint.path[1:]
        ).replace("/", "_")
        self.cache_dir_path = os.path.join(
            self.cache_dir_path, path_from_sparql_endpoint
        )
        self.cache_file_path = os.path.join(
            self.cache_dir_path,
            self.cache_filename,
        )

        self.cache = {}
        self.load_from_cache()

    def _wikidata_uri_to_id(self, uri):
        return uri.split("/")[-1].split("-")[0]



class WikidataShortestPathCache(WikidataBase):
    """WikidataShortestPathCache - class for request shortest path from wikidata service
    with storing cache

    Args:
        cache_dir_path (str, optional): Path to directory with caches. Defaults to "./cache_store".
        sparql_endpoint (_type_, optional): URI for SPARQL endpoint.
            If None, will used SPARQL_ENDPOINT from config. Defaults to None.
        engine (str, optional): Engine of provided SPARQL endpoint. Supported GraphDB and Blazegraph only.
            If None, will used SPARQL_ENGINE from config.
            Defaults to None.

    Raises:
        ValueError: If passed wrong string identifier for engine. Supported only 'grapdb' and 'blazegraph'
    """

    def __init__(
        self,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
        sparql_endpoint: str = None,
        engine: str = None,
    ) -> None:
        super().__init__(cache_dir_path, "wikidata_shortest_paths.pkl", sparql_endpoint)

        self.engine = engine
        if self.engine is None:
            self.engine = 'blazegraph'

        self.engine = self.engine.lower()
        if self.engine not in ["blazegraph", "graphdb"]:
            raise ValueError(
                f'only "blazegraph" and "graphdb" engines supported, but passed {engine}'
            )

        self.cache = {}
        self.load_from_cache()

    def get_shortest_path(
        self,
        item1,
        item2,
        return_only_first=True,
        return_edges=False,
        return_id_only=False,
    ) -> List:
        """get_shortest_path

        Args:
            item1 (str): Identifier of Entity from which path started
            item2 (str): Identifier of End of path Entity
            return_only_first (bool, optional): Graphdb engine, can return a few pathes.
                If False, it will return only first path, if False, it will return all pathes.
                Works only with graphdb engine.
                Defaults to True.
            return_edges (bool, optional): Graphdb engine can return shortes path with edges.
                If False, it will work like Blazegraph, if True
            return_id_only (bool, optional): If True, will return pathes with only ID
                without other URI information. Default False

        Returns:
            list: shortest path or list of shortest pathes
        """
        if item1 is None or item2 is None:
            return None

        if self.engine == "blazegraph":
            if return_edges is True:
                raise ValueError(
                    "For Blazegraph engine, return_only_first must be only True and return_edges must be False"
                )
            elif return_only_first is False:
                logging.warning("For Blazegraph engine, only one path will be returned")

        key = (item1, item2)

        if key in self.cache:
            path_data = self.cache[key]
        else:
            path_data = self._request_path_data(key[0], key[1])
            self.cache[key] = path_data
            self.save_cache()

        if path_data is None:
            return None

        pathes = self._extract_pathes(path_data, return_edges)
        if return_id_only is True:
            pathes = [self._extract_ids_from_path(path) for path in pathes]

        if return_only_first is True:
            return pathes[0]
        else:
            return pathes

    def _extract_ids_from_path(self, path: List[str]) -> List[str]:
        if isinstance(path[0], list):  # with edges case
            results = []
            for _path in path:
                results.append([self._wikidata_uri_to_id(entity) for entity in _path])
            return results

        else:  # without edges case
            return [self._wikidata_uri_to_id(entity) for entity in path]

    def _extract_pathes(self, path_data, return_edges) -> List[List[str]]:
        if self.engine == "blazegraph":
            path = [
                r["out"]["value"]
                for r in sorted(
                    path_data,
                    key=lambda r: float(r["depth"]["value"]),
                )
            ]
            return [path]

        else:
            # pathIndex - index of path. Results can include a lot of pathes
            # edgeIndex - index of edge in each path.
            # path_data sorted by (pathIndex, edgeIndex)
            pathes = [
                (
                    val["pathIndex"]["value"],
                    val["edgeIndex"]["value"],
                    self._rdf4j_edge_value_decode(val["edge"]["value"]),
                )
                for val in path_data
            ]
            pathes = [list(group) for _, group in groupby(pathes, key=lambda k: k[0])]
            pathes = [[path_el[-1] for path_el in path] for path in pathes]

            if not return_edges:
                pathes = [
                    [path[0][0]] + [path_step[-1] for path_step in path]
                    for path in pathes
                ]

            pathes = list(path for path, _ in groupby(pathes))

            # In some cases, gprahDB can return not only shortest path but pathes with shortest path length + 1 pathes
            # For that case, we filter it
            pathes = list(
                next(
                    groupby(sorted(pathes, key=lambda p: len(p)), key=lambda p: len(p))
                )[1]
            )
            return pathes

    def _request_path_data(self, item1, item2):
        if self.engine == "blazegraph":
            query = """
            select * {
                SERVICE gas:service {     
                gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ; 
                gas:in wd:%s ;     
                gas:target wd:%s ;        
                gas:out ?out ;      
                gas:out1 ?depth ;     
                gas:maxIterations 10 ;      
                gas:maxVisited 10000 .                            
                }
            }
            """ % (
                item1,
                item2,
            )
        elif self.engine == "graphdb":
            query = """
            PREFIX path: <http://www.ontotext.com/path#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX dbr: <http://dbpedia.org/resource/>
            PREFIX wd: <http://www.wikidata.org/entity/>

            SELECT ?pathIndex ?edgeIndex ?edge
            WHERE {
                SERVICE path:search {
                    [] path:findPath path:shortestPath ;
                    path:sourceNode wd:%s ;
                    path:destinationNode wd:%s ;
                    path:pathIndex ?pathIndex ;
                    path:resultBindingIndex ?edgeIndex ;
                    path:resultBinding ?edge ;
                    .
                }
            }
            """ % (
                item1,
                item2,
            )
        else:
            raise ValueError(
                f'only "blazegraph" and "graphdb" engines supported, but passed {self.engine}'
            )

        def _try_get_path_data(query, url):
            try:
                request = requests.get(
                    url,
                    params={"format": "json", "query": query},
                    headers={"Accept": "application/json"},
                    timeout=90,
                )
                if request.status_code >= 500 and request.status_code < 600:
                    return None

                data = request.json()

                if len(data["results"]["bindings"]) == 0:
                    return None
                else:
                    return data["results"]["bindings"]

            except (requests.exceptions.Timeout, requests.exceptions.ConnectTimeout):
                return None

            except ValueError as exception:
                logging.warning(
                    f"ValueERROR with request query:    {query}\n{str(exception)}"
                )
                logging.info("sleep 60...")
                time.sleep(60)
                return _try_get_path_data(query, url)

            except Exception as exception:
                logging.error(str(exception))
                raise Exception(exception)

        return _try_get_path_data(query, self.sparql_endpoint)

    def _rdf4j_edge_value_decode(self, rdf4j_edge):
        edge = base64.urlsafe_b64decode(rdf4j_edge.split(":")[-1])
        edge = edge.decode()[2:-2].split(" ")
        return [val[1:-1] for val in edge]


class SubgraphNodeType(str, enum.Enum):
    """SubgraphNodeType - Enum class with types of subgraphs nodes"""

    INTERNAL = "Internal"
    QUESTIONS_ENTITY = "Question entity"
    ANSWER_CANDIDATE_ENTITY = "Answer candidate entity"


class SubgraphsRetriever(WikidataBase):
    """class for extracting subgraphs given the entities and candidate"""

    def __init__(
        self,
        shortest_path: WikidataShortestPathCache = None,
        edge_between_path: bool = False,
        num_request_time: int = 3,
        lang: str = "en",
        sparql_endpoint: str = None,
        cache_dir_path: str = DEFAULT_CACHE_PATH,
    ) -> None:
        super().__init__(
            cache_dir_path, "wikidata_shortest_paths_edges.pkl", sparql_endpoint
        )
        self.cache = {}
        self.load_from_cache()
        if shortest_path is None:
            self.shortest_path = WikidataShortestPathCache()
        else:
           self.shortest_path = shortest_path
        self.edge_between_path = edge_between_path
        self.lang = lang
        self.num_request_time = num_request_time

    def get_subgraph(
        self, entities: List[str], candidate: str, number_of_pathes: Optional[int] = 10
    ):
        """Extract subgraphs given all shortest paths and candidate

        Args:
            entities (List[str]): List of question entities identifiest
            candidate (str): Identifier of answer candidate entity
            number_of_pathes (Optional[int], optional): maximum number of shortest pathes that will queried from KG
                for each pair question entity and candidate entiry.
                Defaults to None.

        Returns:
            _type_: _description_
        """
        # Query shortest pathes between entities and candidate
        pathes = []
        for entity in entities:
            e2c_pathes = self.shortest_path.get_shortest_path(
                entity,
                candidate,
                return_edges=False,
                return_only_first=False,
                return_id_only=True,
            )
            c2e_pathes = self.shortest_path.get_shortest_path(
                candidate,
                entity,
                return_edges=False,
                return_only_first=False,
                return_id_only=True,
            )

            # If pathes not exist in one way, just take other
            if e2c_pathes is None and c2e_pathes is not None:
                pathes.extend(c2e_pathes[:number_of_pathes])
                continue
            elif c2e_pathes is None and e2c_pathes is not None:
                pathes.extend(e2c_pathes[:number_of_pathes])
                continue
            elif (
                e2c_pathes is None and e2c_pathes is None
            ):  # If no shortest path for both directions
                pathes.extend([[entity, candidate]])
            else:
                # Take shortest version of pathes
                # If lengths of shortest pathes same for bouth directions, will take pathes from Question to candidate
                if len(e2c_pathes[0]) > len(c2e_pathes[0]):
                    pathes.extend(c2e_pathes[:number_of_pathes])
                else:
                    pathes.extend(e2c_pathes[:number_of_pathes])

        if self.edge_between_path is True:
            graph = self.subgraph_with_connection(pathes)
        else:
            graph = self.subgraph_without_connection(pathes)

        # Fill node attributes information
        for node in graph:
            if node == candidate:
                graph.nodes[node][
                    "node_type"
                ] = SubgraphNodeType.ANSWER_CANDIDATE_ENTITY
            elif node in entities:
                graph.nodes[node]["node_type"] = SubgraphNodeType.QUESTIONS_ENTITY
            else:
                graph.nodes[node]["node_type"] = SubgraphNodeType.INTERNAL

        return graph, pathes

    def get_distinct_nodes(self, paths):
        """
        given the said paths, return a set of distinct nodes
        """
        # distinct set of our entities in the paths
        h_vertices = set()
        for path in paths:
            for entity in path:
                h_vertices.add(entity)
        return h_vertices

    def subgraph_with_connection(self, paths):
        """
        combine the shortest paths with the connection between the paths
        """
        # get subgraph w/ no edges between
        graph_no_connection = self.subgraph_without_connection(paths)
        for idx, path in enumerate(paths):
            # getting the other paths beside the ones we're on
            if idx < len(paths):
                other_paths = paths[:idx] + paths[idx + 1 :]
            else:
                other_paths = paths[:idx]
            for entity in path:
                # distinct set of our entities in the paths
                other_paths_nodes = self.get_distinct_nodes(other_paths)
                other_paths_nodes.add(entity)

                # only fill in edges of current node with other paths' nodes
                if len(other_paths_nodes) > 1:
                    graph_no_connection = self.fill_edges_in_subgraph(
                        other_paths_nodes, graph_no_connection
                    )
        return graph_no_connection

    def fill_edges_in_subgraph(self, vertices, graph=None):
        """
        given the set of nodes, fill the edges between all the nodes
        """
        if graph is None:
            res = nx.DiGraph()
            for entity in vertices:
                res.add_node(entity)
        else:
            res = graph

        for entity in vertices:
            edges = self.get_edges(entity)

            for result in edges["results"]["bindings"]:
                neighbor_entity = result["o"]["value"].split("/")[-1]
                curr_edge = result["p"]["value"].split("/")[-1]

                if neighbor_entity in vertices:
                    if entity != neighbor_entity:
                        res.add_edge(entity, neighbor_entity, label=curr_edge)

        return res

    def subgraph_without_connection(self, paths):
        """
        combine the shortest paths without the connection between the paths
        """
        res = nx.DiGraph()
        for path in paths:
            # shortest path doesn't return the edges -> fetch the edge for the
            # current short paths
            curr_path = self.fill_edges_in_subgraph(path)
            # combine the currnet subgraph to res subgraph
            res = nx.compose(res, curr_path)
        return res

    def _request_edges(self, entity):
        try:
            # query to get properties of the current entity
            query = """
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?p ?o ?label WHERE 
            {
                BIND(wd:VALUE AS ?q)
                ?q ?p ?o .
                ?o rdfs:label ?label .
                filter( LANG(?label) = 'LANGUAGE' )
            }
            GROUP BY ?p ?o ?label
            """
            query = query.replace("VALUE", entity)
            query = query.replace("LANGUAGE", self.lang)

            request = requests.get(
                self.sparql_endpoint,
                params={"format": "json", "query": query},
                timeout=20,
                headers={"Accept": "application/json"},
            )
            edges = request.json()

            # saving the edges in the cache
            self.cache[entity] = edges
            self.save_cache()
            return edges

        except ValueError:
            logging.info("ValueError")
            return None

    def get_edges(self, entity):
        """
        fetch all of the edges for the current vertice
        """
        # entity already in cache, fetch it
        if entity in self.cache:
            return self.cache.get(entity)

        curr_tries = 0
        edges = self._request_edges(entity)

        # continue to request up to a num_request_time
        while edges is None and curr_tries < self.num_request_time:
            curr_tries += 1
            logging.info("sleep 60...")
            time.sleep(60)
            edges = self._request_edges(entity)

        edges = [] if edges is None else edges
        return edges


def plot_graph_svg(graph):
    viz_graph = graphviz.Digraph(format="svg")

    for node_idx in graph:
        type_node = graph.nodes[node_idx]['node_type']
        node_label = Entity(node_idx).label

        if type_node == SubgraphNodeType.ANSWER_CANDIDATE_ENTITY:
            viz_graph.node(f"{node_label}\n({node_idx})", color="coral", style="filled")
        elif type_node == SubgraphNodeType.QUESTIONS_ENTITY:
            viz_graph.node(f"{node_label}\n({node_idx})", color="deepskyblue", style="filled")
        else:
            viz_graph.node(f"{node_label}\n({node_idx})", color="lightgrey", style="filled")

    for edge in graph.edges:
        n1 = f"{Entity(edge[0]).label}\n({edge[0]})"
        n2 = f"{Entity(edge[1]).label}\n({edge[1]})"
        viz_graph.edge(
            n1,
            n2,
            # label=f'"{edge_label}"',
        )

    return viz_graph