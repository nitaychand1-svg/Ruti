import asyncio
import networkx as nx
from app.modules.logging_config import logger

class Node:
    def __init__(self, name, func, depends_on=None):
        self.name = name
        self.func = func
        self.depends_on = depends_on or []

class TaskGraph:
    def __init__(self):
        self.nodes = {}
        self.graph = nx.DiGraph()

    def add(self, node):
        self.nodes[node.name] = node
        self.graph.add_node(node.name)
        for dep in node.depends_on:
            self.graph.add_edge(dep, node.name)  # dep -> node

    async def run_node(self, node_name, context, results):
        node = self.nodes[node_name]
        deps_results = {dep: results[dep] for dep in node.depends_on}
        try:
            result = await node.func(context, deps_results)
            results[node_name] = result
        except Exception as e:
            logger.error(f"Error in node {node_name}: {e}")
            raise

    async def run(self, start_nodes, context):
        results = {}
        # Topological order
        try:
            order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph has cycles")

        # Run in topological batches (parallel for same level)
        levels = list(nx.topological_generations(self.graph))
        for level in levels:
            tasks = []
            for node in level:
                if node in results: continue  # Already done if no deps
                tasks.append(self.run_node(node, context, results))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=False)

        # Filter to executed nodes
        executed = {k: results[k] for k in results if k not in start_nodes or True}
        return results
