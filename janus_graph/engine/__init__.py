"""Engine facade package.

v0.5.0: single source of truth for knowledge-graph operations shared
between transports (stdio MCP, HTTP daemon). Currently hosts
``search_memory``; future engine modules (e.g. ``add_episode``,
``get_entity``) will land here as their MCP + HTTP transports converge.
"""
