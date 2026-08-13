"""Shared plumbing. Everything else imports from here.

    config   constants and secrets, no logic
    db       the ONLY file that talks to a database
    cache    the ONLY file that touches the network

Changing one of these changes all 13 loaders at once.
"""
