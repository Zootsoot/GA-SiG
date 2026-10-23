"""GA-SiG: GPU-Aided Sensitivity-in-Graph.

The package ranks the species of a detailed kinetic mechanism by the strength
of their sensitivity-based coupling to a set of target species. It has two
modes that share one integrator:

* ``idt``: the sensitivity equation is integrated in time along 0-D reactor
  trajectories up to ignition (ignition-delay-time oriented).
* ``fs``: the sensitivity equation is integrated in residence time along 1-D
  freely propagating flame profiles up to the flame tip (flame-speed oriented).

One JSON file configures a run. See :mod:`gasig.config` for the schema and
``python -m gasig template`` for a template.
"""

__version__ = "0.2.0"
