"""
ST_fitting – Standalone solar-thermal heat-distribution identification from
5-minute data using measured ST energy and observed tank temperature response.

Physics background
------------------
The solar-thermal coil is physically located in the bottom node of the
550 L DHW tank.  However, convective circulation and conduction cause
some fraction of the delivered heat to appear in higher nodes within the
same time step.  Since the grey-box model does not simulate genuine water
circulation, the ``f_st`` distribution vector captures this effective
heat spreading empirically — analogous to how ``f_ashp`` is derived in
``ASHP_fitting``.
"""
