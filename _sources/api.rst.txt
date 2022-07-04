.. _torchlaplace-api:

API
===

Core Functions
---------------

.. automodule:: torchlaplace


   .. autosummary::
   
      laplace_reconstruct
   
laplace_reconstruct
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: torchlaplace.laplace_reconstruct

Inverse Laplace transform algorithms
-------------------------------------

Each Inverse Laplace transform algorithm is a class that has a unified API, of three default parameters of `ilt_reconstruction_terms` (int, default of 33), `torch_float_datatype` (default of torch.float32) and  `torch_complex_datatype` (default of torch.cfloat). Note to use double precision use `torch_float_datatype=torch.double` and `torch_complex_datatype=torch.cdouble`.

.. automodule:: torchlaplace.inverse_laplace

   .. autosummary::

      InverseLaplaceTransformAlgorithmBase
      Fourier
      DeHoog
      FixedTablot
      Stehfest
      CME

InverseLaplaceTransformAlgorithmBase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: InverseLaplaceTransformAlgorithmBase
   :members:

Fourier
^^^^^^^^
.. autoclass:: Fourier
   :members:

DeHoog
^^^^^^^
.. autoclass:: DeHoog
   :members:

FixedTablot
^^^^^^^^^^^^
.. autoclass:: FixedTablot
   :members:

Stehfest
^^^^^^^^^
.. autoclass:: Stehfest
   :members:

CME
^^^^
.. autoclass:: CME
   :members:

Transformation Functions
-------------------------

.. automodule:: torchlaplace.transformations


   .. autosummary::
   
      complex_to_spherical_riemann
      spherical_riemann_to_complex
      spherical_to_complex
      complex_to_spherical

complex_to_spherical_riemann
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: torchlaplace.transformations.complex_to_spherical_riemann

spherical_riemann_to_complex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: torchlaplace.transformations.spherical_riemann_to_complex

spherical_to_complex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: torchlaplace.transformations.spherical_to_complex

complex_to_spherical
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: torchlaplace.transformations.complex_to_spherical
   