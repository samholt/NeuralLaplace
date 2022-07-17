TorchLaplace's documentation
=========================================


**TorchLaplace** is open-source software for differentiable Laplace Reconstructions for modelling any time observation with O(1) complexity.
This library provides Inverse Laplace Transform (ILT) algorithms implemented in PyTorch.
Backpropagation through differential equation (DE) solutions in the Laplace domain is supported using the Riemann stereographic projection for better global representation of the complex Laplace domain.
For usage for DE representations in the Laplace domain in deep learning applications, see reference `[1] <https://arxiv.org/abs/2206.04843>`__ .

**Useful links**:
:ref:`Install<getting_started_ref>` |
`Source Repository <https://github.com/samholt/NeuralLaplace>`__ |
`Issues & Ideas <https://github.com/samholt/NeuralLaplace/issues>`__



.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---
    :img-top: _static/index_getting_started.png

    Getting started
    ^^^^^^^^^^^^^^^

    New to *TorchLaplace*? Check out the getting started guides. They contain an
    introduction to *TorchLaplace'* main concepts and links to additional tutorials.

    +++

    .. link-button:: getting_started
            :type: ref
            :text: To the getting started guides
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/index_user_guide.png

    User guide
    ^^^^^^^^^^

    The user guide provides in-depth information on the
    key concepts of TorchLaplace with useful background information and explanation.

    +++

    .. link-button:: notebooks/user_core
            :type: ref
            :text: To the user guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/index_api.png

    API reference
    ^^^^^^^^^^^^^

    The reference guide contains a detailed description of
    the TorchLaplace API. The reference describes how the methods work and which parameters can
    be used. It assumes that you have an understanding of the key concepts.

    +++

    .. link-button:: torchlaplace-api
            :type: ref
            :text: To the reference guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/index_contribute.png

    Developer guide
    ^^^^^^^^^^^^^^^

    Saw a typo in the documentation? Want to improve
    existing functionalities? The contributing guidelines will guide
    you through the process of improving TorchLaplace.

    +++

    .. link-button:: develop
            :type: ref
            :text: To the development guide
            :classes: btn-block btn-secondary stretched-link

Contents
--------

.. toctree::

   getting_started
   notebooks/user_core
   notebooks/user_ilt
   api
   develop

References
-----------

For usage for DE representations in the Laplace domain and leveraging the stereographic projection and other applications see:

[1] Samuel Holt, Zhaozhi Qian, and Mihaela van der Schaar. "Neural laplace: Learning diverse classes of
differential equations in the laplace domain." *International Conference on Machine Learning.* 2022. `arxiv <https://arxiv.org/abs/2206.04843>`__ 

---

If you found this library useful in your research, please consider citing.

::

   @inproceedings{holt2022neural,
    title={Neural Laplace: Learning diverse classes of differential equations in the Laplace domain},
    author={Holt, Samuel I and Qian, Zhaozhi and van der Schaar, Mihaela},
    booktitle={International Conference on Machine Learning},
    pages={8811--8832},
    year={2022},
    organization={PMLR}
   }