# Sparsity Based Signal Recovery for Galactic Binaries Gravitational Waves

This code was developed by [CEA-IRFU](http://irfu.cea.fr/en/Phocea/Vie_des_labos/Ast/ast_visu.php?id_ast=4698) to answer the problem of Galactic Binaries detection in the LISA Data Challenges of first generation ([LDC Website](https://lisa-ldc.lal.in2p3.fr/ldc)).

## Getting Started

* Download the content of this folder.
* Install Prerequisites.
* Download data
* Set Data folder in jupyter notebook demonstration code
* Launch notebook

### Prerequisites

* python3:
  - `numpy` (version 1.17.2)
  - `matplotlib.pyplot` (`matplotlib` version 3.1.1)
  - `scipy.stats` (`scipy` version 1.3.1)
  - `math`
* LDC code: [LDC Website](https://lisa-ldc.lal.in2p3.fr/ldc) - python modules:
  - `LISAhdf5`
  - `tdi`
  - `LISAParameters`
* Jupyter Notebook (for demonstration code)


### Download Data

The data can be found on LDC website: [LCD Website](https://lisa-ldc.lal.in2p3.fr/ldc) > Challenge 1 > Download datasets



This code is supposed to be run on the LDC1-3 (verification galactic binaries) data. Please download the files:

* LDC1-3_VGB_v2.hdf5
* LDC1-3_VGB_v2_FD_noiseless.hdf5  (noiseless version in order to compute error on solution)

## Running the tests

Run the demo jupyter notebook

## Authors
* Aurore BLELLY (aurore.blelly@cea.fr)
* Jérôme BOBIN
* Hervé MOUTARDE

For more information on the methods developed here, please refer to the corresponding article: A. Blelly, J. Bobin, H. Moutarde, *Sparsity Based Recovery of Galactic Binaries Gravitational Waves*, [arXiv:2005.03696 [gr-qc]](https://arxiv.org/abs/2005.03696).

## License

This project is licensed under the GPL License - see the [LICENSE](../LICENSE) file for details

## Acknowledgments

The authors would like to thank S.Babak, Th.Foglizzo, A.Petiteau, and the LISA Data Challenge members for inspiring discussions and warm support. This work was supported by CNES. It is based on the science case of the LISA space mission. JB was supported by the European Community through the grant LENA (ERC StG - contract no. 678282). JB would like to thank the IPARCOS institute of the Universidad Complutense de Madrid for hosting him.
