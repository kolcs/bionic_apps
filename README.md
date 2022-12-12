[//]: # (```)

[//]: # (cd existing_repo)

[//]: # (git remote add origin https://dev.itk.ppke.hu/kolcs/mcc-flow.git)

[//]: # (git branch -M main)

[//]: # (git push -uf origin main)

[//]: # (```)


[//]: # (## Integrate with your tools)

[//]: # (- [ ] [Set up project integrations]&#40;https://dev.itk.ppke.hu/kolcs/mcc-flow/-/settings/integrations&#41;)


[//]: # (***)

# Bionic Applications

A Brain-Computer Interface and EEG & EMG signal processing tool.

## Description

This project is originally designed for the BCI discipline of the Cybathlon competition by the Ebrainers.

## Installation

1. Download git project
2. Download [miniconda](https://docs.conda.io/en/latest/miniconda.html) and install it.
3. Create a new environment called ''bci'' with python > 3.7 and < 3.11

   `conda create --name bci python=3.10`

4. activate environment

   `conda activate bci`

5. install requirements

   `pip install -r /path/to/requirements.txt`

## Public Databases

- [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/)
- [Giga](http://gigadb.org/dataset/100542)
- [BCI Competition IV 2a](https://www.bbci.de/competition/iv/)
- [TTK](https://hdl.handle.net/21.15109/CONCORDA/UOQQVK)

## Usage

Use examples can be found in folder `examples/`

## Citing

If you use this code in a scientific publication, please cite us as:

```
@misc{kollod_closed_2022,
   title = {Closed loop {BCI} System for Cybathlon 2020},
   url = {http://arxiv.org/abs/2212.04172},
   doi = {10.48550/arXiv.2212.04172},
   number = {{arXiv}:2212.04172},
   publisher = {{arXiv}},
   author = {Köllőd, Csaba and Adolf, András and Márton, Gergely and 
      Wahdow, Moutz and Fadel, Ward and Ulbert, István},
   urldate = {2022-12-09},
   date = {2022-12-08},
   eprinttype = {arxiv},
   eprint = {2212.04172 [cs, eess]},
   keywords = {Computer Science - Human-Computer Interaction, Electrical 
      Engineering and Systems Science - Signal Processing}
}
```

as well as the [MNE-Python](https://mne.tools/) software that is used by bionic_apps:

```
@article{GramfortEtAl2013a,
   title = {{{MEG}} and {{EEG}} Data Analysis with {{MNE}}-{{Python}}},
   author = {Gramfort, Alexandre and Luessi, Martin and Larson, Eric and 
      Engemann, Denis A. and Strohmeier, Daniel and Brodbeck, Christian and 
      Goj, Roman and Jas, Mainak and Brooks, Teon and Parkkonen, Lauri and 
      H{\"a}m{\"a}l{\"a}inen, Matti S.},
   year = {2013},
   volume = {7},
   pages = {1--13},
   doi = {10.3389/fnins.2013.00267},
   journal = {Frontiers in Neuroscience},
   number = {267}
}
```

[//]: # (## Contributing)

[//]: # (State if you are open to contributions and what your requirements are for accepting them.)

[//]: # ()

[//]: # (For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.)

[//]: # ()

[//]: # (You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.)

[//]: # ()

[//]: # (## Authors and acknowledgment)

[//]: # (Show your appreciation to those who have contributed to the project.)

[//]: # ()

## Licensing

Bionic Applications is BSD-licenced (BSD-3-Clause):

> This software is OSI Certified Open Source Software. OSI Certified is a certification mark of the Open Source
> Initiative.
>
>Copyright (c) 2019-2023, authors of Bionic Applications. All rights reserved.
>
>Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
> following conditions are met:
> - Redistributions of source code must retain the above copyright notice, this list of conditions and the following
    disclaimer.
> - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided with the distribution.
> - Neither the names of bionic_apps authors nor the names of any contributors may be used to endorse or promote
    products derived from this software without specific prior written permission.
>
> This software is provided by the copyright holders and contributors "as is" and any express or implied warranties,
> including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are
> disclaimed. In no event shall the copyright owner or contributors be liable for any direct, indirect, incidental,
> special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or
> services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability,
> whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of
> this software, even if advised of the possibility of such damage.


[//]: # (## Project status)

[//]: # (If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.)
