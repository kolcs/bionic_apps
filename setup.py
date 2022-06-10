from setuptools import setup

# todo: Conda install liblsl
#  conda install -c conda-forge liblsl
#  https://niteo.co/blog/setuptools-run-custom-code-in-setup-py
#  https://stackoverflow.com/questions/36539623/how-do-i-find-the-name-of-the-conda-environment-in-which-my-code-is-running

setup(
    name='bionic-apps',
    version='',
    packages=[],
    url='',
    license='',
    author='Köllőd Csaba',
    author_email='kollod.csaba@itk.ppke.hu',
    description='Bionic Applications dependencies install',
    install_requires=['tensorflow', 'scikit-learn', 'pandas', 'matplotlib', 'pylsl', 'mne', 'numpy', 'joblib',
                      'h5py', 'scipy'],
)
