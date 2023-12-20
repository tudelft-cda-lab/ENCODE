from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='encode-netflow',
    version='0.2.1',    
    description='A python package for learning contextual features from (semi-)continuous NetFlow feature data, with the aim to improve machine learning models for network anomaly detection.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/tudelft-cda-lab/ENCODE',
    author='Clinton Cao',
    author_email='c.s.cao@tudelft.nl',
    license='MIT',
    packages=['encode'],
    install_requires=['pandas',
                      'numpy',
                      'scikit-learn',
                      'tqdm'                     
                      ],

    classifiers=[
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
)