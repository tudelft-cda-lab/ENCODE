from setuptools import setup

setup(
    name='encode-netflow',
    version='0.1.0',    
    description='Python package for encoding NetFlow data for use in machine learning',
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
    python_requires='>=3.8',
)