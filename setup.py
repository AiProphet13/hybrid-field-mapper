from setuptools import setup, find_packages

setup(
    name='hybrid_field_mapper',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'dash==2.6.1',
        'plotly==5.10.0',
        'pandas==1.4.3',
        'qiskit==0.37.0',
        'qiskit-ignis==0.7.0',
        'flask==2.0.1',
        'flask-login==0.5.0',
        'flask-sqlalchemy==2.5.1',
        'gunicorn==20.1.0',
        'mpi4py==3.1.3',
        'meep==1.23.0',
        'statsmodels==0.13.2',
        'networkx==2.8.5',
        'slack-sdk==3.21.0',
        'redis==4.3.4',
        'celery==5.2.7',
        'flask-cors==3.0.10',
        'flask-limiter==1.4',
        'sentry-sdk[flask]==1.9.0',
        'dash-extensions==0.0.65',
        'diskcache==5.4.0'
    ],
    author='AiProphet13',
    author_email='your.email@example.com',
    description='Quantum-electromagnetic AI propagation system',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AiProphet13/hybrid-field-mapper',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent'
    ]
)
