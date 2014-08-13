from distutils.core import setup

setup(
        name='Distiller',
        version='0.1.2',
        author='Francisco Canas',
        author_email='mailfrancisco@gmail.com',
        packages=[
            'Distiller',
            'Distiller.test',
            'Distiller.features',
            'Distiller.preprocessing'
        ],
        data_files=[('data', ['data/data.json'])],
        url='https://github.com/FranciscoCanas/Distiller',
        license='LICENSE.txt',
        description='Automatic Keyword Extraction from Document Collections',
        long_description=open('README.md').read(),
        install_requires=[
                "nltk >= 2.0.4"
            ]
        )
