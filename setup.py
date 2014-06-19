from distutils.core import setup

setup(
        name='Distiller',
        version='0.1.0',
        author='Francisco Canas',
        author_email='mailfrancisco@gmail.com',
        packages=['distiller','distiller.src',
            'distiller.test','distiller.src.features',
            'distiller.src.preprocessors'],
        url='https://github.com/FranciscoCanas/Distiller',
        license='LICENSE.txt',
        description='Automatic Keyword Extraction from Document Collections',
        long_description=open('README.txt').read(),
        install_requires=[
                "nltk >= 2.0.4"
            ]
        )
