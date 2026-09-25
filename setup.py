from setuptools import setup

extras = {
    # local MolScribe in the same process (not needed when MolScribe is reached over HTTP/Celery)
    'molscribe': ['MolScribe @ git+https://github.com/mr-racer/molscribe-custom.git@main'],
    'ocr': ['easyocr>=1.6.2'],
    'hub': ['huggingface-hub>=0.11.0'],          # only to download the default MolScribe checkpoint
    'draw': ['matplotlib>=3.5.3'],
    'client': ['requests'],                      # rxnscribe.molscribe_client.HttpMolScribe
    'train': ['pandas>=1.2.4', 'pycocotools>=2.0.4', 'pytorch-lightning>=1.8.6', 'transformers>=4.5.1',
              'matplotlib>=3.5.3'],
}
extras['all'] = sorted({req for reqs in extras.values() for req in reqs})

setup(
    name='RxnScribe',
    version='1.1',
    description='RxnScribe (custom fork: faster inference, shared/remote MolScribe)',
    author='Yujie Qian',
    author_email='yujieq@csail.mit.edu',
    url='https://github.com/mr-racer/rxnscribe-custom',
    packages=['rxnscribe', 'rxnscribe.inference', 'rxnscribe.pix2seq', 'rxnscribe.transformer'],
    package_dir={'rxnscribe': 'rxnscribe'},
    install_requires=[
        'torch',
        'torchvision',
        'numpy>=1.19.5',
        'Pillow>=9.5.0',
        'opencv-python-headless>=4.5.5.64',
    ],
    extras_require=extras,
)
