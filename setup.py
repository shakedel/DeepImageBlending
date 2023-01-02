from setuptools import setup, find_packages

setup(
    name='DeepImageBlending',
    version='0.0.1',
    description='',
    packages=['deep_image_blending'],
    package_dir={'deep_image_blending': '.'},
    python_requires=">=3.8",
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'Pillow',
        'matplotlib',
        'scikit-image',
        'aiohttp',
        'async-timeout',
        'opencv-python'
    ],
)
