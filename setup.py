from setuptools import setup, find_packages

setup(
    name='sesa_audio_separation',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'librosa',
        'soundfile>=0.12.1',
        'gradio>=3.50.0',
        'yt_dlp',
        'gdown',
        'validators',
        'omegaconf',
        'ml-collections',
        'tqdm',
        'psutil',
        'rotary-embedding-torch',
        'noisereduce',
        'loralib',
        'hyper-connections==0.1.11',
    ],
    entry_points={
        'console_scripts': [
            'sesa-audio-separation=sesa_audio_separation.main:main',
        ],
    },
    author='Gecekondu Production',
    description='Professional Audio Source Separation Toolkit',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/test4373/sesa_audio_separation',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
