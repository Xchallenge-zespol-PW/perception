import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'g1_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy<2', 'PyYAML'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='xowski22@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'perception_node = g1_perception.perception_node:main',
            'apriltag_node = g1_perception.apriltag_node:main',
        ],
    },
)
