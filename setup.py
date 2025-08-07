from setuptools import find_packages, setup
import glob

package_name = 'mujoco_simulator_python'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/config',glob.glob('config/*.yaml')),
        (f'share/{package_name}/launch', glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nanhaibei',
    maintainer_email='371743175@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco_simulator_python = mujoco_simulator_python.mujoco_simulator_python:main'
        ],
    },
)
